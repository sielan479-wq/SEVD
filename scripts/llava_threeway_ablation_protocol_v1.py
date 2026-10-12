#!/usr/bin/env python3
"""Independent protocol-v4 POPE entry for the three paper ablations."""
from __future__ import annotations
import argparse, sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
p=argparse.ArgumentParser();p.add_argument('--ablation',choices=('token_level_contrast','real_only','suppression_only'),required=True);p.add_argument('--stage',choices=('smoke','formal'),required=True);p.add_argument('--run-id',required=True);p.add_argument('--limit',type=int);p.add_argument('--config',type=Path,default=ROOT/'configs'/'sccd_llava15.yaml');p.add_argument('--resume',action='store_true');p.add_argument('--confirm-full',action='store_true');a=p.parse_args()
if a.stage=='smoke' and not(a.limit and a.limit>0):p.error('smoke requires positive --limit')
if a.stage=='formal' and (a.limit is not None or not a.confirm_full):p.error('formal requires --confirm-full and forbids --limit')
import evaluation.runtime as runtime
import sccd.verifier as verifier
from evaluation.benchmarks import protocol_pope
from evaluation.binary_protocol_v3 import llava_prompt, resolve
from evaluation.model_loader import load_model
from sccd.engine import SCCDEngine
from sccd.threeway_ablation import SemanticEvidenceAblationVerifier,TokenLevelContrastBinaryEngine
from sccd.vocabulary import ObjectVocabulary
verifier._build_unified_pope_probe=llava_prompt
original_create=runtime.create_context
def create(*x,**k):
 c=original_create(*x,**k);return replace(c,config={**c.config,'sccd_v1':{**c.config['sccd_v1'],'paper_ablation':a.ablation}})
def load(context):
 runtime.emit(context.log_path,f'loading LLaVA three-way ablation={a.ablation}')
 components=load_model(context.config);vocab=ObjectVocabulary.from_synonyms_file(context.config['chair']['synonym_file']);base=SCCDEngine(context.config,components,components[3],vocab,runtime.frozen_method_config(context.config))
 if a.ablation=='token_level_contrast':return TokenLevelContrastBinaryEngine(base)
 base.verifier=SemanticEvidenceAblationVerifier(base.verifier,a.ablation);return base
runtime.create_context=create;protocol_pope.load_engine=load;protocol_pope.resolve_binary_answer=resolve
context=create('pope',a.stage,a.run_id,a.config,a.resume,'sccd_v1');runtime.prepare_run(context)
try:protocol_pope.run(context,a.limit)
except BaseException as e:runtime.set_status(context,'failed',error_type=type(e).__name__,error=str(e));raise
