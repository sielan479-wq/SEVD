"""Qwen backend with a fixed visual-token budget for 24GB GPUs."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from PIL import Image, ImageFilter
from sccd.object_claim import object_from_existence_question
from sccd.risk_gate import decide_action

class QwenSCCDEngineV2:
    max_image_pixels = 1280 * 28 * 28
    def __init__(self, config: dict[str, Any], vocab: Any, gate: Any) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.config,self.vocab,self.gate=config,vocab,gate; self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(config['model_path']),torch_dtype=torch.bfloat16 if self.device=='cuda' else torch.float32,device_map='auto' if self.device=='cuda' else None,low_cpu_mem_usage=True).eval()
        self.processor=AutoProcessor.from_pretrained(str(config['model_path']),use_fast=False); self.tokenizer=self.processor.tokenizer
        self.yes_ids=self._ids(gate.yes_verbalizers); self.no_ids=self._ids(gate.no_verbalizers)
    def _ids(self, words):
        out=[]
        for word in words:
            ids=self.tokenizer(' '+word,add_special_tokens=False).input_ids
            if ids and ids[-1] not in out: out.append(ids[-1])
        return tuple(out)
    def _bounded(self,image):
        w,h=image.size
        if w*h<=self.max_image_pixels:return image
        s=(self.max_image_pixels/(w*h))**.5
        return image.resize((max(1,round(w*s)),max(1,round(h*s))))
    def _inputs(self,question,image):
        content=([{'type':'image'},{'type':'text','text':question}] if image is not None else [{'type':'text','text':question}])
        text=self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        kw={'text':[text],'padding':True,'return_tensors':'pt'}
        if image is not None:kw['images']=[self._bounded(image)]
        return self.processor(**kw).to(self.device)
    def _margin(self,question,image):
        import torch
        inputs=self._inputs(question,image)
        with torch.inference_mode(): logits=self.model(**inputs,use_cache=False).logits[:,-1,:].float().squeeze(0)
        lp=torch.log_softmax(logits,-1); yes=torch.logsumexp(lp[list(self.yes_ids)],0).item(); no=torch.logsumexp(lp[list(self.no_ids)],0).item()
        return float(yes),float(no),float(yes-no)
    def answer_object_existence_sample(self,row,benchmark):
        t=time.time(); claim=object_from_existence_question(str(row.get('question','')),self.vocab); stats={'input_claim_route_count':1,'input_claim_parse_failure_count':0,'verifier_forward_count':0,'unique_claims_scored':0,'changed_step_count':0,'generated_token_count':1,'exception_count':0}
        if claim is None: stats['input_claim_parse_failure_count']=1;stats['runtime_seconds']=time.time()-t;return 'unknown',[],stats
        image=Image.open(str(row['image_path'])).convert('RGB'); q=claim.probe_question+' Please answer yes or no.'
        yr,nr,real=self._margin(q,image); yb,nb,blur=self._margin(q,image.filter(ImageFilter.GaussianBlur(radius=float(self.gate.blur_radius)))); yp,np,prior=self._margin(q,None)
        d=decide_action(real,blur,prior,self.gate); base='yes' if real>0 else 'no'; ans='yes' if d.action=='promote' or (d.action=='keep' and real>0) else 'no'
        stats.update({'verifier_forward_count':3,'unique_claims_scored':1,'changed_step_count':int(ans!=base),'runtime_seconds':time.time()-t})
        tr={'sample_id':str(row.get('sample_id')),'benchmark':benchmark,'claim_source':'input_question','claim_type':'object_existence','generation_path':'input_object_existence','canonical_object':claim.canonical_object,'probe_question':claim.probe_question,'logp_yes_real':yr,'logp_no_real':nr,'logp_yes_blur':yb,'logp_no_blur':nb,'logp_yes_black':yp,'logp_no_black':np,'margin_real':real,'margin_blur':blur,'margin_black':prior,'necessity':d.necessity,'prior_risk':d.prior_risk,'fs':d.fs,'evidence_score':d.evidence_score,'action':d.action,'delta':d.delta,'base_answer':base,'final_answer':ans,'selected_token_before':base,'selected_token_after':ans,'output_changed':ans!=base}
        return ans,[tr],stats
    def answer_pope_sample(self,row): return self.answer_object_existence_sample(row,'pope')
    def generate_chair_sample(self,row,prompt,max_new_tokens,benchmark='chair'):
        import torch
        started=time.time(); image=Image.open(str(row['image_path'])).convert('RGB'); question=prompt.replace('USER: <image>\n','').replace('\nASSISTANT:',''); inputs=self._inputs(question,image)
        with torch.inference_mode(): out=self.model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens)
        answer=self.processor.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return answer,[],{'generated_token_count':len(out[0])-len(inputs.input_ids[0]),'changed_step_count':0,'runtime_seconds':time.time()-started,'verifier_forward_count':0}
