#!/usr/bin/env python3
"""Canonical protocol-v4 POPE/MME runner for Greedy and SEVD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import evaluation.runtime as runtime  # noqa: E402
from evaluation.benchmarks import protocol_mme, protocol_pope  # noqa: E402
from evaluation.binary_protocol_v3 import (  # noqa: E402
    LLaVABaseline,
    QwenFallback,
    QwenGreedy,
    llava_prompt,
    resolve,
)
from evaluation.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    create_context,
    default_run_id,
    prepare_run,
    set_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("llava", "qwen"), required=True)
    parser.add_argument("--benchmark", choices=("pope", "mme"), required=True)
    parser.add_argument("--method", choices=("greedy", "sccd_v1"), required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.stage == "smoke" and not (args.limit and args.limit > 0):
        raise ValueError("smoke requires a positive --limit")
    if args.stage == "formal" and (args.limit is not None or not args.confirm_full):
        raise ValueError("formal requires --confirm-full and forbids --limit")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id")


def bind_llava_loader():
    original_loader = runtime.load_engine
    import sccd.verifier as verifier
    from evaluation.method_engines import BaselineEngine

    verifier._build_unified_pope_probe = llava_prompt

    class ProtocolV4Baseline(LLaVABaseline, BaselineEngine):
        pass

    def load(context):
        if context.method == "sccd_v1":
            return original_loader(context)
        runtime.emit(context.log_path, "loading LLaVA protocol-v4 Greedy components")
        components = runtime.load_model(context.config, context.method)
        return ProtocolV4Baseline(context.config, context.method, components)

    return load


def bind_qwen_loader():
    from evaluation.qwen_engine_v2 import QwenSCCDEngineV2
    from evaluation.qwen_greedy_engine import QwenGreedyEngine
    from evaluation.runtime import frozen_method_config
    from sccd.vocabulary import ObjectVocabulary

    class ProtocolV4SEVD(QwenFallback, QwenSCCDEngineV2):
        pass

    class ProtocolV4Greedy(QwenGreedy, QwenFallback, QwenGreedyEngine):
        pass

    def load(context):
        vocabulary = ObjectVocabulary.from_synonyms_file(
            context.config["chair"]["synonym_file"]
        )
        engine_class = ProtocolV4Greedy if context.method == "greedy" else ProtocolV4SEVD
        runtime.emit(context.log_path, f"loading Qwen protocol-v4 {context.method} components")
        return engine_class(
            context.config,
            vocabulary,
            frozen_method_config(context.config),
        )

    return load


def main() -> None:
    args = parse_args()
    validate_args(args)
    loader = bind_llava_loader() if args.backend == "llava" else bind_qwen_loader()

    runtime.load_engine = loader
    protocol_pope.load_engine = loader
    protocol_mme.load_engine = loader
    protocol_pope.resolve_binary_answer = resolve
    protocol_mme.resolve_binary_answer = resolve

    context = create_context(
        args.benchmark,
        args.stage,
        args.run_id or default_run_id(),
        args.config,
        args.resume,
        args.method,
    )
    prepare_run(context)
    try:
        {"pope": protocol_pope.run, "mme": protocol_mme.run}[args.benchmark](
            context, args.limit
        )
    except BaseException as error:
        set_status(context, "failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
