#!/usr/bin/env python3
"""Minimal public entry point for Greedy/SEVD evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY_BENCHMARKS = ("pope", "mme")
FREE_FORM_BENCHMARKS = ("chair", "amber")
ALL_BENCHMARKS = BINARY_BENCHMARKS + FREE_FORM_BENCHMARKS

sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from evaluation.runtime import create_context, default_run_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("llava", "qwen"), required=True)
    parser.add_argument("--benchmark", choices=(*ALL_BENCHMARKS, "all"), required=True)
    parser.add_argument("--method", choices=("greedy", "sevd"), default="sevd")
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.resume and not args.run_id:
        raise ValueError("--resume requires an explicit --run-id")
    if args.stage == "smoke" and not (args.limit and args.limit > 0):
        raise ValueError("smoke requires a positive --limit")
    if args.stage == "formal" and args.limit is not None:
        raise ValueError("formal forbids --limit")
    if args.stage == "formal" and not (args.confirm_full or args.print_plan):
        raise ValueError("formal requires --confirm-full")
    if args.backend == "qwen" and args.benchmark in FREE_FORM_BENCHMARKS:
        raise ValueError("the released Qwen adapter is validated only for POPE/MME")


def child_command(args: argparse.Namespace, benchmark: str, run_id: str, config: Path) -> list[str]:
    method = "sccd_v1" if args.method == "sevd" else "greedy"
    if benchmark in BINARY_BENCHMARKS:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_binary_protocol_evaluation_v4.py"),
            "--backend", args.backend,
            "--benchmark", benchmark,
            "--method", method,
            "--stage", args.stage,
            "--run-id", run_id,
            "--config", str(config),
        ]
    else:
        if args.backend != "llava":
            raise ValueError(f"{benchmark} is not released for the Qwen adapter")
        command = [
            sys.executable,
            str(ROOT / "evaluation" / "run_full_evaluation.py"),
            "--benchmark", benchmark,
            "--method", method,
            "--stage", args.stage,
            "--run-id", run_id,
            "--config", str(config),
        ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    if args.resume:
        command.append("--resume")
    if args.stage == "formal":
        command.append("--confirm-full")
    return command


def main() -> None:
    args = parse_args()
    validate(args)
    config = args.config or ROOT / "configs" / (
        "sevd_llava15_7b.yaml" if args.backend == "llava" else "sevd_qwen25vl_7b.yaml"
    )
    config = config.expanduser().resolve()
    run_id = args.run_id or default_run_id()
    supported = ALL_BENCHMARKS if args.backend == "llava" else BINARY_BENCHMARKS
    selected = supported if args.benchmark == "all" else (args.benchmark,)
    commands = [child_command(args, benchmark, run_id, config) for benchmark in selected]

    if args.print_plan:
        method = "sccd_v1" if args.method == "sevd" else "greedy"
        contexts = [
            create_context(benchmark, args.stage, run_id, config, args.resume, method)
            for benchmark in selected
        ]
        print(json.dumps({
            "run_id": run_id,
            "backend": args.backend,
            "method": args.method,
            "runs": [
                {"benchmark": item.benchmark, "run_root": str(item.run_root)}
                for item in contexts
            ],
            "commands": commands,
            "will_execute": False,
        }, indent=2))
        return

    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
