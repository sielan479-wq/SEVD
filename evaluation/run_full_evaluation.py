#!/usr/bin/env python3
"""Single CLI entry for isolated SCCD smoke or formal evaluation runs.

This module intentionally contains no benchmark logic. It validates run intent,
creates one run context per benchmark, and dispatches to evaluation/benchmarks/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.benchmarks import RUNNERS  # noqa: E402
from evaluation.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    create_context,
    default_run_id,
    prepare_run,
    set_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=[*RUNNERS, "all"], required=True
    )
    parser.add_argument("--method", choices=["greedy", "sccd_v1"], default="sccd_v1")
    parser.add_argument(
        "--stage",
        choices=["smoke", "formal"],
        required=True,
        help="smoke writes diagnostic-only runs; formal writes publication candidates",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="stable ID shared by all selected benchmarks; required with --resume",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "smoke-only cap per logical partition: POPE split, CHAIR all, "
            "MME task, or AMBER dimension"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--resume", action="store_true", help="continue exactly the supplied run ID"
    )
    parser.add_argument(
        "--confirm-full",
        action="store_true",
        help="required guard for a formal full run",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="print resolved run directories without creating files or loading a model",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    fail = parser.error if parser is not None else lambda message: (_ for _ in ()).throw(ValueError(message))
    if args.resume and not args.run_id:
        fail("--resume requires an explicit --run-id")
    if args.limit is not None and args.limit <= 0:
        fail("--limit must be a positive integer")
    if args.stage == "smoke" and args.limit is None:
        fail("--stage smoke requires --limit")
    if args.stage == "formal" and args.limit is not None:
        fail("--stage formal forbids --limit; formal runs must use the full manifest")
    if args.stage == "formal" and not args.confirm_full and not args.print_plan:
        fail("--stage formal requires --confirm-full")


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_id = args.run_id or default_run_id()
    selected = tuple(RUNNERS) if args.benchmark == "all" else (args.benchmark,)
    contexts = [
        create_context(
            benchmark=benchmark,
            stage=args.stage,
            run_id=run_id,
            config_path=args.config,
            resume=args.resume,
            method=args.method,
        )
        for benchmark in selected
    ]

    if args.print_plan:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "stage": args.stage,
                    "benchmarks": [
                        {"name": context.benchmark, "run_root": str(context.run_root)}
                        for context in contexts
                    ],
                    "will_execute": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    for context in contexts:
        prepare_run(context)
        try:
            RUNNERS[context.benchmark](context, args.limit)
        except BaseException as error:
            set_status(
                context,
                "failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            raise


if __name__ == "__main__":
    main()
