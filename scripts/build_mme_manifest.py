#!/usr/bin/env python3
"""Create the untracked MME manifest from a local official benchmark copy."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from evaluation.mme_manifest import TASKS, build_mme_rows, write_mme_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an untracked 2,374-row MME manifest from local data."
    )
    parser.add_argument(
        "--mme-root",
        type=Path,
        default=Path(os.environ["MME_ROOT"]) if "MME_ROOT" in os.environ else None,
        help="Official MME_Benchmark directory (default: $MME_ROOT).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "manifests" / "mme" / "all.jsonl",
        help="Generated local manifest; ignored by Git by default.",
    )
    args = parser.parse_args()
    if args.mme_root is None:
        parser.error("provide --mme-root or set MME_ROOT")
    rows = build_mme_rows(args.mme_root)
    if len(rows) != 2374:
        raise RuntimeError(
            f"Expected 2,374 MME records across {len(TASKS)} tasks; found {len(rows)}"
        )
    count = write_mme_manifest(rows, args.output)
    print(f"Wrote {count} MME records to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
