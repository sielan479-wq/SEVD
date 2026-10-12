#!/usr/bin/env python3
"""Shared yes/no parser; standalone use requires explicit input/output files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .unified_eval_utils import read_jsonl, write_jsonl


PARSER_ID = "unified_yes_no_parser_v1"
NEG_PATTERNS = (
    r"\bno\b",
    r"\bnot\b",
    r"\bnone\b",
    r"\bwithout\b",
    r"there\s+(is|are)\s+no\b",
    r"does\s+not\s+(show|contain|include)",
    r"do\s+not\s+(see|show)",
    r"isn['’]?t\b",
    r"aren['’]?t\b",
)
POS_PATTERNS = (
    r"\byes\b",
    r"\byeah\b",
    r"\byep\b",
    r"there\s+(is|are)\b",
    r"\bcontains?\b",
    r"\bvisible\b",
    r"\bappears?\b",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def parse_answer(raw_answer: str) -> str:
    text = normalize(raw_answer)
    if not text:
        return "unknown"
    first = re.sub(r"^[^a-z]+", "", text).split(" ", 1)[0]
    if first in {"yes", "yeah", "yep"}:
        return "yes"
    if first in {"no", "nope", "not", "none"}:
        return "no"
    if any(re.search(pattern, text) for pattern in NEG_PATTERNS):
        return "no"
    if any(re.search(pattern, text) for pattern in POS_PATTERNS):
        return "yes"
    return "unknown"


def parse_file(
    manifest_file: Path,
    answer_file: Path,
    output_file: Path,
    method: str,
) -> int:
    manifest = {str(row["sample_id"]): row for row in read_jsonl(manifest_file)}
    output: list[dict[str, Any]] = []
    for answer in read_jsonl(answer_file):
        sample_id = str(answer.get("sample_id"))
        source = manifest.get(sample_id, {})
        raw = answer.get("raw_answer", answer.get("raw_response", answer.get("text", "")))
        output.append(
            {
                "sample_id": sample_id,
                "split": answer.get("split", source.get("split", "")),
                "image_id": answer.get("image_id", source.get("image_id", "")),
                "question": answer.get("question", source.get("question", "")),
                "label": answer.get("label", source.get("label", "")),
                "raw_answer": raw,
                "parsed_answer": parse_answer(raw),
                "method": method,
                "parser": PARSER_ID,
            }
        )
    write_jsonl(output_file, output)
    return len(output)


def self_test() -> None:
    cases = {
        "Yes, there is a dog.": "yes",
        "No, there is no dog.": "no",
        "The image is unclear.": "unknown",
        "not visible": "no",
        "There are two people.": "yes",
    }
    for raw, expected in cases.items():
        assert parse_answer(raw) == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-file", type=Path)
    parser.add_argument("--answer-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--method", default="sccd_v1")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("answer parser self-test passed")
        return
    if not all((args.manifest_file, args.answer_file, args.output_file)):
        parser.error(
            "explicit --manifest-file, --answer-file, and --output-file are required; "
            "formal evaluation must use evaluation/run_full_evaluation.py"
        )
    count = parse_file(
        args.manifest_file, args.answer_file, args.output_file, args.method
    )
    print(f"parsed {count} answers -> {args.output_file}")


if __name__ == "__main__":
    main()
