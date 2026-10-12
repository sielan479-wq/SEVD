#!/usr/bin/env python3
"""POPE metric function; standalone scoring requires explicit files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .unified_eval_utils import read_jsonl, safe_div, write_json


SCORER_ID = "pope_unified_scorer_v1"


def compute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = false_positive = true_negative = false_negative = unknown = 0
    for row in rows:
        label = str(row.get("label", "")).lower()
        prediction = str(row.get("parsed_answer", "unknown")).lower()
        if prediction == "unknown":
            unknown += 1
        if prediction == "yes" and label == "yes":
            true_positive += 1
        elif prediction == "yes" and label == "no":
            false_positive += 1
        elif prediction == "no" and label == "no":
            true_negative += 1
        elif prediction == "no" and label == "yes":
            false_negative += 1
        elif label == "yes":
            false_negative += 1
        elif label == "no":
            false_positive += 1
    total = len(rows)
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "TP": true_positive,
        "FP": false_positive,
        "TN": true_negative,
        "FN": false_negative,
        "unknown": unknown,
        "num_samples": total,
        "accuracy": safe_div(true_positive + true_negative, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": safe_div(
            sum(1 for row in rows if row.get("parsed_answer") == "yes"), total
        ),
    }


def self_test() -> None:
    metrics = compute(
        [
            {"label": "yes", "parsed_answer": "yes"},
            {"label": "no", "parsed_answer": "no"},
            {"label": "no", "parsed_answer": "yes"},
            {"label": "yes", "parsed_answer": "unknown"},
        ]
    )
    assert metrics["TP"] == 1 and metrics["TN"] == 1
    assert metrics["FP"] == 1 and metrics["FN"] == 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-file", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("POPE scorer self-test passed")
        return
    if not args.parsed_file or not args.output_json:
        parser.error(
            "explicit --parsed-file and --output-json are required; formal evaluation "
            "must use evaluation/run_full_evaluation.py"
        )
    metrics = compute(read_jsonl(args.parsed_file))
    metrics["scorer"] = SCORER_ID
    write_json(args.output_json, metrics)
    print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()
