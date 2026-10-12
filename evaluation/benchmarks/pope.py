"""POPE generation, parsing, scoring, and intervention diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.answer_parser import PARSER_ID, parse_answer
from evaluation.pope_metrics import compute as compute_pope
from evaluation.runtime import (
    PROJECT_ROOT,
    RunContext,
    append_jsonl,
    emit,
    existing_ids,
    finish_run,
    load_engine,
    materialize_records,
    record_input_manifest,
    release_engine,
    resolve_image_paths,
    set_status,
    sha256,
    write_intervention_diagnostics,
    write_run_manifest,
    write_text,
)
from evaluation.unified_eval_utils import read_jsonl, write_csv, write_json


SPLITS = ("random", "popular", "adversarial")
EXPECTED_FORMAL_PER_SPLIT = 3000
PREDICTION_FIELDS = (
    "sample_id",
    "split",
    "image_id",
    "question",
    "label",
    "raw_answer",
    "method",
    "run_id",
    "stage",
)
PARSED_FIELDS = PREDICTION_FIELDS + ("parsed_answer", "parser_status", "parser")


def _summary_row(
    split: str, metrics: dict[str, Any], parsed_path: Path, status: str
) -> dict[str, Any]:
    return {
        "method": metrics.get("method", ""),
        "split": split,
        "accuracy": metrics.get("accuracy", ""),
        "precision": metrics.get("precision", ""),
        "recall": metrics.get("recall", ""),
        "f1": metrics.get("f1", ""),
        "yes_ratio": metrics.get("yes_ratio", ""),
        "num_samples": metrics.get("num_samples", ""),
        "parsed_file": str(parsed_path),
        "status": status,
    }


def run(context: RunContext, limit: int | None = None) -> None:
    set_status(context, "freezing_inputs")
    frozen_inputs: list[dict[str, Any]] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        source = PROJECT_ROOT / "data" / "manifests" / "pope" / f"{split}.jsonl"
        rows = resolve_image_paths(
            read_jsonl(source, limit=limit), context.config, "pope"
        )
        if context.stage == "formal" and len(rows) != EXPECTED_FORMAL_PER_SPLIT:
            raise RuntimeError(
                f"Formal POPE requires {EXPECTED_FORMAL_PER_SPLIT} rows for {split}; "
                f"found {len(rows)}"
            )
        rows_by_split[split] = rows
        frozen, count = record_input_manifest(context, f"{split}.jsonl", rows, source)
        frozen_inputs.append(
            {
                "partition": split,
                "source": str(source),
                "source_sha256": sha256(source),
                "frozen": str(frozen),
                "frozen_sha256": sha256(frozen),
                "num_samples": count,
            }
        )

    write_run_manifest(
        context,
        frozen_inputs,
        sum(len(rows) for rows in rows_by_split.values()),
        {
            "canonical_records": "records/{split}.jsonl",
            "predictions": "predictions/{split}.jsonl",
            "parsed": "parsed/{split}.jsonl",
            "traces": "traces/{split}.jsonl",
            "metrics": "metrics/{split}.json + metrics/macro_average.json",
            "diagnostics": "diagnostics/intervention_summary.json",
            "summary": "summaries/summary.json",
        },
        parser=PARSER_ID,
        scorer="pope_unified_scorer_v1",
        aggregation="macro average across random/popular/adversarial",
    )

    set_status(context, "running_generation")
    engine = load_engine(context)
    try:
        for split in SPLITS:
            rows = rows_by_split[split]
            record_path = context.path("records", f"{split}.jsonl")
            done = existing_ids(record_path)
            for index, row in enumerate(rows, 1):
                sample_id = str(row["sample_id"])
                if sample_id in done:
                    continue
                raw_answer, traces, stats = engine.answer_pope_sample(row)
                parsed_answer = parse_answer(raw_answer)
                append_jsonl(
                    record_path,
                    {
                        "sample_id": sample_id,
                        "partition": split,
                        "split": split,
                        "image_id": row["image_id"],
                        "question": row["question"],
                        "label": row["label"],
                        "raw_answer": raw_answer,
                        "parsed_answer": parsed_answer,
                        "parser_status": "ok" if parsed_answer in {"yes", "no"} else "unknown",
                        "parser": PARSER_ID,
                        "method": context.method,
                        "run_id": context.run_id,
                        "stage": context.stage,
                        "stats": stats,
                        "traces": traces,
                    },
                )
                done.add(sample_id)
                if index == 1 or index % 100 == 0 or index == len(rows):
                    emit(context.log_path, f"POPE {split}: {index}/{len(rows)}")
    finally:
        release_engine(engine)

    set_status(context, "materializing_and_scoring")
    all_records: list[dict[str, Any]] = []
    split_metrics: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    integrity_ok = True
    parser_failures = 0
    for split in SPLITS:
        records = read_jsonl(context.path("records", f"{split}.jsonl"))
        all_records.extend(records)
        prediction_path = context.path("predictions", f"{split}.jsonl")
        parsed_path = context.path("parsed", f"{split}.jsonl")
        materialize_records(
            records,
            prediction_path,
            parsed_path,
            context.path("traces", f"{split}.jsonl"),
            PREDICTION_FIELDS,
            PARSED_FIELDS,
        )
        parsed_rows = read_jsonl(parsed_path)
        metrics = compute_pope(parsed_rows)
        metrics.update(
            {
                "method": context.method,
                "run_id": context.run_id,
                "stage": context.stage,
                "split": split,
                "parser": PARSER_ID,
                "scorer": "pope_unified_scorer_v1",
            }
        )
        write_json(context.path("metrics", f"{split}.json"), metrics)
        parser_failures += int(metrics["unknown"])
        split_metrics.append(metrics)
        complete = len(records) == len(rows_by_split[split])
        integrity_ok = integrity_ok and complete
        summary_rows.append(
            _summary_row(split, metrics, parsed_path, "ok" if complete else "incomplete")
        )

    macro = {
        key: sum(float(metrics[key]) for metrics in split_metrics) / len(split_metrics)
        for key in ("accuracy", "precision", "recall", "f1", "yes_ratio")
    }
    macro.update(
        {
            "method": context.method,
            "run_id": context.run_id,
            "stage": context.stage,
            "split": "macro_average",
            "num_samples": sum(int(metrics["num_samples"]) for metrics in split_metrics),
            "num_splits": len(split_metrics),
        }
    )
    write_json(context.path("metrics", "macro_average.json"), macro)
    summary_rows.append(
        _summary_row(
            "macro_average",
            macro,
            context.path("parsed", "{random,popular,adversarial}.jsonl"),
            "ok" if integrity_ok else "incomplete",
        )
    )
    write_csv(
        context.path("summaries", "summary.csv"),
        summary_rows,
        list(summary_rows[0].keys()),
    )
    write_json(
        context.path("summaries", "summary.json"),
        {
            "rows": summary_rows,
            "macro_average": macro,
            "integrity_ok": integrity_ok,
            "parser_failures": parser_failures,
        },
    )
    write_text(
        context.path("summaries", "summary.md"),
        "# POPE SCCD Run Summary\n\n"
        f"- Run ID: `{context.run_id}`\n"
        f"- Stage: `{context.stage}`\n"
        f"- Samples: {macro['num_samples']}\n"
        f"- Accuracy: {macro['accuracy']:.6f}\n"
        f"- Precision: {macro['precision']:.6f}\n"
        f"- Recall: {macro['recall']:.6f}\n"
        f"- F1: {macro['f1']:.6f}\n"
        f"- Parser failures: {parser_failures}\n"
        f"- Integrity: {'passed' if integrity_ok else 'failed'}\n",
    )
    intervention = write_intervention_diagnostics(context, all_records)
    score_valid = integrity_ok and (context.stage == "smoke" or parser_failures == 0)
    finish_run(
        context,
        intervention,
        scorer_ok=score_valid,
        num_samples=len(all_records),
        parser_failures=parser_failures,
    )
    emit(context.log_path, "POPE flow completed")
