"""Protocol-v2 POPE: canonical prompt plus auditable binary fallback."""

from __future__ import annotations

from typing import Any

from evaluation.answer_parser import PARSER_ID
from evaluation.binary_protocol import BINARY_PROTOCOL_ID, resolve_binary_answer
from evaluation.benchmarks.pope import EXPECTED_FORMAL_PER_SPLIT, SPLITS, _summary_row
from evaluation.pope_metrics import compute as compute_pope
from evaluation.runtime import (
    PROJECT_ROOT, RunContext, append_jsonl, emit, existing_ids, finish_run,
    load_engine, materialize_records, record_input_manifest, release_engine,
    resolve_image_paths, set_status, sha256, write_intervention_diagnostics,
    write_run_manifest, write_text,
)
from evaluation.unified_eval_utils import read_jsonl, write_csv, write_json

PREDICTION_FIELDS = (
    "sample_id", "split", "image_id", "question", "label", "raw_answer",
    "method", "run_id", "stage",
)
PARSED_FIELDS = PREDICTION_FIELDS + (
    "parsed_answer", "parser_status", "parser", "binary_protocol",
    "original_raw_answer", "original_parsed_answer", "fallback_triggered",
    "fallback_prompt", "fallback_yes_score", "fallback_no_score", "fallback_answer",
)


def run(context: RunContext, limit: int | None = None) -> None:
    set_status(context, "freezing_inputs")
    frozen_inputs: list[dict[str, Any]] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        source = PROJECT_ROOT / "data" / "manifests" / "pope" / f"{split}.jsonl"
        rows = resolve_image_paths(read_jsonl(source, limit=limit), context.config, "pope")
        if context.stage == "formal" and len(rows) != EXPECTED_FORMAL_PER_SPLIT:
            raise RuntimeError(f"Formal POPE requires {EXPECTED_FORMAL_PER_SPLIT} rows for {split}; found {len(rows)}")
        rows_by_split[split] = rows
        frozen, count = record_input_manifest(context, f"{split}.jsonl", rows, source)
        frozen_inputs.append({"partition": split, "source": str(source), "source_sha256": sha256(source), "frozen": str(frozen), "frozen_sha256": sha256(frozen), "num_samples": count})
    write_run_manifest(
        context, frozen_inputs, sum(map(len, rows_by_split.values())),
        {"canonical_records": "records/{split}.jsonl", "predictions": "predictions/{split}.jsonl", "parsed": "parsed/{split}.jsonl", "traces": "traces/{split}.jsonl", "metrics": "metrics/{split}.json + metrics/macro_average.json", "summary": "summaries/summary.json"},
        parser=PARSER_ID, binary_protocol=BINARY_PROTOCOL_ID, scorer="pope_unified_scorer_v1",
    )
    set_status(context, "running_generation")
    engine = load_engine(context)
    try:
        for split, rows in rows_by_split.items():
            record_path = context.path("records", f"{split}.jsonl")
            done = existing_ids(record_path)
            for index, row in enumerate(rows, 1):
                sample_id = str(row["sample_id"])
                if sample_id in done:
                    continue
                raw_answer, traces, stats = engine.answer_pope_sample(row)
                answer, audit = resolve_binary_answer(engine, row, raw_answer, "pope")
                append_jsonl(record_path, {
                    "sample_id": sample_id, "partition": split, "split": split,
                    "image_id": row["image_id"], "question": row["question"], "label": row["label"],
                    "raw_answer": raw_answer, "parsed_answer": answer,
                    "parser_status": "ok" if answer in {"yes", "no"} else "unknown",
                    "parser": PARSER_ID, "binary_protocol": BINARY_PROTOCOL_ID,
                    "method": context.method, "run_id": context.run_id, "stage": context.stage,
                    "stats": stats, "traces": traces, **audit,
                })
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
    fallback_count = 0
    for split, expected_rows in rows_by_split.items():
        records = read_jsonl(context.path("records", f"{split}.jsonl"))
        all_records.extend(records)
        materialize_records(records, context.path("predictions", f"{split}.jsonl"), context.path("parsed", f"{split}.jsonl"), context.path("traces", f"{split}.jsonl"), PREDICTION_FIELDS, PARSED_FIELDS)
        metrics = compute_pope(read_jsonl(context.path("parsed", f"{split}.jsonl")))
        metrics.update({"method": context.method, "run_id": context.run_id, "stage": context.stage, "split": split, "parser": PARSER_ID, "binary_protocol": BINARY_PROTOCOL_ID, "scorer": "pope_unified_scorer_v1"})
        write_json(context.path("metrics", f"{split}.json"), metrics)
        parser_failures += int(metrics["unknown"])
        fallback_count += sum(bool(r.get("fallback_triggered")) for r in records)
        complete = len(records) == len(expected_rows)
        integrity_ok = integrity_ok and complete
        split_metrics.append(metrics)
        summary_rows.append(_summary_row(split, metrics, context.path("parsed", f"{split}.jsonl"), "ok" if complete else "incomplete"))
    macro = {key: sum(float(m[key]) for m in split_metrics) / len(split_metrics) for key in ("accuracy", "precision", "recall", "f1", "yes_ratio")}
    macro.update({"method": context.method, "run_id": context.run_id, "stage": context.stage, "split": "macro_average", "num_samples": sum(int(m["num_samples"]) for m in split_metrics), "num_splits": len(split_metrics), "binary_protocol": BINARY_PROTOCOL_ID, "fallback_count": fallback_count})
    write_json(context.path("metrics", "macro_average.json"), macro)
    summary_rows.append(_summary_row("macro_average", macro, context.path("parsed", "{random,popular,adversarial}.jsonl"), "ok" if integrity_ok else "incomplete"))
    write_csv(context.path("summaries", "summary.csv"), summary_rows, list(summary_rows[0]))
    write_json(context.path("summaries", "summary.json"), {"rows": summary_rows, "macro_average": macro, "integrity_ok": integrity_ok, "parser_failures": parser_failures, "fallback_count": fallback_count})
    write_text(context.path("summaries", "summary.md"), f"# Protocol-v2 POPE {context.method}\n\n- Samples: {macro['num_samples']}\n- Fallbacks: {fallback_count}\n- Parser failures: {parser_failures}\n")
    intervention = write_intervention_diagnostics(context, all_records)
    finish_run(context, intervention, scorer_ok=integrity_ok and parser_failures == 0, num_samples=len(all_records), parser_failures=parser_failures, binary_fallback_count=fallback_count)
    emit(context.log_path, "Protocol-v2 POPE flow completed")
