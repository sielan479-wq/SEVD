"""Protocol-v2 MME: canonical binary prompts and auditable fallback records."""

from __future__ import annotations

from evaluation.answer_parser import PARSER_ID
from evaluation.binary_protocol import BINARY_PROTOCOL_ID, resolve_binary_answer
from evaluation.benchmarks.mme import (
    EXPECTED_FORMAL_SAMPLES, TASKS, _score, _write_intervention_gate,
)
from evaluation.external_scorers import resolve_external_scorer
from evaluation.runtime import (
    PROJECT_ROOT, RunContext, append_jsonl, emit, ensure, existing_ids, finish_run,
    load_engine, materialize_records, record_input_manifest, release_engine,
    resolve_image_paths, set_status, sha256, write_intervention_diagnostics,
    write_run_manifest, write_text,
)
from evaluation.unified_eval_utils import read_jsonl, write_json
from sccd.binary_prompt import build_llava_binary_prompt

PREDICTION_FIELDS = (
    "sample_id", "task", "image_id", "question", "ground_truth", "raw_response",
    "method", "run_id", "stage", "generation_path", "method_scope_supported",
)
PARSED_FIELDS = PREDICTION_FIELDS + (
    "normalized_answer", "parser_status", "parser", "binary_protocol",
    "original_raw_answer", "original_parsed_answer", "fallback_triggered",
    "fallback_prompt", "fallback_yes_score", "fallback_no_score", "fallback_answer",
)


def run(context: RunContext, limit: int | None = None) -> None:
    source = PROJECT_ROOT / "data" / "manifests" / "mme" / "all.jsonl"
    if not source.is_file():
        raise FileNotFoundError(
            "MME questions and labels are not redistributed. Build the local manifest "
            "from your authorized MME copy with: python scripts/build_mme_manifest.py "
            "--mme-root \"$MME_ROOT\""
        )
    by_task = {task: [] for task in TASKS}
    for row in read_jsonl(source):
        if str(row["task"]) in by_task:
            by_task[str(row["task"])].append(row)
    selected_by_task = {
        task: resolve_image_paths(
            rows[:limit] if limit is not None else rows, context.config, "mme"
        )
        for task, rows in by_task.items()
    }
    selected = [row for task in TASKS for row in selected_by_task[task]]
    if context.stage == "formal" and len(selected) != EXPECTED_FORMAL_SAMPLES:
        raise RuntimeError(f"Formal MME requires {EXPECTED_FORMAL_SAMPLES} rows; found {len(selected)}")
    frozen, count = record_input_manifest(context, "all.jsonl", selected, source)
    scorer = resolve_external_scorer("mme", required=context.stage == "formal")
    write_run_manifest(
        context, [{"partition": "14_tasks", "source": str(source), "source_sha256": sha256(source), "frozen": str(frozen), "frozen_sha256": sha256(frozen), "num_samples": count}], count,
        {"canonical_records": "records/{task}.jsonl", "predictions": "predictions/{task}.jsonl", "parsed": "parsed/{task}.jsonl", "official_scorer_inputs": "parsed/official_inputs/{task}.txt", "traces": "traces/{task}.jsonl", "metrics": "metrics/overall.json", "summary": "summaries/summary.json"},
        parser=PARSER_ID, binary_protocol=BINARY_PROTOCOL_ID,
        scorer=str(scorer) if scorer is not None else None,
        scorer_source="external_not_redistributed",
        scorer_environment_variable="SEVD_MME_SCORER_PATH",
        scorer_sha256=sha256(scorer) if scorer is not None else None,
        tasks=list(TASKS), max_new_tokens=8,
    )
    set_status(context, "running_generation")
    engine = load_engine(context)
    try:
        for task in TASKS:
            rows = selected_by_task[task]
            record_path = context.path("records", f"{task}.jsonl")
            done = existing_ids(record_path)
            for index, row in enumerate(rows, 1):
                sample_id = str(row["sample_id"])
                if sample_id in done:
                    continue
                supported = task == "existence"
                if supported:
                    raw, traces, stats = engine.answer_object_existence_sample(row, benchmark="mme")
                    generation_path = "input_object_existence"
                else:
                    raw, traces, stats = engine.generate_chair_sample(row, build_llava_binary_prompt(row["question"]), 8, benchmark="mme")
                    generation_path = "free_decode_unsupported_claim_type"
                answer, audit = resolve_binary_answer(engine, row, raw, "mme")
                append_jsonl(record_path, {
                    "sample_id": sample_id, "partition": task, "task": task, "image_id": row["image_id"], "question": row["question"], "ground_truth": row["ground_truth"],
                    "raw_response": raw, "normalized_answer": answer, "parser_status": "ok" if answer in {"yes", "no"} else "other", "parser": PARSER_ID, "binary_protocol": BINARY_PROTOCOL_ID,
                    "method": context.method, "run_id": context.run_id, "stage": context.stage, "generation_path": generation_path, "method_scope_supported": supported, "stats": stats, "traces": traces, **audit,
                })
                if index == 1 or index % 50 == 0 or index == len(rows):
                    emit(context.log_path, f"MME {task}: {index}/{len(rows)}")
    finally:
        release_engine(engine)
    set_status(context, "materializing_and_scoring")
    all_records = []
    integrity_ok = True
    parser_failures = fallback_count = 0
    for task in TASKS:
        records = read_jsonl(context.path("records", f"{task}.jsonl"))
        all_records.extend(records)
        materialize_records(records, context.path("predictions", f"{task}.jsonl"), context.path("parsed", f"{task}.jsonl"), context.path("traces", f"{task}.jsonl"), PREDICTION_FIELDS, PARSED_FIELDS)
        integrity_ok = integrity_ok and len(records) == len(selected_by_task[task])
        parser_failures += sum(r.get("normalized_answer") not in {"yes", "no"} for r in records)
        fallback_count += sum(bool(r.get("fallback_triggered")) for r in records)
        official_path = context.path("parsed", "official_inputs", f"{task}.txt")
        ensure(official_path.parent)
        with official_path.open("w", encoding="utf-8") as stream:
            for row in records:
                prediction = row["normalized_answer"] if row.get("normalized_answer") in {"yes", "no"} else "other"
                stream.write(f"{row.get('image_id')}\t{row.get('question')}\t{row.get('ground_truth')}\t{prediction}\n")
    metrics, scorer_ok = _score(context)
    intervention = write_intervention_diagnostics(context, all_records)
    gate = _write_intervention_gate(context, intervention, all_records)
    summary = {"method": context.method, "run_id": context.run_id, "stage": context.stage, "num_samples": len(all_records), "num_tasks": sum(bool(selected_by_task[t]) for t in TASKS), "parser_failures": parser_failures, "fallback_count": fallback_count, "integrity_ok": integrity_ok, "metrics": metrics, "intervention_gate": gate, "binary_protocol": BINARY_PROTOCOL_ID}
    write_json(context.path("summaries", "summary.json"), summary)
    write_text(context.path("summaries", "summary.md"), f"# Protocol-v2 MME {context.method}\n\n- Samples: {len(all_records)}\n- Fallbacks: {fallback_count}\n- Parser failures: {parser_failures}\n")
    finish_run(context, intervention, scorer_ok=scorer_ok and integrity_ok and parser_failures == 0, num_samples=len(all_records), parser_failures=parser_failures, binary_fallback_count=fallback_count, mme_intervention_gate_passed=gate["passed"])
    emit(context.log_path, "Protocol-v2 MME flow completed")
