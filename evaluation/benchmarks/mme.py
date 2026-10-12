"""MME generation, official scoring, and mandatory intervention audit."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation.external_scorers import resolve_external_scorer
from evaluation.answer_parser import PARSER_ID, parse_answer
from evaluation.runtime import (
    PROJECT_ROOT,
    RunContext,
    append_jsonl,
    emit,
    ensure,
    existing_ids,
    finish_run,
    load_engine,
    materialize_records,
    record_input_manifest,
    release_engine,
    set_status,
    sha256,
    write_intervention_diagnostics,
    write_run_manifest,
    write_text,
)
from evaluation.unified_eval_utils import read_jsonl, write_json


PERCEPTION_TASKS = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
)
COGNITION_TASKS = (
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)
TASKS = PERCEPTION_TASKS + COGNITION_TASKS
EXPECTED_FORMAL_SAMPLES = 2374
PREDICTION_FIELDS = (
    "sample_id",
    "task",
    "image_id",
    "question",
    "ground_truth",
    "raw_response",
    "method",
    "run_id",
    "stage",
    "generation_path",
    "method_scope_supported",
)
PARSED_FIELDS = PREDICTION_FIELDS + (
    "normalized_answer",
    "parser_status",
    "parser",
)


def parse_mme_log(text: str) -> dict[str, Any]:
    section: str | None = None
    totals: dict[str, float] = {}
    task_scores: dict[str, float] = {}
    for line in text.splitlines():
        if "Perception" in line and "====" in line:
            section = "perception"
        elif "Cognition" in line and "====" in line:
            section = "cognition"
        total_match = re.search(r"total\s+score:\s*([0-9.eE+-]+)", line)
        if total_match and section:
            totals[section] = float(total_match.group(1))
        task_match = re.search(
            r"^\s*([A-Za-z_]+)\s+score:\s*([0-9.eE+-]+)", line
        )
        if task_match and task_match.group(1) != "total":
            task_scores[task_match.group(1)] = float(task_match.group(2))
    return {
        "perception_total": totals.get("perception"),
        "cognition_total": totals.get("cognition"),
        "overall_total": (
            totals["perception"] + totals["cognition"]
            if "perception" in totals and "cognition" in totals
            else None
        ),
        "task_scores": task_scores,
        "raw_log_excerpt": text[-4000:],
    }


def _score(context: RunContext) -> tuple[dict[str, Any], bool]:
    if context.stage == "smoke":
        metrics = {
            "status": "not_run_for_smoke",
            "reason": "The official MME scorer requires the complete 14-task formal run.",
        }
        write_json(context.path("metrics", "overall.json"), metrics)
        return metrics, True

    calculator = resolve_external_scorer("mme")
    assert calculator is not None
    scoring_log = context.path("logs", "scoring.log")
    command = [
        sys.executable,
        str(calculator),
        "--results_dir",
        str(context.path("parsed", "official_inputs")),
    ]
    ensure(scoring_log.parent)
    with scoring_log.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        process = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
        )
    log_text = scoring_log.read_text(encoding="utf-8", errors="replace")
    metrics = parse_mme_log(log_text)
    complete = all(
        metrics.get(key) is not None
        for key in ("perception_total", "cognition_total", "overall_total")
    )
    ok = process.returncode == 0 and complete
    metrics.update(
        {
            "returncode": process.returncode,
            "status": "ok" if ok else "failed",
            "official_log": str(scoring_log),
            "scorer": str(calculator),
            "scorer_sha256": sha256(calculator),
        }
    )
    write_json(context.path("metrics", "overall.json"), metrics)
    return metrics, ok


def _write_intervention_gate(
    context: RunContext,
    intervention: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if context.method != "sccd_v1":
        gate = {
            "benchmark": "mme", "method": context.method, "run_id": context.run_id,
            "stage": context.stage, "applicable": False, "passed": True,
            "decision": "SCCD-specific claim-routing gate is not applicable to this comparison method.",
        }
        write_json(context.path("diagnostics", "mme_intervention_gate.json"), gate)
        return gate

    existence = intervention["partitions"].get("existence", {})
    existence_records = [row for row in records if row.get("task") == "existence"]
    routed = sum(
        row.get("generation_path") == "input_object_existence"
        for row in existence_records
    )
    route_coverage = routed / len(existence_records) if existence_records else 0.0
    gate = {
        "benchmark": "mme",
        "method": context.method,
        "run_id": context.run_id,
        "stage": context.stage,
        "supported_scope": "MME task=existence only",
        "unsupported_tasks": [task for task in TASKS if task != "existence"],
        "existence_samples": len(existence_records),
        "input_claim_routed_samples": routed,
        "input_claim_route_coverage": route_coverage,
        "nonzero_adjustment_detected": existence.get(
            "nonzero_adjustment_detected", False
        ),
        "effective_intervention_detected": existence.get(
            "effective_intervention_detected"
            , False
        ),
        "samples_with_nonzero_delta": existence.get("samples_with_nonzero_delta", 0),
        "samples_with_output_change": existence.get("samples_with_output_change", 0),
        "passed": bool(
            existence_records
            and route_coverage == 1.0
            and existence.get("nonzero_adjustment_detected", False)
            and existence.get("effective_intervention_detected", False)
        ),
    }
    gate["decision"] = (
        "MME can proceed to a formal effectiveness comparison."
        if gate["passed"]
        else "Do not use this MME run as SCCD effectiveness evidence; inspect traces and routing first."
    )
    write_json(context.path("diagnostics", "mme_intervention_gate.json"), gate)
    return gate


def run(context: RunContext, limit: int | None = None) -> None:
    source = PROJECT_ROOT / "data" / "manifests" / "mme" / "all.jsonl"
    if not source.is_file():
        raise FileNotFoundError(
            "MME questions and labels are not redistributed. Build the local manifest "
            "from your authorized MME copy with: python scripts/build_mme_manifest.py "
            "--mme-root \"$MME_ROOT\""
        )
    source_rows = read_jsonl(source)
    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for row in source_rows:
        task = str(row["task"])
        if task in by_task:
            by_task[task].append(row)
    selected_by_task = {
        task: rows[:limit] if limit is not None else rows
        for task, rows in by_task.items()
    }
    selected = [row for task in TASKS for row in selected_by_task[task]]
    if context.stage == "formal":
        missing_tasks = [task for task in TASKS if not selected_by_task[task]]
        if len(selected) != EXPECTED_FORMAL_SAMPLES or missing_tasks:
            raise RuntimeError(
                "Formal MME requires 2374 rows across all 14 tasks; "
                f"found {len(selected)} rows, missing_tasks={missing_tasks}"
            )
    frozen, count = record_input_manifest(context, "all.jsonl", selected, source)
    scorer = resolve_external_scorer("mme", required=context.stage == "formal")
    write_run_manifest(
        context,
        [
            {
                "partition": "14_tasks",
                "source": str(source),
                "source_sha256": sha256(source),
                "frozen": str(frozen),
                "frozen_sha256": sha256(frozen),
                "num_samples": count,
                "limit_semantics": "per task" if limit is not None else "full",
            }
        ],
        count,
        {
            "canonical_records": "records/{task}.jsonl",
            "predictions": "predictions/{task}.jsonl",
            "parsed": "parsed/{task}.jsonl",
            "official_scorer_inputs": "parsed/official_inputs/{task}.txt",
            "traces": "traces/{task}.jsonl",
            "metrics": "metrics/overall.json",
            "intervention_gate": "diagnostics/mme_intervention_gate.json",
            "summary": "summaries/summary.json",
        },
        parser=PARSER_ID,
        scorer=str(scorer) if scorer is not None else None,
        scorer_source="external_not_redistributed",
        scorer_environment_variable="SEVD_MME_SCORER_PATH",
        scorer_sha256=sha256(scorer) if scorer is not None else None,
        tasks=list(TASKS),
        max_new_tokens=8,
        intervention_requirement=(
            "MME task=existence must have 100% input-claim route coverage, a "
            "non-zero adjustment, and an observed output change. Other MME tasks "
            "are outside the current object-existence claim scope."
        ),
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
                prompt = f"USER: <image>\n{row['question'].strip()}\nASSISTANT:"
                method_scope_supported = task == "existence"
                if method_scope_supported:
                    raw_response, traces, stats = engine.answer_object_existence_sample(
                        row, benchmark="mme"
                    )
                    generation_path = "input_object_existence"
                    if raw_response == "unknown":
                        raise RuntimeError(
                            f"Failed to parse supported MME existence claim: {row['question']}"
                        )
                else:
                    raw_response, traces, stats = engine.generate_chair_sample(
                        row, prompt, 8, benchmark="mme"
                    )
                    generation_path = "free_decode_unsupported_claim_type"
                normalized = parse_answer(raw_response)
                append_jsonl(
                    record_path,
                    {
                        "sample_id": sample_id,
                        "partition": task,
                        "task": task,
                        "image_id": row["image_id"],
                        "question": row["question"],
                        "ground_truth": row["ground_truth"],
                        "raw_response": raw_response,
                        "normalized_answer": normalized,
                        "parser_status": "ok" if normalized in {"yes", "no"} else "other",
                        "parser": PARSER_ID,
                        "method": context.method,
                        "run_id": context.run_id,
                        "stage": context.stage,
                        "generation_path": generation_path,
                        "method_scope_supported": method_scope_supported,
                        "stats": stats,
                        "traces": traces,
                    },
                )
                done.add(sample_id)
                if index == 1 or index % 50 == 0 or index == len(rows):
                    emit(context.log_path, f"MME {task}: {index}/{len(rows)}")
    finally:
        release_engine(engine)

    set_status(context, "materializing_and_scoring")
    all_records: list[dict[str, Any]] = []
    integrity_ok = True
    parser_failures = 0
    for task in TASKS:
        records = read_jsonl(context.path("records", f"{task}.jsonl"))
        all_records.extend(records)
        prediction_path = context.path("predictions", f"{task}.jsonl")
        parsed_path = context.path("parsed", f"{task}.jsonl")
        materialize_records(
            records,
            prediction_path,
            parsed_path,
            context.path("traces", f"{task}.jsonl"),
            PREDICTION_FIELDS,
            PARSED_FIELDS,
        )
        expected = len(selected_by_task[task])
        integrity_ok = integrity_ok and len(records) == expected
        parser_failures += sum(
            1 for row in records if row.get("normalized_answer") not in {"yes", "no"}
        )
        official_path = context.path("parsed", "official_inputs", f"{task}.txt")
        ensure(official_path.parent)
        with official_path.open("w", encoding="utf-8") as stream:
            for row in records:
                prediction = (
                    row["normalized_answer"]
                    if row.get("normalized_answer") in {"yes", "no"}
                    else "other"
                )
                stream.write(
                    f"{row.get('image_id')}\t{row.get('question')}\t"
                    f"{row.get('ground_truth')}\t{prediction}\n"
                )

    metrics, scorer_ok = _score(context)
    intervention = write_intervention_diagnostics(context, all_records)
    gate = _write_intervention_gate(context, intervention, all_records)
    summary = {
        "method": context.method,
        "run_id": context.run_id,
        "stage": context.stage,
        "num_samples": len(all_records),
        "num_tasks": sum(bool(selected_by_task[task]) for task in TASKS),
        "parser_failures": parser_failures,
        "integrity_ok": integrity_ok,
        "metrics": metrics,
        "intervention_gate": gate,
    }
    write_json(context.path("summaries", "summary.json"), summary)
    write_text(
        context.path("summaries", "summary.md"),
        f"# MME {context.method} Run Summary\n\n"
        f"- Run ID: `{context.run_id}`\n"
        f"- Stage: `{context.stage}`\n"
        f"- Samples: {len(all_records)}\n"
        f"- Tasks represented: {summary['num_tasks']}/14\n"
        f"- Parser failures: {parser_failures}\n"
        f"- Overall score: {metrics.get('overall_total', 'not scored')}\n"
        f"- Gate applicable: {gate.get('applicable', True)}\n"
        f"- Non-zero adjustment: {gate.get('nonzero_adjustment_detected', 'not_applicable')}\n"
        f"- Output changed: {gate.get('effective_intervention_detected', 'not_applicable')}\n"
        f"- Intervention gate: {'passed' if gate.get('passed') else 'failed'}\n",
    )
    finish_run(
        context,
        intervention,
        scorer_ok=scorer_ok and integrity_ok and parser_failures == 0,
        num_samples=len(all_records),
        parser_failures=parser_failures,
        mme_intervention_gate_passed=gate["passed"],
    )
    emit(context.log_path, "MME flow completed")
