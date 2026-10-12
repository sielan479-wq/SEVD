"""CHAIR caption generation, strict scoring, and intervention diagnostics."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation.external_scorers import resolve_external_scorer
from evaluation.prompts import build_chair_prompt
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
    resolve_image_paths,
    set_status,
    sha256,
    write_intervention_diagnostics,
    write_run_manifest,
    write_text,
)
from evaluation.unified_eval_utils import read_jsonl, write_json


PREDICTION_FIELDS = (
    "sample_id",
    "image_id",
    "prompt",
    "raw_caption",
    "method",
    "run_id",
    "stage",
)
SCORER_FIELDS = ("sample_id", "image_id", "raw_caption")
EXPECTED_FORMAL_SAMPLES = 5000


def _score(context: RunContext, caption_path: Path) -> tuple[dict[str, Any], bool]:
    if context.stage == "smoke":
        metrics = {
            "status": "not_run_for_smoke",
            "reason": "Strict CHAIR scores are reserved for complete formal runs.",
            "num_samples": len(read_jsonl(caption_path)),
        }
        write_json(context.path("metrics", "metrics.json"), metrics)
        return metrics, True

    scorer = resolve_external_scorer("chair")
    assert scorer is not None
    cache = Path(
        str(
            context.config["chair"].get(
                "scorer_cache",
                PROJECT_ROOT / "evaluation" / "scorers" / "chair_cache.pkl",
            )
        )
    ).expanduser()
    annotation_root = Path(str(context.config["chair"]["annotation_root"])).expanduser()
    raw_output = context.path("metrics", "chair_raw_output.json")
    scoring_log = context.path("logs", "scoring.log")
    command = [
        sys.executable,
        str(scorer),
        "--cap_file",
        str(caption_path),
        "--image_id_key",
        "image_id",
        "--caption_key",
        "raw_caption",
        "--cache",
        str(cache),
        "--coco_path",
        str(annotation_root),
        "--save_path",
        str(raw_output),
    ]
    ensure(scoring_log.parent)
    ensure(cache.parent)
    with scoring_log.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        process = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
        )
    log_text = scoring_log.read_text(encoding="utf-8", errors="replace")
    values = {
        key: float(match.group(1))
        for key in ("CHAIRs", "CHAIRi")
        if (match := re.search(rf"^{key}\s*:\s*([0-9.]+)", log_text, flags=re.M))
    }
    captions = read_jsonl(caption_path)
    complete_metrics = all(key in values for key in ("CHAIRs", "CHAIRi"))
    if process.returncode != 0 and "ZeroDivisionError" in log_text:
        failure_reason = "no matching COCO ground-truth objects; verify image_id normalization"
    elif process.returncode != 0:
        failure_reason = "strict CHAIR scorer exited non-zero; inspect logs/scoring.log"
    elif not complete_metrics:
        failure_reason = "strict CHAIR scorer did not emit both CHAIRs and CHAIRi"
    else:
        failure_reason = None
    ok = process.returncode == 0 and complete_metrics
    metrics = {
        **values,
        "num_samples": len(captions),
        "average_whitespace_tokens": (
            sum(len(str(row.get("raw_caption", "")).split()) for row in captions) / len(captions)
            if captions
            else 0.0
        ),
        "returncode": process.returncode,
        "scorer": str(scorer),
        "scorer_sha256": sha256(scorer),
        "cache": str(cache),
        "cache_sha256": sha256(cache) if cache.exists() else None,
        "annotation_root": str(annotation_root),
        "raw_output": str(raw_output),
        "status": "ok" if ok else "failed",
        "failure_reason": failure_reason,
    }
    write_json(context.path("metrics", "metrics.json"), metrics)
    return metrics, ok


def run(context: RunContext, limit: int | None = None) -> None:
    source = PROJECT_ROOT / "data" / "manifests" / "chair" / "coco_5000.jsonl"
    rows = resolve_image_paths(read_jsonl(source, limit=limit), context.config, "chair")
    if context.stage == "formal" and len(rows) != EXPECTED_FORMAL_SAMPLES:
        raise RuntimeError(
            f"Formal CHAIR requires {EXPECTED_FORMAL_SAMPLES} rows; found {len(rows)}"
        )
    frozen, count = record_input_manifest(context, "coco.jsonl", rows, source)
    scorer = resolve_external_scorer("chair", required=context.stage == "formal")
    write_run_manifest(
        context,
        [
            {
                "partition": "all",
                "source": str(source),
                "source_sha256": sha256(source),
                "frozen": str(frozen),
                "frozen_sha256": sha256(frozen),
                "num_samples": count,
            }
        ],
        count,
        {
            "canonical_records": "records/all.jsonl",
            "captions": "predictions/captions.jsonl",
            "scorer_input": "parsed/chair_input.jsonl",
            "traces": "traces/all.jsonl",
            "metrics": "metrics/metrics.json",
            "diagnostics": "diagnostics/intervention_summary.json",
            "summary": "summaries/summary.json",
        },
        prompt="unified detailed image description",
        max_new_tokens=int(context.config["chair"].get("max_new_tokens", 64)),
        scorer=str(scorer) if scorer is not None else None,
        scorer_source="external_not_redistributed",
        scorer_environment_variable="SEVD_CHAIR_SCORER_PATH",
        scorer_sha256=sha256(scorer) if scorer is not None else None,
        scorer_cache_sha256=(
            sha256(Path(str(context.config["chair"]["scorer_cache"])))
            if Path(str(context.config["chair"]["scorer_cache"])).exists()
            else None
        ),
        annotation_root=str(context.config["chair"]["annotation_root"]),
    )

    record_path = context.path("records", "all.jsonl")
    done = existing_ids(record_path)
    set_status(context, "running_generation")
    engine = load_engine(context)
    try:
        for index, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            if sample_id in done:
                continue
            raw_prompt = row.get("prompt", "Describe the image in detail.")
            prompt = build_chair_prompt(raw_prompt)
            caption, traces, stats = engine.generate_chair_sample(
                row, prompt, int(context.config["chair"].get("max_new_tokens", 64))
            )
            append_jsonl(
                record_path,
                {
                    "sample_id": sample_id,
                    "partition": "all",
                    "image_id": row["image_id"],
                    "prompt": raw_prompt,
                    "raw_caption": caption,
                    "method": context.method,
                    "run_id": context.run_id,
                    "stage": context.stage,
                    "stats": stats,
                    "traces": traces,
                },
            )
            done.add(sample_id)
            if index == 1 or index % 50 == 0 or index == len(rows):
                emit(context.log_path, f"CHAIR: {index}/{len(rows)}")
    finally:
        release_engine(engine)

    set_status(context, "materializing_and_scoring")
    records = read_jsonl(record_path)
    caption_path = context.path("predictions", "captions.jsonl")
    materialize_records(
        records,
        caption_path,
        context.path("parsed", "chair_input.jsonl"),
        context.path("traces", "all.jsonl"),
        PREDICTION_FIELDS,
        SCORER_FIELDS,
    )
    integrity_ok = len(records) == len(rows)
    metrics, scorer_ok = _score(context, caption_path)
    summary = {
        "method": context.method,
        "run_id": context.run_id,
        "stage": context.stage,
        "num_samples": len(records),
        "integrity_ok": integrity_ok,
        "metrics": metrics,
    }
    write_json(context.path("summaries", "summary.json"), summary)
    write_text(
        context.path("summaries", "summary.md"),
        "# CHAIR SCCD Run Summary\n\n"
        f"- Run ID: `{context.run_id}`\n"
        f"- Stage: `{context.stage}`\n"
        f"- Samples: {len(records)}\n"
        f"- CHAIRs: {metrics.get('CHAIRs', 'not scored')}\n"
        f"- CHAIRi: {metrics.get('CHAIRi', 'not scored')}\n"
        f"- Integrity: {'passed' if integrity_ok else 'failed'}\n"
        f"- Scorer: {metrics.get('status')}\n",
    )
    intervention = write_intervention_diagnostics(context, records)
    finish_run(
        context,
        intervention,
        scorer_ok=scorer_ok and integrity_ok,
        num_samples=len(records),
    )
    emit(context.log_path, "CHAIR flow completed")
