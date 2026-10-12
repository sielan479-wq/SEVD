"""AMBER generation, official scoring, and partitioned intervention audit."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    resolve_image_paths,
    set_status,
    sha256,
    write_intervention_diagnostics,
    write_run_manifest,
    write_text,
)
from evaluation.unified_eval_utils import read_jsonl, write_json, write_jsonl


EVALUATION_MODES = {
    "all": "a",
    "generative": "g",
    "discriminative": "d",
    "discriminative_existence": "de",
    "discriminative_attribute": "da",
    "discriminative_relation": "dr",
}
PREDICTION_FIELDS = (
    "sample_id",
    "id",
    "task_type",
    "dimension",
    "image_id",
    "query",
    "response",
    "method",
    "run_id",
    "stage",
)
PARSED_FIELDS = PREDICTION_FIELDS + (
    "normalized_answer",
    "parser_status",
    "parser",
    "original_raw_answer",
    "original_parsed_answer",
    "fallback_triggered",
    "fallback_prompt",
    "fallback_yes_score",
    "fallback_no_score",
    "fallback_answer",
    "repair",
)
EXPECTED_FORMAL_SAMPLES = 15220


def resolve_discriminative_answer(
    engine: Any, row: dict[str, Any], raw_response: str
) -> tuple[str, dict[str, Any]]:
    """Default AMBER parser hook; native runners may install a fallback."""
    parsed = parse_answer(raw_response)
    return parsed, {
        "original_raw_answer": str(raw_response),
        "original_parsed_answer": parsed,
        "fallback_triggered": False,
        "fallback_prompt": None,
        "fallback_yes_score": None,
        "fallback_no_score": None,
        "fallback_answer": None,
    }


def build_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Path]]:
    frozen_manifest = PROJECT_ROOT / "data" / "manifests" / "amber" / "all.jsonl"
    if frozen_manifest.exists():
        rows = resolve_image_paths(read_jsonl(frozen_manifest), config, "amber")
        return rows, [frozen_manifest]

    amber_root = Path(str(config["amber"]["source_root"])).expanduser()
    query_root = amber_root / "data" / "query"
    image_root = Path(
        str(config["amber"].get("image_root", amber_root / "image"))
    ).expanduser()
    all_path = query_root / "query_all.json"
    existence_path = query_root / "query_discriminative-existence.json"
    attribute_path = query_root / "query_discriminative-attribute.json"
    relation_path = query_root / "query_discriminative-relation.json"
    dimensions = {
        **{
            int(row["id"]): "existence"
            for row in json.loads(existence_path.read_text(encoding="utf-8"))
        },
        **{
            int(row["id"]): "attribute"
            for row in json.loads(attribute_path.read_text(encoding="utf-8"))
        },
        **{
            int(row["id"]): "relation"
            for row in json.loads(relation_path.read_text(encoding="utf-8"))
        },
    }
    rows = []
    for row in json.loads(all_path.read_text(encoding="utf-8")):
        item_id = int(row["id"])
        dimension = dimensions.get(item_id, "generative")
        task_type = "generative" if dimension == "generative" else "discriminative"
        image = row.get("image") or f"AMBER_{item_id}.jpg"
        rows.append(
            {
                "sample_id": str(item_id),
                "id": item_id,
                "task_type": task_type,
                "dimension": dimension,
                "image_id": Path(image).stem,
                "image_path": str(image_root / image),
                "query": row["query"],
                "official_source": "data/query/query_all.json",
            }
        )
    return rows, [all_path, existence_path, attribute_path, relation_path]


def _select_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    selected = []
    for dimension in ("generative", "existence", "attribute", "relation"):
        selected.extend(
            [row for row in rows if row["dimension"] == dimension][:limit]
        )
    return sorted(selected, key=lambda row: int(row["id"]))


def parse_amber_log(text: str) -> dict[str, Any]:
    section_map = {
        "generative": "generative",
        "generative task": "generative",
        "discriminative": "discriminative",
        "descriminative": "discriminative",
        "discriminative task": "discriminative",
        "descriminative task": "discriminative",
        "existence": "existence",
        "exsitence": "existence",
        "attribute": "attribute",
        "state": "state",
        "number": "number",
        "action": "action",
        "relation": "relation",
    }
    key_map = {
        "chair": "chair",
        "cover": "cover",
        "hal": "hal",
        "cog": "cog",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    current: str | None = None
    output: dict[str, Any] = {"sections": {}, "raw_log_excerpt": text[-4000:]}
    for line in text.splitlines():
        stripped = line.strip().rstrip(":").lower()
        if stripped in section_map:
            current = section_map[stripped]
            output["sections"].setdefault(current, {})
            continue
        match = re.match(
            r"(CHAIR|Cover|Hal|Cog|Accuracy|Precision|Recall|F1):\s*\t*\s*([0-9.]+)%?",
            line.strip(),
            flags=re.I,
        )
        if match and current:
            metric = key_map[match.group(1).lower()]
            output["sections"].setdefault(current, {})[metric] = float(match.group(2))
    for section, values in output["sections"].items():
        for key, value in values.items():
            output[f"{section}_{key}"] = value
    if "generative_chair" in output and "discriminative_f1" in output:
        output["amber_score"] = 0.5 * (
            1.0
            - float(output["generative_chair"]) / 100.0
            + float(output["discriminative_f1"]) / 100.0
        )
    return output


def _official_groups(
    rows: list[dict[str, Any]], answers: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    answers_by_id = {int(answer["id"]): answer for answer in answers}
    selectors = {
        "all": lambda row: True,
        "generative": lambda row: row["task_type"] == "generative",
        "discriminative": lambda row: row["task_type"] == "discriminative",
        "discriminative_existence": lambda row: row["dimension"] == "existence",
        "discriminative_attribute": lambda row: row["dimension"] == "attribute",
        "discriminative_relation": lambda row: row["dimension"] == "relation",
    }
    return {
        name: [
            answers_by_id[int(row["id"])]
            for row in rows
            if selector(row) and int(row["id"]) in answers_by_id
        ]
        for name, selector in selectors.items()
    }


def _score(
    context: RunContext, official_files: dict[str, Path]
) -> tuple[dict[str, Any], bool]:
    if context.stage == "smoke":
        metrics = {
            "status": "not_run_for_smoke",
            "reason": "Official AMBER scoring is reserved for complete formal runs.",
        }
        write_json(context.path("metrics", "all.json"), metrics)
        return metrics, True

    amber_root = Path(str(context.config["amber"]["source_root"])).expanduser()
    all_ok = True
    metrics_by_part: dict[str, dict[str, Any]] = {}
    required_metric = {
        "all": "amber_score",
        "generative": "generative_chair",
        "discriminative": "discriminative_f1",
        "discriminative_existence": "existence_f1",
        "discriminative_attribute": "attribute_f1",
        "discriminative_relation": "relation_f1",
    }
    for part, evaluation_type in EVALUATION_MODES.items():
        input_file = official_files[part]
        scoring_log = context.path("logs", f"scoring_{evaluation_type}.log")
        command = [
            sys.executable,
            "inference.py",
            "--inference_data",
            str(input_file),
            "--evaluation_type",
            evaluation_type,
        ]
        ensure(scoring_log.parent)
        with scoring_log.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            process = subprocess.run(
                command,
                cwd=str(amber_root),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        metrics = parse_amber_log(scoring_log.read_text(encoding="utf-8", errors="replace"))
        parsed_ok = required_metric[part] in metrics
        part_ok = process.returncode == 0 and parsed_ok
        metrics.update(
            {
                "part": part,
                "evaluation_type": evaluation_type,
                "returncode": process.returncode,
                "status": "ok" if part_ok else "failed",
                "official_log": str(scoring_log),
                "required_metric": required_metric[part],
            }
        )
        all_ok = all_ok and part_ok
        metrics_by_part[part] = metrics
        write_json(context.path("metrics", f"{part}.json"), metrics)
    return metrics_by_part["all"], all_ok


def run(context: RunContext, limit: int | None = None) -> None:
    all_rows, source_files = build_rows(context.config)
    rows = _select_rows(all_rows, limit)
    if context.stage == "formal" and len(rows) != EXPECTED_FORMAL_SAMPLES:
        raise RuntimeError(
            f"Formal AMBER requires {EXPECTED_FORMAL_SAMPLES} rows; found {len(rows)}"
        )
    frozen, count = record_input_manifest(context, "all.jsonl", rows)
    write_run_manifest(
        context,
        [
            {
                "partition": "all",
                "official_sources": [
                    {"path": str(path), "sha256": sha256(path)} for path in source_files
                ],
                "frozen": str(frozen),
                "frozen_sha256": sha256(frozen),
                "num_samples": count,
                "limit_semantics": "per dimension" if limit is not None else "full",
            }
        ],
        count,
        {
            "canonical_records": "records/all.jsonl",
            "predictions": "predictions/all.jsonl",
            "official_scorer_inputs": "predictions/official/*.json",
            "parsed": "parsed/all.jsonl",
            "traces": "traces/all.jsonl",
            "metrics": "metrics/*.json",
            "diagnostics": "diagnostics/intervention_summary.json",
            "summary": "summaries/summary.json",
        },
        parser=PARSER_ID,
        scorer=str(Path(str(context.config["amber"]["source_root"])) / "inference.py"),
        evaluation_modes=EVALUATION_MODES,
        generative_max_new_tokens=64,
        discriminative_max_new_tokens=8,
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
            prompt = f"USER: <image>\n{row['query'].strip()}\nASSISTANT:"
            max_new_tokens = 64 if row["task_type"] == "generative" else 8
            raw_response, traces, stats = engine.generate_chair_sample(
                row, prompt, max_new_tokens, benchmark="amber"
            )
            if row["task_type"] == "discriminative":
                normalized, binary_audit = resolve_discriminative_answer(
                    engine, row, raw_response
                )
            else:
                normalized, binary_audit = "", {}
            response = (
                raw_response
                if row["task_type"] == "generative"
                else "Yes"
                if normalized == "yes"
                else "No"
                if normalized == "no"
                else "other"
            )
            append_jsonl(
                record_path,
                {
                    **row,
                    "partition": row["dimension"],
                    "raw_response": raw_response,
                    "response": response,
                    "normalized_answer": normalized,
                    "parser_status": (
                        "not_applicable"
                        if row["task_type"] == "generative"
                        else "ok"
                        if normalized in {"yes", "no"}
                        else "other"
                    ),
                    "parser": PARSER_ID,
                    "method": context.method,
                    "run_id": context.run_id,
                    "stage": context.stage,
                    "stats": stats,
                    "traces": traces,
                    **binary_audit,
                },
            )
            done.add(sample_id)
            if index == 1 or index % 100 == 0 or index == len(rows):
                emit(context.log_path, f"AMBER: {index}/{len(rows)}")
    finally:
        release_engine(engine)

    set_status(context, "materializing_and_scoring")
    records = read_jsonl(record_path)
    materialize_records(
        records,
        context.path("predictions", "all.jsonl"),
        context.path("parsed", "all.jsonl"),
        context.path("traces", "all.jsonl"),
        PREDICTION_FIELDS,
        PARSED_FIELDS,
    )
    for task_type in ("generative", "discriminative"):
        write_jsonl(
            context.path("parsed", f"{task_type}.jsonl"),
            [row for row in records if row["task_type"] == task_type],
        )
        write_jsonl(
            context.path("traces", f"{task_type}.jsonl"),
            [
                {
                    "sample_id": row["sample_id"],
                    "partition": row["dimension"],
                    "stats": row["stats"],
                    "trace_count": len(row["traces"]),
                    "traces": row["traces"],
                }
                for row in records
                if row["task_type"] == task_type
            ],
        )

    official_answers = [
        {"id": int(row["id"]), "response": row["response"]} for row in records
    ]
    groups = _official_groups(rows, official_answers)
    official_files: dict[str, Path] = {}
    for name, answers in groups.items():
        path = context.path("predictions", "official", f"{name}.json")
        write_json(path, answers)
        official_files[name] = path

    integrity_ok = len(records) == len(rows) and len({row["sample_id"] for row in records}) == len(records)
    parser_failures = sum(
        1
        for row in records
        if row["task_type"] == "discriminative"
        and row.get("normalized_answer") not in {"yes", "no"}
    )
    metrics, scorer_ok = _score(context, official_files)
    intervention = write_intervention_diagnostics(context, records)
    summary = {
        "method": context.method,
        "run_id": context.run_id,
        "stage": context.stage,
        "num_samples": len(records),
        "num_generative": sum(row["task_type"] == "generative" for row in records),
        "num_discriminative": sum(
            row["task_type"] == "discriminative" for row in records
        ),
        "parser_failures": parser_failures,
        "integrity_ok": integrity_ok,
        "metrics": metrics,
        "intervention": intervention["overall"],
    }
    write_json(context.path("summaries", "summary.json"), summary)
    write_text(
        context.path("summaries", "summary.md"),
        "# AMBER SCCD Run Summary\n\n"
        f"- Run ID: `{context.run_id}`\n"
        f"- Stage: `{context.stage}`\n"
        f"- Samples: {len(records)}\n"
        f"- Generative: {summary['num_generative']}\n"
        f"- Discriminative: {summary['num_discriminative']}\n"
        f"- Parser failures: {parser_failures}\n"
        f"- AMBER Score: {metrics.get('amber_score', 'not scored')}\n"
        f"- Output changes detected: {intervention['overall']['samples_with_output_change']}\n"
        f"- Integrity: {'passed' if integrity_ok else 'failed'}\n",
    )
    finish_run(
        context,
        intervention,
        scorer_ok=scorer_ok and integrity_ok and parser_failures == 0,
        num_samples=len(records),
        parser_failures=parser_failures,
    )
    emit(context.log_path, "AMBER flow completed")
