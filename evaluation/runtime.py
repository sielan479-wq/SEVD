"""Shared run identity, provenance, I/O, and intervention diagnostics.

Every evaluation is identified by the tuple
``stage / benchmark / method / run_id`` and writes to one isolated directory.
Benchmark modules own generation and scoring; this module owns everything that
must stay identical across POPE, CHAIR, MME, and AMBER.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation.model_loader import load_model
from evaluation.unified_eval_utils import load_unified_config, read_jsonl, write_json, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sevd_llava15_7b.yaml"
METHOD = "sccd_v1"
SOURCE_METHOD = "object_main_candidate_textonly_v1"
VALID_METHODS = ("greedy", "sccd_v1")
VALID_STAGES = ("smoke", "formal")
VALID_BENCHMARKS = ("pope", "chair", "mme", "amber")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_name(value: str, kind: str) -> str:
    if not _SAFE_NAME.fullmatch(value) or ".." in value:
        raise ValueError(f"Invalid {kind}: {value!r}")
    return value


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure(path.parent)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def existing_ids(path: Path, key: str = "sample_id") -> set[str]:
    if not path.exists():
        return set()
    return {str(row[key]) for row in read_jsonl(path) if key in row}


def write_text(path: Path, value: str) -> None:
    ensure(path.parent)
    path.write_text(value, encoding="utf-8")


def emit(log_path: Path, message: str) -> None:
    line = f"[{now_utc()}] {message}"
    print(line, flush=True)
    ensure(log_path.parent)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


@dataclass(frozen=True)
class RunContext:
    benchmark: str
    stage: str
    run_id: str
    config_path: Path
    config: dict[str, Any]
    run_root: Path
    resume: bool = False
    method: str = METHOD

    @property
    def log_path(self) -> Path:
        return self.run_root / "logs" / "generation.log"

    @property
    def status_path(self) -> Path:
        return self.run_root / "status.json"

    def path(self, *parts: str) -> Path:
        return self.run_root.joinpath(*parts)


def resolve_run_root(
    config: dict[str, Any], stage: str, benchmark: str, method: str, run_id: str
) -> Path:
    output_root = Path(str(config.get("output_root", PROJECT_ROOT / "outputs"))).expanduser().resolve()
    return output_root / stage / benchmark / method / run_id


def create_context(
    benchmark: str,
    stage: str,
    run_id: str,
    config_path: Path = DEFAULT_CONFIG,
    resume: bool = False,
    method: str = METHOD,
) -> RunContext:
    benchmark = validate_name(benchmark, "benchmark")
    stage = validate_name(stage, "stage")
    run_id = validate_name(run_id, "run ID")
    method = validate_name(method, "method")
    if benchmark not in VALID_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method: {method}")
    config_path = config_path.expanduser().resolve()
    config = load_unified_config(config_path)
    run_root = resolve_run_root(config, stage, benchmark, method, run_id)
    return RunContext(benchmark, stage, run_id, config_path, config, run_root, resume, method)


def prepare_run(context: RunContext) -> None:
    if context.run_root.exists() and any(context.run_root.iterdir()) and not context.resume:
        raise FileExistsError(
            f"Run directory already exists: {context.run_root}. "
            "Use a new --run-id or pass --resume to continue this exact run."
        )
    ensure(context.run_root)
    for name in (
        "provenance",
        "records",
        "predictions",
        "parsed",
        "traces",
        "metrics",
        "diagnostics",
        "summaries",
        "logs",
    ):
        ensure(context.path(name))

    config_hash = sha256(context.config_path)
    existing_environment = context.path("provenance", "environment.json")
    if context.resume and existing_environment.exists():
        previous = json.loads(existing_environment.read_text(encoding="utf-8"))
        previous_hash = previous.get("config_sha256")
        if previous_hash and previous_hash != config_hash:
            raise RuntimeError(
                "Refusing to resume with a different config: "
                f"stored={previous_hash}, current={config_hash}"
            )

    write_text(
        context.path("provenance", "config_source.yaml"),
        context.config_path.read_text(encoding="utf-8"),
    )
    write_json(context.path("provenance", "config_resolved.json"), context.config)
    environment = {
        "created_at": now_utc(),
        "stage": context.stage,
        "benchmark": context.benchmark,
        "method": context.method,
        "source_method": context.config.get(context.method, {}).get("source_method", SOURCE_METHOD),
        "run_id": context.run_id,
        "command": shlex.join([sys.executable, *sys.argv]),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "config_path": str(context.config_path),
        "config_sha256": config_hash,
        "project_git_revision": _git_revision(PROJECT_ROOT),
        "llava_git_revision": _git_revision(Path(str(context.config.get("llava_code_path", ""))).expanduser()),
        "model_path": context.config.get("model_path"),
    }
    write_json(existing_environment, environment)
    set_status(context, "prepared")


def set_status(context: RunContext, status: str, **extra: Any) -> None:
    payload: dict[str, Any] = {}
    if context.status_path.exists():
        try:
            payload = json.loads(context.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(
        {
            "updated_at": now_utc(),
            "status": status,
            "stage": context.stage,
            "benchmark": context.benchmark,
            "method": context.method,
            "run_id": context.run_id,
            **extra,
        }
    )
    if status in {
        "completed",
        "completed_no_effective_intervention",
        "smoke_completed",
        "smoke_completed_no_effective_intervention",
    }:
        payload.pop("error_type", None)
        payload.pop("error", None)
    write_json(context.status_path, payload)


def record_input_manifest(
    context: RunContext,
    name: str,
    rows: Iterable[dict[str, Any]],
    source_path: Path | None = None,
) -> tuple[Path, int]:
    rows = list(rows)
    output = context.path("provenance", "input_manifests", name)
    if context.resume and output.exists():
        existing_rows = read_jsonl(output)
        if existing_rows != rows:
            raise RuntimeError(
                f"Refusing to resume because frozen input {name!r} differs from "
                "the currently resolved benchmark input."
            )
        return output, len(existing_rows)
    write_jsonl(output, rows)
    metadata = {
        "name": name,
        "created_at": now_utc(),
        "source_path": str(source_path) if source_path else None,
        "source_sha256": sha256(source_path) if source_path and source_path.exists() else None,
        "frozen_path": str(output),
        "frozen_sha256": sha256(output),
        "num_samples": len(rows),
    }
    write_json(output.with_suffix(output.suffix + ".metadata.json"), metadata)
    return output, len(rows)


def resolve_image_paths(
    rows: Iterable[dict[str, Any]], config: dict[str, Any], benchmark: str
) -> list[dict[str, Any]]:
    """Resolve migrated manifest image paths in the frozen run-local copy."""
    benchmark_config = config.get(benchmark, {})
    image_root_value = benchmark_config.get("image_root") if isinstance(benchmark_config, dict) else None
    image_root = Path(str(image_root_value)).expanduser() if image_root_value else None
    resolved_rows: list[dict[str, Any]] = []
    for row in rows:
        resolved = dict(row)
        source = Path(str(row["image_path"])).expanduser()
        if source.exists():
            candidate = source
        elif image_root is not None:
            # Public manifests use filenames (COCO/AMBER) or benchmark-root
            # relative paths (MME). Absolute paths remain backward compatible.
            relative = source.name if source.is_absolute() else source
            candidate = image_root / relative
        else:
            candidate = source
        if not candidate.exists():
            raise FileNotFoundError(
                f"Image for {benchmark} sample {row.get('sample_id')} does not exist: "
                f"manifest={source}, resolved={candidate}"
            )
        resolved["image_path_manifest"] = str(source)
        resolved["image_path"] = str(candidate.resolve())
        resolved_rows.append(resolved)
    return resolved_rows


def write_run_manifest(
    context: RunContext,
    inputs: list[dict[str, Any]],
    expected_samples: int,
    outputs: dict[str, Any],
    **extra: Any,
) -> None:
    method_config = dict(context.config.get(context.method, {}))
    payload = {
        "created_at": now_utc(),
        "stage": context.stage,
        "benchmark": context.benchmark,
        "method": context.method,
        "source_method": method_config.get("source_method", SOURCE_METHOD),
        "display_name": method_config.get("display_name", context.method),
        "run_id": context.run_id,
        "run_root": str(context.run_root),
        "expected_samples": expected_samples,
        "inputs": inputs,
        "outputs": outputs,
        "method_config": method_config,
        "generation_scope": method_config.get("generation_scope", {
            "claim_type": "object_existence",
            "prior_mode": method_config.get("prior_mode", "text_only_exact"),
            "counterfactual": "gaussian_blur",
            "attribute_enabled": False,
            "count_enabled": False,
            "relation_enabled": False,
        }),
        **extra,
    }
    write_json(context.path("provenance", "run_manifest.json"), payload)


def frozen_method_config(config: dict[str, Any]) -> Any:
    from sccd.risk_gate import FrozenRiskGateConfig

    raw = dict(config.get(METHOD, {}))
    return FrozenRiskGateConfig(
        top_k=int(raw.get("top_k", 50)),
        object_logit_window=float(raw.get("object_logit_window", 3.0)),
        max_claims_per_image=int(raw.get("max_claims_per_image", 8)),
        blur_radius=float(raw.get("blur_radius", 15.0)),
        lambda_prior=float(raw.get("lambda_prior", 0.5)),
        lambda_blur=float(raw.get("lambda_blur", 0.0)),
        prior_mode=str(raw.get("prior_mode", "text_only_exact")),
        suppress_threshold=float(raw.get("suppress_threshold", -0.5)),
        promote_threshold=float(raw.get("promote_threshold", 0.5)),
        margin_floor=float(raw.get("margin_floor", -0.5)),
        suppression_strength=float(raw.get("suppression_strength", 10000.0)),
        promotion_strength=float(raw.get("promotion_strength", 0.5)),
    )


def load_engine(context: RunContext) -> Any:
    if context.method != METHOD:
        from evaluation.method_engines import BaselineEngine

        emit(context.log_path, f"loading LLaVA components for {context.method}")
        return BaselineEngine(context.config, context.method, load_model(context.config, context.method))
    from sccd.engine import SCCDEngine
    from sccd.vocabulary import ObjectVocabulary

    emit(context.log_path, "loading LLaVA components for SCCD")
    components = load_model(context.config)
    vocab = ObjectVocabulary.from_synonyms_file(context.config["chair"]["synonym_file"])
    return SCCDEngine(
        context.config,
        components,
        components[3],
        vocab,
        frozen_method_config(context.config),
    )


def release_engine(engine: Any) -> None:
    try:
        del engine
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def materialize_records(
    records: list[dict[str, Any]],
    prediction_path: Path,
    parsed_path: Path,
    trace_path: Path,
    prediction_fields: tuple[str, ...],
    parsed_fields: tuple[str, ...],
) -> None:
    predictions = [{key: row.get(key) for key in prediction_fields} for row in records]
    parsed = [{key: row.get(key) for key in parsed_fields} for row in records]
    trace_rows = [
        {
            "sample_id": row.get("sample_id"),
            "partition": row.get("partition"),
            "stats": row.get("stats", {}),
            "trace_count": len(row.get("traces", [])),
            "traces": row.get("traces", []),
        }
        for row in records
    ]
    write_jsonl(prediction_path, predictions)
    write_jsonl(parsed_path, parsed)
    write_jsonl(trace_path, trace_rows)


def summarize_interventions(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    stats_totals: defaultdict[str, float] = defaultdict(float)
    nonzero_delta_count = 0
    samples_with_trace = 0
    samples_with_scored_claim = 0
    samples_with_nonzero_delta = 0
    samples_with_output_change = 0
    sample_ids: set[str] = set()

    for row in records:
        sample_ids.add(str(row.get("sample_id")))
        traces = list(row.get("traces") or [])
        stats = dict(row.get("stats") or {})
        if traces:
            samples_with_trace += 1
        if float(stats.get("unique_claims_scored") or 0) > 0:
            samples_with_scored_claim += 1
        sample_nonzero = False
        sample_changed = bool(float(stats.get("changed_step_count") or 0) > 0)
        for trace in traces:
            action_counts[str(trace.get("action", "unknown"))] += 1
            if abs(float(trace.get("delta") or 0.0)) > 0:
                nonzero_delta_count += 1
                sample_nonzero = True
            if trace.get("output_changed") is True:
                sample_changed = True
        if sample_nonzero:
            samples_with_nonzero_delta += 1
        if sample_changed:
            samples_with_output_change += 1
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stats_totals[key] += float(value)

    sample_count = len(sample_ids)
    activated = samples_with_nonzero_delta > 0
    effective = samples_with_output_change > 0
    warnings = []
    if sample_count == 0:
        warnings.append("no canonical sample records")
    if samples_with_scored_claim == 0:
        warnings.append("no semantic claim was scored")
    if nonzero_delta_count == 0:
        warnings.append("no non-zero SCCD logit/decision delta was observed")
    if samples_with_output_change == 0:
        warnings.append("no token or final decision change was observed")
    return {
        "sample_count": sample_count,
        "samples_with_trace": samples_with_trace,
        "samples_with_scored_claim": samples_with_scored_claim,
        "samples_with_nonzero_delta": samples_with_nonzero_delta,
        "samples_with_output_change": samples_with_output_change,
        "nonzero_delta_count": nonzero_delta_count,
        "nonzero_adjustment_detected": activated,
        "effective_intervention_detected": effective,
        "effective_intervention_rate": (samples_with_nonzero_delta / sample_count) if sample_count else 0.0,
        "output_change_rate": (samples_with_output_change / sample_count) if sample_count else 0.0,
        "action_counts": dict(sorted(action_counts.items())),
        "stats_totals": dict(sorted(stats_totals.items())),
        "warnings": warnings,
    }


def write_intervention_diagnostics(
    context: RunContext, records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_partition: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_partition[str(row.get("partition", "all"))].append(row)
    summary = {
        "created_at": now_utc(),
        "benchmark": context.benchmark,
        "method": context.method,
        "run_id": context.run_id,
        "overall": summarize_interventions(records),
        "partitions": {
            name: summarize_interventions(partition_rows)
            for name, partition_rows in sorted(by_partition.items())
        },
    }
    requires_intervention = bool(context.config.get(context.method, {}).get("requires_intervention", context.method != "greedy"))
    summary["requires_intervention"] = requires_intervention
    summary["publication_eligible"] = bool(
        context.stage == "formal"
        and (not requires_intervention or summary["overall"]["effective_intervention_detected"])
    )
    if context.stage != "formal":
        summary["publication_blocker"] = "Smoke runs are diagnostic-only and never publication-eligible."
    elif not summary["publication_eligible"]:
        summary["publication_blocker"] = (
            f"No effective {context.method} intervention was detected. Treat metrics as a no-op "
            "sanity result, not as evidence for method effectiveness."
        )
    write_json(context.path("diagnostics", "intervention_summary.json"), summary)
    return summary


def finish_run(
    context: RunContext,
    intervention: dict[str, Any],
    scorer_ok: bool,
    **extra: Any,
) -> None:
    effective = intervention.get("overall", {}).get(
        "effective_intervention_detected", False
    )
    if not scorer_ok:
        status = "failed_scoring"
    elif context.stage == "smoke" and not effective:
        status = "smoke_completed_no_effective_intervention"
    elif context.stage == "smoke":
        status = "smoke_completed"
    elif not intervention.get("publication_eligible", False):
        status = "completed_no_effective_intervention"
    else:
        status = "completed"
    set_status(
        context,
        status,
        scorer_ok=scorer_ok,
        effective_intervention_detected=effective,
        publication_eligible=intervention.get("publication_eligible", False),
        **extra,
    )
