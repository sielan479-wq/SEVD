"""Configuration and serialization helpers for the current SCCD runner.

Output layout belongs exclusively to :mod:`evaluation.runtime`; this module
deliberately contains no default experiment-result paths.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, MutableMapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sevd_llava15_7b.yaml"
_ENVIRONMENT_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [parse_scalar(part.strip()) for part in inner.split(",")]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", value):
        return float(value)
    return value


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    root: dict[str, Any] = {}
    stack: list[tuple[int, MutableMapping[str, Any]]] = [(-1, root)]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML line {line_number} in {path}: {raw}")
        key, value = stripped.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = parse_scalar(value)
    return _expand_environment(root, path)


def _expand_environment(value: Any, config_path: Path) -> Any:
    """Expand portable ``${NAME}`` references in a parsed configuration.

    Public configs never embed machine-specific paths. Missing variables are
    rejected early instead of becoming misleading unresolved paths later.
    """
    if isinstance(value, dict):
        return {
            key: _expand_environment(item, config_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_environment(item, config_path) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted(
        {
            first or second
            for first, second in _ENVIRONMENT_REFERENCE.findall(value)
            if (first or second) not in os.environ
        }
    )
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Missing environment variable(s) in {config_path}: {names}. "
            "See README.md for the required dataset and model paths."
        )
    return os.path.expanduser(os.path.expandvars(value))


def load_unified_config(path: Path | str | None = DEFAULT_CONFIG) -> dict[str, Any]:
    return load_config(path or DEFAULT_CONFIG)


def ensure_parent(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path | str, data: Any) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def read_jsonl(path: Path | str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(
    path: Path | str, rows: list[dict[str, Any]], fields: list[str]
) -> None:
    path = ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
