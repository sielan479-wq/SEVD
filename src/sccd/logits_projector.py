"""Logit write-back helpers for SCCD."""

from __future__ import annotations

from typing import Any

from .risk_gate import FrozenRiskGateConfig, RiskGateDecision


def project_candidate_logit(original_logit: float, decision: RiskGateDecision, config: FrozenRiskGateConfig | None = None) -> float:
    cfg = config or FrozenRiskGateConfig()
    if decision.action == "suppress":
        return float(original_logit) - float(cfg.suppression_strength)
    if decision.action == "promote":
        return float(original_logit) + float(cfg.promotion_strength)
    return float(original_logit)


def unchanged_trace(action: str, **fields: Any) -> dict[str, Any]:
    row = dict(fields)
    row.update({"action": action, "delta": 0.0, "output_changed": False})
    return row
