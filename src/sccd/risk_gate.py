"""Contrastive semantic-claim score and risk gate used by SCCD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .claim_support import compute_claim_support


@dataclass(frozen=True)
class FrozenRiskGateConfig:
    top_k: int = 50
    object_logit_window: float = 3.0
    max_claims_per_image: int = 8
    blur_radius: float = 15.0
    lambda_prior: float = 0.5
    lambda_blur: float = 0.0
    prior_mode: str = "legacy_black"
    suppress_threshold: float = -0.5
    promote_threshold: float = 0.5
    margin_floor: float = -0.5
    suppression_strength: float = 10000.0
    promotion_strength: float = 0.5
    yes_verbalizers: tuple[str, ...] = ("Yes", " yes", "yes", " Yes")
    no_verbalizers: tuple[str, ...] = ("No", " no", "no", " No")
    verbalizer_aggregation: str = "logsumexp_next_token"
    verifier_prompt_template: str = "unified_yes_no_object_existence"
    method_slug: str = "sccd_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskGateDecision:
    margin_real: float
    margin_blur: float
    margin_black: float
    necessity: float
    prior_risk: float
    fs: float
    evidence_score: float
    action: str
    delta: float


def evidence_score(
    margin_real: float,
    margin_blur: float,
    margin_black: float,
    *,
    lambda_prior: float = 0.5,
    lambda_blur: float = 0.0,
) -> tuple[float, float, float, float]:
    support = compute_claim_support(
        margin_real=margin_real,
        margin_counterfactual=margin_blur,
        margin_text=margin_black,
        prior_weight=lambda_prior,
    )
    necessity = support.counterfactual_term
    prior_risk = support.prior_positive
    fs = support.fs
    score = fs - float(lambda_blur) * max(0.0, float(margin_blur))
    return necessity, prior_risk, fs, score


def decide_action(
    margin_real: float,
    margin_blur: float,
    margin_black: float,
    config: FrozenRiskGateConfig | None = None,
) -> RiskGateDecision:
    cfg = config or FrozenRiskGateConfig()
    necessity, prior_risk, fs, score = evidence_score(
        margin_real,
        margin_blur,
        margin_black,
        lambda_prior=cfg.lambda_prior,
        lambda_blur=cfg.lambda_blur,
    )
    if float(margin_real) > 0.0:
        action = "suppress" if score < cfg.suppress_threshold else "keep"
    else:
        if score > cfg.promote_threshold and float(margin_real) > cfg.margin_floor:
            action = "promote"
        else:
            action = "suppress"
    if action == "suppress":
        delta = -float(cfg.suppression_strength)
    elif action == "promote":
        delta = float(cfg.promotion_strength)
    else:
        delta = 0.0
    return RiskGateDecision(
        margin_real=float(margin_real),
        margin_blur=float(margin_blur),
        margin_black=float(margin_black),
        necessity=necessity,
        prior_risk=prior_risk,
        fs=fs,
        evidence_score=score,
        action=action,
        delta=delta,
    )
