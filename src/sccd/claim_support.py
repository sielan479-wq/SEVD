from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClaimSupportResult:
    margin_real: float
    margin_counterfactual: float
    margin_text: float | None
    counterfactual_term: float
    prior_positive: float
    prior_weight: float
    prior_term: float
    fs: float

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def compute_claim_support(
    margin_real: float,
    margin_counterfactual: float,
    margin_text: float | None,
    prior_weight: float,
) -> ClaimSupportResult:
    real = float(margin_real)
    cf = float(margin_counterfactual)
    text = None if margin_text is None else float(margin_text)
    prior_positive = max(0.0, text or 0.0)
    counterfactual_term = real - cf
    prior_term = float(prior_weight) * prior_positive
    fs = counterfactual_term - prior_term
    return ClaimSupportResult(
        margin_real=real,
        margin_counterfactual=cf,
        margin_text=text,
        counterfactual_term=counterfactual_term,
        prior_positive=prior_positive,
        prior_weight=float(prior_weight),
        prior_term=prior_term,
        fs=fs,
    )
