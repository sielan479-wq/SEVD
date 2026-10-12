"""Paper-facing SCCD ablations; each removes exactly one design commitment."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from .risk_gate import RiskGateDecision


ABLATIONS = ("token_level_contrast", "real_only", "suppression_only")


class SemanticEvidenceAblationVerifier:
    """Real-only and suppression-only variants over frozen SCCD view scores."""

    def __init__(self, base: Any, mode: str) -> None:
        self.base, self.mode = base, mode

    def score(self, claim: Any, image_tensors: dict[str, Any]):
        score = self.base.score(claim, image_tensors)
        decision = score.decision
        if self.mode == "real_only":
            # No counterfactual or text-prior evidence enters this decision.
            action = "keep" if score.real.margin > 0.0 else "suppress"
            decision = RiskGateDecision(
                margin_real=score.real.margin, margin_blur=score.blur.margin,
                margin_black=score.black.margin, necessity=0.0, prior_risk=0.0,
                fs=0.0, evidence_score=score.real.margin, action=action, delta=0.0,
            )
        elif self.mode == "suppression_only" and decision.action == "promote":
            # Preserve the real decoder's negative answer; do not invert it.
            decision = replace(decision, action="keep", delta=0.0)
        return replace(score, decision=decision)


class TokenLevelContrastBinaryEngine:
    """Direct all-token real/blur contrast, without semantic claim verification."""

    def __init__(self, base: Any, alpha: float = 1.0) -> None:
        self.base, self.alpha = base, float(alpha)
        self.tokenizer = base.tokenizer
        self.model = base.model
        self.tokenizer_image_token = base.tokenizer_image_token

    def answer_pope_sample(self, row: dict[str, Any]):
        return self.answer_object_existence_sample(row, "pope")

    def answer_object_existence_sample(self, row: dict[str, Any], benchmark: str):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX
        from sccd.binary_prompt import build_llava_binary_prompt

        started = time.time()
        tensors = self.base._image_tensors(row["image_path"])
        prompt = build_llava_binary_prompt(row["question"])
        ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        generated, traces, changed = [], [], 0
        with torch.inference_mode():
            for step in range(8):
                real = self.model(input_ids=ids, images=tensors["real"], use_cache=False, return_dict=True).logits[:, -1, :].float().squeeze(0)
                blur = self.model(input_ids=ids, images=tensors["blur"], use_cache=False, return_dict=True).logits[:, -1, :].float().squeeze(0)
                adjusted = (1.0 + self.alpha) * real - self.alpha * blur
                before, after = int(real.argmax()), int(adjusted.argmax())
                changed += int(before != after)
                traces.append({"sample_id": str(row.get("sample_id")), "benchmark": benchmark, "step": step, "action": "token_level_real_blur_contrast", "delta": float((adjusted[after] - real[after]).item()), "selected_token_before": before, "selected_token_after": after, "output_changed": before != after})
                generated.append(after)
                ids = torch.cat([ids, torch.tensor([[after]], dtype=ids.dtype, device=ids.device)], dim=1)
                if after == self.tokenizer.eos_token_id: break
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), traces, {"generated_token_count": len(generated), "changed_step_count": changed, "method_forward_count": 2 * len(generated), "runtime_seconds": time.time()-started, "exception_count": 0, "unique_claims_scored": 0, "verifier_forward_count": 0, "input_claim_route_count": 0, "input_claim_parse_failure_count": 0}
