"""Three-view semantic-claim contrastive verifier for SCCD."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Any

from .object_claim import ObjectClaim
from .risk_gate import FrozenRiskGateConfig, RiskGateDecision, decide_action


@dataclass(frozen=True)
class ViewScores:
    logp_yes: float
    logp_no: float
    margin: float
    forward_count: int


@dataclass(frozen=True)
class ObjectVerificationScore:
    claim: ObjectClaim
    real: ViewScores
    blur: ViewScores
    black: ViewScores
    decision: RiskGateDecision
    positive_token_ids: tuple[int, ...]
    negative_token_ids: tuple[int, ...]

    def trace_fields(self) -> dict[str, Any]:
        return {
            "probe_question": self.claim.probe_question,
            "claim_text": self.claim.claim_text,
            "logp_yes_real": self.real.logp_yes,
            "logp_no_real": self.real.logp_no,
            "logp_yes_blur": self.blur.logp_yes,
            "logp_no_blur": self.blur.logp_no,
            "logp_yes_black": self.black.logp_yes,
            "logp_no_black": self.black.logp_no,
            "margin_real": self.decision.margin_real,
            "margin_blur": self.decision.margin_blur,
            "margin_black": self.decision.margin_black,
            "necessity": self.decision.necessity,
            "prior_risk": self.decision.prior_risk,
            "fs": self.decision.fs,
            "evidence_score": self.decision.evidence_score,
            "positive_verbalizer_token_ids": list(self.positive_token_ids),
            "negative_verbalizer_token_ids": list(self.negative_token_ids),
        }


class ObjectExistenceVerifier:
    """Scores `Is there a <object> in the image?` using next-token yes/no logprobs."""

    def __init__(self, model: Any, tokenizer: Any, tokenizer_image_token: Any, config: FrozenRiskGateConfig | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_image_token = tokenizer_image_token
        self.config = config or FrozenRiskGateConfig()
        self.positive_token_ids = self._single_token_ids(self.config.yes_verbalizers)
        self.negative_token_ids = self._single_token_ids(self.config.no_verbalizers)
        overlap = set(self.positive_token_ids) & set(self.negative_token_ids)
        if overlap:
            raise ValueError(f"yes/no verbalizer token overlap: {sorted(overlap)}")
        if not self.positive_token_ids or not self.negative_token_ids:
            raise ValueError("yes/no verbalizer token ids must be non-empty")

    def _single_token_ids(self, texts: tuple[str, ...]) -> tuple[int, ...]:
        seen: set[int] = set()
        out: list[int] = []
        special = set(int(x) for x in getattr(self.tokenizer, "all_special_ids", []) or [])
        for text in texts:
            ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            ids = [int(x) for x in ids if int(x) not in special]
            if not ids:
                continue
            # LLaMA tokenizers may split leading-space verbalizers as
            # [space, yes/no].  The v1 verifier is a next-token class score, so
            # use the verbalizer token itself instead of the shared whitespace.
            token_id = int(ids[-1])
            if token_id not in seen:
                seen.add(token_id)
                out.append(token_id)
        return tuple(out)

    def score(self, claim: ObjectClaim, image_tensors: dict[str, Any]) -> ObjectVerificationScore:
        prompt = _build_unified_pope_probe(claim.probe_question)
        input_ids = self._tokenize_prompt(prompt)
        real = self._score_view(input_ids, image_tensors["real"])
        blur = self._score_view(input_ids, image_tensors["blur"])
        prior_mode = getattr(self.config, "prior_mode", "legacy_black")
        if prior_mode == "legacy_black":
            black = self._score_view(input_ids, image_tensors["black"])
        elif prior_mode == "text_only_exact":
            black = self._score_text(_build_text_only_exact_probe(claim.probe_question))
        elif prior_mode == "text_only_normalized":
            black = self._score_text(_build_text_only_normalized_probe(claim.canonical_object))
        elif prior_mode == "no_prior":
            black = ViewScores(logp_yes=0.0, logp_no=0.0, margin=0.0, forward_count=0)
        else:
            raise ValueError(f"unsupported prior_mode: {prior_mode}")
        decision = decide_action(real.margin, blur.margin, black.margin, self.config)
        return ObjectVerificationScore(
            claim=claim,
            real=real,
            blur=blur,
            black=black,
            decision=decision,
            positive_token_ids=self.positive_token_ids,
            negative_token_ids=self.negative_token_ids,
        )

    def _tokenize_prompt(self, prompt: str) -> Any:
        from llava.constants import IMAGE_TOKEN_INDEX

        ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        return ids.to(next(self.model.parameters()).device)

    def _tokenize_text_prompt(self, prompt: str) -> Any:
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        return ids.to(next(self.model.parameters()).device)

    def _score_text(self, prompt: str) -> ViewScores:
        input_ids = self._tokenize_text_prompt(prompt)
        return self._score_view(input_ids, None)

    def _score_view(self, input_ids: Any, image_tensor: Any) -> ViewScores:
        import torch

        outputs = None
        logits = None
        logp = None
        try:
            with torch.inference_mode():
                kwargs = {
                    "input_ids": input_ids,
                    "use_cache": False,
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "return_dict": True,
                }
                if image_tensor is not None:
                    kwargs["images"] = image_tensor
                outputs = self.model(**kwargs)
                logits = outputs.logits[:, -1, :].float()
                logp = torch.log_softmax(logits, dim=-1).squeeze(0)
                yes = _logsumexp([float(logp[i].item()) for i in self.positive_token_ids])
                no = _logsumexp([float(logp[i].item()) for i in self.negative_token_ids])
                return ViewScores(logp_yes=yes, logp_no=no, margin=yes - no, forward_count=1)
        finally:
            del outputs, logits, logp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _build_unified_pope_probe(question: str) -> str:
    return f"USER: <image>\n{question.strip()}\nPlease answer yes or no.\nASSISTANT:"


def _logsumexp(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("-inf")
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


def _build_text_only_exact_probe(question: str) -> str:
    return f"USER: {question.strip()}\nPlease answer yes or no.\nASSISTANT:"


def _build_text_only_normalized_probe(canonical_object: str) -> str:
    phrase = canonical_object.replace("_", " ").strip()
    article = "an" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"USER: Is there {article} {phrase}?\nPlease answer yes or no.\nASSISTANT:"


def build_text_only_paraphrase_probe(canonical_object: str) -> str:
    phrase = canonical_object.replace("_", " ").strip()
    article = "an" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"USER: Is {article} {phrase} present?\nPlease answer yes or no.\nASSISTANT:"
