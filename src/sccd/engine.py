"""SCCD generation and binary-decision engine."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from .cache import SampleScoreCache
from .object_claim import construct_object_claim, object_from_existence_question
from .verifier import ObjectExistenceVerifier
from .vocabulary import ObjectVocabulary
from .risk_gate import FrozenRiskGateConfig
from .trace_schema import TRACE_SCHEMA_VERSION


@dataclass
class SampleStats:
    object_candidates_in_topk: int = 0
    non_object_candidates_in_topk: int = 0
    non_object_adjustment_count: int = 0
    objects_inside_logit_window: int = 0
    unique_claims_scored: int = 0
    cache_hits: int = 0
    suppress_count: int = 0
    keep_count: int = 0
    promote_count: int = 0
    changed_step_count: int = 0
    generated_token_count: int = 0
    runtime_seconds: float = 0.0
    verifier_forward_count: int = 0
    exception_count: int = 0
    input_claim_route_count: int = 0
    input_claim_parse_failure_count: int = 0

    def add_action(self, action: str) -> None:
        if action == "suppress":
            self.suppress_count += 1
        elif action == "promote":
            self.promote_count += 1
        elif action == "keep":
            self.keep_count += 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SCCDEngine:
    """Semantic object-level engine shared by binary QA and free decoding."""

    def __init__(
        self,
        config: dict[str, Any],
        components: Any,
        tokenizer_image_token: Any,
        vocab: ObjectVocabulary,
        method_config: FrozenRiskGateConfig | None = None,
    ) -> None:
        self.config = config
        self.tokenizer, self.model, self.image_processor = components[0], components[1], components[2]
        self.tokenizer_image_token = tokenizer_image_token
        self.vocab = vocab
        self.method_config = method_config or FrozenRiskGateConfig()
        self.verifier = ObjectExistenceVerifier(self.model, self.tokenizer, tokenizer_image_token, self.method_config)

    def answer_object_existence_sample(
        self, row: dict[str, Any], benchmark: str
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Answer a supported input ``exists(object)`` claim with frozen SCCD."""
        start = time.time()
        stats = SampleStats()
        stats.input_claim_route_count = 1
        sample_id = str(row.get("sample_id"))
        image_id = str(row.get("image_id"))
        image_tensors = self._image_tensors(row["image_path"])
        claim = object_from_existence_question(str(row.get("question", "")), self.vocab)
        if claim is None:
            stats.exception_count += 1
            stats.input_claim_parse_failure_count += 1
            stats.runtime_seconds = time.time() - start
            return "unknown", [], stats.to_dict()
        score = self.verifier.score(claim, image_tensors)
        stats.verifier_forward_count += score.real.forward_count + score.blur.forward_count + score.black.forward_count
        action = score.decision.action
        final_answer = "yes" if action in {"keep", "promote"} and score.decision.margin_real > 0 else "no"
        if action == "promote":
            final_answer = "yes"
        trace = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "sample_id": sample_id,
            "image_id": image_id,
            "split": row.get("split"),
            "task": row.get("task"),
            "benchmark": benchmark,
            "claim_source": "input_question",
            "claim_type": "object_existence",
            "generation_path": "input_object_existence",
            "candidate_rank": 1,
            "candidate_token_id": None,
            "candidate_text": None,
            "canonical_object": claim.canonical_object,
            "original_logit": None,
            "adjusted_logit": None,
            "top1_logit": None,
            "logit_gap_from_top": None,
            **score.trace_fields(),
            "action": action,
            "delta": score.decision.delta,
            "base_answer": "yes" if score.decision.margin_real > 0 else "no",
            "final_answer": final_answer,
            "selected_token_before": "yes" if score.decision.margin_real > 0 else "no",
            "selected_token_after": final_answer,
            "output_changed": final_answer != ("yes" if score.decision.margin_real > 0 else "no"),
            "cache_hit": False,
            "scfd2_used": False,
            "attribute_enabled": False,
            "historical_mixed_method_used": False,
        }
        stats.add_action(action)
        stats.unique_claims_scored = 1
        if trace["output_changed"]:
            stats.changed_step_count = 1
        stats.runtime_seconds = time.time() - start
        return final_answer, [trace], stats.to_dict()

    def answer_pope_sample(
        self, row: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        return self.answer_object_existence_sample(row, benchmark="pope")

    def generate_chair_sample(
        self,
        row: dict[str, Any],
        prompt: str,
        max_new_tokens: int,
        benchmark: str = "chair",
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX

        start = time.time()
        stats = SampleStats()
        sample_id = str(row.get("sample_id"))
        image_id = str(row.get("image_id"))
        score_cache = SampleScoreCache(sample_id)
        score_cache.start(sample_id)
        image_tensors = self._image_tensors(row["image_path"])
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        generated: list[int] = []
        traces: list[dict[str, Any]] = []
        try:
            with torch.inference_mode():
                for step in range(int(max_new_tokens)):
                    outputs = self.model(
                        input_ids=input_ids,
                        images=image_tensors["real"],
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    real_vec = outputs.logits[:, -1, :].float().squeeze(0)
                    selected_before = int(torch.argmax(real_vec).item())
                    adjusted_vec = real_vec.clone()
                    k = min(int(self.method_config.top_k), int(real_vec.numel()))
                    top_values, top_ids = torch.topk(real_vec, k=k)
                    token_ids = [int(x) for x in top_ids.tolist()]
                    token_texts = {tid: self.tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids}
                    top1_logit = float(top_values[0].item())
                    step_traces: list[dict[str, Any]] = []
                    for rank, (tid, value) in enumerate(zip(token_ids, top_values.tolist()), 1):
                        original_logit = float(value)
                        token_text = token_texts[tid]
                        canonical = self.vocab.canonicalize_token(token_text)
                        if canonical is None:
                            stats.non_object_candidates_in_topk += 1
                            continue
                        stats.object_candidates_in_topk += 1
                        gap = top1_logit - original_logit
                        base = {
                            "schema_version": TRACE_SCHEMA_VERSION,
                            "sample_id": sample_id,
                            "image_id": image_id,
                            "benchmark": benchmark,
                            "step": step,
                            "candidate_rank": rank,
                            "candidate_token_id": tid,
                            "candidate_text": token_text,
                            "canonical_object": canonical,
                            "original_logit": original_logit,
                            "top1_logit": top1_logit,
                            "logit_gap_from_top": gap,
                            "selected_token_before": selected_before,
                            "cache_hit": False,
                            "scfd2_used": False,
                            "attribute_enabled": False,
                            "historical_mixed_method_used": False,
                        }
                        if gap > float(self.method_config.object_logit_window):
                            base.update(_empty_score_fields())
                            base.update({"action": "keep_unscored_outside_logit_window", "delta": 0.0, "adjusted_logit": original_logit})
                            step_traces.append(base)
                            continue
                        stats.objects_inside_logit_window += 1
                        cached = score_cache.get(sample_id, canonical)
                        if cached is None and len(score_cache.scores) >= int(self.method_config.max_claims_per_image):
                            base.update(_empty_score_fields(construct_object_claim(canonical).probe_question))
                            base.update({"action": "keep_unscored_claim_budget_exceeded", "delta": 0.0, "adjusted_logit": original_logit})
                            step_traces.append(base)
                            continue
                        if cached is None:
                            score = self.verifier.score(construct_object_claim(canonical), image_tensors)
                            score_cache.set(sample_id, canonical, score)
                            stats.unique_claims_scored += 1
                            stats.verifier_forward_count += score.real.forward_count + score.blur.forward_count + score.black.forward_count
                            cache_hit = False
                        else:
                            score = cached
                            cache_hit = True
                            stats.cache_hits += 1
                        decision = score.decision
                        adjusted = original_logit + decision.delta
                        adjusted_vec[tid] = adjusted
                        stats.add_action(decision.action)
                        base.update(score.trace_fields())
                        base.update({
                            "action": decision.action,
                            "delta": decision.delta,
                            "adjusted_logit": float(adjusted),
                            "cache_hit": cache_hit,
                        })
                        step_traces.append(base)
                    selected_after = int(torch.argmax(adjusted_vec).item())
                    if selected_after != selected_before:
                        stats.changed_step_count += 1
                    for trace in step_traces:
                        trace.update({
                            "selected_token_after": selected_after,
                            "output_changed": selected_after != selected_before,
                        })
                    traces.extend(step_traces)
                    generated.append(selected_after)
                    next_tensor = torch.tensor([[selected_after]], device=input_ids.device, dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, next_tensor], dim=1)
                    if selected_after == getattr(self.tokenizer, "eos_token_id", None):
                        break
        except Exception:
            stats.exception_count += 1
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        stats.generated_token_count = len(generated)
        stats.runtime_seconds = time.time() - start
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, traces, stats.to_dict()

    def _image_tensors(self, image_path: str | Path) -> dict[str, Any]:
        image_path = self._resolve_image_path(image_path)
        image = Image.open(image_path).convert("RGB")
        views = {
            "real": image,
            "blur": image.filter(ImageFilter.GaussianBlur(radius=float(self.method_config.blur_radius))),
            "black": Image.new("RGB", image.size, (0, 0, 0)),
        }
        tensors = {}
        for name, view in views.items():
            tensors[name] = self.image_processor.preprocess(view, return_tensors="pt")["pixel_values"][0].unsqueeze(0).half().cuda()
        return tensors

    def _resolve_image_path(self, image_path: str | Path) -> Path:
        path = Path(image_path).expanduser()
        if path.exists():
            return path
        name = path.name
        candidates = []
        for section_name in ("chair", "pope"):
            section = self.config.get(section_name, {}) if isinstance(self.config, dict) else {}
            root = section.get("image_root") if isinstance(section, dict) else None
            if root:
                candidates.append(Path(str(root)).expanduser() / name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"image not found: {image_path}; tried {[str(c) for c in candidates]}")


# Compatibility alias for source experiments that imported the historical name.
LegacyPreAttributeV1Engine = SCCDEngine


def aggregate_trace_actions(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(t.get("action")) for t in traces)
    non_object_adjustment_count = sum(1 for t in traces if not t.get("canonical_object") and abs(float(t.get("delta") or 0.0)) > 0)
    return {
        "action_counts": dict(counts),
        "non_object_adjustment_count": non_object_adjustment_count,
    }


def _empty_score_fields(probe_question: str | None = None) -> dict[str, Any]:
    return {
        "probe_question": probe_question,
        "claim_text": None,
        "logp_yes_real": None,
        "logp_no_real": None,
        "logp_yes_blur": None,
        "logp_no_blur": None,
        "logp_yes_black": None,
        "logp_no_black": None,
        "margin_real": None,
        "margin_blur": None,
        "margin_black": None,
        "necessity": None,
        "prior_risk": None,
        "fs": None,
        "evidence_score": None,
    }
