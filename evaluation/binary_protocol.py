"""Shared protocol-v2 utilities for binary POPE/MME evaluation.

The fallback deliberately receives only a minimal model-input row.  It never
sees benchmark labels, parser correctness, or any score-derived signal.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from evaluation.answer_parser import parse_answer
from sccd.binary_prompt import build_binary_question, build_llava_binary_prompt


BINARY_PROTOCOL_ID = "binary_prompt_fallback_protocol_v2"


def resolve_binary_answer(
    engine: Any, row: dict[str, Any], raw_answer: str, benchmark: str
) -> tuple[str, dict[str, Any]]:
    """Parse once, then invoke a deterministic fallback only for ``unknown``."""
    original = str(raw_answer)
    parsed = parse_answer(original)
    audit: dict[str, Any] = {
        "original_raw_answer": original,
        "original_parsed_answer": parsed,
        "fallback_triggered": False,
        "fallback_prompt": None,
        "fallback_yes_score": None,
        "fallback_no_score": None,
        "fallback_answer": None,
    }
    if parsed in {"yes", "no"}:
        return parsed, audit
    fallback = getattr(engine, "binary_parser_fallback", None)
    if fallback is None:
        return parsed, audit
    # Do not pass label, ground truth, or prior parsing result into the model
    # fallback. Its only selection criterion is the deterministic log-prob argmax.
    model_row = {
        key: row[key]
        for key in ("sample_id", "image_id", "image_path", "question")
        if key in row
    }
    outcome = fallback(model_row, benchmark)
    answer = str(outcome["answer"]).lower()
    if answer not in {"yes", "no"}:
        raise RuntimeError(f"binary fallback returned invalid answer: {answer!r}")
    audit.update(
        {
            "fallback_triggered": True,
            "fallback_prompt": outcome["prompt"],
            "fallback_yes_score": float(outcome["yes_score"]),
            "fallback_no_score": float(outcome["no_score"]),
            "fallback_answer": answer,
        }
    )
    return answer, audit


class ProtocolBaselineEngineMixin:
    """Canonical prompt overrides for Greedy/VCD/ICD/RITUAL only."""

    def answer_pope_sample(self, row: dict[str, Any]):
        return self._generate(row, build_llava_binary_prompt(row["question"]), 8, "pope")

    def answer_object_existence_sample(self, row: dict[str, Any], benchmark: str):
        return self._generate(
            row, build_llava_binary_prompt(row["question"]), 8, benchmark
        )


class QwenProtocolFallbackMixin:
    """Deterministic label-blind fallback using the engine's frozen verbalizers."""

    def binary_parser_fallback(self, row: dict[str, Any], benchmark: str) -> dict[str, Any]:
        import torch

        prompt = build_binary_question(str(row["question"]))
        image = Image.open(str(row["image_path"])).convert("RGB")
        inputs = self._inputs(prompt, image)
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[:, -1, :].float().squeeze(0)
        logp = torch.log_softmax(logits, dim=-1)
        yes = torch.logsumexp(logp[list(self.yes_ids)], 0).item()
        no = torch.logsumexp(logp[list(self.no_ids)], 0).item()
        return {
            "answer": "yes" if yes > no else "no",
            "prompt": prompt,
            "yes_score": yes,
            "no_score": no,
            "protocol": "qwen_yes_no_verbalizer_argmax_v1",
            "benchmark": benchmark,
        }


class ProtocolQwenGreedyMixin:
    """Greedy binary generation with the same explicit Yes/No instruction."""

    def answer_object_existence_sample(self, row: dict[str, Any], benchmark: str):
        import time
        import torch

        started = time.time()
        image = Image.open(str(row["image_path"])).convert("RGB")
        prompt = build_binary_question(str(row["question"]))
        inputs = self._inputs(prompt, image)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=8)
        text = self.processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text, [], {
            "input_claim_route_count": 0,
            "input_claim_parse_failure_count": 0,
            "unique_claims_scored": 0,
            "verifier_forward_count": 0,
            "changed_step_count": 0,
            "generated_token_count": int(output.shape[1] - inputs.input_ids.shape[1]),
            "runtime_seconds": time.time() - started,
            "exception_count": 0,
        }

    def answer_pope_sample(self, row: dict[str, Any]):
        return self.answer_object_existence_sample(row, "pope")
