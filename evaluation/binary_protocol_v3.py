"""Final canonical binary helpers (v3) used by the protocol rerun entry.

Kept separate from the historical evaluation modules so old formal snapshots and
their provenance remain byte-for-byte untouched.
"""

from __future__ import annotations

import re
from typing import Any

from PIL import Image

from evaluation.answer_parser import parse_answer


PROTOCOL_ID = "binary_prompt_fallback_protocol_v3"


def question_only(question: str) -> str:
    value = str(question).strip()
    if "\\\\n" in value:
        raise ValueError("literal backslash-n in binary question")
    return re.sub(r"\s*please answer yes or no\.\s*$", "", value, flags=re.I)


def llava_prompt(question: str) -> str:
    return f"USER: <image>\n{question_only(question)}\nPlease answer yes or no.\nASSISTANT:"


def qwen_question(question: str) -> str:
    return f"{question_only(question)}\nPlease answer yes or no."


def resolve(engine: Any, row: dict[str, Any], raw_answer: str, benchmark: str):
    original = str(raw_answer)
    parsed = parse_answer(original)
    audit = {"original_raw_answer": original, "original_parsed_answer": parsed, "fallback_triggered": False, "fallback_prompt": None, "fallback_yes_score": None, "fallback_no_score": None, "fallback_answer": None}
    if parsed in {"yes", "no"}:
        return parsed, audit
    callback = getattr(engine, "binary_parser_fallback", None)
    if callback is None:
        return parsed, audit
    model_row = {key: row[key] for key in ("sample_id", "image_id", "image_path", "question") if key in row}
    out = callback(model_row, benchmark)
    answer = str(out["answer"]).lower()
    if answer not in {"yes", "no"}:
        raise RuntimeError("fallback did not produce Yes/No")
    audit.update({"fallback_triggered": True, "fallback_prompt": out["prompt"], "fallback_yes_score": float(out["yes_score"]), "fallback_no_score": float(out["no_score"]), "fallback_answer": answer})
    return answer, audit


class LLaVABaseline:
    def answer_pope_sample(self, row):
        return self._generate(row, llava_prompt(row["question"]), 8, "pope")

    def answer_object_existence_sample(self, row, benchmark):
        return self._generate(row, llava_prompt(row["question"]), 8, benchmark)


class QwenFallback:
    def binary_parser_fallback(self, row, benchmark):
        import torch
        prompt = qwen_question(row["question"])
        image = Image.open(str(row["image_path"])).convert("RGB")
        inputs = self._inputs(prompt, image)
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[:, -1, :].float().squeeze(0)
        logp = torch.log_softmax(logits, dim=-1)
        yes = torch.logsumexp(logp[list(self.yes_ids)], 0).item()
        no = torch.logsumexp(logp[list(self.no_ids)], 0).item()
        return {"answer": "yes" if yes > no else "no", "prompt": prompt, "yes_score": yes, "no_score": no, "benchmark": benchmark}


class QwenGreedy:
    def answer_object_existence_sample(self, row, benchmark):
        import time
        import torch
        started = time.time()
        image = Image.open(str(row["image_path"])).convert("RGB")
        inputs = self._inputs(qwen_question(row["question"]), image)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=8)
        raw = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return raw, [], {"input_claim_route_count": 0, "input_claim_parse_failure_count": 0, "unique_claims_scored": 0, "verifier_forward_count": 0, "changed_step_count": 0, "generated_token_count": int(output.shape[1]-inputs.input_ids.shape[1]), "runtime_seconds": time.time()-started, "exception_count": 0}

    def answer_pope_sample(self, row):
        return self.answer_object_existence_sample(row, "pope")
