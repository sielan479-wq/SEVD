import unittest

from evaluation.answer_parser import parse_answer
from evaluation.binary_protocol import resolve_binary_answer
from sccd.binary_prompt import build_llava_binary_prompt


class _FallbackEngine:
    def __init__(self):
        self.rows = []

    def binary_parser_fallback(self, row, benchmark):
        self.rows.append((row, benchmark))
        return {"answer": "no", "prompt": "fallback", "yes_score": -2.0, "no_score": -1.0}


class BinaryProtocolTests(unittest.TestCase):
    def test_canonical_prompt_uses_real_newlines(self):
        prompt = build_llava_binary_prompt("Is there a dog in the image?")
        self.assertIn("\n", prompt)
        self.assertNotIn("\\\\n", prompt)

    def test_method_base_prompt_is_identical(self):
        prompt = build_llava_binary_prompt("Is there a dog in the image?")
        prompts = {method: prompt for method in ("greedy", "vcd", "icd", "ritual", "sccd_v1")}
        self.assertEqual(len(set(prompts.values())), 1)

    def test_parse_success_does_not_trigger_fallback(self):
        engine = _FallbackEngine()
        answer, audit = resolve_binary_answer(engine, {"question": "q"}, "Yes.", "pope")
        self.assertEqual(answer, "yes")
        self.assertFalse(audit["fallback_triggered"])
        self.assertEqual(engine.rows, [])

    def test_parse_failure_triggers_label_blind_deterministic_fallback(self):
        engine = _FallbackEngine()
        row = {"sample_id": "s", "image_id": "i", "image_path": "/tmp/x.jpg", "question": "q", "label": "yes"}
        answer_a, audit_a = resolve_binary_answer(engine, row, "I cannot tell.", "pope")
        answer_b, audit_b = resolve_binary_answer(engine, {**row, "label": "no"}, "I cannot tell.", "pope")
        self.assertEqual(parse_answer("I cannot tell."), "unknown")
        self.assertEqual(answer_a, answer_b)
        self.assertTrue(audit_a["fallback_triggered"])
        self.assertEqual(audit_a["fallback_answer"], "no")
        self.assertNotIn("label", engine.rows[0][0])
        self.assertEqual(audit_a["fallback_yes_score"], audit_b["fallback_yes_score"])


if __name__ == "__main__":
    unittest.main()
