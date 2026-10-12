import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.runtime import resolve_run_root
from evaluation.unified_eval_utils import load_unified_config


ROOT = Path(__file__).resolve().parents[1]
TEST_ENV = {
    "SEVD_ROOT": str(ROOT),
    "LLAVA_MODEL_PATH": "/models/llava-v1.5-7b",
    "LLAVA_CODE_PATH": "/src/LLaVA",
    "COCO_IMAGE_ROOT": "/datasets/coco/val2014",
    "COCO_ANNOTATION_ROOT": "/datasets/coco/annotations",
    "MME_ROOT": "/datasets/MME_Benchmark",
    "AMBER_ROOT": "/datasets/amber",
}


class ConfigAndProvenanceTests(unittest.TestCase):
    def load(self):
        with patch.dict(os.environ, TEST_ENV, clear=False):
            return load_unified_config(ROOT / "configs" / "sevd_llava15_7b.yaml")

    def test_final_config_is_object_existence_only(self) -> None:
        method = self.load()["sccd_v1"]
        self.assertEqual(method["source_method_slug"], "object_main_candidate_textonly_v1")
        self.assertEqual(method["prior_mode"], "text_only_exact")
        self.assertFalse(method["attribute_enabled"])
        self.assertFalse(method["count_enabled"])
        self.assertFalse(method["relation_enabled"])

    def test_tuned_v3_gate_is_frozen(self) -> None:
        method = self.load()["sccd_v1"]
        self.assertEqual(method["promote_threshold"], 0.45)
        self.assertEqual(method["margin_floor"], -0.55)
        self.assertEqual(method["suppress_threshold"], -1.0)

    def test_run_layout_is_stage_benchmark_method_run(self) -> None:
        config = self.load()
        resolved = resolve_run_root(config, "smoke", "mme", "sccd_v1", "audit")
        self.assertEqual(resolved, ROOT / "outputs" / "smoke" / "mme" / "sccd_v1" / "audit")

    def test_historical_reported_tables_are_not_a_result_source(self) -> None:
        self.assertFalse((ROOT / "results" / "reported").exists())


if __name__ == "__main__":
    unittest.main()
