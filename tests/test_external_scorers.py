import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.external_scorers import resolve_external_scorer


class ExternalScorerTests(unittest.TestCase):
    def test_missing_optional_scorer_is_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_external_scorer("chair", required=False))

    def test_missing_required_scorer_has_actionable_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SEVD_MME_SCORER_PATH"):
                resolve_external_scorer("mme")

    def test_existing_external_scorer_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scorer = Path(directory) / "calculation.py"
            scorer.write_text("# local test scorer\n", encoding="utf-8")
            with patch.dict(
                os.environ, {"SEVD_MME_SCORER_PATH": str(scorer)}, clear=True
            ):
                self.assertEqual(resolve_external_scorer("mme"), scorer.resolve())

    def test_unknown_benchmark_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No external scorer contract"):
            resolve_external_scorer("unknown")


if __name__ == "__main__":
    unittest.main()
