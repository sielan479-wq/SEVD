import unittest

from evaluation.runtime import summarize_interventions


class InterventionDiagnosticsTests(unittest.TestCase):
    def test_nonzero_adjustment_without_output_change_is_not_effective(self) -> None:
        summary = summarize_interventions(
            [
                {
                    "sample_id": "1",
                    "stats": {"unique_claims_scored": 1, "changed_step_count": 0},
                    "traces": [
                        {"action": "suppress", "delta": -10000.0, "output_changed": False}
                    ],
                }
            ]
        )
        self.assertTrue(summary["nonzero_adjustment_detected"])
        self.assertFalse(summary["effective_intervention_detected"])

    def test_output_change_is_an_effective_intervention(self) -> None:
        summary = summarize_interventions(
            [
                {
                    "sample_id": "1",
                    "stats": {"unique_claims_scored": 1, "changed_step_count": 1},
                    "traces": [
                        {"action": "promote", "delta": 0.5, "output_changed": True}
                    ],
                }
            ]
        )
        self.assertEqual(summary["samples_with_output_change"], 1)
        self.assertTrue(summary["effective_intervention_detected"])


if __name__ == "__main__":
    unittest.main()
