import unittest

from sccd.risk_gate import FrozenRiskGateConfig, decide_action, evidence_score


class RiskGateTests(unittest.TestCase):
    def test_faithfulness_score_matches_frozen_formula(self) -> None:
        necessity, prior_risk, fs, score = evidence_score(1.0, 0.25, 0.4)
        self.assertAlmostEqual(necessity, 0.75)
        self.assertAlmostEqual(prior_risk, 0.4)
        self.assertAlmostEqual(fs, 0.55)
        self.assertAlmostEqual(score, 0.55)

    def test_gate_suppresses_unsupported_positive_claim(self) -> None:
        decision = decide_action(0.2, 1.0, 0.0)
        self.assertEqual(decision.action, "suppress")
        self.assertEqual(
            decision.delta, -FrozenRiskGateConfig().suppression_strength
        )

    def test_gate_promotes_supported_near_margin_claim(self) -> None:
        decision = decide_action(-0.1, -0.8, 0.0)
        self.assertEqual(decision.action, "promote")
        self.assertEqual(decision.delta, FrozenRiskGateConfig().promotion_strength)


if __name__ == "__main__":
    unittest.main()
