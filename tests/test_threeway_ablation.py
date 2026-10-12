import unittest

from sccd.object_claim import construct_object_claim
from sccd.risk_gate import RiskGateDecision
from sccd.threeway_ablation import SemanticEvidenceAblationVerifier
from sccd.verifier import ObjectVerificationScore, ViewScores


class BaseVerifier:
    def score(self, *_):
        decision = RiskGateDecision(-0.1, -0.8, 0.0, 0.0, 0.0, 1.0, 1.0, "promote", 0.5)
        return ObjectVerificationScore(
            claim=construct_object_claim("cat"),
            real=ViewScores(0.0, 0.1, -0.1, 1),
            blur=ViewScores(0.0, 0.8, -0.8, 1),
            black=ViewScores(0.0, 0.0, 0.0, 1),
            decision=decision,
            positive_token_ids=(1,),
            negative_token_ids=(2,),
        )


class ThreeWayAblationTests(unittest.TestCase):
    def test_real_only_uses_real_margin(self):
        score = SemanticEvidenceAblationVerifier(BaseVerifier(), "real_only").score(None, None)
        self.assertEqual(score.decision.action, "suppress")

    def test_suppression_only_disables_promotion(self):
        score = SemanticEvidenceAblationVerifier(BaseVerifier(), "suppression_only").score(None, None)
        self.assertEqual(score.decision.action, "keep")


if __name__ == "__main__":
    unittest.main()
