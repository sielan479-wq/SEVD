"""Standard greedy Qwen2.5-VL engine under SCCD's common evaluation contract."""
from __future__ import annotations
from evaluation.qwen_engine_v2 import QwenSCCDEngineV2

class QwenGreedyEngine(QwenSCCDEngineV2):
    def answer_object_existence_sample(self, row, benchmark):
        prompt=f"USER: <image>\n{row['question'].strip()}\nASSISTANT:"
        text, _, stats=self.generate_chair_sample(row,prompt,8,benchmark)
        stats.update({'input_claim_route_count':0,'input_claim_parse_failure_count':0,'unique_claims_scored':0,'verifier_forward_count':0})
        return text, [], stats
    def answer_pope_sample(self,row):
        return self.answer_object_existence_sample(row,'pope')
