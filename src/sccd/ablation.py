"""Single-factor SCCD verifier ablations for the primary LLaVA backbone."""
from dataclasses import replace
from typing import Any
from .risk_gate import decide_action

class AblationVerifier:
    def __init__(self, base: Any, mode: str): self.base,self.mode=base,mode
    def score(self, claim, image_tensors):
        score=self.base.score(claim,image_tensors); d=score.decision
        if self.mode=='no_prior': d=decide_action(score.real.margin,score.blur.margin,0.0,self.base.config)
        elif self.mode=='no_blur': d=decide_action(score.real.margin,score.real.margin,score.black.margin,self.base.config)
        elif self.mode=='no_suppression' and d.action=='suppress' and score.real.margin>0: d=replace(d,action='keep',delta=0.0)
        elif self.mode=='no_promotion' and d.action=='promote': d=replace(d,action='suppress',delta=0.0)
        return replace(score,decision=d)
