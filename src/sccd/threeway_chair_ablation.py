"""CHAIR extension of the paper's token-level contrast ablation."""
from __future__ import annotations
import time
from typing import Any
from .threeway_ablation import TokenLevelContrastBinaryEngine

class TokenLevelContrastChairEngine(TokenLevelContrastBinaryEngine):
    """All-token real/blur contrast at every caption-decoding step."""
    def generate_chair_sample(self,row:dict[str,Any],prompt:str,max_new_tokens:int,benchmark:str='chair'):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX
        started=time.time(); tensors=self.base._image_tensors(row['image_path'])
        ids=self.tokenizer_image_token(prompt,self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
        generated=[];traces=[];changed=0
        with torch.inference_mode():
            for step in range(int(max_new_tokens)):
                real=self.model(input_ids=ids,images=tensors['real'],use_cache=False,return_dict=True).logits[:,-1,:].float().squeeze(0)
                blur=self.model(input_ids=ids,images=tensors['blur'],use_cache=False,return_dict=True).logits[:,-1,:].float().squeeze(0)
                adjusted=(1.0+self.alpha)*real-self.alpha*blur
                before,after=int(real.argmax()),int(adjusted.argmax()); changed+=int(before!=after)
                traces.append({'sample_id':str(row.get('sample_id')),'benchmark':benchmark,'step':step,'action':'token_level_real_blur_contrast','delta':float((adjusted[after]-real[after]).item()),'selected_token_before':before,'selected_token_after':after,'output_changed':before!=after})
                generated.append(after);ids=torch.cat([ids,torch.tensor([[after]],dtype=ids.dtype,device=ids.device)],dim=1)
                if after==self.tokenizer.eos_token_id:break
        return self.tokenizer.decode(generated,skip_special_tokens=True).strip(),traces,{'generated_token_count':len(generated),'changed_step_count':changed,'method_forward_count':2*len(generated),'runtime_seconds':time.time()-started,'exception_count':0,'unique_claims_scored':0,'verifier_forward_count':0,'input_claim_route_count':0,'input_claim_parse_failure_count':0}
