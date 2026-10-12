import unittest
from evaluation.binary_protocol_v3 import llava_prompt, resolve

class Engine:
 def __init__(self): self.seen=[]
 def binary_parser_fallback(self,row,benchmark): self.seen.append(row);return {'answer':'yes','prompt':'p','yes_score':1.,'no_score':0.}

class BinaryProtocolV3(unittest.TestCase):
 def test_real_newline_and_single_instruction(self):
  p=llava_prompt('Is there a dog? Please answer yes or no.')
  self.assertNotIn('\\\\n',p);self.assertEqual(p.lower().count('please answer yes or no.'),1)
 def test_success_skips_fallback(self):
  e=Engine();answer,audit=resolve(e,{'label':'no','question':'q'},'No.','pope');self.assertEqual(answer,'no');self.assertFalse(audit['fallback_triggered']);self.assertFalse(e.seen)
 def test_unknown_uses_label_blind_fallback(self):
  e=Engine();answer,audit=resolve(e,{'label':'no','question':'q','image_path':'x'},'unclear','pope');self.assertEqual(answer,'yes');self.assertTrue(audit['fallback_triggered']);self.assertNotIn('label',e.seen[0])
if __name__=='__main__':unittest.main()
