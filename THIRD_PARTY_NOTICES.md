# Third-party notices

The Apache License 2.0 in the repository root applies to original SEVD
code and documentation. It does not replace the licenses or terms of the
models, datasets, benchmark annotations, or evaluation tools named below.

## CHAIR vocabulary

`assets/chair/synonyms.txt` is derived from the CHAIR release:

- Project: Object Hallucination in Image Captioning (CHAIR)
- Upstream: https://github.com/LisaAnne/Hallucination
- Source: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
- Upstream license statement: BSD 2-Clause License
- SPDX identifier: `BSD-2-Clause`

The upstream README identifies the repository as BSD-2-Clause licensed but
does not provide a separate LICENSE file or a precise copyright notice. SEVD
preserves the upstream authorship, project link, and license designation
without inventing a copyright holder or year.

```
The BSD 2-Clause license conditions and disclaimer are reproduced
below for reference. The upstream repository does not provide a
complete copyright notice for the redistributed vocabulary file;
therefore, no copyright holder or year is asserted here.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

The CHAIR scorer formerly bundled in the internal project is intentionally
not redistributed in this public release. That copy included modifications
from `Maxlinn/CHAIR-metric-standalone`, for which no explicit redistribution
license was found. Users must provide their own authorized scorer installation.

## MME evaluation tool

Official project: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation

Official evaluation tool: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/tools/eval_tool.zip

No explicit software redistribution license was found in the official MME
evaluation archive or Evaluation branch at the time this release was prepared.
SEVD therefore does not redistribute `eval_tool/calculation.py`. Users must
obtain the official tool from its authors and use it under the applicable
upstream terms, or configure a compatible scorer that they are authorized to
use.

## POPE benchmark metadata

The files under `data/manifests/pope/` reproduce benchmark questions and labels
from POPE:

- Project: POPE: Polling-based Object Probing Evaluation for Object Hallucination
- Upstream: https://github.com/RUCAIBox/POPE
- License: MIT License
- Copyright: Copyright (c) 2024 Yifan Li

The upstream MIT terms are:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## AMBER benchmark metadata

`data/manifests/amber/all.jsonl` contains normalized identifiers, task fields,
and queries derived from AMBER:

- Project: AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation
- Upstream: https://github.com/junyangwang0410/AMBER
- License: Apache License 2.0
- Copyright: Copyright 2023 Alibaba X-PLUG Team

The upstream Apache License 2.0 terms are the same license text reproduced in
the repository-root `LICENSE`. This attribution does not transfer ownership of
AMBER material to the SEVD authors.

The AMBER manifest is adapted from the upstream benchmark query
metadata into the SEVD evaluation format. This transformation
includes field normalization and evaluation-specific metadata.
No AMBER image files are redistributed.

## MS COCO identifiers and annotations

The CHAIR manifest contains COCO image identifiers and filenames. SEVD does not
redistribute COCO images or annotation JSON files. COCO annotation files are
provided upstream under Creative Commons Attribution 4.0; individual images
remain governed by the licenses or terms selected by their Flickr uploaders.
See https://cocodataset.org/#termsofuse before downloading or redistributing
COCO material.

## MME local metadata

MME questions, answers, and image metadata are not committed. The local
manifest builder reads them from the user's authorized benchmark copy and
writes `data/manifests/mme/all.jsonl`, which is ignored by Git. Generation of a
local derived file does not grant redistribution rights in the source dataset.

## Models and datasets

SEVD does not redistribute model weights, dataset images, COCO annotation JSON
files, MME data, the complete AMBER package, or the external AMBER scorer. The
limited POPE and AMBER benchmark metadata that is included is identified above
and remains under its upstream license. Users are responsible for obtaining all
other materials from their publishers and complying with the applicable terms.

The repository also interoperates with LLaVA, Qwen2.5-VL, MS COCO, POPE,
CHAIR, MME, and AMBER. Those names remain the property of their respective
owners. Reference to them does not imply endorsement or a transfer of rights.
