#!/usr/bin/env python3
"""LLaVA model loading helpers for SCCD."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any


def add_llava_path(config: dict[str, Any]) -> None:
    llava_path = Path(str(config["llava_code_path"])).expanduser().resolve()
    path_text = str(llava_path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as transformers_set_seed

        transformers_set_seed(seed)
    except ImportError:
        pass


def load_model(config: dict[str, Any], method: str = "sccd_v1"):
    add_llava_path(config)
    from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    set_seed(int(config.get(method, {}).get("seed", 55)))
    model_path = str(config["model_path"])
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name
    )
    model.eval()
    return tokenizer, model, image_processor, tokenizer_image_token
