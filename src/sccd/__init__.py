"""Semantic Claim Contrastive Decoding.

The GPU/Pillow-dependent engine is imported lazily so configuration, CLI help,
path planning, and lightweight component tests do not load the model stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SCCDEngine"]


def __getattr__(name: str) -> Any:
    if name == "SCCDEngine":
        from .engine import SCCDEngine

        return SCCDEngine
    raise AttributeError(name)
