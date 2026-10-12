"""Canonical, method-neutral prompt construction for binary visual QA.

This module is intentionally independent of decoding algorithms. Every LLaVA
binary method receives the same base prompt; methods may only differ in their
visual or contrastive branches after this text has been constructed.
"""

from __future__ import annotations


BINARY_PROMPT_PROTOCOL_ID = "canonical_binary_prompt_v1"


def normalize_binary_question(question: str) -> str:
    """Return one stable question line without literal escape sequences."""
    value = str(question).strip()
    if "\\\\n" in value:
        raise ValueError("binary question contains a literal backslash-n escape sequence")
    return value


def build_binary_question(question: str) -> str:
    """Question supplied to chat-template backbones such as Qwen."""
    return f"{normalize_binary_question(question)}\nPlease answer yes or no."


def build_llava_binary_prompt(question: str) -> str:
    """Exact LLaVA text prompt used before image-token tokenization."""
    return (
        "USER: <image>\n"
        f"{normalize_binary_question(question)}\n"
        "Please answer yes or no.\n"
        "ASSISTANT:"
    )
