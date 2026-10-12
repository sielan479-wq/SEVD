#!/usr/bin/env python3
"""Prompt builders used by SCCD evaluation."""

from __future__ import annotations

import argparse

POPE_TEMPLATE_ID = "unified_yes_no"
CHAIR_TEMPLATE_ID = "unified_caption"
PARSER_ID = "unified_yes_no_parser_v1"


def build_pope_prompt(question: str) -> str:
    return f"USER: <image>\n{question.strip()}\nPlease answer yes or no.\nASSISTANT:"


def build_chair_prompt(prompt: str = "Describe the image in detail.") -> str:
    return f"USER: <image>\n{prompt.strip()}\nASSISTANT:"


def template_text(template_id: str) -> str:
    if template_id == POPE_TEMPLATE_ID:
        return "USER: <image>\\n{question}\\nPlease answer yes or no.\\nASSISTANT:"
    if template_id == CHAIR_TEMPLATE_ID:
        return "USER: <image>\\nDescribe the image in detail.\\nASSISTANT:"
    raise ValueError(f"Unknown prompt template: {template_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert "Please answer yes or no." in build_pope_prompt("Is there a dog in the image?")
        assert build_chair_prompt().startswith("USER: <image>")
        print("unified_prompts self-test passed")
    else:
        print(template_text(POPE_TEMPLATE_ID))
        print(template_text(CHAIR_TEMPLATE_ID))


if __name__ == "__main__":
    main()
