"""CHAIR/COCO object vocabulary for SCCD's current claim instantiation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_TOKEN_RE = re.compile(r"^[\s▁ĠĊ\W_]+|[\s\W_]+$")


@dataclass(frozen=True)
class ObjectVocabulary:
    canonical_objects: tuple[str, ...]
    synonym_to_canonical: dict[str, str]

    @classmethod
    def from_synonyms_file(cls, path: str | Path) -> "ObjectVocabulary":
        path = Path(path).expanduser()
        canonical: list[str] = []
        mapping: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
            if not parts:
                continue
            head = parts[0]
            canonical.append(head)
            for part in parts:
                mapping[part] = head
        return cls(tuple(canonical), mapping)

    def normalize_text(self, text: str) -> str:
        text = text.replace("</s>", "").replace("<unk>", "")
        text = _TOKEN_RE.sub("", text.lower().strip())
        text = text.replace("▁", "").replace("Ġ", "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def canonicalize_phrase(self, text: str) -> str | None:
        norm = self.normalize_text(text)
        if not norm:
            return None
        if norm in self.synonym_to_canonical:
            return self.synonym_to_canonical[norm]
        singular = _simple_singular(norm)
        if singular in self.synonym_to_canonical:
            return self.synonym_to_canonical[singular]
        return None

    def canonicalize_token(self, token_text: str) -> str | None:
        norm = self.normalize_text(token_text)
        if not norm or " " in norm:
            return None
        return self.canonicalize_phrase(norm)


def _simple_singular(text: str) -> str:
    if " " in text:
        return " ".join(_simple_singular(part) for part in text.split())
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith("ses") and len(text) > 4:
        return text[:-2]
    if text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        return text[:-1]
    return text
