"""First-version object-existence claim construction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .vocabulary import ObjectVocabulary


@dataclass(frozen=True)
class ObjectClaim:
    canonical_object: str
    claim_text: str
    probe_question: str


def article_for(text: str) -> str:
    return "an" if text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def construct_object_claim(canonical_object: str) -> ObjectClaim:
    phrase = canonical_object.replace("_", " ").strip()
    return ObjectClaim(
        canonical_object=phrase,
        claim_text=f"{phrase} exists",
        probe_question=f"Is there {article_for(phrase)} {phrase} in the image?",
    )


_EXISTENCE_RE = re.compile(
    r"^(?:is|are)\s+there\s+(?:a|an|the|any)?\s*(.*?)\s+"
    r"in\s+(?:this|the)\s+(?:image|picture)\?"
    r"(?:\s*please\s+answer\s+yes\s+or\s+no\.)?\s*$",
    re.I,
)


def object_from_existence_question(
    question: str, vocab: ObjectVocabulary
) -> ObjectClaim | None:
    """Parse an explicit object-existence question into the frozen claim type.

    This intentionally accepts only the semantics already supported by SCCD:
    ``exists(object)``. Count, color, position, relation, and other MME tasks
    must not call this adapter even if their surface form starts with
    ``Is there``.
    """
    match = _EXISTENCE_RE.search(question.strip())
    if not match:
        return None
    phrase = match.group(1).strip().rstrip("?").strip()
    canonical = vocab.canonicalize_phrase(phrase) or vocab.normalize_text(phrase)
    if not canonical:
        return None
    return construct_object_claim(canonical)


def object_from_pope_question(question: str, vocab: ObjectVocabulary) -> ObjectClaim | None:
    """Compatibility name for the shared explicit existence-question parser."""
    return object_from_existence_question(question, vocab)
