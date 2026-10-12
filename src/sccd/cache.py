"""Sample-local cache for v1 object verifier scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleScoreCache:
    sample_id: str | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def start(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.scores.clear()
        self.hits = 0
        self.misses = 0

    def get(self, sample_id: str, key: str) -> Any | None:
        if sample_id != self.sample_id:
            return None
        if key in self.scores:
            self.hits += 1
            return self.scores[key]
        self.misses += 1
        return None

    def set(self, sample_id: str, key: str, value: Any) -> None:
        if sample_id != self.sample_id:
            raise ValueError("sample-local score cache cannot cross samples")
        self.scores[key] = value
