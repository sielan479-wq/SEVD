"""Resolve locally installed benchmark scorers that are not redistributed.

SEVD deliberately keeps third-party scorer code outside the repository when
an explicit redistribution grant cannot be established. Formal runs record
the resolved path and SHA-256 through benchmark provenance manifests.
"""

from __future__ import annotations

import os
from pathlib import Path


SCORER_ENVIRONMENT = {
    "chair": "SEVD_CHAIR_SCORER_PATH",
    "mme": "SEVD_MME_SCORER_PATH",
}


def resolve_external_scorer(benchmark: str, *, required: bool = True) -> Path | None:
    """Return a validated external scorer path for ``benchmark``."""

    try:
        variable = SCORER_ENVIRONMENT[benchmark]
    except KeyError as error:
        raise ValueError(f"No external scorer contract for benchmark: {benchmark}") from error

    raw = os.environ.get(variable, "").strip()
    if not raw:
        if not required:
            return None
        raise RuntimeError(
            f"{benchmark.upper()} scorer is not bundled with SEVD. Set {variable} "
            "to an authorized local scorer file. See README.md and "
            "THIRD_PARTY_NOTICES.md."
        )

    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{variable} does not point to a file: {path}. "
            "Obtain the scorer from its upstream project under the applicable terms."
        )
    return path
