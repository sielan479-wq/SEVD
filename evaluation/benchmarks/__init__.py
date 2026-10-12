"""Benchmark-specific SCCD evaluation flows."""

from evaluation.benchmarks import amber, chair, mme, pope

RUNNERS = {
    "pope": pope.run,
    "chair": chair.run,
    "mme": mme.run,
    "amber": amber.run,
}

__all__ = ["RUNNERS"]
