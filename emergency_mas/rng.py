"""Seeded random streams.

Each kind of entity draws from its own independent stream, so adding a field to
hospitals never shifts the numbers drawn for incidents. A stream is identified by
(seed, stream id).
"""
from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")

STREAM_IDS = {
    "city": 1, "hotspots": 2, "hospitals": 3, "fleet": 4, "staff": 5, "names": 6,
    "incidents": 7, "victims": 8, "reports": 9, "history": 10, "sites": 11,
}


class Rng:
    """Thin wrapper over numpy's Generator that returns plain Python types."""

    def __init__(self, gen: np.random.Generator, entropy: tuple[int, int]):
        self._g = gen
        self.entropy = entropy

    def random(self) -> float:
        return float(self._g.random())

    def uniform(self, lo: float, hi: float) -> float:
        return float(self._g.uniform(lo, hi))

    def normal(self, mu: float, sd: float) -> float:
        return float(self._g.normal(mu, sd))

    def integers(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi], both ends included."""
        return int(self._g.integers(lo, hi + 1))

    def poisson(self, lam: float) -> int:
        return int(self._g.poisson(lam))

    def binomial(self, n: int, p: float) -> int:
        return int(self._g.binomial(n, p)) if n > 0 else 0

    def beta(self, a: float, b: float) -> float:
        return float(self._g.beta(a, b))

    def bernoulli(self, p: float) -> bool:
        return bool(self._g.random() < p)

    def choice(self, options: Sequence[T], p: Sequence[float] | None = None) -> T:
        return options[int(self._g.choice(len(options), p=p))]

    def weighted(self, weights: dict[str, float]) -> str:
        keys = list(weights)
        w = np.array([weights[k] for k in keys], dtype=float)
        return keys[int(self._g.choice(len(keys), p=w / w.sum()))]

    def seed_int(self) -> int:
        return int(self._g.integers(0, 2**31 - 1))


def make_rng(seed: int, stream: str) -> Rng:
    entropy = (int(seed), STREAM_IDS[stream])
    gen = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(entropy))))
    return Rng(gen, entropy)
