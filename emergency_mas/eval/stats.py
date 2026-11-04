"""Paired comparisons: every policy runs on the same scenarios, so differences are taken seed by seed."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2026


def bootstrap_ci(diffs: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(diffs, size=(BOOTSTRAP_DRAWS, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired(frame: pd.DataFrame, metric: str, a: str, b: str, by: tuple[str, ...] = ("theme", "stress")) -> pd.DataFrame:
    """Mean of `metric` for policies a and b, the mean paired difference a - b with a 95% bootstrap
    interval, the share of seeds where a is better, and a Wilcoxon signed-rank p-value."""
    rows = []
    for key, g in frame.groupby(list(by), sort=False):
        wide = g.pivot_table(index="seed", columns="policy", values=metric)
        if a not in wide or b not in wide:
            continue
        d = (wide[a] - wide[b]).dropna().to_numpy()
        lo, hi = bootstrap_ci(d)
        try:
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "metric": metric, "a": a, "b": b,
                     "n": len(d), "mean_a": wide[a].mean(), "mean_b": wide[b].mean(), "diff": d.mean(),
                     "ci_low": lo, "ci_high": hi, "a_better_share": float(np.mean(d > 0)), "p_value": p})
    return pd.DataFrame(rows)
