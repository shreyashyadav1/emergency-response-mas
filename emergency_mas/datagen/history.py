"""Synthetic past incidents, used as base rates by the Incident Assessment Agent.

Every row pairs what a caller reported with what was really there. The agent can look
up how callers of a given kind of incident tend to under-count, without seeing the truth
of the live incident. The seeds start at 1,000,000 so history never overlaps evaluation seeds.
"""
from __future__ import annotations

import pandas as pd

from .. import params as P
from ..rng import make_rng
from .geography import place_hotspots
from .incidents import generate_demand
from .narrative import TemplateNarrator, seeded_faker

HISTORY_SEED_BASE = 1_000_000


def generate_history(n_incidents: int = 600) -> pd.DataFrame:
    themes, stresses = list(P.THEMES), list(P.STRESS)
    rows: list[dict] = []
    batch = 0
    while len(rows) < n_incidents:
        theme = themes[batch % len(themes)]
        stress = stresses[(batch // len(themes)) % len(stresses)]
        seed = HISTORY_SEED_BASE + batch
        profile = P.THEMES[theme]
        incidents, _ = generate_demand(
            seed=seed, theme_name=theme, stress_name=stress, n_incidents=profile.n_incidents,
            hotspots=place_hotspots(make_rng(seed, "hotspots"), profile.n_hotspots, profile.corridor),
            fake=seeded_faker(make_rng(seed, "names").seed_int()), narrator=TemplateNarrator())
        for i in incidents:
            rows.append({
                "incident_type": i.incident_type, "hazard": i.hazard, "zone": i.zone,
                "reported_victims": i.report.reported_victims, "reported_critical": i.report.reported_critical,
                "reported_hazard": i.report.reported_hazard, "reported_trapped": i.report.reported_trapped,
                "caller_confidence": i.report.confidence,
                "true_victims": i.n_victims, "true_red": i.n_red, "true_yellow": i.n_yellow,
                "true_green": i.n_green, "true_trapped": i.n_trapped, "min_golden_window_min": i.min_golden_window_min,
                "urgency_truth": i.urgency_truth,
            })
        batch += 1
    return pd.DataFrame(rows[:n_incidents])


def base_rates(history: pd.DataFrame) -> pd.DataFrame:
    """Per incident type: how far callers are off, and what share of the victims is RED."""
    grouped = history.groupby("incident_type")
    return pd.DataFrame({
        "incidents": grouped.size(),
        "mean_reported_victims": grouped["reported_victims"].mean(),
        "mean_true_victims": grouped["true_victims"].mean(),
        "true_over_reported_victims": grouped["true_victims"].sum() / grouped["reported_victims"].sum(),
        "red_share_of_victims": grouped["true_red"].sum() / grouped["true_victims"].sum(),
        "true_red_over_reported_critical": grouped["true_red"].sum() / grouped["reported_critical"].sum(),
    }).round(3)
