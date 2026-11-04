"""Stress-knob calibration: how the pressure indices spread over many seeds."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .. import params as P
from .analysis import contention_report, pressure_indices, victim_counts
from .generator import generate_scenario
from .validate import validate_scenario

INDEX_COLUMNS = ["victims", "red", "yellow", "green", "ambulance_pressure", "red_per_als_unit",
                 "er_bay_pressure", "doctor_pressure", "nurse_pressure", "specialist_pressure", "ambulance_pairs"]


def calibration_frame(seeds: Iterable[int] = range(1, 201), themes: Iterable[str] = tuple(P.THEMES),
                      stresses: Iterable[str] = tuple(P.STRESS)) -> pd.DataFrame:
    """One row per (theme, stress, seed) with the indices of the scenario it generates."""
    rows = []
    for theme in themes:
        for stress in stresses:
            for seed in seeds:
                s = generate_scenario(theme, stress, seed)
                rows.append({"theme": theme, "stress": stress, "seed": seed, **victim_counts(s),
                             **pressure_indices(s), "ambulance_pairs": contention_report(s)["n_ambulance_pairs"],
                             "checks_failed": sum(not c.passed for c in validate_scenario(s))})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Median and 10th/90th percentile of every index per theme and stress level."""
    order = {name: i for i, name in enumerate(P.STRESS)}
    grouped = frame.groupby(["theme", "stress"])[INDEX_COLUMNS]
    out = pd.concat({"p10": grouped.quantile(0.10), "p50": grouped.quantile(0.50), "p90": grouped.quantile(0.90)}, axis=1)
    out = out.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return out.sort_index(key=lambda idx: idx.map(order) if idx.name == "stress" else idx)
