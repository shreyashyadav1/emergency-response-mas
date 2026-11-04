"""Pure domain functions shared by the generator, the simulator and the evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from . import params as P

Point = tuple[float, float]


def zone_of(x: float, y: float) -> str:
    r = math.hypot(x - P.CITY_CENTER[0], y - P.CITY_CENTER[1])
    if r < P.ZONE_RADII_KM[0]:
        return "core"
    if r < P.ZONE_RADII_KM[1]:
        return "inner"
    return "outer"


def quadrant_of(x: float, y: float) -> str:
    return ("N" if y >= P.CITY_CENTER[1] else "S") + ("E" if x >= P.CITY_CENTER[0] else "W")


def straight_km(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class TravelModel:
    """Road distance is straight-line distance x circuity, driven at a speed that
    depends on the congestion of both end zones and on a time-of-day multiplier."""

    congestion_profile: tuple[tuple[float, float, float], ...] = ((0.0, 90.0, 1.0),)

    def multiplier(self, t_min: float) -> float:
        for start, end, mult in self.congestion_profile:
            if start <= t_min < end:
                return mult
        return 1.0

    def minutes(self, a: Point, b: Point, speed_kmh: float, depart_min: float = 0.0) -> float:
        zone_factor = 0.5 * (P.ZONE_SPEED_FACTOR[zone_of(*a)] + P.ZONE_SPEED_FACTOR[zone_of(*b)])
        hours = straight_km(a, b) * P.CIRCUITY / (speed_kmh * zone_factor)
        return hours * 60.0 * self.multiplier(depart_min)


def survival_probability(minutes_to_care: float, golden_window: float,
                         survival_if_timely: float, half_life: float) -> float:
    """Flat while care starts inside the golden window, then halves every `half_life` minutes."""
    if minutes_to_care <= golden_window:
        return survival_if_timely
    return survival_if_timely * 0.5 ** ((minutes_to_care - golden_window) / half_life)


@lru_cache(maxsize=4096)
def expected_survival(triage: str, minutes_to_care: int, stabilized: bool = False) -> float:
    """Survival if care starts `minutes_to_care` after the call, averaged over the golden windows this
    triage color can have. Agents use this: they know the model but not each patient's own window."""
    lo, hi = P.GOLDEN_WINDOW_MIN[triage]
    bonus = P.STABILIZATION_BONUS_MIN[triage] if stabilized else 0
    windows = [lo + (hi - lo) * (k + 0.5) / 24 for k in range(24)]
    return sum(survival_probability(minutes_to_care, w + bonus, P.SURVIVAL_IF_TIMELY[triage], P.DECAY_HALF_LIFE_MIN[triage])
               for w in windows) / len(windows)


def urgency_truth(victims: Iterable[tuple[str, float]]) -> float:
    """Oracle urgency of an incident from (triage, golden window) of its victims.
    Used only to score the Incident Assessment Agent; agents never see it."""
    return sum(P.TRIAGE_WEIGHT[triage] * 60.0 / window for triage, window in victims)
