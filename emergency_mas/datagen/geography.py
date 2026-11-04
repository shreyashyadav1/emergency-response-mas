"""City layout: where hospitals, stations and demand hotspots go."""
from __future__ import annotations

import math

from .. import params as P
from ..domain import Point, straight_km
from ..rng import Rng

# Preferred distance from the city center (km) for each hospital zone
ZONE_ANNULUS = {"core": (0.0, 3.5), "inner": (5.0, 8.5), "outer": (10.0, 14.0)}


def clip_point(x: float, y: float, margin: float = 0.5) -> Point:
    lo, hi = margin, P.CITY_KM - margin
    return round(max(lo, min(hi, x)), 3), round(max(lo, min(hi, y)), 3)


def point_in_annulus(rng: Rng, r_lo: float, r_hi: float) -> Point:
    r = math.sqrt(rng.uniform(r_lo ** 2, r_hi ** 2))        # uniform over area
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return clip_point(P.CITY_CENTER[0] + r * math.cos(theta), P.CITY_CENTER[1] + r * math.sin(theta))


def place_hospitals(rng: Rng, zones: list[str]) -> list[Point]:
    """One point per hospital, inside its preferred zone and at least 6 km from the others."""
    placed: list[Point] = []
    for zone in zones:
        lo, hi = ZONE_ANNULUS[zone]
        for _ in range(300):
            candidate = point_in_annulus(rng, lo, hi)
            if all(straight_km(candidate, q) >= P.MIN_HOSPITAL_SEPARATION_KM for q in placed):
                break
        placed.append(candidate)
    return placed


def place_stations(rng: Rng, n: int) -> list[Point]:
    """First station near the center, the rest by farthest-point sampling so coverage is even."""
    candidates = [clip_point(rng.uniform(1.0, P.CITY_KM - 1.0), rng.uniform(1.0, P.CITY_KM - 1.0)) for _ in range(400)]
    chosen = [min(candidates, key=lambda c: straight_km(c, P.CITY_CENTER))]
    while len(chosen) < n:
        chosen.append(max(candidates, key=lambda c: min(straight_km(c, q) for q in chosen)))
    return chosen


def place_hotspots(rng: Rng, n: int, corridor: bool) -> list[Point]:
    """Where demand clusters: scattered spots, or points along one highway."""
    if corridor:
        y0, y1 = rng.uniform(7.0, 17.0), rng.uniform(7.0, 17.0)
        fractions = [(i + 1) / (n + 1) for i in range(n)]
        return [clip_point(3.0 + 18.0 * f, y0 + (y1 - y0) * f) for f in fractions]
    spots: list[Point] = []
    for _ in range(n):
        for _try in range(100):
            candidate = point_in_annulus(rng, 2.0, 9.0)
            if all(straight_km(candidate, q) >= 5.0 for q in spots):
                break
        spots.append(candidate)
    return spots


def incident_location(rng: Rng, cluster_frac: float, sigma_km: float, hotspots: list[Point]) -> Point:
    if hotspots and rng.bernoulli(cluster_frac):
        hx, hy = rng.choice(hotspots)
        return clip_point(rng.normal(hx, sigma_km), rng.normal(hy, sigma_km))
    return clip_point(rng.normal(P.CITY_CENTER[0], 6.0), rng.normal(P.CITY_CENTER[1], 6.0))
