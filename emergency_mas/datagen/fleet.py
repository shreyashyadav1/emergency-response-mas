"""Stations, ambulances and first-responder teams."""
from __future__ import annotations

from .. import params as P
from ..domain import Point, quadrant_of, zone_of
from ..rng import Rng
from ..schema import Ambulance, ResponderTeam, Station
from .geography import clip_point


def make_stations(points: list[Point]) -> list[Station]:
    return [
        Station(station_id=f"S{i}", name=f"Station {i} - {quadrant_of(x, y)} {zone_of(x, y)}",
                x=x, y=y, zone=zone_of(x, y))
        for i, (x, y) in enumerate(points, start=1)
    ]


def _return_leg_position(rng: Rng, station: Station) -> Point:
    """A busy unit is somewhere on its way back to the station."""
    return clip_point(station.x + rng.uniform(-5.0, 5.0), station.y + rng.uniform(-5.0, 5.0))


def make_ambulances(rng: Rng, stations: list[Station], n: int, stress: P.StressProfile) -> list[Ambulance]:
    out: list[Ambulance] = []
    for i in range(n):
        station = stations[i % len(stations)]
        unit = P.AMBULANCE_PATTERN[i % len(P.AMBULANCE_PATTERN)]
        capacity = rng.choice([1, 2], p=[0.4, 0.6]) if unit == "ALS" else 2
        busy = rng.bernoulli(stress.ambulance_busy_prob) and i >= 2      # the first two units start free
        x, y = (_return_leg_position(rng, station) if busy else (station.x, station.y))
        out.append(Ambulance(
            ambulance_id=f"A{i + 1}", unit_type=unit, base_station_id=station.station_id, x=x, y=y,
            capacity=capacity, status="busy" if busy else "available",
            available_from_min=rng.integers(5, 25) if busy else 0,
            crew_size=rng.integers(2, 3), speed_kmh=P.AMBULANCE_SPEED_KMH,
        ))
    return out


def make_responders(rng: Rng, stations: list[Station], n: int, stress: P.StressProfile) -> list[ResponderTeam]:
    out: list[ResponderTeam] = []
    for i in range(n):
        team_type = P.RESPONDER_PATTERN[i % len(P.RESPONDER_PATTERN)]
        spec = P.RESPONDER_TYPES[team_type]
        station = stations[i % len(stations)]
        busy = rng.bernoulli(stress.responder_busy_prob)
        x, y = (_return_leg_position(rng, station) if busy else (station.x, station.y))
        out.append(ResponderTeam(
            team_id=f"R{i + 1}", team_type=team_type, station_id=station.station_id, x=x, y=y,
            capabilities=list(spec["capabilities"]), team_size=spec["size"], speed_kmh=spec["speed_kmh"],
            status="busy" if busy else "available", available_from_min=rng.integers(5, 30) if busy else 0,
        ))
    return out
