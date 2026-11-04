"""Build one complete, reproducible scenario from a seed."""
from __future__ import annotations

import math

from .. import params as P
from ..rng import make_rng
from ..schema import Scenario, ScenarioMeta
from .fleet import make_ambulances, make_responders, make_stations
from .geography import place_hospitals, place_hotspots, place_stations
from .hospitals import archetype_zones, make_hospitals, make_staff
from .incidents import generate_demand
from .narrative import Narrator, TemplateNarrator, seeded_faker


def generate_scenario(
    theme: str = "urban_mci", stress: str = "medium", seed: int = 42, *,
    n_incidents: int | None = None, n_hospitals: int = P.DEFAULT_HOSPITALS,
    n_ambulances: int = P.DEFAULT_AMBULANCES, n_responders: int = P.DEFAULT_RESPONDERS,
    horizon_min: int = P.HORIZON_MIN, narrator: Narrator | None = None,
) -> Scenario:
    """Same arguments, same scenario, byte for byte.

    The city (hospital sites, stations) depends only on the seed, so the same seed gives the
    same map at every stress level and theme. Stress changes how full the hospitals are, how
    many units are busy and how many victims there are.
    """
    if theme not in P.THEMES or stress not in P.STRESS:
        raise ValueError(f"unknown theme {theme!r} or stress {stress!r}")
    th, st = P.THEMES[theme], P.STRESS[stress]
    narrator = narrator or TemplateNarrator()
    names = make_rng(seed, "names")

    hospitals = make_hospitals(
        make_rng(seed, "hospitals"), place_hospitals(make_rng(seed, "sites"), archetype_zones(n_hospitals)), st)
    stations = make_stations(place_stations(make_rng(seed, "city"), max(3, math.ceil(n_ambulances / 2) + 1)))
    fleet_rng = make_rng(seed, "fleet")
    ambulances = make_ambulances(fleet_rng, stations, n_ambulances, st)
    responders = make_responders(fleet_rng, stations, n_responders, st)
    staff = make_staff(make_rng(seed, "staff"), seeded_faker(names.seed_int()), hospitals, st, horizon_min)
    hotspots = place_hotspots(make_rng(seed, "hotspots"), th.n_hotspots, th.corridor)
    incidents, victims = generate_demand(
        seed=seed, theme_name=theme, stress_name=stress, n_incidents=n_incidents or th.n_incidents,
        hotspots=hotspots, fake=seeded_faker(names.seed_int()), narrator=narrator)

    return Scenario(
        meta=ScenarioMeta(
            scenario_id=f"{theme}-{stress}-s{seed}", theme=theme, stress=stress, seed=seed,
            generator_version=P.GENERATOR_VERSION, horizon_min=horizon_min, city_km=P.CITY_KM,
            congestion_profile=[(0.0, float(P.CONGESTION_WINDOW_MIN), st.congestion),
                                (float(P.CONGESTION_WINDOW_MIN), float(horizon_min), 1.0)],
            description=th.description),
        stations=stations, hospitals=hospitals, staff=staff, ambulances=ambulances,
        responders=responders, incidents=incidents, victims=victims)
