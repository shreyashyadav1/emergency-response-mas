"""Stress and contention measures computed from a generated scenario.

These are read-only summaries. They tell us how hard a scenario is and where
the conflicts will be, before any agent has made a decision.
"""
from __future__ import annotations

from collections import Counter

from .. import params as P
from ..domain import TravelModel, straight_km
from ..schema import Scenario


def travel_model(s: Scenario) -> TravelModel:
    return TravelModel(tuple((a, b, m) for a, b, m in s.meta.congestion_profile))


def pressure_indices(s: Scenario) -> dict[str, float]:
    """Demand over supply for each scarce resource. Above 1.0 it cannot all be served."""
    tm = travel_model(s)
    horizon = s.meta.horizon_min
    station = {st.station_id: (st.x, st.y) for st in s.stations}
    n_red = sum(v.triage == "RED" for v in s.victims)
    trips_needed = n_red + -(-(len(s.victims) - n_red) // 2)        # RED ride alone, the rest pair up

    cycle_times = []
    for inc in s.incidents:
        pos = (inc.x, inc.y)
        drive_out = min(tm.minutes(station[a.base_station_id], pos, a.speed_kmh, inc.report_time_min) for a in s.ambulances)
        hosp = min(s.hospitals, key=lambda h: straight_km((h.x, h.y), pos))
        hpos = (hosp.x, hosp.y)
        drive_in = tm.minutes(pos, hpos, P.AMBULANCE_SPEED_KMH, inc.report_time_min)
        drive_back = min(tm.minutes(hpos, p, P.AMBULANCE_SPEED_KMH) for p in station.values())
        cycle_times.append(P.AMBULANCE_TURNOUT_MIN + drive_out + P.SCENE_LOAD_MIN + drive_in
                           + P.HOSPITAL_HANDOFF_MIN + drive_back)
    cycle = sum(cycle_times) / len(cycle_times)
    trip_capacity = len(s.ambulances) * (horizon - 15) / cycle

    bay_minutes_demand = sum(v.treatment_min for v in s.victims)
    bay_minutes_supply = sum(h.er_bays_free for h in s.hospitals) * horizon

    def slot_minutes_supply(role: str, skill: str | None = None) -> float:
        return sum(m.free_slots * max(0, m.on_duty_until_min - m.on_duty_from_min)
                   for m in s.staff if m.role == role and (skill is None or skill in m.skills))

    doctor_demand = sum(v.doctor_slots * v.treatment_min for v in s.victims)
    nurse_demand = sum(v.nurse_slots * v.treatment_min for v in s.victims)

    specialist = 0.0
    for capability in {v.required_capability for v in s.victims if v.doctor_slots} - {"general_emergency"}:
        demand = sum(v.doctor_slots * v.treatment_min for v in s.victims if v.required_capability == capability)
        supply = slot_minutes_supply("doctor", capability)
        specialist = max(specialist, demand / supply if supply > 0 else 99.0)

    n_als = sum(a.unit_type == "ALS" for a in s.ambulances)
    return {
        "ambulance_pressure": round(trips_needed / trip_capacity, 3),
        "red_per_als_unit": round(n_red / max(1, n_als), 3),
        "er_bay_pressure": round(bay_minutes_demand / max(1, bay_minutes_supply), 3),
        "doctor_pressure": round(doctor_demand / max(1.0, slot_minutes_supply("doctor")), 3),
        "nurse_pressure": round(nurse_demand / max(1.0, slot_minutes_supply("nurse")), 3),
        "specialist_pressure": round(specialist, 3),
        "mean_ambulance_cycle_min": round(cycle, 1),
    }


def contention_report(s: Scenario, window_min: int = 20) -> dict:
    """Where the conflicts are, by construction of the data.

    - ambulance_pairs: incident pairs reported within `window_min` minutes whose nearest ambulance is the same unit.
    - single_provider: capabilities needed by RED/YELLOW victims that only one hospital offers.
    - nearest_capable_load: RED/YELLOW victims whose nearest capable hospital is each hospital, next to its free
      ER bays. Shows how nearest-first routing piles patients onto one hospital.
    """
    tm = travel_model(s)
    station = {st.station_id: (st.x, st.y) for st in s.stations}
    nearest: dict[str, str] = {}
    for inc in s.incidents:
        nearest[inc.incident_id] = min(
            s.ambulances,
            key=lambda a: (tm.minutes(station[a.base_station_id], (inc.x, inc.y), a.speed_kmh, inc.report_time_min), a.ambulance_id),
        ).ambulance_id
    pairs = [(a.incident_id, b.incident_id)
             for i, a in enumerate(s.incidents) for b in s.incidents[i + 1:]
             if abs(a.report_time_min - b.report_time_min) <= window_min and nearest[a.incident_id] == nearest[b.incident_id]]

    demand = Counter(v.required_capability for v in s.victims
                     if v.triage in ("RED", "YELLOW") and v.required_capability != "general_emergency")
    single_provider = {}
    for capability, count in demand.items():
        providers = [h.hospital_id for h in s.hospitals if capability in h.capabilities]
        if len(providers) == 1:
            single_provider[capability] = {"hospital": providers[0], "victims": count}

    load = {h.hospital_id: {"red_yellow_nearest_capable": 0, "free_er_bays": h.er_bays_free} for h in s.hospitals}
    by_id = {i.incident_id: i for i in s.incidents}
    for v in s.victims:
        if v.triage == "GREEN":
            continue
        inc = by_id[v.incident_id]
        capable = [h for h in s.hospitals if v.required_capability in h.capabilities]
        target = min(capable, key=lambda h: (straight_km((h.x, h.y), (inc.x, inc.y)), h.hospital_id))
        load[target.hospital_id]["red_yellow_nearest_capable"] += 1

    return {
        "ambulance_pairs": [list(p) for p in pairs],
        "n_ambulance_pairs": len(pairs),
        "capability_demand": dict(demand),
        "single_provider": single_provider,
        "red_yellow_needing_single_provider": sum(d["victims"] for d in single_provider.values()),
        "nearest_capable_load": load,
    }


def victim_counts(s: Scenario) -> dict[str, int]:
    return {
        "incidents": len(s.incidents), "victims": len(s.victims),
        "red": sum(v.triage == "RED" for v in s.victims),
        "yellow": sum(v.triage == "YELLOW" for v in s.victims),
        "green": sum(v.triage == "GREEN" for v in s.victims),
        "trapped": sum(v.trapped for v in s.victims),
        "needs_icu": sum(v.needs_icu for v in s.victims),
        "stations": len(s.stations), "hospitals": len(s.hospitals), "ambulances": len(s.ambulances), "responder_teams": len(s.responders),
        "doctors": sum(m.role == "doctor" for m in s.staff), "nurses": sum(m.role == "nurse" for m in s.staff),
    }
