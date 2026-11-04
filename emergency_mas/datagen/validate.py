"""Validation checks every generated scenario must pass before it is used."""
from __future__ import annotations

from dataclasses import dataclass

from ..schema import Scenario


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


def _unique(ids: list[str]) -> bool:
    return len(ids) == len(set(ids))


def validate_scenario(s: Scenario) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(passed), "" if passed else detail))

    add("schema_roundtrip", Scenario.model_validate(s.model_dump(mode="json")) == s, "JSON round trip changed the data")

    groups = {
        "stations": [x.station_id for x in s.stations], "hospitals": [x.hospital_id for x in s.hospitals],
        "staff": [x.staff_id for x in s.staff], "ambulances": [x.ambulance_id for x in s.ambulances],
        "responders": [x.team_id for x in s.responders], "incidents": [x.incident_id for x in s.incidents],
        "victims": [x.victim_id for x in s.victims],
    }
    add("unique_ids", all(_unique(v) for v in groups.values()), "duplicate identifiers")

    hospital_ids = set(groups["hospitals"]) | {"POOL"}
    station_ids, incident_ids = set(groups["stations"]), set(groups["incidents"])
    add("foreign_keys",
        all(m.home_hospital in hospital_ids for m in s.staff)
        and all(a.base_station_id in station_ids for a in s.ambulances)
        and all(r.station_id in station_ids for r in s.responders)
        and all(v.incident_id in incident_ids for v in s.victims),
        "a record points at something that does not exist")

    add("incident_counts_match_victims",
        all(i.n_red + i.n_yellow + i.n_green == i.n_victims == len(s.victims_of(i.incident_id))
            and i.n_trapped == sum(v.trapped for v in s.victims_of(i.incident_id)) for i in s.incidents),
        "incident totals disagree with the victim table")

    side = s.meta.city_km
    points = [(o.x, o.y) for o in (*s.stations, *s.hospitals, *s.ambulances, *s.responders, *s.incidents)]
    add("inside_city", all(0 <= x <= side and 0 <= y <= side for x, y in points), "a point lies outside the city")

    add("capacity_limits",
        all(h.er_bays_free >= 1 and h.icu_beds_free <= h.icu_beds_total for h in s.hospitals)
        and all(m.baseline_load <= m.workload_limit for m in s.staff)
        and all(m.on_duty_from_min < m.on_duty_until_min for m in s.staff),
        "a capacity or duty window is inconsistent")

    times = [i.report_time_min for i in s.incidents]
    add("arrivals_ordered", times == sorted(times) and times[0] == 0, "incident times are not sorted from minute 0")

    def provider_exists(capability: str) -> bool:
        return any(capability in h.capabilities
                   and any(m.role == "doctor" and capability in m.skills for m in s.staff_of(h.hospital_id))
                   for h in s.hospitals)

    needed = {v.required_capability for v in s.victims}
    missing = sorted(c for c in needed if not provider_exists(c))
    add("every_need_has_a_provider", not missing, f"no hospital with a matching doctor offers {missing}")

    team_capabilities = {c for r in s.responders for c in r.capabilities}
    add("every_scene_need_has_a_team", all(n.capability in team_capabilities for i in s.incidents for n in i.needs),
        "an incident needs a capability no team has")

    add("als_unit_ready_early", any(a.unit_type == "ALS" and a.available_from_min <= 20 for a in s.ambulances),
        "no ALS unit is free in the first 20 minutes")
    return checks
