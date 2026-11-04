"""Helpers every policy uses to read an Observation."""
from __future__ import annotations

from ..schema import Scenario
from ..sim.observe import AmbulanceView, HospitalView, IncidentView, Observation, TeamView

HAZARD_TASK = {"fire": "fire_suppression", "structural": "structural_rescue", "hazmat": "hazmat"}


def ambulances_at(obs: Observation, inc: IncidentView) -> list[AmbulanceView]:
    return [a for a in obs.ambulances if a.ambulance_id in inc.ambulances]


def transport_need(obs: Observation, inc: IncidentView, expected_victims: int | None = None) -> int:
    """Patients still needing a ride, minus the seats already heading to the scene."""
    if inc.revealed:
        waiting = len(inc.waiting())
    else:
        waiting = expected_victims if expected_victims is not None else inc.report.reported_victims
    seats = sum(a.capacity for a in ambulances_at(obs, inc) if a.status in ("to_scene", "at_scene"))
    return max(0, waiting - seats)


def expected_tasks(inc: IncidentView) -> list[str]:
    """Scene work: the real list once a unit is there, otherwise what the caller's report implies."""
    if inc.revealed:
        return [n.capability for n in inc.open_needs()]
    tasks = []
    task = HAZARD_TASK.get(inc.report.reported_hazard)
    if task:
        tasks.append(task)
    if inc.report.reported_trapped:
        tasks.append("extrication")
    tasks.append("triage_stabilization")
    return tasks


def uncovered_tasks(obs: Observation, inc: IncidentView, tasks: list[str] | None = None) -> list[str]:
    covered = {c for r in obs.teams if r.team_id in inc.teams for c in r.capabilities}
    return [c for c in (tasks if tasks is not None else expected_tasks(inc)) if c not in covered]


def capable(obs: Observation, capabilities: set[str]) -> list[HospitalView]:
    return [h for h in obs.hospitals if capabilities <= set(h.capabilities)]


def onboard_capabilities(amb: AmbulanceView) -> set[str]:
    return {v.required_capability for v in amb.onboard}


def inbound(obs: Observation, hospital_id: str) -> list:
    """Patients on their way to a hospital, waiting at its door, or handed over in its corridor."""
    return [v for a in obs.ambulances if a.hospital_id == hospital_id for v in a.onboard] + \
        list(obs.hospital(hospital_id).corridor)


def can_carry(inc: IncidentView, amb: AmbulanceView) -> bool:
    """False only when we know everyone waiting is RED and the unit is BLS."""
    return not inc.revealed or amb.unit_type == "ALS" or any(v.triage != "RED" for v in inc.waiting())


def nearest_unit(obs: Observation, units: list, pos) -> AmbulanceView | TeamView:
    return min(units, key=lambda u: (obs.eta(u, pos), getattr(u, "ambulance_id", None) or u.team_id))


def needs_destination(amb: AmbulanceView) -> bool:
    return bool(amb.onboard) and amb.hospital_id is None and amb.status in ("loading", "loaded")


def pool_round_robin(scenario: Scenario) -> dict[str, str]:
    """The uncoordinated default: pool clinicians are split evenly across hospitals in id order."""
    hospitals = sorted(h.hospital_id for h in scenario.hospitals)
    pool = sorted(m.staff_id for m in scenario.staff if m.home_hospital == "POOL")
    return {sid: hospitals[i % len(hospitals)] for i, sid in enumerate(pool)}
