"""What a policy is allowed to see each minute.

Ground truth stays in the simulator. Until a unit reaches an incident, the only
information about it is the caller's report. Hospitals publish their capacity on
a shared status board; their internal detail (who treats whom, until when) is
there too, for the hospital's own agent.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import params as P
from ..domain import Point, TravelModel
from ..schema import IncidentReport
from .state import AMBULANCE_DISPATCHABLE, TEAM_DISPATCHABLE, World, drive_minutes


@dataclass(frozen=True)
class VictimView:
    victim_id: str
    incident_id: str
    triage: str
    required_capability: str
    needs_icu: bool
    trapped: bool
    status: str
    ambulance_id: str | None
    hospital_id: str | None


@dataclass(frozen=True)
class NeedView:
    capability: str
    work_left: float
    started_at: int | None
    done: bool

    @property
    def started(self) -> bool:
        return self.started_at is not None


@dataclass(frozen=True)
class IncidentView:
    incident_id: str
    incident_type: str
    x: float
    y: float
    report_time: int
    report: IncidentReport
    revealed: bool
    access_open: bool
    extricated: bool
    victims: tuple[VictimView, ...]       # empty until a unit has been on scene
    needs: tuple[NeedView, ...]           # empty until a unit has been on scene
    ambulances: tuple[str, ...]           # ambulances heading there, waiting or loading
    teams: tuple[str, ...]                # teams heading there or working

    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    def waiting(self) -> list[VictimView]:
        """Victims still at the scene and not being loaded."""
        return [v for v in self.victims if v.status == "at_scene"]

    @property
    def cleared(self) -> bool:
        return self.revealed and not self.waiting()

    def open_needs(self) -> list[NeedView]:
        return [n for n in self.needs if not n.done]


@dataclass(frozen=True)
class AmbulanceView:
    ambulance_id: str
    unit_type: str
    capacity: int
    status: str
    pos: Point
    base: Point
    speed_kmh: float
    incident_id: str | None
    hospital_id: str | None
    onboard: tuple[VictimView, ...]
    free_at: int
    since: int                   # minute the current status began

    @property
    def dispatchable(self) -> bool:
        return self.status in AMBULANCE_DISPATCHABLE and not self.onboard

    @property
    def from_rest(self) -> bool:
        return self.status in ("idle", "at_scene")


@dataclass(frozen=True)
class TeamView:
    team_id: str
    team_type: str
    capabilities: tuple[str, ...]
    status: str
    pos: Point
    speed_kmh: float
    incident_id: str | None
    task: str | None
    free_at: int

    @property
    def dispatchable(self) -> bool:
        return self.status in TEAM_DISPATCHABLE

    @property
    def from_rest(self) -> bool:
        return self.status == "idle"


@dataclass(frozen=True)
class StaffView:
    staff_id: str
    role: str
    specialty: str
    skills: tuple[str, ...]
    home_hospital: str
    hospital_id: str | None
    lendable: bool
    available: bool
    free_slots: int
    patients: int
    arrives_at: int
    on_duty_from: int
    on_duty_until: int
    workload_limit: int
    baseline_load: int


@dataclass(frozen=True)
class CareView:
    victim_id: str
    triage: str
    care_end: int | None
    boarding: bool
    needs_icu: bool
    doctor_id: str | None
    nurse_ids: tuple[str, ...]


@dataclass(frozen=True)
class HospitalView:
    hospital_id: str
    name: str
    pos: Point
    trauma_level: int
    capabilities: tuple[str, ...]
    bays_total: int
    baseline_occupied: int
    regular_free: int
    surge_total: int
    surge_used: int
    surge_free: int
    surge_ready_at: int | None
    icu_free: int
    reserve_bays_for_red: int
    diversion_threshold: float
    max_lend_doctors: int
    max_lend_nurses: int
    door: tuple[str, ...]
    corridor: tuple[VictimView, ...]      # patients handed over in the corridor, waiting for a bay
    in_care: tuple[CareView, ...]
    staff: tuple[StaffView, ...]          # people working here now or on their way here

    @property
    def free_bays(self) -> int:
        return self.regular_free + self.surge_free

    @property
    def occupancy(self) -> float:
        open_surge = self.surge_total if self.surge_ready_at is not None else 0
        used = self.bays_total - self.regular_free + self.surge_used
        return used / max(1, self.bays_total + open_surge)

    def free_slots(self, role: str, skill: str | None = None) -> int:
        return sum(m.free_slots for m in self.staff
                   if m.role == role and m.available and (skill is None or skill in m.skills))


@dataclass(frozen=True)
class Observation:
    t: int
    horizon: int
    incidents: tuple[IncidentView, ...]   # reported incidents only
    ambulances: tuple[AmbulanceView, ...]
    teams: tuple[TeamView, ...]
    hospitals: tuple[HospitalView, ...]
    pool: tuple[StaffView, ...]           # pool clinicians not assigned to a hospital yet
    events: tuple[dict, ...]              # what happened since the previous decision
    travel: TravelModel

    def incident(self, incident_id: str) -> IncidentView:
        return next(i for i in self.incidents if i.incident_id == incident_id)

    def hospital(self, hospital_id: str) -> HospitalView:
        return next(h for h in self.hospitals if h.hospital_id == hospital_id)

    def ambulance(self, ambulance_id: str) -> AmbulanceView:
        return next(a for a in self.ambulances if a.ambulance_id == ambulance_id)

    def drive(self, a: Point, b: Point, speed_kmh: float = P.AMBULANCE_SPEED_KMH) -> int:
        return drive_minutes(self.travel, a, b, speed_kmh, self.t)

    def eta(self, unit: AmbulanceView | TeamView, dest: Point) -> int:
        """Minutes from now until the unit could be at `dest` if dispatched now."""
        wait = max(0, unit.free_at - self.t)
        return wait + (P.TURNOUT_MIN if unit.from_rest else 0) + self.drive(unit.pos, dest, unit.speed_kmh)


def _victim_view(w: World, victim_id: str) -> VictimView:
    vs = w.victims[victim_id]
    return VictimView(vs.v.victim_id, vs.v.incident_id, vs.v.triage, vs.v.required_capability, vs.v.needs_icu,
                      vs.v.trapped and not w.incidents[vs.v.incident_id].extricated(), vs.status,
                      vs.ambulance_id, vs.hospital_id)


def _staff_view(m, t: int) -> StaffView:
    return StaffView(m.s.staff_id, m.s.role, m.s.specialty, tuple(m.s.skills), m.s.home_hospital, m.hospital_id,
                     m.s.lendable, m.available(t), m.free_slots(t), len(m.patients), m.arrives_at,
                     m.s.on_duty_from_min, m.s.on_duty_until_min, m.s.workload_limit, m.s.baseline_load)


def build_observation(w: World, t: int, events: list[dict]) -> Observation:
    incidents = []
    for iid, inc in w.incidents.items():
        if not inc.visible:
            continue
        revealed = inc.revealed_at is not None
        victims = tuple(_victim_view(w, vs.v.victim_id) for vs in w.victims_of(iid)) if revealed else ()
        needs = tuple(NeedView(c, max(0.0, left), inc.work_started.get(c), c in inc.work_done)
                      for c, left in inc.work_left.items()) if revealed else ()
        ambulances = tuple(a.a.ambulance_id for a in w.ambulances.values()
                           if a.incident_id == iid and a.status in ("to_scene", "at_scene", "loading"))
        teams = tuple(r.r.team_id for r in w.teams.values() if r.incident_id == iid and r.status in ("to_scene", "working"))
        incidents.append(IncidentView(iid, inc.inc.incident_type, inc.inc.x, inc.inc.y, inc.inc.report_time_min,
                                      inc.inc.report, revealed, inc.access_open(t), inc.extricated(), victims, needs,
                                      ambulances, teams))

    ambulances = tuple(
        AmbulanceView(a.a.ambulance_id, a.a.unit_type, a.a.capacity, a.status, a.position(t), a.base, a.a.speed_kmh,
                      a.incident_id, a.hospital_id, tuple(_victim_view(w, v) for v in a.onboard),
                      a.until if a.status == "busy_prior" else t, a.status_since)
        for a in w.ambulances.values())
    teams = tuple(
        TeamView(r.r.team_id, r.r.team_type, tuple(r.r.capabilities), r.status, r.position(t), r.r.speed_kmh,
                 r.incident_id, r.task, r.until if r.status == "busy_prior" else t)
        for r in w.teams.values())

    hospitals = []
    for hid, hs in w.hospitals.items():
        care = tuple(CareView(v, w.victims[v].v.triage, w.victims[v].care_end, w.victims[v].boarding,
                              w.victims[v].v.needs_icu, w.victims[v].doctor_id, w.victims[v].nurse_ids)
                     for v in hs.in_care)
        staff = tuple(_staff_view(m, t) for m in w.staff.values() if m.hospital_id == hid)
        h = hs.h
        hospitals.append(HospitalView(
            hid, h.name, (h.x, h.y), h.trauma_level, tuple(h.capabilities), h.er_bays_total, h.er_bays_occupied,
            hs.regular_free(), h.surge_bays, hs.surge_used, hs.surge_free(t), hs.surge_ready_at, hs.icu_free,
            h.reserve_bays_for_red, h.diversion_threshold, h.max_staff_lend_doctors, h.max_staff_lend_nurses,
            tuple(hs.door), tuple(_victim_view(w, v) for v in hs.corridor), care, staff))
    pool = tuple(_staff_view(m, t) for m in w.staff.values() if m.hospital_id is None)
    return Observation(t, w.scenario.meta.horizon_min, tuple(incidents), ambulances, teams, tuple(hospitals), pool,
                       tuple(events), w.travel)
