"""Mutable runtime state of one simulation run. Only the simulator writes it."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import params as P
from ..domain import Point, TravelModel
from ..schema import Ambulance, Hospital, Incident, ResponderTeam, Scenario, StaffMember, Victim

TRIAGE_RANK = {"RED": 0, "YELLOW": 1, "GREEN": 2}
HAZARD_TASK = {"fire": "fire_suppression", "structural": "structural_rescue", "hazmat": "hazmat"}
AMBULANCE_DISPATCHABLE = ("idle", "returning", "to_scene", "at_scene")
TEAM_DISPATCHABLE = ("idle", "returning", "to_scene")


def drive_minutes(travel: TravelModel, a: Point, b: Point, speed_kmh: float, depart: float) -> int:
    """Whole minutes of driving, at least one."""
    return max(1, math.ceil(travel.minutes(a, b, speed_kmh, depart)))


@dataclass
class Trip:
    origin: Point
    dest: Point
    depart: int
    arrive: int

    def position(self, t: float) -> Point:
        if t <= self.depart:
            return self.origin
        if t >= self.arrive:
            return self.dest
        f = (t - self.depart) / (self.arrive - self.depart)
        return (self.origin[0] + f * (self.dest[0] - self.origin[0]), self.origin[1] + f * (self.dest[1] - self.origin[1]))


@dataclass
class VictimState:
    v: Victim
    report_time: int
    status: str = "at_scene"          # at_scene, loading, onboard, corridor, in_care, done
    ambulance_id: str | None = None
    hospital_id: str | None = None
    stabilized_at: int | None = None
    picked_up_at: int | None = None
    care_start: int | None = None
    care_end: int | None = None
    doctor_id: str | None = None
    nurse_ids: tuple[str, ...] = ()
    bay: str | None = None            # "regular" or "surge" while the victim holds a bay
    boarding: bool = False            # treatment over, waiting in the ER bay for an ICU bed
    survival: float | None = None     # set when care starts

    @property
    def window(self) -> float:
        """Golden window in minutes after the call, extended if the victim was stabilized in time."""
        bonus = 0
        if self.stabilized_at is not None and self.stabilized_at - self.report_time <= self.v.golden_window_min:
            bonus = P.STABILIZATION_BONUS_MIN[self.v.triage]
        return self.v.golden_window_min + bonus


@dataclass
class IncidentState:
    inc: Incident
    visible: bool = False
    revealed_at: int | None = None    # first unit on scene; triage becomes known
    work_left: dict[str, float] = field(default_factory=dict)
    work_started: dict[str, int] = field(default_factory=dict)
    work_done: dict[str, int] = field(default_factory=dict)
    teams: list[str] = field(default_factory=list)    # teams on scene now

    @property
    def hazard_task(self) -> str | None:
        task = HAZARD_TASK.get(self.inc.hazard)
        return task if task in self.work_left else None

    def access_open(self, t: int) -> bool:
        """Victims can be handed to ambulances: a unit is on scene and any hazard is being handled."""
        if self.revealed_at is None:
            return False
        task = self.hazard_task
        if task is None:
            return True
        start = self.work_started.get(task)
        return start is not None and t >= start + P.HAZARD_ACCESS_MIN

    def extricated(self) -> bool:
        return "extrication" not in self.work_left or "extrication" in self.work_done


@dataclass
class AmbulanceState:
    a: Ambulance
    base: Point
    pos: Point
    status: str                       # busy_prior, idle, to_scene, at_scene, loading, loaded, to_hospital, at_door, handoff, returning
    until: int = 0                    # end of a timed activity: busy_prior, loading, handoff
    trip: Trip | None = None
    incident_id: str | None = None
    hospital_id: str | None = None
    onboard: list[str] = field(default_factory=list)   # victims in the ambulance and not yet admitted
    status_since: int = 0
    busy_minutes: int = 0
    ramp_minutes: int = 0
    loads: int = 0

    def position(self, t: float) -> Point:
        if self.trip is not None and self.status in ("to_scene", "to_hospital", "returning"):
            return self.trip.position(t)
        return self.pos


@dataclass
class TeamState:
    r: ResponderTeam
    base: Point
    pos: Point
    status: str                       # busy_prior, idle, to_scene, working, returning
    until: int = 0
    trip: Trip | None = None
    incident_id: str | None = None
    task: str | None = None
    status_since: int = 0
    busy_minutes: int = 0

    def position(self, t: float) -> Point:
        if self.trip is not None and self.status in ("to_scene", "returning"):
            return self.trip.position(t)
        return self.pos


@dataclass
class StaffState:
    s: StaffMember
    hospital_id: str | None           # where the person works now; None for an unassigned pool member
    arrives_at: int = 0               # not available before this minute (moving between hospitals, pool reporting)
    patients: list[str] = field(default_factory=list)

    def available(self, t: int) -> bool:
        return (self.hospital_id is not None and t >= self.arrives_at
                and self.s.on_duty_from_min <= t < self.s.on_duty_until_min)

    def free_slots(self, t: int) -> int:
        if not self.available(t):
            return 0
        return max(0, self.s.workload_limit - self.s.baseline_load - len(self.patients))


@dataclass
class HospitalState:
    h: Hospital
    icu_free: int
    regular_used: int = 0
    surge_used: int = 0
    surge_ready_at: int | None = None
    door: list[str] = field(default_factory=list)       # ambulances waiting at the ER door, in arrival order
    corridor: list[str] = field(default_factory=list)   # patients handed over in the corridor, waiting for a bay
    in_care: list[str] = field(default_factory=list)    # victims holding a bay (treatment or ICU boarding)

    def regular_free(self) -> int:
        return self.h.er_bays_total - self.h.er_bays_occupied - self.regular_used

    def surge_free(self, t: int) -> int:
        if self.surge_ready_at is None or t < self.surge_ready_at:
            return 0
        return self.h.surge_bays - self.surge_used


@dataclass
class World:
    scenario: Scenario
    travel: TravelModel
    stations: dict[str, Point]
    incidents: dict[str, IncidentState]
    victims: dict[str, VictimState]
    ambulances: dict[str, AmbulanceState]
    teams: dict[str, TeamState]
    staff: dict[str, StaffState]
    hospitals: dict[str, HospitalState]

    @classmethod
    def from_scenario(cls, s: Scenario) -> "World":
        stations = {st.station_id: (st.x, st.y) for st in s.stations}
        incidents = {i.incident_id: IncidentState(i, work_left={n.capability: float(n.work_min) for n in i.needs})
                     for i in s.incidents}
        report = {i.incident_id: i.report_time_min for i in s.incidents}
        victims = {v.victim_id: VictimState(v, report_time=report[v.incident_id]) for v in s.victims}
        ambulances = {}
        for a in s.ambulances:
            busy = a.status == "busy"
            ambulances[a.ambulance_id] = AmbulanceState(a, base=stations[a.base_station_id], pos=(a.x, a.y),
                                                        status="busy_prior" if busy else "idle", until=a.available_from_min)
        teams = {}
        for r in s.responders:
            busy = r.status == "busy"
            teams[r.team_id] = TeamState(r, base=stations[r.station_id], pos=(r.x, r.y),
                                         status="busy_prior" if busy else "idle", until=r.available_from_min)
        staff = {m.staff_id: StaffState(m, hospital_id=None if m.home_hospital == "POOL" else m.home_hospital)
                 for m in s.staff}
        hospitals = {h.hospital_id: HospitalState(h, icu_free=h.icu_beds_free) for h in s.hospitals}
        return cls(s, TravelModel(tuple((a, b, m) for a, b, m in s.meta.congestion_profile)), stations,
                   incidents, victims, ambulances, teams, staff, hospitals)

    def victims_of(self, incident_id: str) -> list[VictimState]:
        return [vs for vs in self.victims.values() if vs.v.incident_id == incident_id]
