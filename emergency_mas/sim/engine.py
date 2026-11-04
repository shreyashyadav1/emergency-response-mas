"""Discrete-time simulator: one tick per minute over the scenario horizon.

Each minute: calls that come in become visible, units drive and work, hospitals
admit and discharge, then the policy sees an Observation and returns actions and
messages. The simulator checks every action against the rules before applying it,
so a policy can never break capacity, capability or crew rules.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .. import params as P
from ..domain import Point, survival_probability
from ..schema import Scenario
from .actions import (Action, ActionResult, AssignStaff, DispatchAmbulance, DispatchTeam, OpenSurge, ReturnToBase,
                      SetDestination)
from .observe import build_observation
from .policy import Decision, Policy
from .state import (AMBULANCE_DISPATCHABLE, TEAM_DISPATCHABLE, TRIAGE_RANK, AmbulanceState, HospitalState, StaffState,
                    TeamState, Trip, VictimState, World, drive_minutes)


@dataclass
class RunResult:
    scenario_id: str
    policy: str
    metrics: dict
    ticks: list[dict] = field(default_factory=list)


class Simulator:
    def __init__(self, scenario: Scenario, policy: Policy, record: bool = True):
        self.scenario = scenario
        self.policy = policy
        self.w = World.from_scenario(scenario)
        self.record = record
        self.ticks: list[dict] = []
        self.events: list[dict] = []
        self.counts: Counter[str] = Counter()
        self.samples: dict[str, list[float]] = {"bays": [], "staff": []}
        self._carry: list[dict] = []

    # ------------------------------------------------------------------ run
    def run(self) -> RunResult:
        from .metrics import compute_metrics

        self.policy.reset(self.scenario)
        for t in range(self.scenario.meta.horizon_min):
            self.step(t)
        return RunResult(self.scenario.meta.scenario_id, self.policy.name, compute_metrics(self), self.ticks)

    def step(self, t: int) -> None:
        self.events = []
        self._calls(t)
        self._advance(t)
        obs = build_observation(self.w, t, self._carry + self.events)
        decision = self.policy.decide(obs)
        mark = len(self.events)
        results = [self._apply(action, t) for action in decision.actions]
        self._settle(t)
        self._account(t)
        for m in decision.messages:
            self.counts[f"msg:{getattr(m, 'type', 'message')}"] += 1
        if self.record:
            self.ticks.append(self._snapshot(t, decision, results))
        self._carry = self.events[mark:]

    # ------------------------------------------------------------------ helpers
    def _event(self, t: int, kind: str, **data) -> None:
        self.events.append({"t": t, "kind": kind, **data})

    @staticmethod
    def _set(unit, status: str, t: int) -> None:
        unit.status = status
        unit.status_since = t

    def _compatible(self, capabilities: set[str]) -> bool:
        return any(capabilities <= set(hs.h.capabilities) for hs in self.w.hospitals.values())

    # ------------------------------------------------------------------ the world moves
    def _calls(self, t: int) -> None:
        for iid, inc in self.w.incidents.items():
            if not inc.visible and inc.inc.report_time_min <= t:
                inc.visible = True
                r = inc.inc.report
                self._event(t, "call", incident=iid, reported_victims=r.reported_victims,
                            reported_critical=r.reported_critical, narrative=r.narrative)

    def _advance(self, t: int) -> None:
        w = self.w
        for amb in w.ambulances.values():
            if amb.status == "busy_prior" and t >= amb.until:
                self._set(amb, "idle", t)
                self._event(t, "unit_available", unit=amb.a.ambulance_id)
            elif amb.status in ("to_scene", "to_hospital", "returning") and t >= amb.trip.arrive:
                amb.pos, amb.trip = amb.trip.dest, None
                if amb.status == "to_scene":
                    self._ambulance_on_scene(amb, t)
                elif amb.status == "to_hospital":
                    self._ambulance_at_door(amb, t)
                else:
                    self._set(amb, "idle", t)
                    self._event(t, "unit_home", unit=amb.a.ambulance_id)
        for team in w.teams.values():
            if team.status == "busy_prior" and t >= team.until:
                self._set(team, "idle", t)
                self._event(t, "unit_available", unit=team.r.team_id)
            elif team.status in ("to_scene", "returning") and t >= team.trip.arrive:
                team.pos, team.trip = team.trip.dest, None
                if team.status == "to_scene":
                    self._team_on_scene(team, t)
                else:
                    self._set(team, "idle", t)
                    self._event(t, "unit_home", unit=team.r.team_id)
        self._scene_work(t)
        for amb in w.ambulances.values():
            if amb.status == "loading" and t >= amb.until:
                self._loading_done(amb, t)
            elif amb.status == "handoff" and t >= amb.until:
                amb.hospital_id = None
                self._set(amb, "idle", t)
                self._event(t, "handoff_done", unit=amb.a.ambulance_id)
        for amb in w.ambulances.values():
            if amb.status == "at_scene":
                self._try_load(amb, t)
        self._treatments(t)
        self._timed_events(t)
        self._admissions(t)
        self._auto_return(t)

    def _reveal(self, iid: str, t: int) -> None:
        inc = self.w.incidents[iid]
        if inc.revealed_at is not None:
            return
        inc.revealed_at = t
        at_scene = [vs for vs in self.w.victims_of(iid) if vs.status == "at_scene"]
        self._event(t, "scene_assessed", incident=iid,
                    triage={k: sum(vs.v.triage == k for vs in at_scene) for k in TRIAGE_RANK},
                    trapped=sum(vs.v.trapped for vs in at_scene), hazard=inc.inc.hazard)

    def _ambulance_on_scene(self, amb: AmbulanceState, t: int) -> None:
        self._set(amb, "at_scene", t)
        self._event(t, "on_scene", unit=amb.a.ambulance_id, incident=amb.incident_id)
        self._reveal(amb.incident_id, t)

    def _team_on_scene(self, team: TeamState, t: int) -> None:
        self._set(team, "working", t)
        self.w.incidents[team.incident_id].teams.append(team.r.team_id)
        self._event(t, "on_scene", unit=team.r.team_id, incident=team.incident_id)
        self._reveal(team.incident_id, t)

    def _scene_work(self, t: int) -> None:
        for iid, inc in self.w.incidents.items():
            for tid in list(inc.teams):
                team = self.w.teams[tid]
                task = next((c for c in P.TASK_ORDER if c in inc.work_left and c not in inc.work_done
                             and c in team.r.capabilities), None)
                if task is None:
                    inc.teams.remove(tid)
                    team.task, team.incident_id = None, None
                    self._set(team, "idle", t)
                    self._event(t, "team_done", unit=tid, incident=iid)
                    continue
                if task not in inc.work_started:
                    inc.work_started[task] = t
                    self._event(t, "work_started", incident=iid, task=task, unit=tid)
                team.task = task
                inc.work_left[task] -= 1
                if inc.work_left[task] <= 0:
                    inc.work_done[task] = t
                    self._event(t, "work_done", incident=iid, task=task)
                    if task == "triage_stabilization":
                        for vs in self.w.victims_of(iid):
                            if vs.status == "at_scene" and vs.stabilized_at is None:
                                vs.stabilized_at = t
            hz = inc.hazard_task
            if hz and inc.work_started.get(hz) is not None and t == inc.work_started[hz] + P.HAZARD_ACCESS_MIN:
                self._event(t, "scene_access_open", incident=iid)

    def _try_load(self, amb: AmbulanceState, t: int) -> None:
        iid = amb.incident_id
        inc = self.w.incidents[iid]
        waiting = [vs for vs in self.w.victims_of(iid) if vs.status == "at_scene"]
        if not waiting:
            self._leave_scene(amb, t, "scene_clear")
            return
        open_ = inc.access_open(t)
        loadable = [vs for vs in waiting if open_ and (not vs.v.trapped or inc.extricated())
                    and (amb.a.unit_type == "ALS" or vs.v.triage != "RED")]
        if not loadable:
            if amb.a.unit_type == "BLS" and all(vs.v.triage == "RED" for vs in waiting):
                self.counts["wasted_trips"] += 1
                self._leave_scene(amb, t, "wasted_trip")
            return
        loadable.sort(key=lambda vs: (TRIAGE_RANK[vs.v.triage], vs.v.victim_id))
        chosen: list[VictimState] = []
        for vs in loadable:
            if len(chosen) == amb.a.capacity:
                break
            if vs.v.triage == "RED" and any(c.v.triage == "RED" for c in chosen):
                continue                      # at most one RED patient per load
            if not self._compatible({c.v.required_capability for c in chosen} | {vs.v.required_capability}):
                continue                      # every patient on board must fit one hospital
            chosen.append(vs)
        for vs in chosen:
            vs.status, vs.ambulance_id = "loading", amb.a.ambulance_id
            if amb.a.unit_type == "ALS" and vs.v.triage == "RED" and vs.stabilized_at is None:
                vs.stabilized_at = t          # the ALS crew starts advanced care on scene
        amb.onboard = [vs.v.victim_id for vs in chosen]
        amb.until = t + P.LOAD_MIN
        self._set(amb, "loading", t)
        self._event(t, "loading", unit=amb.a.ambulance_id, incident=iid,
                    victims=amb.onboard, triage=[vs.v.triage for vs in chosen])

    def _leave_scene(self, amb: AmbulanceState, t: int, why: str) -> None:
        self._event(t, why, unit=amb.a.ambulance_id, incident=amb.incident_id)
        amb.incident_id = None
        self._set(amb, "idle", t)

    def _loading_done(self, amb: AmbulanceState, t: int) -> None:
        for vid in amb.onboard:
            vs = self.w.victims[vid]
            vs.status, vs.picked_up_at = "onboard", t
        amb.loads += 1
        self._set(amb, "loaded", t)
        if amb.hospital_id:
            self._drive_to_hospital(amb, t)

    def _drive_to_hospital(self, amb: AmbulanceState, t: int) -> None:
        hs = self.w.hospitals[amb.hospital_id]
        origin, dest = amb.position(t), (hs.h.x, hs.h.y)
        amb.trip = Trip(origin, dest, t, t + drive_minutes(self.w.travel, origin, dest, amb.a.speed_kmh, t))
        amb.incident_id = None
        self._set(amb, "to_hospital", t)
        self._event(t, "transport", unit=amb.a.ambulance_id, hospital=amb.hospital_id, eta=amb.trip.arrive,
                    victims=list(amb.onboard))

    def _ambulance_at_door(self, amb: AmbulanceState, t: int) -> None:
        self._set(amb, "at_door", t)
        self.w.hospitals[amb.hospital_id].door.append(amb.a.ambulance_id)
        self._event(t, "at_hospital", unit=amb.a.ambulance_id, hospital=amb.hospital_id)

    def _clinicians(self, hs: HospitalState, t: int, role: str) -> list[StaffState]:
        return sorted((m for m in self.w.staff.values()
                       if m.hospital_id == hs.h.hospital_id and m.s.role == role and m.free_slots(t) > 0),
                      key=lambda m: (-m.free_slots(t), m.s.staff_id))

    def _pick_doctor(self, hs: HospitalState, t: int, v) -> StaffState | None:
        here = self._clinicians(hs, t, "doctor")
        skilled = [m for m in here if v.required_capability in m.s.skills]
        if skilled:
            return skilled[0]
        if v.required_capability == "general_emergency":
            return here[0] if here else None
        if v.triage == "YELLOW" and P.YELLOW_GENERALIST_OK:
            return next((m for m in here if "general_emergency" in m.s.skills), None)
        return None

    def _pick_nurses(self, hs: HospitalState, t: int, v) -> list[StaffState]:
        here = self._clinicians(hs, t, "nurse")
        chosen: list[StaffState] = []
        if v.nurse_slots and v.required_capability != "general_emergency":
            first = next((m for m in here if v.required_capability in m.s.skills), None)
            if first is None and not (v.triage == "YELLOW" and P.YELLOW_GENERALIST_OK):
                return []
            if first is not None:
                chosen.append(first)
        for m in here:
            if len(chosen) >= v.nurse_slots:
                break
            if m not in chosen:
                chosen.append(m)
        return chosen

    def _admit(self, vs: VictimState, hs: HospitalState, t: int) -> bool:
        v = vs.v
        if v.required_capability not in hs.h.capabilities:
            return False
        if hs.regular_free() > 0:
            bay = "regular"
        elif hs.surge_free(t) > 0:
            bay = "surge"
        else:
            self.counts["blocked:no_bay"] += 1
            return False
        doctor = None
        if v.doctor_slots:
            doctor = self._pick_doctor(hs, t, v)
            if doctor is None:
                self.counts[f"blocked:no_doctor:{v.required_capability}"] += 1
                return False
        nurses = self._pick_nurses(hs, t, v)
        if len(nurses) < v.nurse_slots:
            self.counts[f"blocked:no_nurse:{v.required_capability}"] += 1
            return False
        if bay == "regular":
            hs.regular_used += 1
        else:
            hs.surge_used += 1
        hs.in_care.append(v.victim_id)
        if doctor is not None:
            doctor.patients.append(v.victim_id)
        for n in nurses:
            n.patients.append(v.victim_id)
        vs.bay, vs.status, vs.hospital_id = bay, "in_care", hs.h.hospital_id
        vs.doctor_id = doctor.s.staff_id if doctor is not None else None
        vs.nurse_ids = tuple(n.s.staff_id for n in nurses)
        vs.care_start, vs.care_end = t, t + v.treatment_min
        vs.survival = survival_probability(t - vs.report_time, vs.window, v.survival_if_timely, v.decay_half_life_min)
        self._event(t, "admitted", victim=v.victim_id, triage=v.triage, hospital=hs.h.hospital_id,
                    minutes_to_care=t - vs.report_time, survival=round(vs.survival, 3), bay=bay)
        return True

    def _admissions(self, t: int) -> None:
        for hs in self.w.hospitals.values():
            for vid in sorted(hs.corridor, key=lambda v: (TRIAGE_RANK[self.w.victims[v].v.triage], v)):
                if self._admit(self.w.victims[vid], hs, t):
                    hs.corridor.remove(vid)
            for aid in list(hs.door):
                amb = self.w.ambulances[aid]
                for vid in sorted(amb.onboard, key=lambda v: (TRIAGE_RANK[self.w.victims[v].v.triage], v)):
                    if self._admit(self.w.victims[vid], hs, t):
                        amb.onboard.remove(vid)
                if amb.onboard and t - amb.status_since >= P.OFFLOAD_LIMIT_MIN:
                    for vid in amb.onboard:
                        self.w.victims[vid].status = "corridor"
                        hs.corridor.append(vid)
                    self.counts["corridor_handovers"] += len(amb.onboard)
                    self._event(t, "corridor_handover", unit=aid, hospital=hs.h.hospital_id, victims=list(amb.onboard))
                    amb.onboard = []
                if not amb.onboard:
                    hs.door.remove(aid)
                    amb.until = t + P.HANDOFF_MIN
                    self._set(amb, "handoff", t)

    def _free_bay(self, hs: HospitalState, vs: VictimState) -> None:
        if vs.bay == "regular":
            hs.regular_used -= 1
        elif vs.bay == "surge":
            hs.surge_used -= 1
        hs.in_care.remove(vs.v.victim_id)
        vs.bay = None

    def _treatments(self, t: int) -> None:
        for hs in self.w.hospitals.values():
            for vid in list(hs.in_care):
                vs = self.w.victims[vid]
                if vs.boarding or vs.care_end != t:
                    continue
                for sid in (vs.doctor_id, *vs.nurse_ids):
                    if sid:
                        self.w.staff[sid].patients.remove(vid)
                vs.status = "done"
                if vs.v.needs_icu and hs.icu_free == 0:
                    vs.boarding = True        # no ICU bed: the patient keeps the ER bay
                    self._event(t, "boarding", victim=vid, hospital=hs.h.hospital_id)
                    continue
                if vs.v.needs_icu:
                    hs.icu_free -= 1
                self._free_bay(hs, vs)
                self._event(t, "discharged_from_er", victim=vid, hospital=hs.h.hospital_id)

    def _timed_events(self, t: int) -> None:
        for hid, hs in self.w.hospitals.items():
            if hs.surge_ready_at == t:
                self._event(t, "surge_open", hospital=hid, bays=hs.h.surge_bays)
        for m in self.w.staff.values():
            if m.hospital_id is not None and m.arrives_at == t and t > 0:
                self._event(t, "staff_arrived", staff=m.s.staff_id, hospital=m.hospital_id)

    def _auto_return(self, t: int) -> None:
        for unit in (*self.w.ambulances.values(), *self.w.teams.values()):
            if unit.status == "idle" and unit.pos != unit.base and t - unit.status_since >= P.IDLE_RETURN_MIN:
                self._start_trip(unit, unit.base, t, rest=False)
                self._set(unit, "returning", t)

    def _start_trip(self, unit, dest: Point, t: int, rest: bool) -> None:
        origin = unit.position(t)
        speed = unit.a.speed_kmh if isinstance(unit, AmbulanceState) else unit.r.speed_kmh
        depart = t + (P.TURNOUT_MIN if rest else 0)
        unit.trip = Trip(origin, dest, depart, depart + drive_minutes(self.w.travel, origin, dest, speed, t))

    # ------------------------------------------------------------------ actions
    def _apply(self, action: Action, t: int) -> ActionResult:
        handler = {
            DispatchAmbulance: self._do_dispatch_ambulance, DispatchTeam: self._do_dispatch_team,
            SetDestination: self._do_set_destination, ReturnToBase: self._do_return,
            AssignStaff: self._do_assign_staff, OpenSurge: self._do_open_surge,
        }[type(action)]
        ok, reason = handler(action, t)
        self.counts[f"{type(action).__name__}:{'ok' if ok else 'rejected'}"] += 1
        if not ok:
            self._event(t, "action_rejected", action=type(action).__name__, reason=reason, **action.__dict__)
        return ActionResult(action, ok, reason)

    def _do_dispatch_ambulance(self, a: DispatchAmbulance, t: int) -> tuple[bool, str]:
        amb, inc = self.w.ambulances.get(a.ambulance_id), self.w.incidents.get(a.incident_id)
        if amb is None or inc is None:
            return False, "unknown ambulance or incident"
        if not inc.visible:
            return False, "incident not reported yet"
        if amb.status not in AMBULANCE_DISPATCHABLE or amb.onboard:
            return False, f"ambulance is {amb.status}"
        if amb.incident_id == a.incident_id and amb.status in ("to_scene", "at_scene"):
            return False, "already assigned to this incident"
        if inc.revealed_at is not None:
            waiting = [vs for vs in self.w.victims_of(a.incident_id) if vs.status == "at_scene"]
            if not waiting:
                return False, "nobody left at the scene"
            if amb.a.unit_type == "BLS" and all(vs.v.triage == "RED" for vs in waiting):
                return False, "a BLS unit cannot carry the RED patients waiting there"
        rest = amb.status in ("idle", "at_scene")
        self._start_trip(amb, (inc.inc.x, inc.inc.y), t, rest=rest)
        amb.incident_id = a.incident_id
        self._set(amb, "to_scene", t)
        self._event(t, "dispatched", unit=a.ambulance_id, incident=a.incident_id, eta=amb.trip.arrive)
        return True, ""

    def _do_dispatch_team(self, a: DispatchTeam, t: int) -> tuple[bool, str]:
        team, inc = self.w.teams.get(a.team_id), self.w.incidents.get(a.incident_id)
        if team is None or inc is None:
            return False, "unknown team or incident"
        if not inc.visible:
            return False, "incident not reported yet"
        if team.status not in TEAM_DISPATCHABLE:
            return False, f"team is {team.status}"
        if team.incident_id == a.incident_id and team.status == "to_scene":
            return False, "already assigned to this incident"
        if not any(c in team.r.capabilities and c not in inc.work_done for c in inc.work_left):
            return False, "no open work this team can do"
        self._start_trip(team, (inc.inc.x, inc.inc.y), t, rest=team.status == "idle")
        team.incident_id = a.incident_id
        self._set(team, "to_scene", t)
        self._event(t, "dispatched", unit=a.team_id, incident=a.incident_id, eta=team.trip.arrive)
        return True, ""

    def _do_set_destination(self, a: SetDestination, t: int) -> tuple[bool, str]:
        amb, hs = self.w.ambulances.get(a.ambulance_id), self.w.hospitals.get(a.hospital_id)
        if amb is None or hs is None:
            return False, "unknown ambulance or hospital"
        if not amb.onboard:
            return False, "no patients on board"
        needed = {self.w.victims[v].v.required_capability for v in amb.onboard}
        if not needed <= set(hs.h.capabilities):
            return False, f"{a.hospital_id} lacks {sorted(needed - set(hs.h.capabilities))}"
        if amb.hospital_id == a.hospital_id:
            return False, "already going there"
        if amb.status == "loading":
            amb.hospital_id = a.hospital_id
            self._event(t, "destination", unit=a.ambulance_id, hospital=a.hospital_id)
            return True, ""
        if amb.status in ("loaded", "to_hospital", "at_door"):
            if amb.status == "at_door":
                self.w.hospitals[amb.hospital_id].door.remove(amb.a.ambulance_id)
            diverted = amb.status != "loaded"
            amb.hospital_id = a.hospital_id
            self._drive_to_hospital(amb, t)
            if diverted:
                self.counts["diversions"] += 1
                self._event(t, "diverted", unit=a.ambulance_id, hospital=a.hospital_id)
            return True, ""
        return False, f"ambulance is {amb.status}"

    def _do_return(self, a: ReturnToBase, t: int) -> tuple[bool, str]:
        unit = self.w.ambulances.get(a.unit_id) or self.w.teams.get(a.unit_id)
        if unit is None:
            return False, "unknown unit"
        if unit.status not in ("idle", "at_scene") or getattr(unit, "onboard", []):
            return False, f"unit is {unit.status}"
        self._start_trip(unit, unit.base, t, rest=False)
        if isinstance(unit, AmbulanceState):
            unit.incident_id = None
        self._set(unit, "returning", t)
        return True, ""

    def _do_assign_staff(self, a: AssignStaff, t: int) -> tuple[bool, str]:
        m, hs = self.w.staff.get(a.staff_id), self.w.hospitals.get(a.hospital_id)
        if m is None or hs is None:
            return False, "unknown staff or hospital"
        pool = m.s.home_hospital == "POOL"
        if not pool and not m.s.lendable:
            return False, "unit-bound specialist cannot be lent"
        if m.patients:
            return False, "still treating patients"
        if m.hospital_id == a.hospital_id:
            return False, "already there"
        if t >= m.s.on_duty_until_min:
            return False, "off duty"
        if not pool and a.hospital_id != m.s.home_hospital:
            home = self.w.hospitals[m.s.home_hospital]
            lent = sum(1 for o in self.w.staff.values() if o.s.home_hospital == m.s.home_hospital
                       and o.s.role == m.s.role and o.hospital_id not in (None, o.s.home_hospital))
            limit = home.h.max_staff_lend_doctors if m.s.role == "doctor" else home.h.max_staff_lend_nurses
            if lent >= limit:
                return False, f"{m.s.home_hospital} lends at most {limit} {m.s.role}s"
        if m.hospital_id is None:
            m.arrives_at = max(m.s.on_duty_from_min, t + P.POOL_REPORT_DELAY_MIN) if t >= m.s.on_duty_from_min \
                else m.s.on_duty_from_min
        else:
            here = self.w.hospitals[m.hospital_id].h
            m.arrives_at = t + drive_minutes(self.w.travel, (here.x, here.y), (hs.h.x, hs.h.y), P.AMBULANCE_SPEED_KMH, t) \
                + P.STAFF_TRANSFER_EXTRA_MIN
        source = m.hospital_id or "POOL"
        m.hospital_id = a.hospital_id
        self.counts["staff_moves"] += 1
        self._event(t, "staff_assigned", staff=a.staff_id, role=m.s.role, specialty=m.s.specialty,
                    source=source, hospital=a.hospital_id, arrives=m.arrives_at)
        return True, ""

    def _do_open_surge(self, a: OpenSurge, t: int) -> tuple[bool, str]:
        hs = self.w.hospitals.get(a.hospital_id)
        if hs is None:
            return False, "unknown hospital"
        if hs.surge_ready_at is not None:
            return False, "surge already opened"
        if hs.h.surge_bays == 0:
            return False, "no surge bays"
        hs.surge_ready_at = t + P.SURGE_ACTIVATION_MIN
        self.counts["surges"] += 1
        self._event(t, "surge_ordered", hospital=a.hospital_id, ready=hs.surge_ready_at)
        return True, ""

    # ------------------------------------------------------------------ end of minute
    def _settle(self, t: int) -> None:
        for amb in self.w.ambulances.values():
            if amb.status == "loaded" and amb.hospital_id is None and t - amb.status_since >= P.DESTINATION_GRACE_MIN:
                needed = {self.w.victims[v].v.required_capability for v in amb.onboard}
                options = [hs for hs in self.w.hospitals.values() if needed <= set(hs.h.capabilities)]
                pos = amb.position(t)
                best = min(options, key=lambda hs: (drive_minutes(self.w.travel, pos, (hs.h.x, hs.h.y), amb.a.speed_kmh, t),
                                                    hs.h.hospital_id))
                amb.hospital_id = best.h.hospital_id
                self.counts["fallback_destinations"] += 1
                self._event(t, "destination_fallback", unit=amb.a.ambulance_id, hospital=amb.hospital_id)
                self._drive_to_hospital(amb, t)

    def _account(self, t: int) -> None:
        for amb in self.w.ambulances.values():
            self.counts[f"ambulance_minutes:{amb.status}"] += 1
            if amb.status not in ("idle", "returning"):
                amb.busy_minutes += 1
            if amb.status == "at_door":
                amb.ramp_minutes += 1
        for team in self.w.teams.values():
            if team.status not in ("idle", "returning"):
                team.busy_minutes += 1
        used = total = 0
        for hs in self.w.hospitals.values():
            assert 0 <= hs.regular_used <= hs.h.er_bays_total - hs.h.er_bays_occupied, "regular bays over capacity"
            assert 0 <= hs.surge_used <= (hs.h.surge_bays if hs.surge_ready_at is not None else 0), "surge over capacity"
            open_surge = hs.h.surge_bays if hs.surge_ready_at is not None and t >= hs.surge_ready_at else 0
            used += hs.h.er_bays_occupied + hs.regular_used + hs.surge_used
            total += hs.h.er_bays_total + open_surge
        self.samples["bays"].append(used / total)
        slots_used = slots = 0
        for m in self.w.staff.values():
            assert m.s.baseline_load + len(m.patients) <= m.s.workload_limit or not m.patients, "staff over workload"
            if m.available(t):
                slots += m.s.workload_limit
                slots_used += m.s.baseline_load + len(m.patients)
        self.samples["staff"].append(slots_used / max(1, slots))

    # ------------------------------------------------------------------ run log
    def _snapshot(self, t: int, decision: Decision, results: list[ActionResult]) -> dict:
        w = self.w
        incidents = []
        for iid, inc in w.incidents.items():
            if not inc.visible:
                continue
            vs = w.victims_of(iid)
            waiting = [v for v in vs if v.status == "at_scene"]
            incidents.append({
                "id": iid, "type": inc.inc.incident_type, "x": inc.inc.x, "y": inc.inc.y,
                "reported": inc.inc.report.reported_victims, "revealed": inc.revealed_at is not None,
                "waiting": ({k: sum(v.v.triage == k for v in waiting) for k in TRIAGE_RANK}
                            if inc.revealed_at is not None else None),
                "worst": next((k for k in TRIAGE_RANK if any(v.v.triage == k for v in vs)), "GREEN"),
                "access": inc.access_open(t), "needs": {c: max(0.0, left) for c, left in inc.work_left.items()},
                "teams": list(inc.teams),
            })
        ambulances = [{"id": a.a.ambulance_id, "type": a.a.unit_type, "status": a.status,
                       "x": round(a.position(t)[0], 3), "y": round(a.position(t)[1], 3),
                       "incident": a.incident_id, "hospital": a.hospital_id, "onboard": len(a.onboard)}
                      for a in w.ambulances.values()]
        teams = [{"id": r.r.team_id, "type": r.r.team_type, "status": r.status,
                  "x": round(r.position(t)[0], 3), "y": round(r.position(t)[1], 3),
                  "incident": r.incident_id, "task": r.task} for r in w.teams.values()]
        hospitals = []
        for hid, hs in w.hospitals.items():
            staff_here = [m for m in w.staff.values() if m.hospital_id == hid and m.available(t)]
            hospitals.append({
                "id": hid, "bays_used": hs.h.er_bays_occupied + hs.regular_used + hs.surge_used,
                "bays_total": hs.h.er_bays_total + (hs.h.surge_bays if hs.surge_ready_at is not None and t >= hs.surge_ready_at else 0),
                "surge": "closed" if hs.surge_ready_at is None else ("open" if t >= hs.surge_ready_at else "opening"),
                "icu_free": hs.icu_free, "door": len(hs.door), "corridor": len(hs.corridor),
                "doctor_slots_free": sum(m.free_slots(t) for m in staff_here if m.s.role == "doctor"),
                "doctor_slots": sum(m.s.workload_limit for m in staff_here if m.s.role == "doctor"),
                "nurse_slots_free": sum(m.free_slots(t) for m in staff_here if m.s.role == "nurse"),
                "nurse_slots": sum(m.s.workload_limit for m in staff_here if m.s.role == "nurse"),
                "lent_in": sum(1 for m in staff_here if m.s.home_hospital != hid),
            })
        served = [v for v in w.victims.values() if v.care_start is not None]
        return {
            "t": t,
            "incidents": incidents, "ambulances": ambulances, "teams": teams, "hospitals": hospitals,
            "served": {k: sum(v.v.triage == k for v in served) for k in TRIAGE_RANK},
            "events": self.events,
            "actions": [r.as_dict() for r in results],
            "messages": [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in decision.messages],
            "notes": decision.notes,
        }
