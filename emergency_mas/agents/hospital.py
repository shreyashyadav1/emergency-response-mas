"""Hospital Coordination Agent: one per hospital.

It guards its own bays, staff and capabilities. For each admission request it
projects its state to the ambulance's arrival time (patients already on their way
are served first), keeps bays back for RED patients, and answers accept, reject
or, when negotiation is on, a counter-offer: "yes if you lend me a trauma nurse"
or "yes in 12 minutes". It opens surge bays on its own when the door is filling.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .. import params as P
from ..policies.common import inbound
from ..sim.actions import OpenSurge
from ..sim.observe import HospitalView, Observation, StaffView, VictimView
from ..sim.state import TRIAGE_RANK
from .messages import ALLOCATION_AGENT, DISPATCH_AGENT, Message, MessageBus, MsgType, ProposedAction, hospital_agent

COUNTER_WAIT_LIMIT_MIN = 45   # a hospital only offers "yes, later" if room is expected within 45 minutes


@dataclass(frozen=True)
class Reply:
    kind: str                     # accept, reject, counter
    reason: str = ""
    condition: str | None = None  # counters: "staff:<role>:<skill>" or "after:<minutes>"
    wait: int | None = None       # counters "after": minutes from now until room is expected


def free_at(hv: HospitalView, at: int, extra: tuple[StaffView, ...] = ()) -> tuple[dict[str, int], dict[str, StaffView]]:
    """Free workload slots per clinician at minute `at`, counting patients who will have left by then."""
    releases: Counter[str] = Counter()
    for c in hv.in_care:
        if not c.boarding and c.care_end is not None and c.care_end <= at:
            for sid in (c.doctor_id, *c.nurse_ids):
                if sid:
                    releases[sid] += 1
    free, meta = {}, {}
    for m in (*hv.staff, *extra):
        if m.arrives_at > at or not (m.on_duty_from <= at < m.on_duty_until):
            continue
        free[m.staff_id] = max(0, m.workload_limit - m.baseline_load - m.patients + releases[m.staff_id])
        meta[m.staff_id] = m
    return free, meta


def bays_at(hv: HospitalView, at: int) -> int:
    """Bays free at minute `at`: free now, plus surge if ready by then, plus patients who will have left."""
    icu = hv.icu_free
    released = 0
    for c in sorted(hv.in_care, key=lambda c: c.care_end or 10**9):
        if c.boarding or c.care_end is None or c.care_end > at:
            continue
        if c.needs_icu:
            if icu == 0:
                continue              # no ICU bed: this patient will keep the bay
            icu -= 1
        released += 1
    surge = hv.surge_total - hv.surge_used if hv.surge_ready_at is not None and hv.surge_ready_at <= at else 0
    return hv.regular_free + surge + released


def fit_staff(victims: list[VictimView], free: dict[str, int], meta: dict[str, StaffView]) -> tuple[bool, str]:
    """Give each victim the doctor and nurse slots the simulator will ask for. Changes `free` in place."""
    for v in sorted(victims, key=lambda v: (TRIAGE_RANK[v.triage], v.victim_id)):
        need, skill = P.STAFF_NEED[v.triage], v.required_capability
        if need["doctors"]:
            doctors = sorted((s for s in free if meta[s].role == "doctor" and free[s] > 0),
                             key=lambda s: (-free[s], s))
            pick = next((s for s in doctors if skill in meta[s].skills), None)
            if pick is None and skill == "general_emergency" and doctors:
                pick = doctors[0]
            if pick is None and v.triage == "YELLOW" and P.YELLOW_GENERALIST_OK:
                pick = next((s for s in doctors if "general_emergency" in meta[s].skills), None)
            if pick is None:
                return False, f"doctor:{skill}"
            free[pick] -= 1
        nurses = sorted((s for s in free if meta[s].role == "nurse" and free[s] > 0), key=lambda s: (-free[s], s))
        chosen = []
        if skill != "general_emergency":
            first = next((s for s in nurses if skill in meta[s].skills), None)
            if first is None and not (v.triage == "YELLOW" and P.YELLOW_GENERALIST_OK):
                return False, f"nurse:{skill}"
            if first is not None:
                chosen.append(first)
        for s in nurses:
            if len(chosen) >= need["nurses"]:
                break
            if s not in chosen:
                chosen.append(s)
        if len(chosen) < need["nurses"]:
            return False, "nurse:any"
        for s in chosen:
            free[s] -= 1
    return True, ""


class HospitalCoordinationAgent:
    def __init__(self, hospital_id: str, negotiation: bool):
        self.hid = hospital_id
        self.name = hospital_agent(hospital_id)
        self.negotiation = negotiation
        self.tally: Counter[str] = Counter()

    # ------------------------------------------------------------------ admissions
    def evaluate(self, obs: Observation, victims: list[VictimView], eta: int,
                 extra_staff: tuple[StaffView, ...] = ()) -> Reply:
        hv = obs.hospital(self.hid)
        needed = {v.required_capability for v in victims}
        if not needed <= set(hv.capabilities):
            return Reply("reject", f"cannot treat {sorted(needed - set(hv.capabilities))[0]}")
        at = obs.t + eta
        ids = {v.victim_id for v in victims}
        coming = [v for v in inbound(obs, self.hid) if v.victim_id not in ids]
        has_red = any(v.triage == "RED" for v in victims)
        if not has_red and all(v.triage == "GREEN" for v in victims) and hv.occupancy >= hv.diversion_threshold:
            return Reply("reject", "on diversion for minor patients")
        room = bays_at(hv, at) - len(coming) - (0 if has_red else hv.reserve_bays_for_red)
        free, meta = free_at(hv, at, extra_staff)
        fit_staff(coming, free, meta)
        staffed, missing = fit_staff(victims, free, meta)
        if room >= len(victims) and staffed:
            return Reply("accept", "room and staff at arrival")
        if room >= len(victims):
            if self.negotiation:
                return Reply("counter", f"a bay is free but no {missing.replace(':', ' for ')}", f"staff:{missing}")
            return Reply("reject", f"no {missing.replace(':', ' for ')} free")
        wait = self.minutes_to_room(obs, len(victims) - room, at)
        if self.negotiation and wait is not None and wait <= COUNTER_WAIT_LIMIT_MIN:
            return Reply("counter", f"full now, room expected in {wait} min", f"after:{wait}", wait)
        reserved = "" if has_red or hv.reserve_bays_for_red == 0 else f" (holding {hv.reserve_bays_for_red} for RED)"
        return Reply("reject", "no bay at arrival" + reserved)

    def reply(self, obs: Observation, victims: list[VictimView], eta: int) -> Reply:
        """The answer to an admission request. For the rules agent it is the evaluation itself."""
        return self.evaluate(obs, victims, eta)

    def minutes_to_room(self, obs: Observation, extra_bays: int, at: int) -> int | None:
        """Minutes from now until `extra_bays` more bays should be free, from scheduled discharges."""
        hv = obs.hospital(self.hid)
        ends = sorted(c.care_end for c in hv.in_care
                      if not c.boarding and c.care_end is not None and c.care_end > at and not c.needs_icu)
        if extra_bays <= 0:
            return max(0, at - obs.t)
        if len(ends) < extra_bays:
            return None
        return ends[extra_bays - 1] - obs.t

    def answer(self, bus: MessageBus, request: Message, reply: Reply, round_: int) -> Message:
        self.tally[reply.kind] += 1
        kind = {"accept": MsgType.ADMISSION_ACCEPT, "reject": MsgType.ADMISSION_REJECT,
                "counter": MsgType.COUNTER_PROPOSAL}[reply.kind]
        action = ProposedAction(kind="admit" if reply.kind == "accept" else reply.kind,
                                unit_id=request.proposed_action.unit_id, hospital_id=self.hid,
                                victim_ids=request.proposed_action.victim_ids, eta_min=request.proposed_action.eta_min,
                                condition=reply.condition)
        return bus.send(self.name, request.sender, kind, action, incident_id=request.incident_id,
                        rationale=reply.reason, in_reply_to=request.msg_id, round=round_)

    # ------------------------------------------------------------------ surge and lending
    def surge(self, obs: Observation, bus: MessageBus) -> list:
        hv = obs.hospital(self.hid)
        if hv.surge_ready_at is not None or hv.surge_total == 0:
            return []
        pressure = len(inbound(obs, self.hid)) - hv.free_bays
        if pressure < 0 or hv.free_slots("nurse") < 2:
            return []
        bus.send(self.name, DISPATCH_AGENT, MsgType.STATUS_UPDATE,
                 ProposedAction(kind="open_surge", hospital_id=self.hid, value=hv.surge_total),
                 rationale=f"{len(inbound(obs, self.hid))} patients inbound for {hv.free_bays} free bays; "
                           f"opening {hv.surge_total} surge bays in {P.SURGE_ACTIVATION_MIN} min")
        return [OpenSurge(self.hid)]

    def offer(self, obs: Observation, role: str, skill: str) -> StaffView | None:
        """A clinician this hospital can spare for someone else, keeping a slot in reserve for itself."""
        hv = obs.hospital(self.hid)
        lent = sum(1 for h in obs.hospitals for m in h.staff
                   if m.home_hospital == self.hid and m.hospital_id != self.hid and m.role == role)
        if lent >= (hv.max_lend_doctors if role == "doctor" else hv.max_lend_nurses):
            return None
        free, meta = free_at(hv, obs.t + 30)
        fit_staff(inbound(obs, self.hid), free, meta)
        spare = sum(n for s, n in free.items() if meta[s].role == role)
        candidates = sorted((m for m in hv.staff if m.home_hospital == self.hid and m.lendable and m.role == role
                             and m.available and m.patients == 0 and (skill in m.skills or skill == "any")),
                            key=lambda m: (-m.free_slots, m.staff_id))
        for m in candidates:
            if spare - free.get(m.staff_id, 0) >= 1:
                return m
        return None

    def note(self, obs: Observation) -> str:
        hv = obs.hospital(self.hid)
        surge = "closed" if hv.surge_ready_at is None else ("open" if obs.t >= hv.surge_ready_at else "opening")
        return (f"{hv.free_bays} free bays, {len(inbound(obs, self.hid))} inbound, {len(hv.door)} at door, surge {surge}; "
                f"accepted {self.tally['accept']}, countered {self.tally['counter']}, rejected {self.tally['reject']}")


__all__ = ["HospitalCoordinationAgent", "Reply", "ALLOCATION_AGENT"]
