"""Resource Allocation Agent: the referee between dispatch and the hospitals.

Three jobs. When several incidents want the same unit, it awards units so the total
value is highest (an assignment problem) and tells each loser what it gets instead.
When a hospital counters an admission request, it compares "later", "only if
staffed" and "somewhere farther" by when the patient would actually start care,
and arranges the staff loan from the regional pool or another hospital. And it
sends the pool's clinicians to the hospitals that are about to run short.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .. import params as P
from ..policies.common import capable, inbound, onboard_capabilities
from ..sim.actions import AssignStaff, SetDestination
from ..sim.observe import AmbulanceView, HospitalView, Observation, StaffView
from .assessment import Assessment
from .hospital import HospitalCoordinationAgent
from .messages import ALLOCATION_AGENT, DISPATCH_AGENT, Message, MessageBus, MsgType, ProposedAction, hospital_agent

DOOR_PATIENCE_MIN = 10        # after 10 minutes at a full ER door, look for a hospital that can take the patient sooner
FORECAST_SHARE = 0.3          # weight of patients still at scenes when deciding where pool staff are needed


def _uid(unit) -> str:
    return getattr(unit, "ambulance_id", None) or unit.team_id


class ResourceAllocationAgent:
    name = ALLOCATION_AGENT

    def __init__(self) -> None:
        self.tally: Counter[str] = Counter()
        self.used_staff: set[str] = set()
        self.reasons: dict[str, str] = {}      # unit -> why it went where it went, when an LLM agent chose

    def start(self) -> None:
        self.used_staff = set()
        self.reasons = {}

    # ------------------------------------------------------------------ contested units
    def assign(self, label: str, slots: list, units: list, v: np.ndarray,
               bidders: dict[int, list[int]]) -> list[tuple[int, int]]:
        """Which slot each unit serves: the assignment with the highest total value."""
        rows, cols = linear_sum_assignment(-v)
        return [(int(i), int(j)) for i, j in zip(rows, cols) if v[i, j] > 0]

    def award(self, label: str, slots: list, units: list, values: list[list[float]], bus: MessageBus) -> list[tuple[int, int]]:
        v = np.array(values, dtype=float)
        if v.size == 0 or v.max() <= 0:
            return []
        bidders: dict[int, list[int]] = defaultdict(list)
        for i in range(len(slots)):
            if v[i].max() > 0:
                bidders[int(v[i].argmax())].append(i)
        pairs = self.assign(label, slots, units, v, bidders)
        winner = {j: i for i, j in pairs}
        alternative = {i: j for i, j in pairs}
        settled: dict[int, Message] = {}
        for j, group in sorted(bidders.items()):
            incidents = sorted({slots[i].incident_id for i in group})
            if len(incidents) < 2:
                continue
            self.tally["conflicts"] += 1
            unit = _uid(units[j])
            notice = bus.send(DISPATCH_AGENT, self.name, MsgType.CONFLICT_NOTICE,
                              ProposedAction(kind=f"dispatch_{label}", unit_id=unit, value=round(float(v[group, j].max()), 3)),
                              incident_id=incidents[0],
                              rationale=f"{unit} is the first choice of " + ", ".join(
                                  f"{slots[i].incident_id} ({slots[i].kind}, value {v[i, j]:.2f})" for i in group))
            settled[j] = notice
        for i, j in pairs:
            unit, slot = _uid(units[j]), slots[i]
            notice = settled.get(j)
            if notice is None:
                why = f"{unit} is the best free {label} for {slot.incident_id}" + ("" if label == "ambulance" else f" ({slot.kind})")
            else:
                losers = [k for k in bidders[j] if k != i]
                parts = []
                for k in losers:
                    alt = alternative.get(k)
                    parts.append(f"{slots[k].incident_id} takes {_uid(units[alt])} instead" if alt is not None
                                 else f"{slots[k].incident_id} waits for the next free {label}")
                why = f"{unit} to {slot.incident_id} {self.reasons.get(unit, 'maximizes total value')}; " + "; ".join(parts)
            bus.send(self.name, DISPATCH_AGENT, MsgType.ALLOCATION_DECISION,
                     ProposedAction(kind=f"dispatch_{label}", unit_id=unit, value=round(float(v[i, j]), 3)),
                     incident_id=slot.incident_id, rationale=why,
                     in_reply_to=notice.msg_id if notice else None, round=2 if notice else 1)
        return pairs

    # ------------------------------------------------------------------ admissions
    def _find_staff(self, obs: Observation, target: str, role: str, skill: str,
                    hospitals: dict[str, HospitalCoordinationAgent], bus: MessageBus,
                    request: Message) -> tuple[StaffView, int, str] | None:
        found: list[tuple[int, str, StaffView]] = []
        for m in obs.pool:
            if m.staff_id in self.used_staff or m.role != role or obs.t >= m.on_duty_until:
                continue
            if skill != "any" and skill not in m.skills:
                continue
            arrival = (max(m.on_duty_from, obs.t + P.POOL_REPORT_DELAY_MIN) if obs.t >= m.on_duty_from
                       else m.on_duty_from)
            found.append((arrival, "the regional pool", m))
        dst = obs.hospital(target)
        for hid, hca in hospitals.items():
            if hid == target:
                continue
            ask = bus.send(self.name, hospital_agent(hid), MsgType.STAFF_REQUEST,
                           ProposedAction(kind="lend_staff", hospital_id=target, condition=f"{role}:{skill}"),
                           incident_id=request.incident_id, rationale=f"{target} needs a {role} who can handle {skill}",
                           in_reply_to=request.msg_id, round=2)
            m = hca.offer(obs, role, skill)
            if m is None or m.staff_id in self.used_staff:
                bus.send(hca.name, self.name, MsgType.STAFF_OFFER, ProposedAction(kind="reject", hospital_id=target),
                         incident_id=request.incident_id, rationale="nobody to spare without falling short",
                         in_reply_to=ask.msg_id, round=2)
                continue
            arrival = obs.t + obs.drive(obs.hospital(hid).pos, dst.pos) + P.STAFF_TRANSFER_EXTRA_MIN
            bus.send(hca.name, self.name, MsgType.STAFF_OFFER,
                     ProposedAction(kind="lend_staff", staff_id=m.staff_id, hospital_id=target, eta_min=arrival - obs.t),
                     incident_id=request.incident_id, rationale=f"can lend {m.staff_id} ({m.specialty})",
                     in_reply_to=ask.msg_id, round=2)
            found.append((arrival, hid, m))
        if not found:
            return None
        arrival, source, m = min(found, key=lambda f: (f[0], f[1], f[2].staff_id))
        return replace(m, hospital_id=target, arrives_at=arrival), arrival, source

    def pick_settlement(self, obs: Observation, amb: AmbulanceView, options: list) -> tuple:
        """The option where care starts soonest; ties go to the simpler deal, then the hospital id.
        Each option is (care start, rank, hospital, staff actions, why, request), rank 0 accept, 1 later, 2 with staff."""
        return min(options, key=lambda o: (o[0], o[1], o[2]))

    def settle_admission(self, obs: Observation, amb: AmbulanceView, replies: list, bus: MessageBus,
                         hospitals: dict[str, HospitalCoordinationAgent]) -> tuple[str, list]:
        """Pick the hospital where this load starts care soonest, arranging staff if that is what it takes."""
        first_id, _, first_reply, _ = replies[0]
        if first_reply.kind == "accept":
            return first_id, []
        victims = list(amb.onboard)
        options = []
        for hid, eta, reply, request in replies:
            if reply.kind == "accept":
                options.append((eta, 0, hid, [], f"{hid} accepts on arrival (ETA {eta} min)", request))
            elif reply.kind == "counter" and reply.condition.startswith("after:"):
                wait = int(reply.condition.split(":")[1])
                options.append((max(eta, wait), 1, hid, [], f"{hid} has room in {wait} min (ETA {eta})", request))
            elif reply.kind == "counter" and reply.condition.startswith("staff:"):
                _, role, skill = reply.condition.split(":")
                found = self._find_staff(obs, hid, role, skill, hospitals, bus, request)
                if found is None:
                    continue
                staff, arrival, source = found
                if hospitals[hid].evaluate(obs, victims, eta, extra_staff=(staff,)).kind != "accept":
                    continue
                options.append((max(eta, arrival - obs.t), 2, hid, [AssignStaff(staff.staff_id, hid)],
                                f"{hid} accepts if {staff.staff_id} ({staff.specialty}) comes from {source}", request))
        if options:
            care, rank, hid, extra, why, request = self.pick_settlement(obs, amb, options)
        else:
            candidates = []
            for h in capable(obs, onboard_capabilities(amb)):
                eta = obs.drive(amb.pos, h.pos)
                wait = hospitals[h.hospital_id].minutes_to_room(obs, 1, obs.t + eta)
                candidates.append((max(eta, wait if wait is not None else 10**6), eta, h.hospital_id))
            care, _, hid = min(candidates)
            rank, extra, request = 3, [], None
            why = f"nobody can take them on arrival; {hid} should free a bay first"
        self.tally["negotiated_admissions"] += 1
        if extra:
            self.used_staff.add(extra[0].staff_id)
            self.tally["staff_loans"] += 1
        incident = victims[0].incident_id
        if request is not None and rank > 0:
            bus.send(self.name, hospital_agent(hid), MsgType.ALLOCATION_DECISION,
                     ProposedAction(kind="admit", unit_id=amb.ambulance_id, hospital_id=hid,
                                    victim_ids=[v.victim_id for v in victims], eta_min=care,
                                    staff_id=extra[0].staff_id if extra else None),
                     incident_id=incident, rationale=f"agreed: {why}", in_reply_to=request.msg_id,
                     round=3 if extra else 2)
        bus.send(self.name, DISPATCH_AGENT, MsgType.ALLOCATION_DECISION,
                 ProposedAction(kind="set_destination", unit_id=amb.ambulance_id, hospital_id=hid, eta_min=care),
                 incident_id=incident, rationale=why, in_reply_to=(request or replies[0][3]).msg_id,
                 round=min(P.MAX_NEGOTIATION_ROUNDS, rank + 1))
        return hid, extra

    def rebalance(self, obs: Observation, bus: MessageBus, hospitals: dict[str, HospitalCoordinationAgent]) -> list:
        """Move an ambulance stuck at a full ER door to a hospital that can start care sooner."""
        actions = []
        for amb in obs.ambulances:
            if amb.status != "at_door" or not amb.onboard or obs.t - amb.since < DOOR_PATIENCE_MIN:
                continue
            wait_here = hospitals[amb.hospital_id].minutes_to_room(obs, 1, obs.t)
            wait_here = 60 if wait_here is None else wait_here
            for h in sorted(capable(obs, onboard_capabilities(amb)), key=lambda h: (obs.drive(amb.pos, h.pos), h.hospital_id)):
                eta = obs.drive(amb.pos, h.pos)
                if h.hospital_id == amb.hospital_id or eta + 5 >= wait_here:
                    continue
                request = bus.send(self.name, hospital_agent(h.hospital_id), MsgType.ADMISSION_REQUEST,
                                   ProposedAction(kind="admit", unit_id=amb.ambulance_id, hospital_id=h.hospital_id,
                                                  victim_ids=[v.victim_id for v in amb.onboard], eta_min=eta),
                                   incident_id=amb.onboard[0].incident_id,
                                   rationale=f"{amb.ambulance_id} has waited {obs.t - amb.since} min at {amb.hospital_id}, "
                                             f"room there in about {wait_here} min")
                reply = hospitals[h.hospital_id].evaluate(obs, list(amb.onboard), eta)
                hospitals[h.hospital_id].answer(bus, request, reply, 1)
                if reply.kind == "accept":
                    self.tally["diversions"] += 1
                    actions.append(SetDestination(amb.ambulance_id, h.hospital_id))
                    bus.send(self.name, DISPATCH_AGENT, MsgType.ALLOCATION_DECISION,
                             ProposedAction(kind="set_destination", unit_id=amb.ambulance_id, hospital_id=h.hospital_id,
                                            eta_min=eta),
                             incident_id=amb.onboard[0].incident_id,
                             rationale=f"divert to {h.hospital_id}: care in {eta} min instead of about {wait_here}",
                             in_reply_to=request.msg_id, round=2)
                    break
        return actions

    # ------------------------------------------------------------------ regional pool
    @staticmethod
    def _need(obs: Observation, h: HospitalView, m: StaffView, assess: dict[str, Assessment]) -> float:
        skills = set(m.skills)
        coming = inbound(obs, h.hospital_id)
        if m.role == "doctor":
            demand = sum(1 for v in coming if v.triage != "GREEN" and v.required_capability in skills)
            free = sum(s.free_slots for s in h.staff if s.role == "doctor" and s.available and skills & set(s.skills))
        else:
            demand = sum(P.STAFF_NEED[v.triage]["nurses"] for v in coming)
            free = h.free_slots("nurse")
        forecast = 0.0
        for inc in obs.incidents:
            a = assess.get(inc.incident_id)
            if a is None:
                continue
            nearest = min(obs.hospitals, key=lambda x: (obs.drive(inc.pos, x.pos), x.hospital_id))
            if nearest.hospital_id == h.hospital_id:
                forecast += (a.red + a.yellow) if m.role == "doctor" else (2 * a.red + a.yellow + a.green)
        return demand + FORECAST_SHARE * forecast - free

    def allocate_pool(self, obs: Observation, bus: MessageBus, assess: dict[str, Assessment]) -> list:
        actions = []
        order = {h.hospital_id: k for k, h in enumerate(sorted(obs.hospitals, key=lambda h: h.hospital_id))}
        for m in sorted(obs.pool, key=lambda m: m.staff_id):
            if m.staff_id in self.used_staff or m.on_duty_from > obs.t + P.POOL_REPORT_DELAY_MIN:
                continue
            scores = [(self._need(obs, h, m, assess), h.free_bays, -order[h.hospital_id], h) for h in obs.hospitals]
            need, _, _, best = max(scores, key=lambda s: s[:3])
            self.used_staff.add(m.staff_id)
            self.tally["pool_assignments"] += 1
            actions.append(AssignStaff(m.staff_id, best.hospital_id))
            bus.send(self.name, hospital_agent(best.hospital_id), MsgType.ALLOCATION_DECISION,
                     ProposedAction(kind="assign_staff", staff_id=m.staff_id, hospital_id=best.hospital_id,
                                    value=round(need, 2)),
                     rationale=f"pool {m.specialty} {m.staff_id} to {best.hospital_id}: expected patients minus free "
                               f"slots is {need:+.1f} there, the highest of any hospital")
        return actions

    def note(self) -> str:
        t = self.tally
        return (f"{t['conflicts']} unit conflicts settled, {t['negotiated_admissions']} admissions negotiated, "
                f"{t['staff_loans']} staff loans, {t['pool_assignments']} pool clinicians placed, {t['diversions']} diversions")
