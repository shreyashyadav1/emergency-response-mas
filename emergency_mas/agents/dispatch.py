"""Dispatch Agent: proposes which ambulances and teams go where, and finds a hospital for every load.

Every incident bids for units. A unit's value for an incident is the triage weight
it would carry (or unblock) per minute of the unit's time, so a nearby ALS unit at
a scene with a RED patient is worth more than the same unit three incidents away.
Two domain rules come with it: never park an ambulance at a scene that a hazard
still keeps closed, and keep fire engines for fires and extrications rather than
routine triage. Without negotiation the highest-priority incident takes its
favorite unit first; with negotiation the Resource Allocation Agent settles it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import params as P
from ..domain import expected_survival
from ..policies.common import HAZARD_TASK, ambulances_at, capable, needs_destination, onboard_capabilities, uncovered_tasks
from ..sim.actions import DispatchAmbulance, DispatchTeam, SetDestination
from ..sim.observe import AmbulanceView, IncidentView, Observation, TeamView
from .assessment import Assessment
from .hospital import HospitalCoordinationAgent, Reply
from .messages import DISPATCH_AGENT, MessageBus, MsgType, ProposedAction, hospital_agent

HAZARD_TASKS = set(HAZARD_TASK.values())
EARLY_SLACK_MIN = 5          # an ambulance may reach a closed scene at most 5 min before it opens
MAX_NEW_UNITS_PER_INCIDENT = 2
WAIT_IF_NOT_SENT_MIN = 30    # how much longer a patient is assumed to wait if this unit goes elsewhere
TRANSPORT_FLOOR = 0.002      # a small value for any ride, so idle units still pick up minor patients


@dataclass
class Slot:
    """One more unit an incident would like this minute."""

    incident_id: str
    pos: tuple[float, float]
    priority: float
    kind: str                                   # "ambulance" or a scene task
    remaining: dict[str, float] = field(default_factory=dict)
    access: int = 0                             # ambulances: minutes until victims can be handed over
    to_hospital: int = 0                        # ambulances: minutes from the scene to the nearest hospital
    weight: float = 0.0                         # teams: how much the task matters
    elapsed: int = 0                            # minutes since the call
    stabilized: bool = False                    # a triage team has finished on scene


def load(remaining: dict[str, float], unit_type: str, capacity: int) -> dict[str, float]:
    """What one ambulance would carry: one RED at most and only on ALS, then YELLOW, then GREEN."""
    carried = {k: 0.0 for k in P.TRIAGE}
    seats = float(capacity)
    if unit_type == "ALS" and remaining["RED"] > 0:
        carried["RED"] = min(1.0, remaining["RED"])
        seats -= carried["RED"]
    for k in ("YELLOW", "GREEN"):
        carried[k] = min(seats, remaining[k])
        seats -= carried[k]
    return carried


def after(remaining: dict[str, float], carried: dict[str, float]) -> dict[str, float]:
    return {k: max(0.0, remaining[k] - carried[k]) for k in P.TRIAGE}


def weight_of(counts: dict[str, float]) -> float:
    return sum(P.TRIAGE_WEIGHT[k] * counts[k] for k in P.TRIAGE)


class DispatchAgent:
    name = DISPATCH_AGENT

    def __init__(self) -> None:
        self.last: list[str] = []
        self.route_reasons: dict[str, str] = {}    # ambulance -> why its hospitals are asked in this order (LLM agent)

    # ------------------------------------------------------------------ demand
    @staticmethod
    def remaining(obs: Observation, inc: IncidentView, a: Assessment) -> dict[str, float]:
        if inc.revealed:
            rem = {k: float(sum(v.triage == k for v in inc.waiting())) for k in P.TRIAGE}
        else:
            rem = {"RED": a.red, "YELLOW": a.yellow, "GREEN": a.green}
        for amb in ambulances_at(obs, inc):
            if amb.status in ("to_scene", "at_scene"):
                rem = after(rem, load(rem, amb.unit_type, amb.capacity))
        return rem

    @staticmethod
    def access(obs: Observation, inc: IncidentView, a: Assessment) -> int | None:
        """Minutes until victims can be handed to an ambulance; None while nobody is coming to open the scene."""
        heading = [r for r in obs.teams if r.team_id in inc.teams]
        hazard = next((n.capability for n in inc.needs if n.capability in HAZARD_TASKS), None) if inc.revealed \
            else HAZARD_TASK.get(a.hazard)
        wait = 0
        if hazard:
            need = next((n for n in inc.needs if n.capability == hazard), None)
            if need is not None and need.done:
                wait = 0
            elif need is not None and need.started:
                wait = max(0, need.started_at + P.HAZARD_ACCESS_MIN - obs.t)
            else:
                crews = [r for r in heading if hazard in r.capabilities]
                if not crews:
                    return None
                wait = min(obs.drive(r.pos, inc.pos, r.speed_kmh) for r in crews) + P.HAZARD_ACCESS_MIN
        waiting = inc.waiting()
        if inc.revealed and waiting and all(v.trapped for v in waiting):
            need = next((n for n in inc.needs if n.capability == "extrication"), None)
            crews = [r for r in heading if "extrication" in r.capabilities]
            if need is None or need.done:
                pass
            elif need.started:
                wait = max(wait, int(need.work_left))
            elif crews:
                wait = max(wait, min(obs.drive(r.pos, inc.pos, r.speed_kmh) for r in crews) + int(need.work_left))
            else:
                return None
        return wait

    def ambulance_slots(self, obs: Observation, assess: dict[str, Assessment]) -> list[Slot]:
        slots = []
        nearest_hospital = {i.incident_id: min(obs.drive(i.pos, h.pos) for h in obs.hospitals) for i in obs.incidents}
        for inc in obs.incidents:
            a = assess[inc.incident_id]
            rem = self.remaining(obs, inc, a)
            if sum(rem.values()) < 0.5:
                continue
            access = self.access(obs, inc, a)
            if access is None:
                continue
            for _ in range(MAX_NEW_UNITS_PER_INCIDENT):
                if sum(rem.values()) < 0.5:
                    break
                stabilized = any(n.capability == "triage_stabilization" and n.done for n in inc.needs)
                slots.append(Slot(inc.incident_id, inc.pos, a.priority, "ambulance", dict(rem), access,
                                  nearest_hospital[inc.incident_id], elapsed=obs.t - inc.report_time,
                                  stabilized=stabilized))
                rem = after(rem, load(rem, "ALS", 2))
        return slots

    @staticmethod
    def ambulance_value(obs: Observation, slot: Slot, amb: AmbulanceView) -> float:
        """Expected survivors this ride buys, per minute of the ambulance's time."""
        carried = load(slot.remaining, amb.unit_type, amb.capacity)
        if sum(carried.values()) <= 0:
            return 0.0
        eta = obs.eta(amb, slot.pos)
        if eta + EARLY_SLACK_MIN < slot.access:
            return 0.0
        to_care = slot.elapsed + max(eta, slot.access) + P.LOAD_MIN + slot.to_hospital
        gain = 0.0
        for k, n in carried.items():
            if n <= 0:
                continue
            stabilized = slot.stabilized or (k == "RED" and amb.unit_type == "ALS")
            gain += n * (expected_survival(k, to_care, stabilized)
                         - expected_survival(k, to_care + WAIT_IF_NOT_SENT_MIN, stabilized) + TRANSPORT_FLOOR)
        return gain / (to_care - slot.elapsed + P.HANDOFF_MIN)

    def team_slots(self, obs: Observation, assess: dict[str, Assessment]) -> list[Slot]:
        slots = []
        for inc in obs.incidents:
            a = assess[inc.incident_id]
            tasks = uncovered_tasks(obs, inc, None if inc.revealed else a.tasks)
            for task in tasks:
                if task in HAZARD_TASKS:
                    weight = a.priority + 5
                elif task == "extrication":
                    weight = (sum(P.TRIAGE_WEIGHT[v.triage] for v in inc.waiting() if v.trapped) if inc.revealed
                              else 0.5 * (P.TRIAGE_WEIGHT["RED"] * a.red + P.TRIAGE_WEIGHT["YELLOW"] * a.yellow))
                else:
                    weight = 0.6 * (P.TRIAGE_WEIGHT["RED"] * a.red + P.TRIAGE_WEIGHT["YELLOW"] * a.yellow) \
                        + (0 if inc.revealed else 3)
                if weight > 0:
                    slots.append(Slot(inc.incident_id, inc.pos, a.priority, task, weight=weight))
        return slots

    @staticmethod
    def team_value(obs: Observation, slot: Slot, team: TeamView, spare_fire_engines: bool) -> float:
        if slot.kind not in team.capabilities:
            return 0.0
        if slot.kind == "triage_stabilization" and team.team_type != "paramedic_qrv" and not spare_fire_engines:
            return 0.0
        return slot.weight / (obs.eta(team, slot.pos) + 5)

    # ------------------------------------------------------------------ dispatching
    def dispatch(self, obs: Observation, assess: dict[str, Assessment], bus: MessageBus, allocator) -> list:
        actions: list = []
        self.last = []
        units = [a for a in obs.ambulances if a.dispatchable and a.status in ("idle", "returning")]
        slots = self.ambulance_slots(obs, assess)
        if units and slots:
            values = [[self.ambulance_value(obs, s, u) for u in units] for s in slots]
            for i, j in self._resolve("ambulance", slots, units, values, obs, bus, allocator):
                actions.append(DispatchAmbulance(units[j].ambulance_id, slots[i].incident_id))
                self.last.append(f"{units[j].ambulance_id}->{slots[i].incident_id}")
        teams = [r for r in obs.teams if r.dispatchable and r.status in ("idle", "returning")]
        tslots = self.team_slots(obs, assess)
        if teams and tslots:
            qrv_free = any(r.team_type == "paramedic_qrv" for r in teams)
            fire_work = any(s.kind in ("fire_suppression", "extrication") for s in tslots)
            spare = not qrv_free and not fire_work
            values = [[self.team_value(obs, s, r, spare) for r in teams] for s in tslots]
            for i, j in self._resolve("team", tslots, teams, values, obs, bus, allocator):
                actions.append(DispatchTeam(teams[j].team_id, tslots[i].incident_id))
                self.last.append(f"{teams[j].team_id}->{tslots[i].incident_id} ({tslots[i].kind})")
        return actions

    def _resolve(self, label: str, slots: list[Slot], units: list, values: list[list[float]], obs: Observation,
                 bus: MessageBus, allocator) -> list[tuple[int, int]]:
        if allocator is not None:
            return allocator.award(label, slots, units, values, bus)
        taken: set[int] = set()
        pairs = []
        for i in sorted(range(len(slots)), key=lambda i: (-slots[i].priority, slots[i].incident_id)):
            best = max(((values[i][j], j) for j in range(len(units)) if j not in taken and values[i][j] > 0),
                       default=None)
            if best is not None:
                taken.add(best[1])
                pairs.append((i, best[1]))
                unit = units[best[1]]
                uid = getattr(unit, "ambulance_id", None) or unit.team_id
                bus.send(self.name, "broadcast", MsgType.STATUS_UPDATE,
                         ProposedAction(kind=f"dispatch_{label}", unit_id=uid, value=round(best[0], 3),
                                        eta_min=obs.eta(unit, slots[i].pos)),
                         incident_id=slots[i].incident_id,
                         rationale=f"highest-priority incident takes its best free {label}"
                                   + ("" if label == "ambulance" else f" for {slots[i].kind}"))
        return pairs

    # ------------------------------------------------------------------ hospitals
    def hospital_order(self, obs: Observation, amb: AmbulanceView) -> list:
        """The hospitals to ask for this load, in order: the three nearest that can treat everyone on board."""
        return sorted(capable(obs, onboard_capabilities(amb)), key=lambda h: (obs.drive(amb.pos, h.pos), h.hospital_id))[:3]

    def route(self, obs: Observation, bus: MessageBus, hospitals: dict[str, HospitalCoordinationAgent],
              allocator) -> list:
        actions: list = []
        for amb in obs.ambulances:
            if not needs_destination(amb):
                continue
            incident_id = amb.onboard[0].incident_id
            left = max(0, P.LOAD_MIN - (obs.t - amb.since)) if amb.status == "loading" else 0
            options = self.hospital_order(obs, amb)
            colors = ", ".join(v.triage for v in amb.onboard)
            why = self.route_reasons.pop(amb.ambulance_id, "")
            replies: list[tuple[str, int, Reply, object]] = []
            for k, h in enumerate(options):
                eta = left + obs.drive(amb.pos, h.pos)
                request = bus.send(self.name, hospital_agent(h.hospital_id), MsgType.ADMISSION_REQUEST,
                                   ProposedAction(kind="admit", unit_id=amb.ambulance_id, hospital_id=h.hospital_id,
                                                  victim_ids=[v.victim_id for v in amb.onboard], eta_min=eta),
                                   incident_id=incident_id, rationale=f"{amb.ambulance_id} bringing {colors}, ETA {eta} min"
                                   + (f"; asked first: {why}" if why and k == 0 else ""))
                reply = hospitals[h.hospital_id].reply(obs, list(amb.onboard), eta)
                hospitals[h.hospital_id].answer(bus, request, reply, 1)
                replies.append((h.hospital_id, eta, reply, request))
                if reply.kind == "accept":
                    break
            if allocator is not None:
                choice, extra = allocator.settle_admission(obs, amb, replies, bus, hospitals)
                actions += extra
            else:
                choice = next((hid for hid, _, r, _ in replies if r.kind == "accept"), options[0].hospital_id)
            actions.append(SetDestination(amb.ambulance_id, choice))
        return actions

    def note(self) -> str:
        return "dispatched " + ", ".join(self.last) if self.last else "no new dispatches this minute"
