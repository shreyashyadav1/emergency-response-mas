"""M3: the four agents with their judgment calls made by a language model through the OpenAI Agents SDK.

Each class keeps everything its rules parent does (state, messages, checks) and hands one kind of
decision to its SDK agent in emergency_mas/llm/agents.py:

- Incident Assessment: estimate the victims of a new call from the caller's own words and the base rates;
- Dispatch: the order in which to ask hospitals to take an ambulance's patients;
- Hospital Coordination: whether to make a counter-offer or refuse, when a hospital cannot simply accept;
- Resource Allocation: which incident gets a contested unit, and how to settle a countered admission.

The rules agents compute the options and the numbers the tools return. An answer that is missing,
malformed or names an option that was not offered leaves the rules decision in place and is
counted in the run's LLM stats. The simulator still checks every resulting action.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .. import params as P
from ..llm.agents import AwardsReply, CounterReply, EstimateReply, RoutingReply, SettlementReply
from ..llm.backend import Backend, Situation
from ..policies.common import inbound
from ..schema import Scenario
from ..sim.observe import AmbulanceView, IncidentView, Observation, VictimView
from .allocation import ResourceAllocationAgent, _uid
from .assessment import Assessment, IncidentAssessmentAgent, tasks_for
from .dispatch import DispatchAgent
from .hospital import HospitalCoordinationAgent, Reply, bays_at
from .system import AgentSystem

BLOCKED = -1e6                                  # value that keeps the assignment solver away from a pair


def _pretty(text: str) -> str:
    return text.replace("_", " ")


def _offer(condition: str | None) -> str:
    """A counter-offer condition in words."""
    if not condition:
        return "none"
    if condition.startswith("after:"):
        return f"yes in {condition.split(':')[1]} minutes"
    _, role, skill = condition.split(":")
    return f"yes if a {role} who can handle {_pretty(skill)} is lent to us"


class LLMAssessmentAgent(IncidentAssessmentAgent):
    def __init__(self, llm: Backend) -> None:
        super().__init__()
        self.llm = llm

    def _from_report(self, inc: IncidentView) -> Assessment:
        rules = super()._from_report(inc)
        r = inc.report
        prompt = (f"Minute {inc.report_time}. A 911 call reports a {_pretty(inc.incident_type)} ({inc.incident_id}).\n"
                  f"The caller's words: \"{r.narrative}\"\n"
                  f"The caller's counts: {r.reported_victims} hurt, {r.reported_critical} critical, hazard "
                  f"{r.reported_hazard}, someone trapped: {'yes' if r.reported_trapped else 'no'}. "
                  f"Caller reliability {r.confidence:.2f} (1 is fully reliable).\nEstimate the victims.")
        data = {"base_rates": {"incident_type": inc.incident_type,
                               **{k: round(v, 3) for k, v in self.rates[inc.incident_type].items()}},
                "statistical_estimate": {"red": round(rules.red, 2), "yellow": round(rules.yellow, 2),
                                         "green": round(rules.green, 2), "hazard": rules.hazard, "trapped": rules.trapped}}
        reply = self.llm.ask("assessment", "assessment", prompt, Situation(data), EstimateReply)
        if reply is None:
            return rules
        total = reply.red + reply.yellow + reply.green
        if total <= 0 or total > 4 * max(r.reported_victims, 1) + 10:
            self.llm.reject()
            return rules
        self.llm.compare("assessment", round(reply.red) != round(rules.red) or round(total) != round(rules.victims)
                         or reply.hazard != rules.hazard or reply.trapped != rules.trapped)
        return Assessment(inc.incident_id, "report", reply.red, reply.yellow, reply.green, reply.trapped, reply.hazard,
                          tasks_for(reply.hazard, reply.trapped), r.confidence, note=f"LLM: {reply.reason}")


class LLMDispatchAgent(DispatchAgent):
    def __init__(self, llm: Backend) -> None:
        super().__init__()
        self.llm = llm

    def hospital_order(self, obs: Observation, amb: AmbulanceView) -> list:
        options = super().hospital_order(obs, amb)
        if len(options) < 2:
            return options
        left = max(0, P.LOAD_MIN - (obs.t - amb.since)) if amb.status == "loading" else 0
        rows = []
        for h in options:
            eta = left + obs.drive(amb.pos, h.pos)
            coming = len(inbound(obs, h.hospital_id))
            rows.append({"hospital": h.hospital_id, "name": h.name, "minutes_to_arrive": eta, "bays_free_now": h.free_bays,
                         "bays_expected_free_at_arrival": bays_at(h, obs.t + eta) - coming,
                         "patients_already_heading_there": coming, "ambulances_waiting_at_door": len(h.door),
                         "occupancy": round(h.occupancy, 2), "bays_held_for_red": h.reserve_bays_for_red})
        onboard = ", ".join(f"{v.triage} needing {_pretty(v.required_capability)}" for v in amb.onboard)
        prompt = (f"Minute {obs.t}. Ambulance {amb.ambulance_id} ({amb.unit_type}) is taking patients from "
                  f"{amb.onboard[0].incident_id}: {onboard}.\nHospitals that can treat everyone on board: "
                  f"{', '.join(h.hospital_id for h in options)}. Choose the order in which to ask them.")
        reply = self.llm.ask("routing", "dispatch", prompt, Situation({"hospital_options": rows}), RoutingReply)
        if reply is None:
            return options
        by_id = {h.hospital_id: h for h in options}
        order = [hid for hid in dict.fromkeys(reply.ask_order) if hid in by_id]
        if not order:
            self.llm.reject()
            return options
        self.llm.compare("routing", order[0] != options[0].hospital_id)
        self.route_reasons[amb.ambulance_id] = reply.reason
        return [by_id[hid] for hid in order] + [h for h in options if h.hospital_id not in order]


class LLMHospitalAgent(HospitalCoordinationAgent):
    def __init__(self, hospital_id: str, negotiation: bool, llm: Backend) -> None:
        super().__init__(hospital_id, negotiation)
        self.llm = llm

    def reply(self, obs: Observation, victims: list[VictimView], eta: int) -> Reply:
        rules = self.evaluate(obs, victims, eta)
        if rules.kind != "counter":
            return rules
        hv = obs.hospital(self.hid)
        ids = {v.victim_id for v in victims}
        coming = [v for v in inbound(obs, self.hid) if v.victim_id not in ids]
        data = {"capacity_at_arrival": {
            "hospital": self.hid, "minutes_until_arrival": eta,
            "bays_expected_free_at_arrival": bays_at(hv, obs.t + eta) - len(coming),
            "bays_held_for_red": hv.reserve_bays_for_red, "patients_already_on_their_way": len(coming),
            "red_patients_already_on_their_way": sum(v.triage == "RED" for v in coming),
            "on_this_ambulance": [f"{v.triage} needing {_pretty(v.required_capability)}" for v in victims],
            "why_not_a_plain_yes": rules.reason, "counter_offer_on_the_table": _offer(rules.condition),
            "occupancy": round(hv.occupancy, 2), "ambulances_waiting_at_door": len(hv.door)}}
        prompt = (f"Minute {obs.t}. Dispatch asks {self.hid} to admit {len(victims)} patient(s) "
                  f"({', '.join(v.triage for v in victims)}) arriving in {eta} min. You cannot simply accept: "
                  f"{rules.reason}. Decide whether to make the counter-offer or refuse.")
        reply = self.llm.ask("counter", "hospital", prompt, Situation(data), CounterReply)
        if reply is None:
            return rules
        self.llm.compare("counter", reply.decision == "reject")
        if reply.decision == "reject":
            return Reply("reject", f"declines to counter: {reply.reason}")
        return Reply("counter", f"{rules.reason}; {reply.reason}", rules.condition, rules.wait)


class LLMAllocationAgent(ResourceAllocationAgent):
    def __init__(self, llm: Backend) -> None:
        super().__init__()
        self.llm = llm
        self.t = 0

    def assign(self, label: str, slots: list, units: list, v: np.ndarray, bidders: dict[int, list[int]]) -> list[tuple[int, int]]:
        best = super().assign(label, slots, units, v, bidders)
        contested = {j: g for j, g in bidders.items() if len({slots[i].incident_id for i in g}) >= 2}
        if not contested:
            return best
        pick = {j: i for i, j in best}
        rows = []
        for j, group in sorted(contested.items()):
            bids = []
            for i in group:
                others = [(float(v[i, k]), _uid(units[k])) for k in range(v.shape[1]) if k != j and v[i, k] > 0]
                alt = max(others, default=None)
                bids.append({"incident": slots[i].incident_id, "needs": _pretty(slots[i].kind),
                             "value": round(float(v[i, j]), 4), "priority": round(slots[i].priority, 1),
                             "best_other_unit": alt[1] if alt else None, "its_value": round(alt[0], 4) if alt else 0.0})
            unit = units[j]
            rows.append({"unit": _uid(unit), "unit_type": getattr(unit, "unit_type", None) or _pretty(unit.team_type),
                         "bids": bids, "optimizer_pick": slots[pick[j]].incident_id if j in pick else None})
        names = ", ".join(r["unit"] for r in rows)
        prompt = (f"Minute {self.t}. Contested units: several incidents want the same {label} this minute ({names}). "
                  "Award each contested unit to one incident.")
        reply = self.llm.ask("conflict", "allocation", prompt, Situation({"contested_units": rows}), AwardsReply)
        if reply is None:
            return best
        fixed: dict[int, int] = {}
        for award in reply.awards:
            j = next((j for j in contested if _uid(units[j]) == award.unit), None)
            i = next((i for i in contested.get(j, []) if slots[i].incident_id == award.incident), None)
            if j is not None and i is not None and j not in fixed:
                fixed[j] = i
        if not fixed:
            self.llm.reject()
            return best
        self.llm.compare("conflict", any(pick.get(j) is None or slots[pick[j]].incident_id != slots[i].incident_id
                                         for j, i in fixed.items()))
        v2 = v.copy()
        for j, i in fixed.items():
            v2[:, j] = BLOCKED
            v2[i, :] = BLOCKED
            v2[i, j] = v[i, j]
        rows_, cols_ = linear_sum_assignment(-v2)
        pairs = [(int(i), int(j)) for i, j in zip(rows_, cols_) if v[i, j] > 0 and v2[i, j] > BLOCKED / 2]
        for j in fixed:
            self.reasons[_uid(units[j])] = f"chosen by the LLM referee ({reply.reason})"
        return pairs

    def pick_settlement(self, obs: Observation, amb: AmbulanceView, options: list) -> tuple:
        if len(options) < 2:
            return super().pick_settlement(obs, amb, options)
        deals = {0: "accepts on arrival", 1: "has room later", 2: "accepts if a clinician is moved there"}
        rows = [{"option": k, "hospital": o[2], "deal": deals[o[1]], "care_starts_in_min": int(o[0]), "detail": o[4]}
                for k, o in enumerate(options)]
        prompt = (f"Minute {obs.t}. Admission settlement: ambulance {amb.ambulance_id} carries "
                  f"{', '.join(v.triage for v in amb.onboard)}, and hospitals answered with counter-offers. "
                  "Pick how to settle it.")
        reply = self.llm.ask("settlement", "allocation", prompt, Situation({"settlement_options": rows}), SettlementReply)
        if reply is None:
            return super().pick_settlement(obs, amb, options)
        if not 0 <= reply.option < len(options):
            self.llm.reject()
            return super().pick_settlement(obs, amb, options)
        self.llm.compare("settlement", options[reply.option] is not super().pick_settlement(obs, amb, options))
        care, rank, hid, extra, why, request = options[reply.option]
        return care, rank, hid, extra, f"{why}; LLM referee: {reply.reason}", request


class LLMAgentSystem(AgentSystem):
    """M3: M2's agents and negotiation, with the judgment calls made by SDK agents on a language model."""

    def __init__(self, llm: Backend) -> None:
        super().__init__(negotiation=True)
        self.llm = llm
        self.name = "M3_llm_agents"

    def reset(self, scenario: Scenario) -> None:
        super().reset(scenario)
        self.assessor = LLMAssessmentAgent(self.llm)
        self.dispatcher = LLMDispatchAgent(self.llm)
        self.hospitals = {h.hospital_id: LLMHospitalAgent(h.hospital_id, True, self.llm) for h in scenario.hospitals}
        self.allocator = LLMAllocationAgent(self.llm)

    def decide(self, obs: Observation):
        self.allocator.t = obs.t
        return super().decide(obs)
