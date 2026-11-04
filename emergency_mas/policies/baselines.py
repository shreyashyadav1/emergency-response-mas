"""Two baselines without agents, so the agent system has something honest to beat.

B0 is what one dispatcher with no coordination would do. B1 adds triage on the
caller's report and the hospital status board, but still no negotiation.
"""
from __future__ import annotations

from ..schema import Scenario
from ..sim.actions import AssignStaff, DispatchAmbulance, DispatchTeam, SetDestination
from ..sim.observe import IncidentView, Observation
from ..sim.policy import Decision
from .common import (can_carry, capable, inbound, nearest_unit, needs_destination, onboard_capabilities,
                     pool_round_robin, transport_need, uncovered_tasks)


class _Baseline:
    name = "baseline"

    def reset(self, scenario: Scenario) -> None:
        self.pool_plan = pool_round_robin(scenario)

    def _pool(self, obs: Observation) -> list:
        return [AssignStaff(m.staff_id, self.pool_plan[m.staff_id]) for m in obs.pool if m.staff_id in self.pool_plan]

    def _teams(self, obs: Observation, order: list[IncidentView]) -> list:
        actions, free = [], [r for r in obs.teams if r.status in ("idle", "returning")]
        for inc in order:
            for task in uncovered_tasks(obs, inc):
                options = [r for r in free if task in r.capabilities]
                if options:
                    team = nearest_unit(obs, options, inc.pos)
                    actions.append(DispatchTeam(team.team_id, inc.incident_id))
                    free.remove(team)
        return actions


class NearestFirst(_Baseline):
    """B0: earliest call first, nearest free unit, nearest hospital that can treat the patient."""

    name = "B0_nearest_first"

    def decide(self, obs: Observation) -> Decision:
        order = sorted(obs.incidents, key=lambda i: (i.report_time, i.incident_id))
        actions = self._pool(obs)
        free = [a for a in obs.ambulances if a.dispatchable and a.status in ("idle", "returning")]
        for inc in order:
            need = transport_need(obs, inc)
            while need > 0:
                usable = [a for a in free if can_carry(inc, a)]
                if not usable:
                    break
                amb = nearest_unit(obs, usable, inc.pos)
                actions.append(DispatchAmbulance(amb.ambulance_id, inc.incident_id))
                free.remove(amb)
                need -= amb.capacity
        actions += self._teams(obs, order)
        for amb in obs.ambulances:
            if needs_destination(amb):
                options = capable(obs, onboard_capabilities(amb))
                best = min(options, key=lambda h: (obs.drive(amb.pos, h.pos), h.hospital_id))
                actions.append(SetDestination(amb.ambulance_id, best.hospital_id))
        return Decision(actions)


def report_priority(inc: IncidentView) -> float:
    """B1's triage: the caller's critical count matters most, the true colors once a unit is there."""
    if inc.revealed:
        w = inc.waiting()
        return 10 * sum(v.triage == "RED" for v in w) + 4 * sum(v.triage == "YELLOW" for v in w) + sum(
            v.triage == "GREEN" for v in w)
    return 10 * inc.report.reported_critical + inc.report.reported_victims


class PriorityGreedy(_Baseline):
    """B1: highest reported priority first, ALS where the report says critical, and the nearest
    hospital that shows a free bay and staff on the status board."""

    name = "B1_priority_greedy"

    def decide(self, obs: Observation) -> Decision:
        order = sorted(obs.incidents, key=lambda i: (-report_priority(i), i.report_time, i.incident_id))
        actions = self._pool(obs)
        free = [a for a in obs.ambulances if a.dispatchable and a.status in ("idle", "returning")]
        for inc in order:
            need = transport_need(obs, inc)
            if inc.revealed:
                red = sum(v.triage == "RED" for v in inc.waiting())
            else:
                red = inc.report.reported_critical
            while need > 0:
                usable = [a for a in free if can_carry(inc, a)]
                if not usable:
                    break
                als = [a for a in usable if a.unit_type == "ALS"]
                amb = nearest_unit(obs, als if red > 0 and als else usable, inc.pos)
                actions.append(DispatchAmbulance(amb.ambulance_id, inc.incident_id))
                free.remove(amb)
                need -= amb.capacity
                red -= 1 if amb.unit_type == "ALS" else 0
        actions += self._teams(obs, order)
        for amb in obs.ambulances:
            if not needs_destination(amb):
                continue
            options = sorted(capable(obs, onboard_capabilities(amb)), key=lambda h: (obs.drive(amb.pos, h.pos), h.hospital_id))
            doctor_skill = {v.required_capability for v in amb.onboard if v.triage != "GREEN"}

            def has_room(h) -> bool:
                room = h.free_bays - len(inbound(obs, h.hospital_id))
                return room >= len(amb.onboard) and all(h.free_slots("doctor", s) > 0 for s in doctor_skill)

            best = next((h for h in options if has_room(h)), options[0])
            actions.append(SetDestination(amb.ambulance_id, best.hospital_id))
        return Decision(actions)
