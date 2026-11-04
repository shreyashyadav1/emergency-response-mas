"""The four agents working together, as one policy the simulator can run.

negotiation=True is the full system (M2). negotiation=False is the M1 ablation:
the same agents and messages, but hospitals only accept or reject, contested units
go to whoever has the higher priority, there are no staff loans or diversions, and
the regional pool is split evenly as in the baselines.
"""
from __future__ import annotations

from ..policies.common import pool_round_robin
from ..schema import Scenario
from ..sim.actions import AssignStaff
from ..sim.observe import Observation
from ..sim.policy import Decision
from .allocation import ResourceAllocationAgent
from .assessment import IncidentAssessmentAgent
from .dispatch import DispatchAgent
from .hospital import HospitalCoordinationAgent
from .messages import MessageBus


class AgentSystem:
    def __init__(self, negotiation: bool = True):
        self.negotiation = negotiation
        self.name = "M2_agents_negotiation" if negotiation else "M1_agents_no_negotiation"

    def reset(self, scenario: Scenario) -> None:
        self.bus = MessageBus()
        self.assessor = IncidentAssessmentAgent()
        self.dispatcher = DispatchAgent()
        self.hospitals = {h.hospital_id: HospitalCoordinationAgent(h.hospital_id, self.negotiation)
                          for h in scenario.hospitals}
        self.allocator = ResourceAllocationAgent() if self.negotiation else None
        self.pool_plan = pool_round_robin(scenario)

    def decide(self, obs: Observation) -> Decision:
        self.bus.start(obs.t)
        if self.allocator is not None:
            self.allocator.start()
        assess = self.assessor.assess(obs, self.bus)
        actions: list = []
        if self.allocator is not None:
            actions += self.allocator.allocate_pool(obs, self.bus, assess)
        else:
            actions += [AssignStaff(m.staff_id, self.pool_plan[m.staff_id]) for m in obs.pool]
        actions += self.dispatcher.dispatch(obs, assess, self.bus, self.allocator)
        actions += self.dispatcher.route(obs, self.bus, self.hospitals, self.allocator)
        if self.allocator is not None:
            actions += self.allocator.rebalance(obs, self.bus, self.hospitals)
        for hca in self.hospitals.values():
            actions += hca.surge(obs, self.bus)
        notes = {self.assessor.name: self.assessor.note(), self.dispatcher.name: self.dispatcher.note()}
        notes.update({hca.name: hca.note(obs) for hca in self.hospitals.values()})
        if self.allocator is not None:
            notes[self.allocator.name] = self.allocator.note()
        return Decision(actions, self.bus.step_messages(), notes)
