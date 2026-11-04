"""Incident Assessment Agent: turns a noisy caller report into an estimate other agents can plan with.

Callers under-count, so the agent corrects the report with base rates learned from
600 synthetic past incidents. Once a unit is on scene the estimate is replaced by
what the crew actually sees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .. import params as P
from ..policies.common import HAZARD_TASK
from ..sim.observe import IncidentView, Observation
from .messages import ASSESSMENT_AGENT, DISPATCH_AGENT, MessageBus, MsgType, ProposedAction

TYPICAL_HAZARD = {"building_fire": "fire", "structural_collapse": "structural", "hazmat_release": "hazmat"}
# E[60 / window] for a window drawn uniformly from the triage color's range
URGENCY_PER_MINUTE = {k: 60.0 * math.log(hi / lo) / (hi - lo) for k, (lo, hi) in P.GOLDEN_WINDOW_MIN.items()}


def tasks_for(hazard: str, trapped: bool) -> list[str]:
    """Scene tasks implied by a hazard and trapped victims, in the order teams work on them."""
    return ([HAZARD_TASK[hazard]] if hazard in HAZARD_TASK else []) + (["extrication"] if trapped else []) \
        + ["triage_stabilization"]


@lru_cache(maxsize=1)
def base_rates() -> dict[str, dict[str, float]]:
    """Learned from 600 synthetic past incidents, per incident type: how callers under-count, how true RED
    counts relate to what callers said (a two-term least-squares fit), the triage mix, how often someone is trapped."""
    from ..datagen.history import generate_history

    h = generate_history(600)
    rates = {}
    for kind, g in h.groupby("incident_type"):
        victims = max(1, g["true_victims"].sum())
        other = (g["reported_victims"] - g["reported_critical"]).clip(lower=0)
        x = np.column_stack([g["reported_critical"], other])
        coef, *_ = np.linalg.lstsq(x, g["true_red"].to_numpy(dtype=float), rcond=None)
        red_share, yellow_share = g["true_red"].sum() / victims, g["true_yellow"].sum() / victims
        rates[kind] = {
            "victims_per_reported": g["true_victims"].sum() / max(1, g["reported_victims"].sum()),
            "red_per_critical": max(0.0, float(coef[0])),
            "red_per_other": max(0.0, float(coef[1])),
            "yellow_of_rest": yellow_share / max(1e-9, 1 - red_share),
            "trapped_rate": float((g["true_trapped"] > 0).mean()),
        }
    return rates


@dataclass
class Assessment:
    incident_id: str
    source: str                       # "report" until a unit is on scene, then "scene"
    red: float
    yellow: float
    green: float
    trapped: bool
    hazard: str
    tasks: list[str] = field(default_factory=list)
    confidence: float = 0.5
    note: str = ""                    # the LLM agent's reasoning, when it made this estimate

    @property
    def victims(self) -> float:
        return self.red + self.yellow + self.green

    @property
    def priority(self) -> float:
        """Estimated urgency on the same scale as the oracle: triage weight x 60 / golden window, using
        the expected 60/window for each color because the agent cannot see individual windows."""
        return sum(P.TRIAGE_WEIGHT[k] * URGENCY_PER_MINUTE[k] * n
                   for k, n in (("RED", self.red), ("YELLOW", self.yellow), ("GREEN", self.green)))


class IncidentAssessmentAgent:
    name = ASSESSMENT_AGENT

    def __init__(self) -> None:
        self.rates = base_rates()
        self.current: dict[str, Assessment] = {}
        self.first: dict[str, Assessment] = {}     # first estimate, kept to score the agent afterwards

    def _from_report(self, inc: IncidentView) -> Assessment:
        """Expected counts (not rounded) given what the caller said and what past incidents of this type looked like."""
        r, rate = inc.report, self.rates[inc.incident_type]
        victims = max(float(r.reported_victims), r.reported_victims * rate["victims_per_reported"])
        other = max(0, r.reported_victims - r.reported_critical)
        red = min(victims, rate["red_per_critical"] * r.reported_critical + rate["red_per_other"] * other)
        yellow = (victims - red) * rate["yellow_of_rest"]
        if r.reported_hazard in ("fire", "structural", "hazmat"):
            hazard = r.reported_hazard
        elif r.reported_hazard == "unknown":
            hazard = TYPICAL_HAZARD.get(inc.incident_type, "none")
        else:
            hazard = "none"
        trapped = r.reported_trapped or rate["trapped_rate"] > 0.6
        return Assessment(inc.incident_id, "report", red, yellow, victims - red - yellow, trapped, hazard,
                          tasks_for(hazard, trapped), r.confidence)

    @staticmethod
    def _from_scene(inc: IncidentView, previous: Assessment | None) -> Assessment:
        waiting = inc.waiting()
        count = {k: sum(v.triage == k for v in waiting) for k in P.TRIAGE}
        hazard = previous.hazard if previous else "none"
        open_tasks = [n.capability for n in inc.open_needs()]
        for task_hazard, task in HAZARD_TASK.items():
            if task in [n.capability for n in inc.needs]:
                hazard = task_hazard
        return Assessment(inc.incident_id, "scene", count["RED"], count["YELLOW"], count["GREEN"],
                          any(v.trapped for v in waiting), hazard, open_tasks, 1.0)

    def assess(self, obs: Observation, bus: MessageBus) -> dict[str, Assessment]:
        for inc in obs.incidents:
            old = self.current.get(inc.incident_id)
            new = self._from_scene(inc, old) if inc.revealed else (old or self._from_report(inc))
            changed = old is None or old.source != new.source or abs(old.priority - new.priority) >= 1
            self.current[inc.incident_id] = new
            self.first.setdefault(inc.incident_id, new)
            if changed:
                bus.send(self.name, DISPATCH_AGENT, MsgType.ASSESSMENT, ProposedAction(
                    kind="assess", estimate={"red": new.red, "yellow": new.yellow, "green": new.green,
                                             "priority": round(new.priority, 1), "confidence": round(new.confidence, 2)}),
                    incident_id=inc.incident_id,
                    rationale=(f"scene count: {new.red:.0f} RED, {new.yellow:.0f} YELLOW, {new.green:.0f} GREEN"
                               if new.source == "scene" else
                               f"caller said {inc.report.reported_victims} hurt, {inc.report.reported_critical} critical; "
                               f"base rates say about {new.victims:.0f} victims, {new.red:.0f} RED"
                               + (f"; expect {new.hazard} hazard" if new.hazard != "none" else "")
                               + (f"; {new.note}" if new.note else "")))
        return self.current

    def note(self) -> str:
        if not self.current:
            return "waiting for calls"
        top = max(self.current.values(), key=lambda a: a.priority)
        scene = sum(a.source == "scene" for a in self.current.values())
        return (f"{len(self.current)} incidents assessed ({scene} confirmed on scene); "
                f"top priority {top.incident_id}: about {top.red:.0f} RED of {top.victims:.0f}")
