"""The four agents as OpenAI Agents SDK agents.

Each has its own instructions (its job, objective and decision policy), its own function tools,
and a reply schema. For one decision the agent gets a short prompt, calls its tools for the
numbers (taken from what the agent can observe and from the rules agents' scoring), and answers
with one JSON object. The answer is validated against the schema and against the options it was
offered; the simulator still checks every action that follows from it.

Replies are plain JSON text rather than the SDK's structured-output mode, because some
OpenAI-compatible servers, Gemini's among them, do not accept a response format and tools in the
same request. Tools are declared without strict schemas for the same reason.
"""
from __future__ import annotations

import json
from typing import Literal

from agents import Agent, ModelSettings, RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from .backend import Situation


# ---------------------------------------------------------------------- replies
class EstimateReply(BaseModel):
    red: float = Field(ge=0, description="Expected RED (critical) victims.")
    yellow: float = Field(ge=0, description="Expected YELLOW (serious) victims.")
    green: float = Field(ge=0, description="Expected GREEN (minor) victims.")
    hazard: Literal["none", "fire", "structural", "hazmat"]
    trapped: bool
    reason: str = Field(max_length=400)


class RoutingReply(BaseModel):
    ask_order: list[str] = Field(min_length=1, description="Hospital ids, the one to ask first first.")
    reason: str = Field(max_length=400)


class CounterReply(BaseModel):
    decision: Literal["counter", "reject"]
    reason: str = Field(max_length=400)


class Award(BaseModel):
    unit: str
    incident: str


class AwardsReply(BaseModel):
    awards: list[Award]
    reason: str = Field(max_length=400)


class SettlementReply(BaseModel):
    option: int
    reason: str = Field(max_length=400)


# ---------------------------------------------------------------------- tools
def _data(ctx: RunContextWrapper[Situation], name: str) -> str:
    value = ctx.context.data.get(name)
    return json.dumps(value, default=str) if value is not None else f"{name} has nothing for this decision"


@function_tool(strict_mode=False)
def base_rates(ctx: RunContextWrapper[Situation]) -> str:
    """How 600 past incidents of this type compared with what callers said: true victims per reported victim,
    RED patients per caller-reported critical patient and per other reported victim, the YELLOW share of the
    non-RED victims, and how often someone was trapped."""
    return _data(ctx, "base_rates")


@function_tool(strict_mode=False)
def statistical_estimate(ctx: RunContextWrapper[Situation]) -> str:
    """The estimate the base rates alone give for this call: expected RED, YELLOW and GREEN victims, the
    likely hazard, and whether someone is probably trapped."""
    return _data(ctx, "statistical_estimate")


@function_tool(strict_mode=False)
def hospital_options(ctx: RunContextWrapper[Situation]) -> str:
    """The hospitals that can treat everyone on this ambulance: minutes until arrival, ER bays free now and
    expected free at arrival, patients already heading there, ambulances waiting at the door, occupancy."""
    return _data(ctx, "hospital_options")


@function_tool(strict_mode=False)
def capacity_at_arrival(ctx: RunContextWrapper[Situation]) -> str:
    """This hospital when the ambulance arrives: bays expected free, bays held back for RED patients,
    patients already on their way, who is on the ambulance, what is missing, and the counter-offer on the table."""
    return _data(ctx, "capacity_at_arrival")


@function_tool(strict_mode=False)
def contested_units(ctx: RunContextWrapper[Situation]) -> str:
    """Each unit that several incidents want this minute: the incidents bidding for it, what the unit is worth
    to each (expected survivors gained per minute of the unit's time), each incident's priority, the best other
    unit each could get instead, and the incident an optimizer maximizing total value would pick."""
    return _data(ctx, "contested_units")


@function_tool(strict_mode=False)
def settlement_options(ctx: RunContextWrapper[Situation]) -> str:
    """The ways to settle this admission: which hospital, what deal (accepts on arrival, has room later, or
    accepts if a clinician is moved there), and in how many minutes these patients' care would start."""
    return _data(ctx, "settlement_options")


# ---------------------------------------------------------------------- instructions
ASSESSMENT = """You are the Incident Assessment Agent in a city's emergency-response system.
Your job: turn one noisy 911 call into the best estimate of how many victims there are and how badly they are hurt, so the other agents send the right units to the right place first.
Your policy: callers under-count victims and misjudge who is critical. Call base_rates to see how past incidents of this type compared with what callers said, and statistical_estimate for the estimate those rates give. Start from that estimate. Move away from it only when the caller's own words give a concrete reason, such as people described as unconscious, not breathing, trapped or badly burned, or a scene that sounds larger or smaller than the counts. Expected counts may be fractional.
Answer with only a JSON object, no other text:
{"red": <number>, "yellow": <number>, "green": <number>, "hazard": "none" | "fire" | "structural" | "hazmat", "trapped": true | false, "reason": "<one short sentence>"}"""

DISPATCH = """You are the Dispatch Agent in a city's emergency-response system.
Your job now: an ambulance is loading patients and needs a hospital. Decide the order in which to ask hospitals to admit them; each hospital answers yes, no, or with a counter-offer.
Your objective: these patients should start treatment as soon as possible, without crowding one emergency room while another has room.
Your policy: call hospital_options. Prefer the nearest hospital that should have a bay free when the ambulance arrives. Ask a farther hospital first only when the nearest is expected to be full, already has ambulances waiting at its door, or has many patients already heading there, and the extra drive is short compared with the likely wait.
Answer with only a JSON object, no other text:
{"ask_order": ["<hospital id>", ...], "reason": "<one short sentence>"}"""

HOSPITAL = """You are the Hospital Coordination Agent for one hospital in a city's emergency-response system.
Your job now: an ambulance asked to bring patients and you cannot simply accept. Either you are full now but expect room soon, or you have a bay but lack a clinician with the right skill. You can make the counter-offer on the table ("yes in N minutes" or "yes if a clinician is lent to us") or refuse, so the ambulance tries another hospital.
Your objective: every patient who reaches your hospital is treated safely and soon, and the region as a whole saves the most lives.
Your policy: call capacity_at_arrival. Make the counter-offer when it would start these patients' care reasonably soon and does not take a bay you hold back for RED patients from a non-RED patient. Refuse when the wait would be long, when ambulances are already queuing at your door, or when accepting would crowd out critical patients already on their way.
Answer with only a JSON object, no other text:
{"decision": "counter" | "reject", "reason": "<one short sentence>"}"""

ALLOCATION = """You are the Resource Allocation Agent, the referee between dispatch and the hospitals in a city's emergency-response system. You make two kinds of decision; the prompt says which one this is.
Contested units: several incidents want the same ambulance or team this minute. Call contested_units and award every contested unit to one of the incidents bidding for it. The value numbers estimate expected survivors gained per minute of the unit's time; prefer the award that saves the most lives overall, and count what each losing incident can get instead. The optimizer's pick maximizes the summed value; depart from it only for a clear reason, such as a critical patient who would otherwise wait with nothing else coming.
Answer: {"awards": [{"unit": "<unit id>", "incident": "<incident id>"}, ...], "reason": "<one short sentence>"}
Admission settlement: hospitals answered an ambulance's request with counter-offers. Call settlement_options and pick the option that starts these patients' care soonest; choose a slower one only for a clear reason.
Answer: {"option": <option number>, "reason": "<one short sentence>"}
Answer with only the JSON object, no other text."""

SETTINGS = ModelSettings(temperature=0)


def build_agents(model) -> dict[str, Agent]:
    """The four agents, all on `model` (an SDK Model). With no model they still exist, for dry runs and replays."""
    kw = {"model": model} if model is not None else {}
    return {
        "assessment": Agent(name="Incident Assessment Agent", instructions=ASSESSMENT,
                            tools=[base_rates, statistical_estimate], model_settings=SETTINGS, **kw),
        "dispatch": Agent(name="Dispatch Agent", instructions=DISPATCH, tools=[hospital_options],
                          model_settings=SETTINGS, **kw),
        "hospital": Agent(name="Hospital Coordination Agent", instructions=HOSPITAL, tools=[capacity_at_arrival],
                          model_settings=SETTINGS, **kw),
        "allocation": Agent(name="Resource Allocation Agent", instructions=ALLOCATION,
                            tools=[contested_units, settlement_options], model_settings=SETTINGS, **kw),
    }
