"""Structured messages between agents.

Every message names its sender, recipient, type, the incident it is about and the
action it proposes. All of them are logged, so the dashboard can replay who asked
for what, who refused, and how each conflict was settled.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ASSESSMENT_AGENT = "incident_assessment"
DISPATCH_AGENT = "dispatch"
ALLOCATION_AGENT = "resource_allocation"


def hospital_agent(hospital_id: str) -> str:
    return f"hospital:{hospital_id}"


class MsgType(StrEnum):
    ASSESSMENT = "assessment"
    DISPATCH_PROPOSAL = "dispatch_proposal"
    CONFLICT_NOTICE = "conflict_notice"
    ALLOCATION_DECISION = "allocation_decision"
    ADMISSION_REQUEST = "admission_request"
    ADMISSION_ACCEPT = "admission_accept"
    ADMISSION_REJECT = "admission_reject"
    COUNTER_PROPOSAL = "counter_proposal"
    STAFF_REQUEST = "staff_request"
    STAFF_OFFER = "staff_offer"
    STATUS_UPDATE = "status_update"


ActionKind = Literal[
    "assess", "dispatch_ambulance", "dispatch_team", "admit", "reject", "counter", "lend_staff",
    "assign_staff", "open_surge", "set_destination", "inform",
]


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ActionKind
    unit_id: str | None = None
    hospital_id: str | None = None
    staff_id: str | None = None
    victim_ids: list[str] = Field(default_factory=list)
    eta_min: float | None = None
    value: float | None = None
    condition: str | None = Field(default=None, description="For counters: 'staff:<role>:<skill>' or 'after:<minutes>'.")
    estimate: dict[str, float] | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    msg_id: int
    t: int
    sender: str
    recipient: str
    type: MsgType
    incident_id: str | None = None
    proposed_action: ProposedAction
    rationale: str = ""
    in_reply_to: int | None = None
    round: int = Field(default=1, ge=1)


class MessageBus:
    """Collects the messages of one decision step and keeps the whole history."""

    def __init__(self) -> None:
        self._next_id = 1
        self.history: list[Message] = []
        self._step: list[Message] = []
        self.t = 0

    def start(self, t: int) -> None:
        self.t = t
        self._step = []

    def send(self, sender: str, recipient: str, type: MsgType, action: ProposedAction, *,
             incident_id: str | None = None, rationale: str = "", in_reply_to: int | None = None,
             round: int = 1) -> Message:
        msg = Message(msg_id=self._next_id, t=self.t, sender=sender, recipient=recipient, type=type,
                      incident_id=incident_id, proposed_action=action, rationale=rationale,
                      in_reply_to=in_reply_to, round=round)
        self._next_id += 1
        self._step.append(msg)
        self.history.append(msg)
        return msg

    def step_messages(self) -> list[Message]:
        return list(self._step)
