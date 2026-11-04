"""Actions a policy can ask the simulator to take. The simulator may refuse any of them."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DispatchAmbulance:
    ambulance_id: str
    incident_id: str


@dataclass(frozen=True)
class DispatchTeam:
    team_id: str
    incident_id: str


@dataclass(frozen=True)
class SetDestination:
    """Choose (or change) the hospital for the patients an ambulance is loading or carrying."""
    ambulance_id: str
    hospital_id: str


@dataclass(frozen=True)
class ReturnToBase:
    unit_id: str


@dataclass(frozen=True)
class AssignStaff:
    """Send a pool clinician to a hospital, or lend a clinician from one hospital to another."""
    staff_id: str
    hospital_id: str


@dataclass(frozen=True)
class OpenSurge:
    hospital_id: str


Action = DispatchAmbulance | DispatchTeam | SetDestination | ReturnToBase | AssignStaff | OpenSurge


def describe(action: Action) -> dict:
    return {"type": type(action).__name__, **asdict(action)}


@dataclass(frozen=True)
class ActionResult:
    action: Action
    ok: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {**describe(self.action), "ok": self.ok, "reason": self.reason}
