"""The contract between the simulator and anything that makes decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..schema import Scenario
from .actions import Action
from .observe import Observation


@dataclass
class Decision:
    actions: list[Action] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)   # agent messages, logged for the dashboard
    notes: dict[str, str] = field(default_factory=dict)  # one status line per agent, for the dashboard


class Policy(Protocol):
    name: str

    def reset(self, scenario: Scenario) -> None: ...

    def decide(self, obs: Observation) -> Decision: ...
