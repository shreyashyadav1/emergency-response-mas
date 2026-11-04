"""Plain-language caller narratives, written from the reported facts only.

The default narrator is template based so every scenario is exactly
reproducible and free to regenerate. The Narrator protocol leaves room for an
LLM narrator whose output is cached by prompt hash.
"""
from __future__ import annotations

from typing import Protocol

from faker import Faker

from ..rng import Rng

_FAKE: Faker | None = None


def seeded_faker(seed: int) -> Faker:
    """One shared Faker, re-seeded on every use so its output depends only on the seed passed in."""
    global _FAKE
    if _FAKE is None:
        _FAKE = Faker("en_US")
    _FAKE.seed_instance(seed)
    return _FAKE


_STREET_SUFFIXES = ["Street", "Avenue", "Road", "Boulevard", "Lane", "Way"]
_BUILDINGS = ["apartment block", "warehouse", "office building", "restaurant", "parking garage", "school annex"]

_OPENERS = {
    "multi_vehicle_accident": [
        "Caller reports a crash involving several vehicles on {street}.",
        "Multiple cars collided on {street} and traffic has stopped.",
        "Chain-reaction collision on {street}, caller is standing nearby.",
    ],
    "building_fire": [
        "Fire reported in a {building} on {street}.",
        "Caller sees flames and heavy smoke from a {building} on {street}.",
        "Fire alarm and visible fire at a {building} on {street}.",
    ],
    "structural_collapse": [
        "Part of a {building} on {street} has collapsed.",
        "Caller heard a loud bang and a {building} on {street} came down.",
        "A floor gave way in a {building} on {street}.",
    ],
    "hazmat_release": [
        "Chemical leak reported near {street}, caller can smell fumes.",
        "A storage tank is leaking on {street} and people are coughing.",
        "Unknown vapor cloud near {street}, workers are evacuating.",
    ],
    "crowd_incident": [
        "Crowd surge at an event on {street}, people are on the ground.",
        "Several people collapsed in a crowded area on {street}.",
        "Heat and crowding at a gathering on {street}, many feel faint.",
    ],
}
_HAZARD_PHRASE = {
    "fire": "Flames and smoke are visible.",
    "structural": "Debris is still coming down.",
    "hazmat": "There is a strong chemical smell.",
}


class Narrator(Protocol):
    name: str

    def describe(self, rng: Rng, fake: Faker, incident_type: str, *, reported_victims: int,
                 reported_critical: int, reported_hazard: str, reported_trapped: bool,
                 confidence: float) -> str: ...


class TemplateNarrator:
    name = "template"

    def describe(self, rng: Rng, fake: Faker, incident_type: str, *, reported_victims: int,
                 reported_critical: int, reported_hazard: str, reported_trapped: bool,
                 confidence: float) -> str:
        street = f"{fake.last_name()} {rng.choice(_STREET_SUFFIXES)}"
        opener = rng.choice(_OPENERS[incident_type]).format(street=street, building=rng.choice(_BUILDINGS))
        if confidence < 0.45:
            count = f"Caller thinks maybe {reported_victims} people are hurt but cannot be sure."
        else:
            count = f"About {reported_victims} {'person is' if reported_victims == 1 else 'people are'} hurt."
        if reported_critical > 0:
            verb = "looks" if reported_critical == 1 else "look"
            critical = f"{reported_critical} {verb} badly hurt."
        else:
            critical = "No one seems critical yet."
        parts = [opener, count, critical]
        if reported_trapped:
            parts.append("Someone is trapped.")
        if reported_hazard in _HAZARD_PHRASE:
            parts.append(_HAZARD_PHRASE[reported_hazard])
        return " ".join(parts)
