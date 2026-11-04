"""Write a scenario to disk: CSV tables, a canonical JSON file and a manifest with hashes."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
from importlib.metadata import version
from pathlib import Path

from .. import params as P
from ..domain import straight_km
from ..schema import Scenario
from .analysis import contention_report, pressure_indices, travel_model, victim_counts
from .validate import validate_scenario


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scenario_json(s: Scenario) -> str:
    return json.dumps(s.model_dump(mode="json"), sort_keys=True, indent=1) + "\n"


def scenario_hash(s: Scenario) -> str:
    return sha256(scenario_json(s))


def _join(values) -> str:
    return "|".join(str(v) for v in values)


def table_rows(s: Scenario) -> dict[str, list[dict]]:
    """One flat table per entity type. List fields are joined with '|'."""
    hospitals = [{**h.model_dump(mode="json"), "capabilities": _join(h.capabilities),
                  "er_bays_free": h.er_bays_free, "beds_available": h.beds_available} for h in s.hospitals]
    staff = [{**m.model_dump(mode="json"), "skills": _join(m.skills), "free_slots": m.free_slots} for m in s.staff]
    responders = [{**r.model_dump(mode="json"), "capabilities": _join(r.capabilities)} for r in s.responders]
    incidents = []
    for i in s.incidents:
        row = i.model_dump(mode="json", exclude={"needs", "report"})
        row["needs"] = _join(f"{n.capability}:{n.work_min}min x{n.min_teams}" for n in i.needs)
        row.update({f"report_{k}": v for k, v in i.report.model_dump(mode="json").items()})
        incidents.append(row)
    return {
        "stations": [x.model_dump(mode="json") for x in s.stations],
        "hospitals": hospitals,
        "staff": staff,
        "ambulances": [x.model_dump(mode="json") for x in s.ambulances],
        "responders": responders,
        "incidents": incidents,
        "victims": [x.model_dump(mode="json") for x in s.victims],
        "travel_times": travel_rows(s),
    }


def travel_rows(s: Scenario) -> list[dict]:
    """Base travel times between the places agents care about (departing at minute 0)."""
    tm = travel_model(s)
    rows: list[dict] = []

    def add(kind: str, a_id: str, a_pos, b_id: str, b_pos, speed: float) -> None:
        rows.append({"kind": kind, "from_id": a_id, "to_id": b_id,
                     "road_km": round(straight_km(a_pos, b_pos) * P.CIRCUITY, 3),
                     "minutes_t0": round(tm.minutes(a_pos, b_pos, speed, 0.0), 2)})

    for a in s.ambulances:
        for i in s.incidents:
            add("ambulance_to_incident", a.ambulance_id, (a.x, a.y), i.incident_id, (i.x, i.y), a.speed_kmh)
    for r in s.responders:
        for i in s.incidents:
            add("responder_to_incident", r.team_id, (r.x, r.y), i.incident_id, (i.x, i.y), r.speed_kmh)
    for i in s.incidents:
        for h in s.hospitals:
            add("incident_to_hospital", i.incident_id, (i.x, i.y), h.hospital_id, (h.x, h.y), P.AMBULANCE_SPEED_KMH)
    for h1 in s.hospitals:
        for h2 in s.hospitals:
            if h1 is not h2:
                add("hospital_to_hospital", h1.hospital_id, (h1.x, h1.y), h2.hospital_id, (h2.x, h2.y), P.AMBULANCE_SPEED_KMH)
    return rows


def to_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{p: version(p) for p in ("numpy", "pandas", "pydantic", "faker")}}


def write_scenario(s: Scenario, out_root: str | Path) -> dict:
    """Write every file for one scenario and return its manifest."""
    out = Path(out_root) / s.meta.scenario_id
    out.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, rows in table_rows(s).items():
        text = to_csv(rows)
        (out / f"{name}.csv").write_text(text, encoding="utf-8")
        hashes[f"{name}.csv"] = sha256(text)
    text = scenario_json(s)
    (out / "scenario.json").write_text(text, encoding="utf-8")
    hashes["scenario.json"] = sha256(text)

    manifest = {
        "scenario_id": s.meta.scenario_id, "theme": s.meta.theme, "stress": s.meta.stress, "seed": s.meta.seed,
        "generator_version": s.meta.generator_version, "scenario_sha256": hashes["scenario.json"],
        "counts": victim_counts(s), "pressure": pressure_indices(s), "contention": contention_report(s),
        "checks": {c.name: c.passed for c in validate_scenario(s)},
        "files": hashes, "libraries": library_versions(),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
