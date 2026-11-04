"""What the dashboard shows, computed from one run log.

A run log is a header (scenario, policy, final metrics) and one record per simulated minute,
written by `Simulator._snapshot`. A `Replay` holds one log and the scenario it ran on, and its
methods return the data behind each panel at a given minute. Nothing here imports Streamlit, so
the tests cover it directly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from statistics import mean

import pandas as pd

from ..datagen import generate_scenario
from ..eval.report import OUTCOMES
from ..eval.runner import POLICY_LABEL, read_run_log, run_log_header, run_one
from ..schema import Scenario

TRIAGE = ("RED", "YELLOW", "GREEN")

AGENT_JOBS = {
    "incident_assessment": ("Incident Assessment", "Turns each noisy 911 call into an estimate of victims and urgency"),
    "dispatch": ("Dispatch", "Bids for ambulances and teams, and asks hospitals to take each load"),
    "resource_allocation": ("Resource Allocation", "The referee: settles unit conflicts, arranges staff loans, places the pool"),
}
HOSPITAL_JOB = "Guards its bays and staff: accepts, rejects, or counters with a condition"

TYPE_LABEL = {
    "assessment": "Assessment", "dispatch_proposal": "Dispatch proposal", "conflict_notice": "Unit conflict",
    "allocation_decision": "Decision", "admission_request": "Admission request", "admission_accept": "Accept",
    "admission_reject": "Reject", "counter_proposal": "Counter-offer", "staff_request": "Staff request",
    "staff_offer": "Staff offer", "status_update": "Status update",
}

UNIT_STATE = {"busy_prior": "unavailable", "idle": "free", "returning": "free"}     # every other status is in use
STATUS_TEXT = {
    "busy_prior": "busy with an earlier call", "idle": "at its station", "to_scene": "driving to",
    "at_scene": "on scene at", "loading": "loading patients at", "loaded": "loaded at", "to_hospital": "driving to",
    "at_door": "waiting at the full ER door of", "handoff": "handing over at", "returning": "returning to its station",
    "working": "working at",
}
TO_HOSPITAL = {"to_hospital", "at_door", "handoff"}
TO_INCIDENT = {"to_scene", "at_scene", "loading", "loaded", "working"}
NEGOTIATING = {"counter_proposal", "admission_reject", "staff_request"}
NEVER = 10**9


def agent_name(agent_id: str) -> str:
    if agent_id.startswith("hospital:"):
        return f"Hospital {agent_id.split(':', 1)[1]}"
    return AGENT_JOBS.get(agent_id, (agent_id, ""))[0]


def message_text(m: dict) -> str:
    return m["rationale"].replace("_", " ")


@dataclass
class Thread:
    """One negotiation: a unit conflict or an admission request, with every reply that follows from it."""
    kind: str                  # "unit conflict" or "admission"
    t: int
    unit: str | None
    incident: str | None
    messages: list[dict]

    @property
    def negotiated(self) -> bool:
        """More than a plain yes: a conflict, a counter-offer, a refusal or a staff search."""
        return self.kind == "unit conflict" or any(m["type"] in NEGOTIATING for m in self.messages)

    @property
    def rounds(self) -> int:
        return max(m["round"] for m in self.messages)

    @property
    def outcome(self) -> str:
        decisions = [m for m in self.messages if m["type"] == "allocation_decision"]
        if decisions:
            more = f" (+{len(decisions) - 1} more)" if len(decisions) > 1 else ""
            return message_text(decisions[-1]) + more
        accepts = [m for m in self.messages if m["type"] == "admission_accept"]
        if accepts:
            return f"{agent_name(accepts[-1]['sender'])} accepts"
        return message_text(self.messages[-1])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"Round": m["round"], "From": agent_name(m["sender"]), "To": agent_name(m["recipient"]),
                              "Type": TYPE_LABEL.get(m["type"], m["type"]), "Message": message_text(m)}
                             for m in self.messages])


@dataclass
class Replay:
    """One run, minute by minute, and the scenario it ran on."""
    header: dict
    ticks: list[dict]
    scenario: Scenario

    @property
    def policy(self) -> str:
        """B0, B1, M1 or M2: the log stores the policy's full name, which starts with its key."""
        return self.header["policy"].split("_", 1)[0]

    @property
    def label(self) -> str:
        return POLICY_LABEL.get(self.policy, self.header["policy"])

    @property
    def horizon(self) -> int:
        return len(self.ticks)

    @property
    def metrics(self) -> dict:
        return self.header["metrics"]

    def tick(self, t: int) -> dict:
        return self.ticks[min(max(t, 0), self.horizon - 1)]

    @cached_property
    def messages(self) -> list[dict]:
        return [m for tick in self.ticks for m in tick["messages"]]

    @cached_property
    def events(self) -> list[dict]:
        return [e for tick in self.ticks for e in tick["events"]]

    def first(self, kind: str, key: str) -> dict[str, int]:
        """Minute of the first event of this kind, for each value of `key`."""
        out: dict[str, int] = {}
        for e in self.events:
            if e["kind"] == kind:
                out.setdefault(e[key], e["t"])
        return out

    @cached_property
    def triage_counts(self) -> dict[str, Counter]:
        counts: dict[str, Counter] = {i.incident_id: Counter() for i in self.scenario.incidents}
        for v in self.scenario.victims:
            counts[v.incident_id][v.triage] += 1
        return counts

    # ------------------------------------------------------------------ performance
    @cached_property
    def series(self) -> pd.DataFrame:
        """Running totals at every minute. At the last minute they equal the run's final metrics."""
        called, reached = self.first("call", "incident"), self.first("scene_assessed", "incident")
        admitted = self.first("admitted", "victim")
        reds: dict[str, list[str]] = {}
        for v in self.scenario.victims:
            if v.triage == "RED":
                reds.setdefault(v.incident_id, []).append(v.victim_id)
        per_minute = {name: Counter(m["t"] for m in self.messages if keep(m))
                      for name, keep in (("messages", lambda m: True),
                                         ("conflicts", lambda m: m["type"] == "conflict_notice"))}
        per_minute["staff_moves"] = Counter(e["t"] for e in self.events if e["kind"] == "staff_assigned")
        totals = Counter()
        rows = []
        for tick in self.ticks:
            t = tick["t"]
            totals["door_minutes"] += sum(a["status"] == "at_door" for a in tick["ambulances"])
            for name, counts in per_minute.items():
                totals[name] += counts[t]
            response = [reached[i] - called[i] for i in reached if reached[i] <= t]
            waiting = sum(1 for i, victims in reds.items()
                          if called.get(i, NEVER) <= t and any(admitted.get(v, NEVER) > t for v in victims))
            rows.append({"t": t, "victims_in_care": sum(tick["served"].values()), "red_in_care": tick["served"]["RED"],
                         "critical_waiting": waiting, "first_response": mean(response) if response else None,
                         "incidents_called": sum(1 for c in called.values() if c <= t), **totals})
        return pd.DataFrame(rows).set_index("t")

    # ------------------------------------------------------------------ map
    def map_layers(self, t: int) -> dict[str, pd.DataFrame]:
        """Incidents, units and hospitals at minute t, with what the hover text needs."""
        tick = self.tick(t)
        static = {i.incident_id: i for i in self.scenario.incidents}
        incidents = []
        for inc in tick["incidents"]:
            # before a unit arrives nobody has left the scene, so everyone is still waiting
            waiting = inc["waiting"] if inc["revealed"] else dict(self.triage_counts[inc["id"]])
            needs = [f"{task.replace('_', ' ')} ({left:.0f} min left)" for task, left in inc["needs"].items() if left > 0]
            incidents.append({
                "id": inc["id"], "number": int(inc["id"].rsplit("-", 1)[-1]), "x": inc["x"], "y": inc["y"],
                "type": inc["type"].replace("_", " "), "victims": static[inc["id"]].n_victims,
                "waiting": sum(waiting.values()), "state": next((k for k in TRIAGE if waiting.get(k)), "cleared"),
                **{k: waiting.get(k, 0) for k in TRIAGE}, "reached": inc["revealed"], "reported": inc["reported"],
                "access": inc["access"], "needs": ", ".join(needs) or "none", "teams": ", ".join(inc["teams"]) or "none",
            })
        hospitals = {h.hospital_id: h for h in self.scenario.hospitals}
        places = {h.hospital_id: (h.x, h.y) for h in self.scenario.hospitals}
        places.update({i["id"]: (i["x"], i["y"]) for i in tick["incidents"]})
        units = []
        for kind, records in (("ambulance", tick["ambulances"]), ("team", tick["teams"])):
            for u in records:
                where = (u.get("hospital") if u["status"] in TO_HOSPITAL
                         else u.get("incident") if u["status"] in TO_INCIDENT else None)
                target = places.get(where)
                units.append({
                    "id": u["id"], "kind": kind, "type": u["type"].replace("_", " "), "status": u["status"],
                    "state": UNIT_STATE.get(u["status"], "in use"), "x": u["x"], "y": u["y"], "where": where or "",
                    "tx": target[0] if target else None, "ty": target[1] if target else None,
                    "doing": f"{STATUS_TEXT.get(u['status'], u['status'])} {where or ''}".strip(),
                    "onboard": u.get("onboard", 0), "task": (u.get("task") or "").replace("_", " "),
                })
        rows = []
        for h in tick["hospitals"]:
            s = hospitals[h["id"]]
            rows.append({"id": h["id"], "name": s.name, "short": s.name.split()[0], "x": s.x, "y": s.y,
                         "bays_used": h["bays_used"], "bays_total": h["bays_total"], "door": h["door"],
                         "corridor": h["corridor"], "icu_free": h["icu_free"], "icu_total": s.icu_beds_total,
                         "surge": h["surge"], "doctors_free": h["doctor_slots_free"], "nurses_free": h["nurse_slots_free"],
                         "lent_in": h["lent_in"]})
        return {"incidents": pd.DataFrame(incidents), "units": pd.DataFrame(units), "hospitals": pd.DataFrame(rows)}

    # ------------------------------------------------------------------ resources
    def resources(self, t: int) -> pd.DataFrame:
        """How many of each resource are in use, free or unavailable at minute t."""
        tick = self.tick(t)
        hospitals = {h.hospital_id: h for h in self.scenario.hospitals}
        rows = []

        def add(resource: str, in_use: int, free: int, unavailable: int = 0) -> None:
            rows.extend((resource, state, n) for state, n in (("in use", in_use), ("free", free), ("unavailable", unavailable)))

        for resource, records in (("Ambulances", tick["ambulances"]), ("Responder teams", tick["teams"])):
            state = Counter(UNIT_STATE.get(u["status"], "in use") for u in records)
            add(resource, state["in use"], state["free"], state["unavailable"])
        hs = tick["hospitals"]
        used, total = sum(h["bays_used"] for h in hs), sum(h["bays_total"] for h in hs)
        add("ER bays", used, total - used, sum(hospitals[h["id"]].surge_bays for h in hs if h["surge"] != "open"))
        icu_free = sum(h["icu_free"] for h in hs)
        add("ICU beds", sum(hospitals[h["id"]].icu_beds_total for h in hs) - icu_free, icu_free)
        for role in ("doctor", "nurse"):
            slots, free = sum(h[f"{role}_slots"] for h in hs), sum(h[f"{role}_slots_free"] for h in hs)
            add(f"{role.title()}s (patient slots)", slots - free, free)
        return pd.DataFrame(rows, columns=["resource", "state", "count"])

    # ------------------------------------------------------------------ agents and negotiation
    def agents(self, t: int, last: int = 3) -> list[dict]:
        """One card per agent: its status line at minute t, message counts so far and its latest messages."""
        notes = self.tick(t)["notes"]
        seen = [m for m in self.messages if m["t"] <= t]
        cards = []
        for agent_id, note in notes.items():
            mine = [m for m in seen if agent_id in (m["sender"], m["recipient"])]
            recent = []
            for m in reversed(mine[-last:]):
                sent = m["sender"] == agent_id
                recent.append({"t": m["t"], "direction": "to" if sent else "from",
                               "other": agent_name(m["recipient"] if sent else m["sender"]),
                               "type": TYPE_LABEL.get(m["type"], m["type"]), "text": message_text(m)})
            cards.append({"id": agent_id, "name": agent_name(agent_id),
                          "job": HOSPITAL_JOB if agent_id.startswith("hospital:") else AGENT_JOBS.get(agent_id, ("", ""))[1],
                          "status": note.replace("_", " ").replace("->", " to "), "sent": sum(m["sender"] == agent_id for m in mine),
                          "received": sum(m["recipient"] == agent_id for m in mine), "recent": recent})
        return cards

    @cached_property
    def threads(self) -> list[Thread]:
        """Every negotiation, oldest first. A message belongs to the conflict notice or admission request
        it answers, followed back through `in_reply_to`; the requests one ambulance sends to several
        hospitals in the same minute form one thread."""
        by_id = {m["msg_id"]: m for m in self.messages}

        def root(m: dict) -> dict:
            while m.get("in_reply_to") in by_id:
                m = by_id[m["in_reply_to"]]
            return m

        groups: dict[tuple, list[dict]] = {}
        for m in self.messages:
            r = root(m)
            if r["type"] == "conflict_notice":
                key = ("unit conflict", r["t"], r["msg_id"])
            elif r["type"] == "admission_request":
                key = ("admission", r["t"], r["proposed_action"]["unit_id"])
            else:
                continue
            groups.setdefault(key, []).append(m)
        out = []
        for (kind, t, _), msgs in groups.items():
            msgs.sort(key=lambda m: m["msg_id"])
            out.append(Thread(kind, t, msgs[0]["proposed_action"].get("unit_id"), msgs[0].get("incident_id"), msgs))
        return sorted(out, key=lambda th: th.messages[0]["msg_id"])

    def threads_until(self, t: int, negotiated_only: bool = False) -> list[Thread]:
        return [th for th in self.threads if th.t <= t and (th.negotiated or not negotiated_only)]


# ---------------------------------------------------------------------- loading
def live(theme: str, stress: str, seed: int, policy: str) -> Replay:
    """Run one policy on one scenario now, recording every minute."""
    scenario = generate_scenario(theme, stress, seed)
    _, result = run_one(theme, stress, seed, policy, record=True)
    return Replay(run_log_header(result, scenario.meta.model_dump(mode="json")), result.ticks, scenario)


def from_log(path: str | Path) -> Replay:
    """Replay a log saved by `python -m emergency_mas.eval replay`."""
    header, ticks = read_run_log(path)
    meta = header["scenario"]
    scenario = generate_scenario(meta["theme"], meta["stress"], meta["seed"])
    if scenario.meta.generator_version != meta["generator_version"]:
        raise ValueError(f"{path} was written by generator {meta['generator_version']}; this code is "
                         f"{scenario.meta.generator_version}, so its scenario would not match")
    return Replay(header, ticks, scenario)


def saved_logs(folder: str | Path) -> dict[str, dict[str, Path]]:
    """Saved logs by scenario id, then by policy key, from files named `<scenario_id>__<policy>.jsonl`."""
    found: dict[str, dict[str, Path]] = {}
    for path in sorted(Path(folder).glob("*__*.jsonl")):
        scenario_id, policy = path.stem.rsplit("__", 1)
        found.setdefault(scenario_id, {})[policy] = path
    return found


# ---------------------------------------------------------------------- comparisons
def compare_outcomes(a: Replay, b: Replay) -> pd.DataFrame:
    """The final metrics of two runs side by side, labelled as in docs/RESULTS.md."""
    rows = []
    for key, label, digits in [*OUTCOMES, ("messages", "Messages between agents", 0), ("staff_moves", "Staff moved", 0)]:
        x, y = a.metrics.get(key), b.metrics.get(key)
        rows.append({"Outcome": label, a.policy: "n/a" if x is None else f"{x:.{digits}f}",
                     b.policy: "n/a" if y is None else f"{y:.{digits}f}",
                     "Difference": "n/a" if x is None or y is None else f"{round(x - y, digits) + 0.0:+.{digits}f}"})
    return pd.DataFrame(rows)


def seed_context(runs: pd.DataFrame, theme: str, stress: str, seed: int, policy: str, against: str,
                 metric: str = "weighted_survival") -> dict | None:
    """Where this seed sits among the experiment's seeds for the same setting, so one run is not read as the average."""
    if runs.empty or policy == against:
        return None
    wide = runs[(runs.theme == theme) & (runs.stress == stress)].pivot_table(index="seed", columns="policy", values=metric)
    if policy not in wide or against not in wide:
        return None
    diff = (wide[policy] - wide[against]).dropna()
    if diff.empty:
        return None
    out = {"seeds": len(diff), "ahead_share": float((diff > 0).mean()), "median_diff": float(diff.median())}
    if seed in diff.index:
        out["diff"] = float(diff[seed])
        out["rank"] = int((diff < diff[seed]).sum()) + 1          # 1 = the smallest gain
    return out
