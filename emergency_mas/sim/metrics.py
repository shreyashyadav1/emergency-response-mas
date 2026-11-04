"""Outcome metrics for one run, computed from the final simulator state.

Victims who never start definitive care inside the horizon are scored as if care
started at the end of the horizon. That flatters every policy equally, so the
comparison between policies stays fair.
"""
from __future__ import annotations

from statistics import median

from .. import params as P
from ..domain import survival_probability


def _outcome(vs, horizon: int) -> float:
    if vs.survival is not None:
        return vs.survival
    return survival_probability(horizon - vs.report_time, vs.window, vs.v.survival_if_timely, vs.v.decay_half_life_min)


def compute_metrics(sim) -> dict:
    w, horizon = sim.w, sim.scenario.meta.horizon_min
    victims = list(w.victims.values())
    outcomes = {vs.v.victim_id: _outcome(vs, horizon) for vs in victims}
    weight = {vs.v.victim_id: P.TRIAGE_WEIGHT[vs.v.triage] for vs in victims}

    m: dict = {
        "victims": len(victims),
        "expected_survivors": round(sum(outcomes.values()), 3),
        "expected_deaths": round(len(victims) - sum(outcomes.values()), 3),
        "weighted_survival": round(sum(outcomes[k] * weight[k] for k in outcomes) / max(1e-9, sum(weight.values())), 4),
    }
    for triage in P.TRIAGE:
        group = [vs for vs in victims if vs.v.triage == triage]
        served = [vs for vs in group if vs.care_start is not None]
        times = [vs.care_start - vs.report_time for vs in served]
        key = triage.lower()
        m[f"{key}_victims"] = len(group)
        m[f"{key}_served"] = len(served)
        m[f"{key}_in_window"] = sum(1 for vs in served if vs.care_start - vs.report_time <= vs.window)
        m[f"{key}_survivors"] = round(sum(outcomes[vs.v.victim_id] for vs in group), 3)
        m[f"{key}_median_min_to_care"] = round(median(times), 1) if times else None
    m["victims_served"] = sum(1 for vs in victims if vs.care_start is not None)

    response = [inc.revealed_at - inc.inc.report_time_min for inc in w.incidents.values() if inc.revealed_at is not None]
    m["incidents_reached"] = len(response)
    m["mean_first_response_min"] = round(sum(response) / len(response), 1) if response else None
    critical = [inc for inc in w.incidents.values() if inc.inc.n_red > 0]
    m["critical_incidents"] = len(critical)
    reds = {inc.inc.incident_id: [vs for vs in w.victims_of(inc.inc.incident_id) if vs.v.triage == "RED"] for inc in critical}
    m["unserved_critical_incidents"] = sum(1 for group in reds.values() if any(vs.care_start is None for vs in group))
    m["late_critical_incidents"] = sum(
        1 for group in reds.values()
        if any(vs.care_start is None or vs.care_start - vs.report_time > vs.window for vs in group))

    n_amb = max(1, len(w.ambulances))
    m["ambulance_utilization"] = round(sum(a.busy_minutes for a in w.ambulances.values()) / (n_amb * horizon), 3)
    m["team_utilization"] = round(sum(r.busy_minutes for r in w.teams.values()) / (max(1, len(w.teams)) * horizon), 3)
    m["er_bay_utilization"] = round(sum(sim.samples["bays"]) / max(1, len(sim.samples["bays"])), 3)
    m["staff_utilization"] = round(sum(sim.samples["staff"]) / max(1, len(sim.samples["staff"])), 3)
    m["ambulance_loads"] = sum(a.loads for a in w.ambulances.values())
    m["ramp_wait_min"] = sum(a.ramp_minutes for a in w.ambulances.values())
    m["boarding_patients"] = sum(1 for vs in victims if vs.boarding)
    m["corridor_handovers"] = sim.counts["corridor_handovers"]

    c = sim.counts
    m["wasted_trips"] = c["wasted_trips"]
    m["diversions"] = c["diversions"]
    m["fallback_destinations"] = c["fallback_destinations"]
    m["staff_moves"] = c["staff_moves"]
    m["surges_opened"] = c["surges"]
    m["actions_rejected"] = sum(v for k, v in c.items() if k.endswith(":rejected"))
    m["messages"] = sum(v for k, v in c.items() if k.startswith("msg:"))
    for k, v in sorted(c.items()):
        if k.startswith("msg:"):
            m[f"messages_{k[4:]}"] = v
    return m
