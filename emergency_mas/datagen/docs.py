"""Render the data dictionary and the assumptions from the code, so the docs cannot drift."""
from __future__ import annotations

from .. import params as P
from .. import schema as S

_MODELS = [
    ("Station", S.Station), ("Hospital", S.Hospital), ("StaffMember", S.StaffMember), ("Ambulance", S.Ambulance),
    ("ResponderTeam", S.ResponderTeam), ("Incident", S.Incident), ("IncidentReport", S.IncidentReport),
    ("ResponderNeed", S.ResponderNeed), ("Victim", S.Victim), ("ScenarioMeta", S.ScenarioMeta),
]


def _type_name(annotation) -> str:
    text = str(annotation).replace("typing.", "")
    if text.startswith("<class '"):
        text = text[8:-2]
    return text.replace("|", "\\|")


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out + [""]


def data_dictionary_md() -> str:
    lines = [
        "# Data dictionary", "",
        "Generated from `emergency_mas/schema.py` by `python -m emergency_mas.datagen docs`. Do not edit by hand.", "",
        "Each scenario folder holds one CSV per entity, `scenario.json` (everything, nested) and `MANIFEST.json`",
        "(counts, stress indices, validation results, SHA-256 of every file). List fields are joined with `|` in CSV.", "",
        "| File | Rows |", "|---|---|",
        "| `stations.csv` | Stations that host ambulances and responder teams |",
        "| `hospitals.csv` | One row per hospital; adds `er_bays_free` and `beds_available` |",
        "| `staff.csv` | Doctors and nurses, including the regional on-call pool; adds `free_slots` |",
        "| `ambulances.csv`, `responders.csv` | The fleet at t=0 |",
        "| `incidents.csv` | One row per incident; `needs` is `capability:minutes xteams`, report fields are prefixed `report_` |",
        "| `victims.csv` | One row per treatable victim, with the ground truth |",
        "| `travel_times.csv` | `kind, from_id, to_id, road_km, minutes_t0` for ambulance/responder to incident, incident to hospital, hospital to hospital |",
        "",
    ]
    for name, model in _MODELS:
        lines += [f"## {name}", "", (model.__doc__ or "").strip(), ""]
        lines += _table(["Field", "Type", "Meaning"],
                        [[f"`{f}`", _type_name(info.annotation), info.description or ""] for f, info in model.model_fields.items()])
    return "\n".join(lines)


def assumptions_md() -> str:
    L: list[str] = [
        "# Modeling assumptions", "",
        "Generated from `emergency_mas/params.py` by `python -m emergency_mas.datagen docs`.",
        "The numbers are plausible and internally consistent. They are **not** calibrated to clinical or operational data.",
        "They are tuned so that ambulances, hospital beds and specialist staff are contested.", "",
        "## City and travel", "",
        f"- Square city, {P.CITY_KM:g} km on a side, coordinates in km. Three congestion zones around the center:",
        f"  core (< {P.ZONE_RADII_KM[0]:g} km), inner (< {P.ZONE_RADII_KM[1]:g} km), outer.",
        f"- Road distance = straight-line distance x {P.CIRCUITY}. Ambulances drive {P.AMBULANCE_SPEED_KMH:g} km/h times a zone factor "
        f"({', '.join(f'{k} {v:g}' for k, v in P.ZONE_SPEED_FACTOR.items())}; the mean of the two end zones is used).",
        f"- A congestion multiplier (see stress levels) lengthens every trip that departs in the first {P.CONGESTION_WINDOW_MIN} minutes.",
        f"- Each scenario covers {P.HORIZON_MIN} simulated minutes. The first call arrives at minute 0.", "",
        "## Triage and time sensitivity", "",
        "Victims are tagged RED (immediate), YELLOW (delayed) or GREEN (minor). People beyond help on arrival are counted on the incident "
        "but are not part of the allocation problem.", "",
    ]
    L += _table(
        ["Triage", "Weight", "Golden window (min)", "Survival if timely", "Half-life after window (min)", "Doctor slots", "Nurse slots", "ER stay (min)"],
        [[t, P.TRIAGE_WEIGHT[t], f"{P.GOLDEN_WINDOW_MIN[t][0]}-{P.GOLDEN_WINDOW_MIN[t][1]}", P.SURVIVAL_IF_TIMELY[t],
          P.DECAY_HALF_LIFE_MIN[t], P.STAFF_NEED[t]["doctors"], P.STAFF_NEED[t]["nurses"],
          f"{P.STAFF_NEED[t]['treatment_min'][0]}-{P.STAFF_NEED[t]['treatment_min'][1]}"] for t in P.TRIAGE])
    L += ["Survival is flat while care starts inside the golden window, then halves every half-life: "
          "`p = p0 * 0.5 ** ((t - window) / half_life)`. This is the outcome the evaluation maximizes. "
          "The weights set how much each color counts in the oracle urgency score (weight x 60 / window) and in the "
          "severity-weighted survival metric.", ""]

    L += ["## Stress levels", ""]
    L += _table(
        ["Level", "Victim scale", "RED boost", "ER occupancy", "ICU occupancy", "Ambulance busy at t=0", "Responder busy at t=0", "Staff not yet on duty", "Congestion x"],
        [[n, s.victim_scale, s.red_boost, s.er_occupancy, s.icu_occupancy, s.ambulance_busy_prob, s.responder_busy_prob,
          s.staff_absent_prob, s.congestion] for n, s in P.STRESS.items()])

    L += ["## Themes", ""]
    L += _table(
        ["Theme", "Incidents", "Arrival window (min)", "Burst share", "Clustered share", "Hotspot spread (km)", "Mix"],
        [[n, t.n_incidents, t.arrival_window_min, t.burst_frac, t.cluster_frac, t.hotspot_sigma_km,
          ", ".join(f"{k} {v:g}" for k, v in t.type_weights.items())] for n, t in P.THEMES.items()])
    L += [f"Presets: {', '.join(f'`{k}` = {v['theme']} / {v['stress']}' for k, v in P.PRESETS.items())}. "
          f"Default fleet: {P.DEFAULT_HOSPITALS} hospitals, {P.DEFAULT_AMBULANCES} ambulances, {P.DEFAULT_RESPONDERS} responder teams.", ""]

    L += ["## Incident types", ""]
    L += _table(
        ["Type", "Victims (min-max, mean)", "RED / YELLOW / GREEN", "Injuries (non-minor)", "Trapped share", "Hazard"],
        [[n, f"{p.victims[0]}-{p.victims[1]}, {p.victims[2]:g}",
          " / ".join(f"{p.triage_mix[t]:g}" for t in P.TRIAGE),
          ", ".join(f"{k} {v:g}" for k, v in p.injuries.items()), p.trapped_frac,
          ", ".join(f"{k} {v:g}" for k, v in p.hazards.items())] for n, p in P.INCIDENT_PROFILES.items()])

    L += ["## Hospitals", ""]
    L += _table(
        ["Archetype", "Level", "ER bays", "ICU beds", "Doctors", "Nurses", "Capabilities", "Reserve for RED", "Diversion at"],
        [[a["name"], a["level"], a["er_bays"], a["icu_beds"], sum(a["doctors"].values()), sum(a["nurses"].values()),
          ", ".join(a["capabilities"]), a["reserve_red"], a["diversion"]] for a in P.HOSPITAL_ARCHETYPES[:3]])
    L += [f"The first three archetypes are the default hospitals. A regional on-call pool of "
          f"{sum(P.STAFF_POOL['doctors'].values())} doctors and {sum(P.STAFF_POOL['nurses'].values())} nurses reports for duty "
          f"{P.POOL_ARRIVAL_MIN[0]}-{P.POOL_ARRIVAL_MIN[1]} minutes after the first call and is contested by every hospital. "
          f"Surge bays equal {P.SURGE_BAY_FRAC:.0%} of regular bays and take {P.SURGE_ACTIVATION_MIN} minutes to open.",
          "Staff have a workload limit (patients at once) and a baseline load from the normal census. Unit-bound specialists "
          "always keep one free slot; generalists can be lent to another hospital.", ""]

    L += ["## Responder teams", ""]
    L += _table(["Type", "Capabilities", "People", "Speed (km/h)"],
                [[n, ", ".join(v["capabilities"]), v["size"], v["speed_kmh"]] for n, v in P.RESPONDER_TYPES.items()])
    L += ["Ambulances alternate ALS / BLS. ALS units can carry a RED patient, BLS units cannot. At most one RED patient per load.", ""]

    L += ["## Noisy first report", "",
          "The Incident Assessment Agent sees only what the caller says. Caller confidence is Beta(4, 2). "
          "The victim count is the truth times `1 + N(-0.10, 0.30 x (1 - confidence))`, clipped to 0.4-1.5, so callers under-count on average. "
          f"Callers cannot triage: each RED victim is reported as critical with probability "
          f"`{P.CRITICAL_DETECT_BASE} + {P.CRITICAL_DETECT_PER_CONFIDENCE} x confidence`, and each YELLOW victim is mistaken for "
          f"critical with probability `{P.CRITICAL_FALSE_ALARM} x (1 - confidence)`. Hazards and trapped people are mentioned with probability "
          "`0.70 + 0.30 x confidence`. The narrative is built from the reported facts only.", "",
          "## Simulation rules", "",
          f"These apply to every policy, baseline or agent. One tick per minute for {P.HORIZON_MIN} minutes.", "",
          f"- A unit leaving from rest starts driving {P.TURNOUT_MIN} min after dispatch. Loading takes {P.LOAD_MIN} min, and the "
          f"ambulance stays {P.HANDOFF_MIN} min at the ER after admission.",
          f"- Hazard scenes: victims reach the casualty collection point {P.HAZARD_ACCESS_MIN} min after a team with the matching "
          "capability starts work. Trapped victims wait for extrication. Teams work tasks in the order "
          + ", ".join(f"`{x}`" for x in P.TASK_ORDER) + "; two teams on one task finish it twice as fast.",
          f"- Stabilization extends the golden window once: RED +{P.STABILIZATION_BONUS_MIN['RED']} min, YELLOW "
          f"+{P.STABILIZATION_BONUS_MIN['YELLOW']} min. It comes from a finished triage task, or from an ALS crew starting to load a RED patient.",
          "- Only ALS units carry RED patients, at most one per load. Everyone in one ambulance must fit one hospital's capabilities.",
          "- Admission needs the capability, a free regular or open surge bay, a doctor slot with the matching skill and the nurse "
          "slots. Any doctor may take a general emergency. A YELLOW patient may be taken by an emergency-trained doctor and any "
          "nurse when no specialist is free; a RED patient always needs the specialist and a nurse with the matching skill.",
          f"- After {P.OFFLOAD_LIMIT_MIN} min at a full ER door the crew hands its patients over in the corridor and leaves. "
          "Corridor patients are admitted first when room frees up; their clock keeps running.",
          "- A patient who needs ICU keeps the ER bay after treatment if no ICU bed is free. The normal census stays constant.",
          f"- Staff lent between hospitals arrive after the drive plus {P.STAFF_TRANSFER_EXTRA_MIN} min. A pool clinician assigned after "
          f"coming on duty arrives {P.POOL_REPORT_DELAY_MIN} min later. Unit-bound specialists cannot be lent.",
          f"- Surge bays open {P.SURGE_ACTIVATION_MIN} min after the order. Idle units drive home after {P.IDLE_RETURN_MIN} min. "
          f"A loaded ambulance without a destination goes to the nearest capable hospital after {P.DESTINATION_GRACE_MIN} min.",
          f"- Every negotiation ends with a decision within {P.MAX_NEGOTIATION_ROUNDS} rounds.",
          "- Victims still waiting at the end of the horizon are scored as if care started then.", "",
          "## Reproducibility", "",
          "- One master seed. Every kind of entity draws from its own stream, `SeedSequence([seed, stream id])`, so adding a field to one "
          "entity never shifts the numbers of another.",
          "- The same seed gives the same map (hospital sites and stations) at every stress level and theme. Stress changes the load.",
          "- Output files contain no timestamps. Each folder has a `MANIFEST.json` with the SHA-256 of every file and the library versions. "
          "`python -m emergency_mas.datagen verify` regenerates the frozen scenarios and compares hashes.",
          "- Library versions are pinned in `requirements-lock.txt`; NumPy does not promise identical streams across major versions.", "",
          "## Known simplifications", "",
          "- Survival parameters are illustrative, not clinical.",
          "- Travel time uses zones and one congestion window, not a road graph.",
          "- ER bays free up only when a treatment time ends. ICU and ward beds do not free up during the horizon.",
          "- GREEN victims are transported like everyone else. Walking-wounded self-transport is not modeled.",
          ""]
    return "\n".join(L)
