"""Incidents, their victims and the noisy first report."""
from __future__ import annotations

from faker import Faker

from .. import params as P
from ..domain import Point, quadrant_of, urgency_truth, zone_of
from ..rng import Rng, make_rng
from ..schema import Incident, IncidentReport, ResponderNeed, Victim
from .geography import incident_location
from .narrative import Narrator


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def arrival_times(rng: Rng, theme: P.ThemeProfile, n: int) -> list[int]:
    """Report times in whole minutes, sorted, with the first call at minute 0."""
    raw: list[int] = []
    for _ in range(n):
        if rng.bernoulli(theme.burst_frac):
            t = rng.normal(theme.burst_center_min, theme.burst_sd_min)
        else:
            t = rng.uniform(0.0, theme.arrival_window_min)
        raw.append(max(0, int(round(t))))
    raw.sort()
    return [t - raw[0] for t in raw]


def make_victim(rng: Rng, prof: P.IncidentProfile, stress: P.StressProfile, incident_id: str, number: int) -> Victim:
    mix = dict(prof.triage_mix)
    mix["RED"] *= 1.0 + stress.red_boost
    triage = rng.weighted(mix)
    age_group = rng.weighted(P.AGE_GROUP_WEIGHTS)
    if triage == "GREEN" and rng.bernoulli(P.MINOR_SHARE_OF_GREEN):
        injury = "minor_injury"
    else:
        injury = rng.weighted(prof.injuries)
    if triage == "RED":
        needs_icu = rng.bernoulli(P.ICU_PROB_RED[injury])
    elif triage == "YELLOW":
        needs_icu = rng.bernoulli(P.ICU_PROB_YELLOW)
    else:
        needs_icu = False
    trapped = triage != "GREEN" and rng.bernoulli(prof.trapped_frac)
    need = P.STAFF_NEED[triage]
    return Victim(
        victim_id=f"{incident_id}-V{number}", incident_id=incident_id, triage=triage, age_group=age_group,
        injury_type=injury, required_capability=P.INJURY_CAPABILITY[injury][triage], needs_icu=needs_icu,
        trapped=trapped, golden_window_min=rng.integers(*P.GOLDEN_WINDOW_MIN[triage]),
        survival_if_timely=P.SURVIVAL_IF_TIMELY[triage], decay_half_life_min=P.DECAY_HALF_LIFE_MIN[triage],
        doctor_slots=need["doctors"], nurse_slots=need["nurses"], treatment_min=rng.integers(*need["treatment_min"]),
    )


def make_needs(rng: Rng, hazard: str, n_trapped: int) -> list[ResponderNeed]:
    needs = [ResponderNeed(capability="triage_stabilization",
                           work_min=rng.integers(*P.RESPONDER_WORK_MIN["triage_stabilization"]), min_teams=1)]
    if n_trapped > 0:
        work = rng.integers(*P.RESPONDER_WORK_MIN["extrication"]) + 5 * (n_trapped - 1)
        needs.append(ResponderNeed(capability="extrication", work_min=work, min_teams=2 if n_trapped >= 4 else 1))
    hazard_capability = {"fire": "fire_suppression", "structural": "structural_rescue", "hazmat": "hazmat"}.get(hazard)
    if hazard_capability:
        needs.append(ResponderNeed(capability=hazard_capability,
                                   work_min=rng.integers(*P.RESPONDER_WORK_MIN[hazard_capability]), min_teams=1))
    return needs


def make_report(rng: Rng, fake: Faker, narrator: Narrator, incident_type: str, report_time: int, hazard: str,
                n_victims: int, n_red: int, n_yellow: int, n_trapped: int) -> IncidentReport:
    """Callers under-count victims on average and cannot triage: they miss some critical
    patients and mistake some delayed ones for critical. Noise grows as confidence falls."""
    confidence = round(rng.beta(4.0, 2.0), 3)
    count_factor = _clip(1.0 + rng.normal(-0.10, 0.30 * (1.0 - confidence)), 0.4, 1.5)
    reported_victims = max(1, round(n_victims * count_factor))
    detect = P.CRITICAL_DETECT_BASE + P.CRITICAL_DETECT_PER_CONFIDENCE * confidence
    false_alarm = P.CRITICAL_FALSE_ALARM * (1.0 - confidence)
    reported_critical = min(reported_victims, rng.binomial(n_red, detect) + rng.binomial(n_yellow, false_alarm))
    p_mentioned = 0.70 + 0.30 * confidence
    if hazard == "none":
        reported_hazard = "none"
    else:
        reported_hazard = hazard if rng.bernoulli(p_mentioned) else "unknown"
    reported_trapped = n_trapped > 0 and rng.bernoulli(p_mentioned)
    narrative = narrator.describe(
        rng, fake, incident_type, reported_victims=reported_victims, reported_critical=reported_critical,
        reported_hazard=reported_hazard, reported_trapped=reported_trapped, confidence=confidence)
    return IncidentReport(
        report_time_min=report_time, narrative=narrative, reported_victims=reported_victims,
        reported_critical=reported_critical, reported_hazard=reported_hazard,
        reported_trapped=reported_trapped, confidence=confidence)


def generate_demand(*, seed: int, theme_name: str, stress_name: str, n_incidents: int,
                    hotspots: list[Point], fake: Faker, narrator: Narrator) -> tuple[list[Incident], list[Victim]]:
    theme, stress = P.THEMES[theme_name], P.STRESS[stress_name]
    r_inc, r_vic, r_rep = (make_rng(seed, s) for s in ("incidents", "victims", "reports"))
    incidents: list[Incident] = []
    victims: list[Victim] = []
    for number, report_time in enumerate(arrival_times(r_inc, theme, n_incidents), start=1):
        incident_id = f"INC-{number:02d}"
        incident_type = r_inc.weighted(theme.type_weights)
        prof = P.INCIDENT_PROFILES[incident_type]
        x, y = incident_location(r_inc, theme.cluster_frac, theme.hotspot_sigma_km, hotspots)
        hazard = r_inc.weighted(prof.hazards)
        lo, hi, mean = prof.victims
        n_victims = max(lo, min(round(hi * stress.victim_scale), r_inc.poisson(mean * stress.victim_scale)))

        group = [make_victim(r_vic, prof, stress, incident_id, k) for k in range(1, n_victims + 1)]
        n_trapped = sum(v.trapped for v in group)
        n_red = sum(v.triage == "RED" for v in group)
        incidents.append(Incident(
            incident_id=incident_id, incident_type=incident_type, x=x, y=y, zone=zone_of(x, y),
            quadrant=quadrant_of(x, y), report_time_min=report_time, hazard=hazard, n_victims=n_victims,
            n_red=n_red, n_yellow=sum(v.triage == "YELLOW" for v in group),
            n_green=sum(v.triage == "GREEN" for v in group), n_trapped=n_trapped,
            unsalvageable_on_scene=r_inc.poisson(prof.unsalvageable_mean),
            min_golden_window_min=min(v.golden_window_min for v in group),
            urgency_truth=round(urgency_truth((v.triage, v.golden_window_min) for v in group), 3),
            needs=make_needs(r_inc, hazard, n_trapped),
            report=make_report(r_rep, fake, narrator, incident_type, report_time, hazard, n_victims, n_red,
                               sum(v.triage == "YELLOW" for v in group), n_trapped),
        ))
        victims.extend(group)
    return incidents, victims
