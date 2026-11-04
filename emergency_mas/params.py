"""Modeling assumptions for the synthetic emergency-response world.

Every number the generator uses lives in this file so it can be audited, changed
and cited in the report. The values are plausible and internally consistent, and
they are tuned so that resources are contested. They are NOT calibrated to
clinical or operational data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

GENERATOR_VERSION = "0.2.0"   # 0.2.0: callers misjudge who is critical (binomial detection and false alarms)

# ---------------------------------------------------------------------------
# City and travel
# ---------------------------------------------------------------------------
CITY_KM = 24.0                    # square city, coordinates in km
CITY_CENTER = (12.0, 12.0)
ZONE_RADII_KM = (4.5, 9.0)        # core < 4.5 <= inner < 9.0 <= outer
ZONE_SPEED_FACTOR = {"core": 0.60, "inner": 0.80, "outer": 1.00}
CIRCUITY = 1.30                   # road km = straight-line km x 1.30
AMBULANCE_SPEED_KMH = 55.0        # emergency driving, before the zone factor
HORIZON_MIN = 240                 # simulated minutes per scenario
CONGESTION_WINDOW_MIN = 90        # the stress congestion multiplier applies to the first 90 min
MIN_HOSPITAL_SEPARATION_KM = 6.0

# Operational constants shared with the simulator (used here only to size the stress indices)
AMBULANCE_TURNOUT_MIN = 1.5       # dispatch to wheels rolling
SCENE_LOAD_MIN = 10.0             # loading time on scene
HOSPITAL_HANDOFF_MIN = 8.0        # hand-over at the ER door
SURGE_ACTIVATION_MIN = 15         # time to open surge bays

# ---------------------------------------------------------------------------
# Triage (START-style colors) and time sensitivity
# ---------------------------------------------------------------------------
TRIAGE = ("RED", "YELLOW", "GREEN")
TRIAGE_LABEL = {"RED": "Immediate", "YELLOW": "Delayed", "GREEN": "Minor"}
TRIAGE_WEIGHT = {"RED": 10.0, "YELLOW": 4.0, "GREEN": 1.0}   # value of treating a victim in time

# Minutes after the call by which definitive care must start (uniform in range)
GOLDEN_WINDOW_MIN = {"RED": (25, 60), "YELLOW": (90, 240), "GREEN": (300, 720)}
# Survival probability if care starts inside the window
SURVIVAL_IF_TIMELY = {"RED": 0.92, "YELLOW": 0.99, "GREEN": 1.00}
# After the window closes, survival halves every `half-life` minutes
DECAY_HALF_LIFE_MIN = {"RED": 25, "YELLOW": 120, "GREEN": 600}

# Hospital resources one victim consumes (slots) and how long they hold an ER bay
STAFF_NEED = {
    "RED": {"doctors": 1, "nurses": 2, "treatment_min": (90, 150)},
    "YELLOW": {"doctors": 1, "nurses": 1, "treatment_min": (45, 90)},
    "GREEN": {"doctors": 0, "nurses": 1, "treatment_min": (20, 40)},
}
AGE_GROUP_WEIGHTS = {"child": 0.12, "adult": 0.68, "elderly": 0.20}

# ---------------------------------------------------------------------------
# Injuries and hospital capabilities
# ---------------------------------------------------------------------------
CAPABILITIES = (
    "general_emergency", "trauma_surgery", "neurosurgery", "burn_unit",
    "cardiac_cath", "pediatric_emergency", "toxicology_decon", "critical_care",
)

# Capability a hospital must offer to take a victim, by injury type and triage color
INJURY_CAPABILITY = {
    "blunt_trauma": {"RED": "trauma_surgery", "YELLOW": "general_emergency", "GREEN": "general_emergency"},
    "head_injury": {"RED": "neurosurgery", "YELLOW": "general_emergency", "GREEN": "general_emergency"},
    "crush_injury": {"RED": "trauma_surgery", "YELLOW": "trauma_surgery", "GREEN": "general_emergency"},
    "burn": {"RED": "burn_unit", "YELLOW": "burn_unit", "GREEN": "general_emergency"},
    "smoke_inhalation": {"RED": "critical_care", "YELLOW": "general_emergency", "GREEN": "general_emergency"},
    "chemical_exposure": {"RED": "toxicology_decon", "YELLOW": "toxicology_decon", "GREEN": "general_emergency"},
    "cardiac_event": {"RED": "cardiac_cath", "YELLOW": "general_emergency", "GREEN": "general_emergency"},
    "minor_injury": {"RED": "general_emergency", "YELLOW": "general_emergency", "GREEN": "general_emergency"},
}
# Probability a RED victim needs an ICU bed, by injury; YELLOW and GREEN below
ICU_PROB_RED = {
    "blunt_trauma": 0.50, "head_injury": 0.80, "crush_injury": 0.60, "burn": 0.90,
    "smoke_inhalation": 0.80, "chemical_exposure": 0.70, "cardiac_event": 0.60, "minor_injury": 0.0,
}
ICU_PROB_YELLOW = 0.10
MINOR_SHARE_OF_GREEN = 0.85       # share of GREEN victims whose injury is just "minor_injury"

# ---------------------------------------------------------------------------
# Incident types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IncidentProfile:
    label: str
    victims: tuple[int, int, float]          # (min, max, Poisson mean) before the stress scale
    triage_mix: dict[str, float]
    injuries: dict[str, float]               # injury mix for non-minor victims
    trapped_frac: float                      # share of RED/YELLOW victims who are trapped
    hazards: dict[str, float]
    unsalvageable_mean: float                # Poisson mean of victims beyond help on arrival


INCIDENT_PROFILES = {
    "multi_vehicle_accident": IncidentProfile(
        "Multi-vehicle accident", (1, 7, 3.5),
        {"RED": 0.16, "YELLOW": 0.30, "GREEN": 0.54},
        {"blunt_trauma": 0.60, "head_injury": 0.25, "crush_injury": 0.15},
        0.25, {"none": 0.85, "fire": 0.15}, 0.10),
    "building_fire": IncidentProfile(
        "Building fire", (2, 9, 4.5),
        {"RED": 0.20, "YELLOW": 0.32, "GREEN": 0.48},
        {"burn": 0.45, "smoke_inhalation": 0.45, "blunt_trauma": 0.10},
        0.20, {"fire": 1.0}, 0.30),
    "structural_collapse": IncidentProfile(
        "Structural collapse", (3, 10, 5.5),
        {"RED": 0.26, "YELLOW": 0.34, "GREEN": 0.40},
        {"crush_injury": 0.55, "blunt_trauma": 0.30, "head_injury": 0.15},
        0.55, {"structural": 1.0}, 0.40),
    "hazmat_release": IncidentProfile(
        "Hazmat release", (2, 8, 4.0),
        {"RED": 0.13, "YELLOW": 0.32, "GREEN": 0.55},
        {"chemical_exposure": 0.85, "smoke_inhalation": 0.15},
        0.0, {"hazmat": 1.0}, 0.05),
    "crowd_incident": IncidentProfile(
        "Crowd incident", (2, 9, 4.5),
        {"RED": 0.12, "YELLOW": 0.30, "GREEN": 0.58},
        {"cardiac_event": 0.35, "blunt_trauma": 0.45, "head_injury": 0.20},
        0.0, {"none": 1.0}, 0.05),
}

# On-scene work (minutes, uniform in range) that responder teams must complete
RESPONDER_WORK_MIN = {
    "triage_stabilization": (10, 18),
    "extrication": (20, 40),
    "fire_suppression": (30, 60),
    "structural_rescue": (45, 90),
    "hazmat": (30, 50),
}

# How callers judge "critical": each RED victim is recognized with probability
# CRITICAL_DETECT_BASE + CRITICAL_DETECT_PER_CONFIDENCE x confidence, and each YELLOW
# victim is mistaken for critical with probability CRITICAL_FALSE_ALARM x (1 - confidence).
CRITICAL_DETECT_BASE = 0.60
CRITICAL_DETECT_PER_CONFIDENCE = 0.30
CRITICAL_FALSE_ALARM = 0.15

# ---------------------------------------------------------------------------
# Responder teams
# ---------------------------------------------------------------------------
RESPONDER_TYPES = {
    "fire_engine": {"capabilities": ("fire_suppression", "extrication", "triage_stabilization"), "size": 4, "speed_kmh": 45.0},
    "heavy_rescue": {"capabilities": ("structural_rescue", "extrication"), "size": 6, "speed_kmh": 40.0},
    "hazmat": {"capabilities": ("hazmat",), "size": 5, "speed_kmh": 40.0},
    "paramedic_qrv": {"capabilities": ("triage_stabilization",), "size": 2, "speed_kmh": 60.0},
}
RESPONDER_PATTERN = ("fire_engine", "paramedic_qrv", "heavy_rescue", "hazmat", "fire_engine", "paramedic_qrv")
AMBULANCE_PATTERN = ("ALS", "BLS", "ALS", "BLS", "ALS")

# ---------------------------------------------------------------------------
# Hospitals and medical staff
# ---------------------------------------------------------------------------
DOCTOR_SPECIALTIES = {
    "emergency_medicine": {"skills": ("general_emergency",), "limit": 3},
    "trauma_surgeon": {"skills": ("trauma_surgery", "general_emergency"), "limit": 2},
    "neurosurgeon": {"skills": ("neurosurgery",), "limit": 2},
    "burn_specialist": {"skills": ("burn_unit",), "limit": 2},
    "cardiologist": {"skills": ("cardiac_cath",), "limit": 2},
    "pediatrician": {"skills": ("pediatric_emergency", "general_emergency"), "limit": 2},
    "toxicologist": {"skills": ("toxicology_decon",), "limit": 2},
    "intensivist": {"skills": ("critical_care",), "limit": 2},
}
NURSE_SPECIALTIES = {
    "er_nurse": {"skills": ("general_emergency",), "limit": 3},
    "trauma_nurse": {"skills": ("general_emergency", "trauma_surgery"), "limit": 2},
    "critical_care_nurse": {"skills": ("critical_care", "burn_unit", "cardiac_cath", "neurosurgery", "toxicology_decon"), "limit": 2},
    "pediatric_nurse": {"skills": ("general_emergency", "pediatric_emergency"), "limit": 2},
}
# Generalists can be lent to another hospital; unit-bound specialists cannot
LENDABLE_SPECIALTIES = ("emergency_medicine", "intensivist", "pediatrician", "er_nurse", "trauma_nurse", "critical_care_nurse")
SHIFT_CHANGE_PROB = 0.10          # share of on-duty staff whose shift ends inside the horizon
SURGE_BAY_FRAC = 0.30             # surge bays as a share of regular ER bays

HOSPITAL_ARCHETYPES = (
    {
        "name": "Harborview Trauma Center", "level": 1, "zone": "core",
        "er_bays": 14, "icu_beds": 12, "ward_free": (10, 22),
        "capabilities": ("general_emergency", "trauma_surgery", "neurosurgery", "burn_unit",
                         "cardiac_cath", "toxicology_decon", "critical_care"),
        "doctors": {"emergency_medicine": 2, "trauma_surgeon": 2, "neurosurgeon": 1, "burn_specialist": 1,
                    "cardiologist": 1, "toxicologist": 1, "intensivist": 1},
        "nurses": {"er_nurse": 4, "trauma_nurse": 3, "critical_care_nurse": 4, "pediatric_nurse": 1},
        "reserve_red": 2, "diversion": 0.90, "lend": (1, 2),
    },
    {
        "name": "Eastgate Regional Medical Center", "level": 2, "zone": "inner",
        "er_bays": 10, "icu_beds": 8, "ward_free": (8, 16),
        "capabilities": ("general_emergency", "trauma_surgery", "cardiac_cath", "pediatric_emergency",
                         "toxicology_decon", "critical_care"),
        "doctors": {"emergency_medicine": 2, "trauma_surgeon": 1, "cardiologist": 1, "pediatrician": 1,
                    "toxicologist": 1, "intensivist": 1},
        "nurses": {"er_nurse": 3, "trauma_nurse": 2, "critical_care_nurse": 3, "pediatric_nurse": 1},
        "reserve_red": 1, "diversion": 0.85, "lend": (1, 3),
    },
    {
        "name": "Northfield Community Hospital", "level": 3, "zone": "outer",
        "er_bays": 6, "icu_beds": 3, "ward_free": (4, 10),
        "capabilities": ("general_emergency", "critical_care"),
        "doctors": {"emergency_medicine": 2, "intensivist": 1},
        "nurses": {"er_nurse": 4, "critical_care_nurse": 1},
        "reserve_red": 0, "diversion": 0.80, "lend": (1, 2),
    },
    {
        "name": "Westbrook Specialty Hospital", "level": 2, "zone": "inner",
        "er_bays": 8, "icu_beds": 6, "ward_free": (6, 12),
        "capabilities": ("general_emergency", "burn_unit", "neurosurgery", "pediatric_emergency", "critical_care"),
        "doctors": {"emergency_medicine": 2, "burn_specialist": 1, "neurosurgeon": 1, "pediatrician": 1, "intensivist": 1},
        "nurses": {"er_nurse": 3, "critical_care_nurse": 2, "pediatric_nurse": 1},
        "reserve_red": 1, "diversion": 0.85, "lend": (1, 2),
    },
    {
        "name": "Riverside Community Hospital", "level": 3, "zone": "outer",
        "er_bays": 6, "icu_beds": 3, "ward_free": (4, 10),
        "capabilities": ("general_emergency", "critical_care"),
        "doctors": {"emergency_medicine": 2, "intensivist": 1},
        "nurses": {"er_nurse": 4, "critical_care_nurse": 1},
        "reserve_red": 0, "diversion": 0.80, "lend": (1, 2),
    },
)
# Regional on-call pool that every hospital can ask for (contested)
STAFF_POOL = {
    "doctors": {"emergency_medicine": 1, "trauma_surgeon": 1, "intensivist": 1},
    "nurses": {"er_nurse": 3, "critical_care_nurse": 2, "trauma_nurse": 1},
}
POOL_ARRIVAL_MIN = (15, 45)       # the pool reports for duty 15-45 minutes after the first call
RECALL_DELAY_MIN = (20, 60)       # absent staff are recalled and arrive 20-60 minutes late

# ---------------------------------------------------------------------------
# Stress levels and themes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StressProfile:
    victim_scale: float          # multiplies the mean number of victims per incident
    red_boost: float             # relative increase of the RED probability
    er_occupancy: float          # mean ER bay occupancy at t=0
    icu_occupancy: float
    ward_free_frac: float        # position inside the archetype's free-ward-bed range
    ambulance_busy_prob: float   # an ambulance is still on an earlier call at t=0
    responder_busy_prob: float
    staff_absent_prob: float     # rostered staff who are not yet on duty at t=0
    congestion: float            # travel-time multiplier during the first CONGESTION_WINDOW_MIN


STRESS = {
    "low": StressProfile(0.7, 0.00, 0.40, 0.50, 1.0, 0.00, 0.00, 0.05, 1.00),
    "medium": StressProfile(1.0, 0.10, 0.60, 0.70, 0.6, 0.20, 0.10, 0.12, 1.15),
    "high": StressProfile(1.25, 0.25, 0.75, 0.82, 0.3, 0.40, 0.25, 0.20, 1.30),
}


@dataclass(frozen=True)
class ThemeProfile:
    description: str
    n_incidents: int
    type_weights: dict[str, float]
    n_hotspots: int
    cluster_frac: float          # share of incidents placed around a hotspot
    hotspot_sigma_km: float
    corridor: bool               # hotspots lie along one road
    arrival_window_min: int
    burst_frac: float            # share of incidents arriving in the opening burst
    burst_center_min: float
    burst_sd_min: float


THEMES = {
    "urban_mci": ThemeProfile(
        "Mixed urban mass-casualty day: crashes, fires, a collapse, a spill and a crowd event.",
        12, {"multi_vehicle_accident": 0.30, "building_fire": 0.22, "structural_collapse": 0.08,
             "hazmat_release": 0.15, "crowd_incident": 0.25},
        2, 0.40, 1.6, False, 90, 0.35, 15.0, 10.0),
    "earthquake": ThemeProfile(
        "Earthquake: collapses and fires cluster around an epicenter and arrive in a burst.",
        15, {"structural_collapse": 0.38, "building_fire": 0.24, "multi_vehicle_accident": 0.14,
             "hazmat_release": 0.10, "crowd_incident": 0.14},
        2, 0.70, 2.8, False, 60, 0.60, 10.0, 8.0),
    "highway_pileup": ThemeProfile(
        "Fog pile-up on one highway corridor plus a tanker spill: clustered crashes along a single road.",
        10, {"multi_vehicle_accident": 0.80, "hazmat_release": 0.20},
        3, 0.85, 0.9, True, 45, 0.50, 6.0, 6.0),
}

PRESETS = {
    "baseline": {"theme": "urban_mci", "stress": "medium"},
    "calm": {"theme": "urban_mci", "stress": "low"},
    "crunch": {"theme": "urban_mci", "stress": "high"},
    "surge": {"theme": "earthquake", "stress": "high"},
    "pileup": {"theme": "highway_pileup", "stress": "high"},
}

# Default fleet sizes (the assignment suggests 3 hospitals, 5 ambulances, 10-15 incidents)
DEFAULT_HOSPITALS = 3
DEFAULT_AMBULANCES = 5
DEFAULT_RESPONDERS = 6

# ---------------------------------------------------------------------------
# Simulation rules (whole minutes; the simulator ticks once per minute)
# ---------------------------------------------------------------------------
TURNOUT_MIN = math.ceil(AMBULANCE_TURNOUT_MIN)   # a unit leaving from rest starts driving 2 minutes after dispatch
LOAD_MIN = int(SCENE_LOAD_MIN)                   # loading patients on scene
HANDOFF_MIN = int(HOSPITAL_HANDOFF_MIN)          # ambulance stays at the ER door this long after admission
HAZARD_ACCESS_MIN = 10            # once a hazard team starts work, victims reach the casualty collection point 10 min later
STABILIZATION_BONUS_MIN = {"RED": 15, "YELLOW": 30, "GREEN": 0}   # on-scene stabilization extends the golden window
                                  # (given once, by a triage team finishing or by an ALS crew starting to load the patient)
TASK_ORDER = ("fire_suppression", "structural_rescue", "hazmat", "extrication", "triage_stabilization")
STAFF_TRANSFER_EXTRA_MIN = 5      # hand-over on top of the drive when staff move between hospitals
POOL_REPORT_DELAY_MIN = 10        # a pool clinician assigned after coming on duty reaches the hospital 10 min later
IDLE_RETURN_MIN = 5               # an idle unit away from its station drives home after 5 minutes
DESTINATION_GRACE_MIN = 1         # a loaded ambulance with no destination goes to the nearest capable hospital after 1 min
OFFLOAD_LIMIT_MIN = 30            # after 30 min at a full ER door the crew hands patients over in the corridor and leaves
YELLOW_GENERALIST_OK = True       # a YELLOW patient may be taken by an emergency-trained doctor and any nurse
                                  # when no specialist is free; RED patients always need the specialist
MAX_NEGOTIATION_ROUNDS = 3        # request, counter, decision

