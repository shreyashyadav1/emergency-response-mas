"""Typed records for the synthetic emergency-response world.

Field descriptions double as the data dictionary (docs/DATA_DICTIONARY.md is
generated from them), so every field says what it means and its unit.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Zone = Literal["core", "inner", "outer"]
Triage = Literal["RED", "YELLOW", "GREEN"]
Capability = Literal[
    "general_emergency", "trauma_surgery", "neurosurgery", "burn_unit",
    "cardiac_cath", "pediatric_emergency", "toxicology_decon", "critical_care",
]
Hazard = Literal["none", "fire", "structural", "hazmat"]
IncidentType = Literal[
    "multi_vehicle_accident", "building_fire", "structural_collapse", "hazmat_release", "crowd_incident",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Station(_Model):
    station_id: str = Field(description="Station identifier, e.g. S1. Hosts ambulances and responder teams.")
    name: str = Field(description="Station name.")
    x: float = Field(description="Position in km from the west edge of the city.")
    y: float = Field(description="Position in km from the south edge of the city.")
    zone: Zone = Field(description="Congestion zone of the station.")


class Hospital(_Model):
    hospital_id: str = Field(description="Hospital identifier, e.g. H1.")
    name: str = Field(description="Hospital name.")
    x: float = Field(description="Position in km from the west edge.")
    y: float = Field(description="Position in km from the south edge.")
    zone: Zone = Field(description="Congestion zone.")
    trauma_level: int = Field(ge=1, le=3, description="1 = comprehensive trauma center, 3 = community hospital.")
    capabilities: list[Capability] = Field(description="Treatment capabilities on offer.")
    er_bays_total: int = Field(ge=0, description="Emergency-room treatment bays.")
    er_bays_occupied: int = Field(ge=0, description="Bays already in use at t=0 (the normal census).")
    surge_bays: int = Field(ge=0, description="Extra bays that can open after the surge activation delay if staff allow.")
    icu_beds_total: int = Field(ge=0, description="ICU beds.")
    icu_beds_free: int = Field(ge=0, description="ICU beds free at t=0.")
    ward_beds_free: int = Field(ge=0, description="General ward beds free at t=0.")
    reserve_bays_for_red: int = Field(ge=0, description="Bays the hospital agent prefers to hold back for RED arrivals.")
    diversion_threshold: float = Field(ge=0, le=1, description="ER occupancy above which the hospital asks to divert YELLOW and GREEN patients.")
    max_staff_lend_doctors: int = Field(ge=0, description="Most doctors the hospital will lend at one time.")
    max_staff_lend_nurses: int = Field(ge=0, description="Most nurses the hospital will lend at one time.")

    @property
    def er_bays_free(self) -> int:
        return self.er_bays_total - self.er_bays_occupied

    @property
    def beds_available(self) -> int:
        return self.icu_beds_free + self.ward_beds_free


class StaffMember(_Model):
    staff_id: str = Field(description="D## for doctors, N## for nurses.")
    name: str = Field(description="Synthetic name (Faker, seeded).")
    role: Literal["doctor", "nurse"] = Field(description="Doctor or nurse.")
    specialty: str = Field(description="Specialty, e.g. trauma_surgeon or er_nurse.")
    skills: list[Capability] = Field(description="Capabilities this person can support.")
    home_hospital: str = Field(description="Hospital id, or POOL for the regional on-call pool.")
    on_duty_from_min: int = Field(ge=0, description="Minute the person starts working (0 = already on duty).")
    on_duty_until_min: int = Field(ge=0, description="Minute the shift ends.")
    workload_limit: int = Field(ge=1, description="Most patients handled at once.")
    baseline_load: int = Field(ge=0, description="Patients already assigned at t=0 (normal census).")
    lendable: bool = Field(description="Can be lent to another hospital. Unit-bound specialists cannot.")

    @property
    def free_slots(self) -> int:
        return max(0, self.workload_limit - self.baseline_load)


class Ambulance(_Model):
    ambulance_id: str = Field(description="Ambulance identifier, e.g. A1.")
    unit_type: Literal["ALS", "BLS"] = Field(description="ALS (advanced life support) can carry RED patients; BLS cannot.")
    base_station_id: str = Field(description="Home station.")
    x: float = Field(description="Position at t=0, km from the west edge.")
    y: float = Field(description="Position at t=0, km from the south edge.")
    capacity: int = Field(ge=1, le=2, description="Stretchers. At most one RED patient per load.")
    status: Literal["available", "busy"] = Field(description="busy = still finishing an earlier call at t=0.")
    available_from_min: int = Field(ge=0, description="Minute the unit can take a new call.")
    crew_size: int = Field(ge=2, description="Crew members aboard.")
    speed_kmh: float = Field(gt=0, description="Emergency driving speed before the zone and congestion factors.")


class ResponderTeam(_Model):
    team_id: str = Field(description="Responder team identifier, e.g. R1.")
    team_type: Literal["fire_engine", "heavy_rescue", "hazmat", "paramedic_qrv"] = Field(description="Kind of team.")
    station_id: str = Field(description="Home station.")
    x: float = Field(description="Position at t=0, km from the west edge.")
    y: float = Field(description="Position at t=0, km from the south edge.")
    capabilities: list[str] = Field(description="On-scene capabilities.")
    team_size: int = Field(ge=1, description="People on the team.")
    speed_kmh: float = Field(gt=0, description="Driving speed before the zone and congestion factors.")
    status: Literal["available", "busy"] = Field(description="busy = still on an earlier job at t=0.")
    available_from_min: int = Field(ge=0, description="Minute the team can take a new job.")


class ResponderNeed(_Model):
    capability: str = Field(description="On-scene capability required.")
    work_min: int = Field(ge=1, description="Minutes of work once a capable team is on scene.")
    min_teams: int = Field(ge=1, description="Teams required at the same time.")


class IncidentReport(_Model):
    """What the first caller says. Noisy and incomplete; the Incident Assessment Agent sees this, not the truth."""

    report_time_min: int = Field(ge=0, description="Minute the call arrives.")
    narrative: str = Field(description="Caller's account in plain language (generated from the reported facts only).")
    reported_victims: int = Field(ge=0, description="Caller's count of people hurt. Tends to under-count.")
    reported_critical: int = Field(ge=0, description="Caller's count of people who look critical.")
    reported_hazard: Hazard | Literal["unknown"] = Field(description="Hazard the caller mentions. 'unknown' if not mentioned.")
    reported_trapped: bool = Field(description="Caller says someone is trapped.")
    confidence: float = Field(ge=0, le=1, description="How reliable the caller is. Lower means noisier counts.")


class Incident(_Model):
    incident_id: str = Field(description="INC-## in order of arrival.")
    incident_type: IncidentType = Field(description="Kind of incident.")
    x: float = Field(description="Position in km from the west edge.")
    y: float = Field(description="Position in km from the south edge.")
    zone: Zone = Field(description="Congestion zone.")
    quadrant: Literal["NE", "NW", "SE", "SW"] = Field(description="Quadrant of the city.")
    report_time_min: int = Field(ge=0, description="Minute the incident is reported.")
    hazard: Hazard = Field(description="True on-scene hazard.")
    n_victims: int = Field(ge=0, description="True number of treatable victims.")
    n_red: int = Field(ge=0, description="True RED (immediate) victims.")
    n_yellow: int = Field(ge=0, description="True YELLOW (delayed) victims.")
    n_green: int = Field(ge=0, description="True GREEN (minor) victims.")
    n_trapped: int = Field(ge=0, description="Victims who cannot be loaded until extricated.")
    unsalvageable_on_scene: int = Field(ge=0, description="People beyond help on arrival. Not part of the allocation problem.")
    min_golden_window_min: int = Field(ge=0, description="Shortest golden window among the victims, minutes after the call.")
    urgency_truth: float = Field(ge=0, description="Oracle urgency (sum of triage weight x 60 / window). Hidden from agents; scores the assessment.")
    needs: list[ResponderNeed] = Field(description="On-scene work responder teams must do.")
    report: IncidentReport = Field(description="The caller's noisy report.")


class Victim(_Model):
    victim_id: str = Field(description="INC-##-V# victim identifier.")
    incident_id: str = Field(description="Incident the victim belongs to.")
    triage: Triage = Field(description="True triage color. Hidden from agents until responders assess on scene.")
    age_group: Literal["child", "adult", "elderly"] = Field(description="Age group.")
    injury_type: str = Field(description="Main injury.")
    required_capability: Capability = Field(description="Capability the receiving hospital must offer.")
    needs_icu: bool = Field(description="Needs an ICU bed after the ER.")
    trapped: bool = Field(description="Cannot be loaded until extrication finishes.")
    golden_window_min: int = Field(ge=1, description="Minutes after the call by which definitive care must start.")
    survival_if_timely: float = Field(ge=0, le=1, description="Survival probability if care starts inside the window.")
    decay_half_life_min: int = Field(ge=1, description="After the window closes, survival halves every this many minutes.")
    doctor_slots: int = Field(ge=0, description="Doctor workload slots the victim uses in the ER.")
    nurse_slots: int = Field(ge=0, description="Nurse workload slots the victim uses in the ER.")
    treatment_min: int = Field(ge=1, description="Minutes the victim holds an ER bay.")


class ScenarioMeta(_Model):
    scenario_id: str = Field(description="theme-stress-seed identifier.")
    theme: str = Field(description="Scenario theme.")
    stress: str = Field(description="Stress level: low, medium or high.")
    seed: int = Field(description="Master seed.")
    generator_version: str = Field(description="Generator version.")
    horizon_min: int = Field(description="Simulated minutes.")
    city_km: float = Field(description="Side of the square city in km.")
    congestion_profile: list[tuple[float, float, float]] = Field(description="(start min, end min, travel-time multiplier).")
    description: str = Field(description="One-line description of the theme.")


class Scenario(_Model):
    meta: ScenarioMeta
    stations: list[Station]
    hospitals: list[Hospital]
    staff: list[StaffMember]
    ambulances: list[Ambulance]
    responders: list[ResponderTeam]
    incidents: list[Incident]
    victims: list[Victim]

    def hospital(self, hospital_id: str) -> Hospital:
        return next(h for h in self.hospitals if h.hospital_id == hospital_id)

    def incident(self, incident_id: str) -> Incident:
        return next(i for i in self.incidents if i.incident_id == incident_id)

    def victims_of(self, incident_id: str) -> list[Victim]:
        return [v for v in self.victims if v.incident_id == incident_id]

    def staff_of(self, hospital_id: str) -> list[StaffMember]:
        return [s for s in self.staff if s.home_hospital == hospital_id]
