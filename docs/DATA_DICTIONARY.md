# Data dictionary

Generated from `emergency_mas/schema.py` by `python -m emergency_mas.datagen docs`. Do not edit by hand.

Each scenario folder holds one CSV per entity, `scenario.json` (everything, nested) and `MANIFEST.json`
(counts, stress indices, validation results, SHA-256 of every file). List fields are joined with `|` in CSV.

| File | Rows |
|---|---|
| `stations.csv` | Stations that host ambulances and responder teams |
| `hospitals.csv` | One row per hospital; adds `er_bays_free` and `beds_available` |
| `staff.csv` | Doctors and nurses, including the regional on-call pool; adds `free_slots` |
| `ambulances.csv`, `responders.csv` | The fleet at t=0 |
| `incidents.csv` | One row per incident; `needs` is `capability:minutes xteams`, report fields are prefixed `report_` |
| `victims.csv` | One row per treatable victim, with the ground truth |
| `travel_times.csv` | `kind, from_id, to_id, road_km, minutes_t0` for ambulance/responder to incident, incident to hospital, hospital to hospital |

## Station



| Field | Type | Meaning |
|---|---|---|
| `station_id` | str | Station identifier, e.g. S1. Hosts ambulances and responder teams. |
| `name` | str | Station name. |
| `x` | float | Position in km from the west edge of the city. |
| `y` | float | Position in km from the south edge of the city. |
| `zone` | Literal['core', 'inner', 'outer'] | Congestion zone of the station. |

## Hospital



| Field | Type | Meaning |
|---|---|---|
| `hospital_id` | str | Hospital identifier, e.g. H1. |
| `name` | str | Hospital name. |
| `x` | float | Position in km from the west edge. |
| `y` | float | Position in km from the south edge. |
| `zone` | Literal['core', 'inner', 'outer'] | Congestion zone. |
| `trauma_level` | int | 1 = comprehensive trauma center, 3 = community hospital. |
| `capabilities` | list[Literal['general_emergency', 'trauma_surgery', 'neurosurgery', 'burn_unit', 'cardiac_cath', 'pediatric_emergency', 'toxicology_decon', 'critical_care']] | Treatment capabilities on offer. |
| `er_bays_total` | int | Emergency-room treatment bays. |
| `er_bays_occupied` | int | Bays already in use at t=0 (the normal census). |
| `surge_bays` | int | Extra bays that can open after the surge activation delay if staff allow. |
| `icu_beds_total` | int | ICU beds. |
| `icu_beds_free` | int | ICU beds free at t=0. |
| `ward_beds_free` | int | General ward beds free at t=0. |
| `reserve_bays_for_red` | int | Bays the hospital agent prefers to hold back for RED arrivals. |
| `diversion_threshold` | float | ER occupancy above which the hospital asks to divert YELLOW and GREEN patients. |
| `max_staff_lend_doctors` | int | Most doctors the hospital will lend at one time. |
| `max_staff_lend_nurses` | int | Most nurses the hospital will lend at one time. |

## StaffMember



| Field | Type | Meaning |
|---|---|---|
| `staff_id` | str | D## for doctors, N## for nurses. |
| `name` | str | Synthetic name (Faker, seeded). |
| `role` | Literal['doctor', 'nurse'] | Doctor or nurse. |
| `specialty` | str | Specialty, e.g. trauma_surgeon or er_nurse. |
| `skills` | list[Literal['general_emergency', 'trauma_surgery', 'neurosurgery', 'burn_unit', 'cardiac_cath', 'pediatric_emergency', 'toxicology_decon', 'critical_care']] | Capabilities this person can support. |
| `home_hospital` | str | Hospital id, or POOL for the regional on-call pool. |
| `on_duty_from_min` | int | Minute the person starts working (0 = already on duty). |
| `on_duty_until_min` | int | Minute the shift ends. |
| `workload_limit` | int | Most patients handled at once. |
| `baseline_load` | int | Patients already assigned at t=0 (normal census). |
| `lendable` | bool | Can be lent to another hospital. Unit-bound specialists cannot. |

## Ambulance



| Field | Type | Meaning |
|---|---|---|
| `ambulance_id` | str | Ambulance identifier, e.g. A1. |
| `unit_type` | Literal['ALS', 'BLS'] | ALS (advanced life support) can carry RED patients; BLS cannot. |
| `base_station_id` | str | Home station. |
| `x` | float | Position at t=0, km from the west edge. |
| `y` | float | Position at t=0, km from the south edge. |
| `capacity` | int | Stretchers. At most one RED patient per load. |
| `status` | Literal['available', 'busy'] | busy = still finishing an earlier call at t=0. |
| `available_from_min` | int | Minute the unit can take a new call. |
| `crew_size` | int | Crew members aboard. |
| `speed_kmh` | float | Emergency driving speed before the zone and congestion factors. |

## ResponderTeam



| Field | Type | Meaning |
|---|---|---|
| `team_id` | str | Responder team identifier, e.g. R1. |
| `team_type` | Literal['fire_engine', 'heavy_rescue', 'hazmat', 'paramedic_qrv'] | Kind of team. |
| `station_id` | str | Home station. |
| `x` | float | Position at t=0, km from the west edge. |
| `y` | float | Position at t=0, km from the south edge. |
| `capabilities` | list[str] | On-scene capabilities. |
| `team_size` | int | People on the team. |
| `speed_kmh` | float | Driving speed before the zone and congestion factors. |
| `status` | Literal['available', 'busy'] | busy = still on an earlier job at t=0. |
| `available_from_min` | int | Minute the team can take a new job. |

## Incident



| Field | Type | Meaning |
|---|---|---|
| `incident_id` | str | INC-## in order of arrival. |
| `incident_type` | Literal['multi_vehicle_accident', 'building_fire', 'structural_collapse', 'hazmat_release', 'crowd_incident'] | Kind of incident. |
| `x` | float | Position in km from the west edge. |
| `y` | float | Position in km from the south edge. |
| `zone` | Literal['core', 'inner', 'outer'] | Congestion zone. |
| `quadrant` | Literal['NE', 'NW', 'SE', 'SW'] | Quadrant of the city. |
| `report_time_min` | int | Minute the incident is reported. |
| `hazard` | Literal['none', 'fire', 'structural', 'hazmat'] | True on-scene hazard. |
| `n_victims` | int | True number of treatable victims. |
| `n_red` | int | True RED (immediate) victims. |
| `n_yellow` | int | True YELLOW (delayed) victims. |
| `n_green` | int | True GREEN (minor) victims. |
| `n_trapped` | int | Victims who cannot be loaded until extricated. |
| `unsalvageable_on_scene` | int | People beyond help on arrival. Not part of the allocation problem. |
| `min_golden_window_min` | int | Shortest golden window among the victims, minutes after the call. |
| `urgency_truth` | float | Oracle urgency (sum of triage weight x 60 / window). Hidden from agents; scores the assessment. |
| `needs` | list[emergency_mas.schema.ResponderNeed] | On-scene work responder teams must do. |
| `report` | emergency_mas.schema.IncidentReport | The caller's noisy report. |

## IncidentReport

What the first caller says. Noisy and incomplete; the Incident Assessment Agent sees this, not the truth.

| Field | Type | Meaning |
|---|---|---|
| `report_time_min` | int | Minute the call arrives. |
| `narrative` | str | Caller's account in plain language (generated from the reported facts only). |
| `reported_victims` | int | Caller's count of people hurt. Tends to under-count. |
| `reported_critical` | int | Caller's count of people who look critical. |
| `reported_hazard` | Union[Literal['none', 'fire', 'structural', 'hazmat'], Literal['unknown']] | Hazard the caller mentions. 'unknown' if not mentioned. |
| `reported_trapped` | bool | Caller says someone is trapped. |
| `confidence` | float | How reliable the caller is. Lower means noisier counts. |

## ResponderNeed



| Field | Type | Meaning |
|---|---|---|
| `capability` | str | On-scene capability required. |
| `work_min` | int | Minutes of work once a capable team is on scene. |
| `min_teams` | int | Teams required at the same time. |

## Victim



| Field | Type | Meaning |
|---|---|---|
| `victim_id` | str | INC-##-V# victim identifier. |
| `incident_id` | str | Incident the victim belongs to. |
| `triage` | Literal['RED', 'YELLOW', 'GREEN'] | True triage color. Hidden from agents until responders assess on scene. |
| `age_group` | Literal['child', 'adult', 'elderly'] | Age group. |
| `injury_type` | str | Main injury. |
| `required_capability` | Literal['general_emergency', 'trauma_surgery', 'neurosurgery', 'burn_unit', 'cardiac_cath', 'pediatric_emergency', 'toxicology_decon', 'critical_care'] | Capability the receiving hospital must offer. |
| `needs_icu` | bool | Needs an ICU bed after the ER. |
| `trapped` | bool | Cannot be loaded until extrication finishes. |
| `golden_window_min` | int | Minutes after the call by which definitive care must start. |
| `survival_if_timely` | float | Survival probability if care starts inside the window. |
| `decay_half_life_min` | int | After the window closes, survival halves every this many minutes. |
| `doctor_slots` | int | Doctor workload slots the victim uses in the ER. |
| `nurse_slots` | int | Nurse workload slots the victim uses in the ER. |
| `treatment_min` | int | Minutes the victim holds an ER bay. |

## ScenarioMeta



| Field | Type | Meaning |
|---|---|---|
| `scenario_id` | str | theme-stress-seed identifier. |
| `theme` | str | Scenario theme. |
| `stress` | str | Stress level: low, medium or high. |
| `seed` | int | Master seed. |
| `generator_version` | str | Generator version. |
| `horizon_min` | int | Simulated minutes. |
| `city_km` | float | Side of the square city in km. |
| `congestion_profile` | list[tuple[float, float, float]] | (start min, end min, travel-time multiplier). |
| `description` | str | One-line description of the theme. |
