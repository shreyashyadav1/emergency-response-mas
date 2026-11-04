# Modeling assumptions

Generated from `emergency_mas/params.py` by `python -m emergency_mas.datagen docs`.
The numbers are plausible and internally consistent. They are **not** calibrated to clinical or operational data.
They are tuned so that ambulances, hospital beds and specialist staff are contested.

## City and travel

- Square city, 24 km on a side, coordinates in km. Three congestion zones around the center:
  core (< 4.5 km), inner (< 9 km), outer.
- Road distance = straight-line distance x 1.3. Ambulances drive 55 km/h times a zone factor (core 0.6, inner 0.8, outer 1; the mean of the two end zones is used).
- A congestion multiplier (see stress levels) lengthens every trip that departs in the first 90 minutes.
- Each scenario covers 240 simulated minutes. The first call arrives at minute 0.

## Triage and time sensitivity

Victims are tagged RED (immediate), YELLOW (delayed) or GREEN (minor). People beyond help on arrival are counted on the incident but are not part of the allocation problem.

| Triage | Weight | Golden window (min) | Survival if timely | Half-life after window (min) | Doctor slots | Nurse slots | ER stay (min) |
|---|---|---|---|---|---|---|---|
| RED | 10.0 | 25-60 | 0.92 | 25 | 1 | 2 | 90-150 |
| YELLOW | 4.0 | 90-240 | 0.99 | 120 | 1 | 1 | 45-90 |
| GREEN | 1.0 | 300-720 | 1.0 | 600 | 0 | 1 | 20-40 |

Survival is flat while care starts inside the golden window, then halves every half-life: `p = p0 * 0.5 ** ((t - window) / half_life)`. This is the outcome the evaluation maximizes. The weights set how much each color counts in the oracle urgency score (weight x 60 / window) and in the severity-weighted survival metric.

## Stress levels

| Level | Victim scale | RED boost | ER occupancy | ICU occupancy | Ambulance busy at t=0 | Responder busy at t=0 | Staff not yet on duty | Congestion x |
|---|---|---|---|---|---|---|---|---|
| low | 0.7 | 0.0 | 0.4 | 0.5 | 0.0 | 0.0 | 0.05 | 1.0 |
| medium | 1.0 | 0.1 | 0.6 | 0.7 | 0.2 | 0.1 | 0.12 | 1.15 |
| high | 1.25 | 0.25 | 0.75 | 0.82 | 0.4 | 0.25 | 0.2 | 1.3 |

## Themes

| Theme | Incidents | Arrival window (min) | Burst share | Clustered share | Hotspot spread (km) | Mix |
|---|---|---|---|---|---|---|
| urban_mci | 12 | 90 | 0.35 | 0.4 | 1.6 | multi_vehicle_accident 0.3, building_fire 0.22, structural_collapse 0.08, hazmat_release 0.15, crowd_incident 0.25 |
| earthquake | 15 | 60 | 0.6 | 0.7 | 2.8 | structural_collapse 0.38, building_fire 0.24, multi_vehicle_accident 0.14, hazmat_release 0.1, crowd_incident 0.14 |
| highway_pileup | 10 | 45 | 0.5 | 0.85 | 0.9 | multi_vehicle_accident 0.8, hazmat_release 0.2 |

Presets: `baseline` = urban_mci / medium, `calm` = urban_mci / low, `crunch` = urban_mci / high, `surge` = earthquake / high, `pileup` = highway_pileup / high. Default fleet: 3 hospitals, 5 ambulances, 6 responder teams.

## Incident types

| Type | Victims (min-max, mean) | RED / YELLOW / GREEN | Injuries (non-minor) | Trapped share | Hazard |
|---|---|---|---|---|---|
| multi_vehicle_accident | 1-7, 3.5 | 0.16 / 0.3 / 0.54 | blunt_trauma 0.6, head_injury 0.25, crush_injury 0.15 | 0.25 | none 0.85, fire 0.15 |
| building_fire | 2-9, 4.5 | 0.2 / 0.32 / 0.48 | burn 0.45, smoke_inhalation 0.45, blunt_trauma 0.1 | 0.2 | fire 1 |
| structural_collapse | 3-10, 5.5 | 0.26 / 0.34 / 0.4 | crush_injury 0.55, blunt_trauma 0.3, head_injury 0.15 | 0.55 | structural 1 |
| hazmat_release | 2-8, 4 | 0.13 / 0.32 / 0.55 | chemical_exposure 0.85, smoke_inhalation 0.15 | 0.0 | hazmat 1 |
| crowd_incident | 2-9, 4.5 | 0.12 / 0.3 / 0.58 | cardiac_event 0.35, blunt_trauma 0.45, head_injury 0.2 | 0.0 | none 1 |

## Hospitals

| Archetype | Level | ER bays | ICU beds | Doctors | Nurses | Capabilities | Reserve for RED | Diversion at |
|---|---|---|---|---|---|---|---|---|
| Harborview Trauma Center | 1 | 14 | 12 | 9 | 12 | general_emergency, trauma_surgery, neurosurgery, burn_unit, cardiac_cath, toxicology_decon, critical_care | 2 | 0.9 |
| Eastgate Regional Medical Center | 2 | 10 | 8 | 7 | 9 | general_emergency, trauma_surgery, cardiac_cath, pediatric_emergency, toxicology_decon, critical_care | 1 | 0.85 |
| Northfield Community Hospital | 3 | 6 | 3 | 3 | 5 | general_emergency, critical_care | 0 | 0.8 |

The first three archetypes are the default hospitals. A regional on-call pool of 3 doctors and 6 nurses reports for duty 15-45 minutes after the first call and is contested by every hospital. Surge bays equal 30% of regular bays and take 15 minutes to open.
Staff have a workload limit (patients at once) and a baseline load from the normal census. Unit-bound specialists always keep one free slot; generalists can be lent to another hospital.

## Responder teams

| Type | Capabilities | People | Speed (km/h) |
|---|---|---|---|
| fire_engine | fire_suppression, extrication, triage_stabilization | 4 | 45.0 |
| heavy_rescue | structural_rescue, extrication | 6 | 40.0 |
| hazmat | hazmat | 5 | 40.0 |
| paramedic_qrv | triage_stabilization | 2 | 60.0 |

Ambulances alternate ALS / BLS. ALS units can carry a RED patient, BLS units cannot. At most one RED patient per load.

## Noisy first report

The Incident Assessment Agent sees only what the caller says. Caller confidence is Beta(4, 2). The victim count is the truth times `1 + N(-0.10, 0.30 x (1 - confidence))`, clipped to 0.4-1.5, so callers under-count on average. Callers cannot triage: each RED victim is reported as critical with probability `0.6 + 0.3 x confidence`, and each YELLOW victim is mistaken for critical with probability `0.15 x (1 - confidence)`. Hazards and trapped people are mentioned with probability `0.70 + 0.30 x confidence`. The narrative is built from the reported facts only.

## Simulation rules

These apply to every policy, baseline or agent. One tick per minute for 240 minutes.

- A unit leaving from rest starts driving 2 min after dispatch. Loading takes 10 min, and the ambulance stays 8 min at the ER after admission.
- Hazard scenes: victims reach the casualty collection point 10 min after a team with the matching capability starts work. Trapped victims wait for extrication. Teams work tasks in the order `fire_suppression`, `structural_rescue`, `hazmat`, `extrication`, `triage_stabilization`; two teams on one task finish it twice as fast.
- Stabilization extends the golden window once: RED +15 min, YELLOW +30 min. It comes from a finished triage task, or from an ALS crew starting to load a RED patient.
- Only ALS units carry RED patients, at most one per load. Everyone in one ambulance must fit one hospital's capabilities.
- Admission needs the capability, a free regular or open surge bay, a doctor slot with the matching skill and the nurse slots. Any doctor may take a general emergency. A YELLOW patient may be taken by an emergency-trained doctor and any nurse when no specialist is free; a RED patient always needs the specialist and a nurse with the matching skill.
- After 30 min at a full ER door the crew hands its patients over in the corridor and leaves. Corridor patients are admitted first when room frees up; their clock keeps running.
- A patient who needs ICU keeps the ER bay after treatment if no ICU bed is free. The normal census stays constant.
- Staff lent between hospitals arrive after the drive plus 5 min. A pool clinician assigned after coming on duty arrives 10 min later. Unit-bound specialists cannot be lent.
- Surge bays open 15 min after the order. Idle units drive home after 5 min. A loaded ambulance without a destination goes to the nearest capable hospital after 1 min.
- Every negotiation ends with a decision within 3 rounds.
- Victims still waiting at the end of the horizon are scored as if care started then.

## Reproducibility

- One master seed. Every kind of entity draws from its own stream, `SeedSequence([seed, stream id])`, so adding a field to one entity never shifts the numbers of another.
- The same seed gives the same map (hospital sites and stations) at every stress level and theme. Stress changes the load.
- Output files contain no timestamps. Each folder has a `MANIFEST.json` with the SHA-256 of every file and the library versions. `python -m emergency_mas.datagen verify` regenerates the frozen scenarios and compares hashes.
- Library versions are pinned in `requirements-lock.txt`; NumPy does not promise identical streams across major versions.

## Known simplifications

- Survival parameters are illustrative, not clinical.
- Travel time uses zones and one congestion window, not a road graph.
- ER bays free up only when a treatment time ends. ICU and ward beds do not free up during the horizon.
- GREEN victims are transported like everyone else. Walking-wounded self-transport is not modeled.
