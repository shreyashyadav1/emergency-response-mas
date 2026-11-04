"""Hospitals and their medical staff."""
from __future__ import annotations

import math

from faker import Faker

from .. import params as P
from ..domain import Point, zone_of
from ..rng import Rng
from ..schema import Hospital, StaffMember


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def archetype_zones(n: int) -> list[str]:
    return [P.HOSPITAL_ARCHETYPES[i % len(P.HOSPITAL_ARCHETYPES)]["zone"] for i in range(n)]


def make_hospitals(rng: Rng, points: list[Point], stress: P.StressProfile) -> list[Hospital]:
    hospitals: list[Hospital] = []
    for i, (x, y) in enumerate(points):
        arch = P.HOSPITAL_ARCHETYPES[i % len(P.HOSPITAL_ARCHETYPES)]
        round_no = i // len(P.HOSPITAL_ARCHETYPES)
        er_total = arch["er_bays"]
        occupancy = _clip(rng.normal(stress.er_occupancy, 0.07), 0.20, 0.92)
        er_occupied = min(er_total - 1, round(er_total * occupancy))          # always leave one bay
        icu_occupancy = _clip(rng.normal(stress.icu_occupancy, 0.08), 0.30, 0.95)
        icu_free = round(arch["icu_beds"] * (1.0 - icu_occupancy))
        if arch["level"] <= 2:
            icu_free = max(1, icu_free)
        ward_lo, ward_hi = arch["ward_free"]
        ward_free = round(ward_lo + (ward_hi - ward_lo) * stress.ward_free_frac * rng.uniform(0.7, 1.3))
        lend_doctors, lend_nurses = arch["lend"]
        hospitals.append(Hospital(
            hospital_id=f"H{i + 1}",
            name=arch["name"] if round_no == 0 else f"{arch['name']} {round_no + 1}",
            x=x, y=y, zone=zone_of(x, y), trauma_level=arch["level"],
            capabilities=list(arch["capabilities"]),
            er_bays_total=er_total, er_bays_occupied=er_occupied,
            surge_bays=math.ceil(P.SURGE_BAY_FRAC * er_total),
            icu_beds_total=arch["icu_beds"], icu_beds_free=icu_free,
            ward_beds_free=int(_clip(ward_free, 0, ward_hi * 1.3)),
            reserve_bays_for_red=arch["reserve_red"], diversion_threshold=arch["diversion"],
            max_staff_lend_doctors=lend_doctors, max_staff_lend_nurses=lend_nurses,
        ))
    return hospitals


def make_staff(rng: Rng, fake: Faker, hospitals: list[Hospital], stress: P.StressProfile,
               horizon_min: int) -> list[StaffMember]:
    """Doctors and nurses for every hospital, then the regional on-call pool."""
    people: list[tuple[str, str, str, float]] = []        # (home, role, specialty, census factor)
    for i, hosp in enumerate(hospitals):
        arch = P.HOSPITAL_ARCHETYPES[i % len(P.HOSPITAL_ARCHETYPES)]
        census = hosp.er_bays_occupied / hosp.er_bays_total
        for specialty, count in arch["doctors"].items():
            people += [(hosp.hospital_id, "doctor", specialty, census)] * count
        for specialty, count in arch["nurses"].items():
            people += [(hosp.hospital_id, "nurse", specialty, census)] * count
    for specialty, count in P.STAFF_POOL["doctors"].items():
        people += [("POOL", "doctor", specialty, 0.0)] * count
    for specialty, count in P.STAFF_POOL["nurses"].items():
        people += [("POOL", "nurse", specialty, 0.0)] * count

    staff: list[StaffMember] = []
    counters = {"doctor": 0, "nurse": 0}
    for home, role, specialty, census in people:
        spec = (P.DOCTOR_SPECIALTIES if role == "doctor" else P.NURSE_SPECIALTIES)[specialty]
        counters[role] += 1
        first, last = fake.first_name(), fake.last_name()
        name = f"Dr. {first} {last}" if role == "doctor" else f"{first} {last}, RN"
        limit = spec["limit"]
        unit_bound = specialty not in P.LENDABLE_SPECIALTIES
        baseline = int(_clip(round(limit * census * rng.uniform(0.8, 1.6)), 0, limit - 1 if unit_bound else limit))
        if home == "POOL":
            start, end = rng.integers(*P.POOL_ARRIVAL_MIN), horizon_min
        else:
            start = rng.integers(*P.RECALL_DELAY_MIN) if rng.bernoulli(stress.staff_absent_prob) else 0
            end = rng.integers(90, 200) if rng.bernoulli(P.SHIFT_CHANGE_PROB) else horizon_min
            end = max(end, start + 45)
        staff.append(StaffMember(
            staff_id=f"{'D' if role == 'doctor' else 'N'}{counters[role]:02d}", name=name, role=role,
            specialty=specialty, skills=list(spec["skills"]), home_hospital=home,
            on_duty_from_min=start, on_duty_until_min=min(end, horizon_min),
            workload_limit=limit, baseline_load=baseline,
            lendable=specialty in P.LENDABLE_SPECIALTIES,
        ))
    return staff
