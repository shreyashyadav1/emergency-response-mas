"""The data layer must be reproducible, valid and shaped the way the assumptions say."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import pandas as pd
import pytest

from emergency_mas import params as P
from emergency_mas.datagen import generate_scenario, validate_scenario
from emergency_mas.datagen.analysis import contention_report, pressure_indices
from emergency_mas.datagen.docs import assumptions_md, data_dictionary_md
from emergency_mas.datagen.export import scenario_hash, write_scenario
from emergency_mas.datagen.history import HISTORY_SEED_BASE, base_rates, generate_history
from emergency_mas.schema import Scenario

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- reproducibility
def test_same_seed_gives_identical_bytes():
    assert scenario_hash(generate_scenario("urban_mci", "medium", 7)) == scenario_hash(generate_scenario("urban_mci", "medium", 7))


def test_different_seeds_differ():
    assert scenario_hash(generate_scenario("urban_mci", "medium", 7)) != scenario_hash(generate_scenario("urban_mci", "medium", 8))


def test_other_work_in_between_changes_nothing():
    first = scenario_hash(generate_scenario("earthquake", "high", 3))
    generate_scenario("urban_mci", "low", 99)
    assert scenario_hash(generate_scenario("earthquake", "high", 3)) == first


def test_identical_across_processes_with_different_hash_seeds():
    code = ("from emergency_mas.datagen import generate_scenario; "
            "from emergency_mas.datagen.export import scenario_hash; "
            "print(scenario_hash(generate_scenario('urban_mci', 'medium', 11)))")
    digests = {
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip()
        for h in (0, 1, 12345)
    }
    assert len(digests) == 1


def test_city_is_shared_across_stress_levels():
    low, high = generate_scenario("urban_mci", "low", 5), generate_scenario("urban_mci", "high", 5)
    assert [(h.x, h.y) for h in low.hospitals] == [(h.x, h.y) for h in high.hospitals]
    assert [(s.x, s.y) for s in low.stations] == [(s.x, s.y) for s in high.stations]


def test_entity_streams_are_independent():
    small = generate_scenario("urban_mci", "medium", 9, n_incidents=10)
    large = generate_scenario("urban_mci", "medium", 9, n_incidents=14)
    assert (small.hospitals, small.staff, small.ambulances, small.stations, small.responders) == \
           (large.hospitals, large.staff, large.ambulances, large.stations, large.responders)
    assert small.incidents != large.incidents


# ---------------------------------------------------------------- validity
@pytest.mark.parametrize("preset", list(P.PRESETS))
def test_every_check_passes(preset):
    spec = P.PRESETS[preset]
    for seed in range(1, 26):
        failed = [c for c in validate_scenario(generate_scenario(spec["theme"], spec["stress"], seed)) if not c.passed]
        assert not failed, f"{preset} seed {seed}: {failed}"


def test_scenarios_have_the_requested_size():
    s = generate_scenario("urban_mci", "medium", 1)
    assert (len(s.hospitals), len(s.ambulances), len(s.responders), len(s.incidents)) == (3, 5, 6, 12)
    big = generate_scenario("earthquake", "high", 1, n_hospitals=5, n_ambulances=8, n_responders=9, n_incidents=15)
    assert (len(big.hospitals), len(big.ambulances), len(big.responders), len(big.incidents)) == (5, 8, 9, 15)
    assert not [c for c in validate_scenario(big) if not c.passed]


def test_unknown_theme_is_rejected():
    with pytest.raises(ValueError):
        generate_scenario("flood", "medium", 1)


# ---------------------------------------------------------------- the assumptions hold
def test_stress_levels_are_ordered():
    seeds = range(1, 41)
    for index in ("ambulance_pressure", "er_bay_pressure", "specialist_pressure"):
        means = {level: mean(pressure_indices(generate_scenario("urban_mci", level, s))[index] for s in seeds) for level in P.STRESS}
        assert means["low"] < means["medium"] < means["high"], (index, means)


def test_medium_stress_is_contested_and_low_is_not():
    seeds = range(1, 41)
    medium = mean(pressure_indices(generate_scenario("urban_mci", "medium", s))["ambulance_pressure"] for s in seeds)
    low = mean(pressure_indices(generate_scenario("urban_mci", "low", s))["ambulance_pressure"] for s in seeds)
    assert medium > 1.0 > low


def test_triage_mix_matches_the_assumed_mix():
    counts: Counter = Counter()
    for seed in range(1, 151):
        s = generate_scenario("urban_mci", "medium", seed)
        kinds = {i.incident_id: i.incident_type for i in s.incidents}
        counts.update(v.triage for v in s.victims if kinds[v.incident_id] == "multi_vehicle_accident")
    mix = dict(P.INCIDENT_PROFILES["multi_vehicle_accident"].triage_mix)
    mix["RED"] *= 1.0 + P.STRESS["medium"].red_boost
    total, norm = sum(counts.values()), sum(mix.values())
    for triage in P.TRIAGE:
        assert abs(counts[triage] / total - mix[triage] / norm) < 0.03, (triage, counts)


def test_callers_undercount_on_average_but_not_wildly():
    ratios = [i.report.reported_victims / i.n_victims
              for seed in range(1, 61) for i in generate_scenario("urban_mci", "medium", seed).incidents]
    assert 0.80 < mean(ratios) < 1.0


def test_a_scenario_contains_real_conflicts():
    s = generate_scenario("urban_mci", "medium", 42)
    report = contention_report(s)
    assert report["n_ambulance_pairs"] > 0
    loads = report["nearest_capable_load"]
    assert any(v["red_yellow_nearest_capable"] > v["free_er_bays"] for v in loads.values())


# ---------------------------------------------------------------- files and docs
def test_files_match_the_scenario_and_the_manifest(tmp_path):
    s = generate_scenario("urban_mci", "medium", 42)
    manifest = write_scenario(s, tmp_path)
    folder = tmp_path / s.meta.scenario_id
    assert Scenario.model_validate_json((folder / "scenario.json").read_text()) == s
    assert hashlib.sha256((folder / "scenario.json").read_bytes()).hexdigest() == manifest["scenario_sha256"] == scenario_hash(s)
    for name, expected in (("hospitals", len(s.hospitals)), ("staff", len(s.staff)), ("ambulances", len(s.ambulances)),
                           ("responders", len(s.responders)), ("incidents", len(s.incidents)), ("victims", len(s.victims))):
        assert len(pd.read_csv(folder / f"{name}.csv")) == expected
    assert all(manifest["checks"].values())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    assert json.loads((folder / "MANIFEST.json").read_text())["seed"] == 42


def test_manifest_has_no_dates_and_is_repeatable(tmp_path):
    first = write_scenario(generate_scenario("urban_mci", "low", 2), tmp_path / "a")
    second = write_scenario(generate_scenario("urban_mci", "low", 2), tmp_path / "b")
    assert first == second
    text = (tmp_path / "a" / "urban_mci-low-s2" / "MANIFEST.json").read_text()
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)


def test_docs_cover_every_field():
    dictionary = data_dictionary_md()
    for field in ("golden_window_min", "reserve_bays_for_red", "workload_limit", "urgency_truth", "reported_victims"):
        assert f"`{field}`" in dictionary
    assert "survival" in assumptions_md().lower()


# ---------------------------------------------------------------- history
def test_history_is_reproducible_and_disjoint_from_evaluation_seeds():
    a, b = generate_history(80), generate_history(80)
    pd.testing.assert_frame_equal(a, b)
    assert HISTORY_SEED_BASE > 100_000
    rates = base_rates(generate_history(300))
    assert (rates["true_over_reported_victims"] > 1.0).all()
