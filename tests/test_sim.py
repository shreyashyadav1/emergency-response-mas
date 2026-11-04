"""The simulator must enforce the rules whatever a policy asks for, and be exactly repeatable."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from emergency_mas.agents import AgentSystem
from emergency_mas.datagen import generate_scenario
from emergency_mas.policies import NearestFirst, PriorityGreedy
from emergency_mas.sim import Decision, DispatchAmbulance, DispatchTeam, OpenSurge, SetDestination, Simulator
from emergency_mas.sim.observe import build_observation
from emergency_mas.sim.state import World

MAKERS = {"B0": NearestFirst, "B1": PriorityGreedy, "M1": lambda: AgentSystem(False), "M2": lambda: AgentSystem(True)}


@pytest.fixture(scope="module")
def runs():
    scenario = generate_scenario("urban_mci", "high", 3)
    out = {}
    for key, make in MAKERS.items():
        sim = Simulator(scenario, make())
        out[key] = (scenario, sim, sim.run())
    return out


def test_same_run_twice_is_identical():
    s = generate_scenario("urban_mci", "medium", 5)
    a, b = Simulator(s, AgentSystem(True)).run(), Simulator(s, AgentSystem(True)).run()
    assert a.metrics == b.metrics
    assert json.dumps(a.ticks, sort_keys=True) == json.dumps(b.ticks, sort_keys=True)


def test_beds_are_never_over_capacity(runs):
    for _, _, result in runs.values():
        for tick in result.ticks:
            for h in tick["hospitals"]:
                assert h["bays_used"] <= h["bays_total"] + 0, (tick["t"], h)


def test_red_patients_ride_only_in_als_units_one_at_a_time(runs):
    for scenario, _, result in runs.values():
        unit_type = {a.ambulance_id: a.unit_type for a in scenario.ambulances}
        capacity = {a.ambulance_id: a.capacity for a in scenario.ambulances}
        for tick in result.ticks:
            for e in tick["events"]:
                if e["kind"] == "loading":
                    assert len(e["victims"]) <= capacity[e["unit"]]
                    reds = e["triage"].count("RED")
                    assert reds <= 1
                    if reds:
                        assert unit_type[e["unit"]] == "ALS"


def test_every_admission_is_once_at_a_capable_hospital_in_time_order(runs):
    for scenario, sim, result in runs.values():
        caps = {h.hospital_id: set(h.capabilities) for h in scenario.hospitals}
        admitted = Counter(e["victim"] for tick in result.ticks for e in tick["events"] if e["kind"] == "admitted")
        assert all(n == 1 for n in admitted.values())
        for vs in sim.w.victims.values():
            if vs.care_start is not None:
                assert vs.v.required_capability in caps[vs.hospital_id]
                assert vs.report_time <= vs.picked_up_at <= vs.care_start < scenario.meta.horizon_min
                assert 0 < vs.survival <= vs.v.survival_if_timely


def test_bookkeeping_is_consistent_at_the_end(runs):
    for _, sim, _ in runs.values():
        w = sim.w
        for hs in w.hospitals.values():
            assert hs.regular_used + hs.surge_used == len(hs.in_care)
        for m in w.staff.values():
            for vid in m.patients:
                assert w.victims[vid].status == "in_care"
            assert m.s.baseline_load + len(m.patients) <= m.s.workload_limit or not m.patients


def test_impossible_requests_are_refused_and_change_nothing():
    class Unreasonable:
        name = "unreasonable"

        def reset(self, scenario):
            pass

        def decide(self, obs):
            return Decision([DispatchAmbulance("A9", "INC-01"), SetDestination("A1", "H3"), DispatchTeam("R4", "INC-99"),
                             DispatchAmbulance("A1", "INC-15"), OpenSurge("H7")])

    result = Simulator(generate_scenario("urban_mci", "medium", 1), Unreasonable()).run()
    assert result.metrics["victims_served"] == 0
    assert result.metrics["actions_rejected"] >= 5 * 240


def test_observation_hides_the_truth_until_a_unit_arrives():
    w = World.from_scenario(generate_scenario("urban_mci", "medium", 2))
    for inc in w.incidents.values():
        inc.visible = True
    obs = build_observation(w, 0, [])
    assert obs.incidents and all(not i.revealed and not i.victims and not i.needs for i in obs.incidents)


@pytest.mark.parametrize("preset", ["calm", "baseline", "crunch", "surge", "pileup"])
def test_every_policy_runs_on_every_preset(preset):
    from emergency_mas import params as P

    spec = P.PRESETS[preset]
    scenario = generate_scenario(spec["theme"], spec["stress"], 1)
    for make in MAKERS.values():
        result = Simulator(scenario, make(), record=False).run()
        assert result.metrics["victims_served"] > 0
        assert 0 < result.metrics["weighted_survival"] <= 1
