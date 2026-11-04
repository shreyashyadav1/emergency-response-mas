"""The dashboard shows the run that happened: its totals end at the final metrics, saved logs replay
the same minutes, every negotiation message lands in one thread, and the app itself renders."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from emergency_mas import params as P
from emergency_mas.dashboard import figures as F
from emergency_mas.dashboard import replay as R
from emergency_mas.eval.cli import main as eval_cli

ROOT = Path(__file__).resolve().parents[1]
NEGOTIATION = {"conflict_notice", "admission_request", "admission_accept", "admission_reject", "counter_proposal",
               "staff_request", "staff_offer"}


@pytest.fixture(scope="module")
def m2():
    return R.live("urban_mci", "high", 3, "M2")


@pytest.fixture(scope="module")
def b0():
    return R.live("urban_mci", "high", 3, "B0")


def test_running_totals_end_at_the_final_metrics(m2, b0):
    for r in (m2, b0):
        last, m = r.series.iloc[-1], r.metrics
        assert last.victims_in_care == m["victims_served"]
        assert last.red_in_care == m["red_served"]
        assert last.critical_waiting == m["unserved_critical_incidents"]
        assert last.door_minutes == m["ramp_wait_min"]
        assert round(last.first_response, 1) == m["mean_first_response_min"]
        assert last.messages == m["messages"]
        assert last.staff_moves == m["staff_moves"]
        assert (r.series.victims_in_care.diff().dropna() >= 0).all()


def test_a_saved_log_replays_the_same_run(m2, tmp_path):
    assert eval_cli(["replay", "--stress", "high", "--seed", "3", "--policies", "M2", "--out", str(tmp_path)]) == 0
    saved = R.from_log(R.saved_logs(tmp_path)["urban_mci-high-s3"]["M2"])
    assert saved.policy == "M2"
    assert saved.metrics == m2.metrics
    assert saved.ticks == json.loads(json.dumps(m2.ticks))


def test_every_negotiation_message_lands_in_one_thread(m2):
    placed = [m["msg_id"] for th in m2.threads for m in th.messages]
    assert len(placed) == len(set(placed))
    assert {m["msg_id"] for m in m2.messages if m["type"] in NEGOTIATION} <= set(placed)
    destinations = [m for m in m2.messages if m["type"] == "allocation_decision" and m["proposed_action"]["kind"] == "set_destination"]
    assert destinations and all(m["in_reply_to"] is not None for m in destinations)
    for th in m2.threads:
        assert th.rounds <= P.MAX_NEGOTIATION_ROUNDS
        if any(m["type"] == "counter_proposal" for m in th.messages):
            assert any(m["type"] == "allocation_decision" and m["recipient"] == "dispatch" for m in th.messages)


def test_resources_add_up(m2):
    s = m2.scenario
    for t in (0, 60, 239):
        totals = m2.resources(t).groupby("resource")["count"].sum()
        assert totals["Ambulances"] == len(s.ambulances)
        assert totals["Responder teams"] == len(s.responders)
        assert totals["ICU beds"] == sum(h.icu_beds_total for h in s.hospitals)
        assert totals["ER bays"] == sum(h.er_bays_total + h.surge_bays for h in s.hospitals)


def test_map_layers_follow_the_world(m2):
    for t in (0, 30, 120, 239):
        layers = m2.map_layers(t)
        incidents, units = layers["incidents"], layers["units"]
        if len(incidents):
            assert incidents.waiting.between(0, incidents.victims).all()
            assert set(incidents.state) <= {"RED", "YELLOW", "GREEN", "cleared"}
        assert len(units) == len(m2.scenario.ambulances) + len(m2.scenario.responders)
        assert units.dropna(subset=["tx"])["where"].ne("").all()
    assert len(m2.map_layers(239)["incidents"]) == len(m2.scenario.incidents)


def test_agent_cards_only_for_agent_policies(m2, b0):
    cards = m2.agents(120)
    assert {"Incident Assessment", "Dispatch", "Resource Allocation"} <= {c["name"] for c in cards}
    assert all(len(c["recent"]) <= 3 for c in cards)
    assert b0.agents(120) == [] and b0.threads == []


def test_agent_cards_show_what_is_pending(m2):
    for t in (15, 45, 90, 150):
        tick, cards = m2.tick(t), {c["id"]: c for c in m2.agents(t)}
        pending = m2.pending(t, most=99)
        assert all(isinstance(c["pending"], list) for c in cards.values())
        for h in tick["hospitals"]:
            door = [u["id"] for u in tick["ambulances"] if u["status"] == "at_door" and u["hospital"] == h["id"]]
            assert all(u in " ".join(pending[f"hospital:{h['id']}"]) for u in door)
    first_call = min(e["t"] for e in m2.events if e["kind"] == "call")
    assert m2.pending(first_call)["incident_assessment"]                 # nobody is on scene at the first call yet
    moves = [e for e in m2.events if e["kind"] == "staff_assigned" and e["arrives"] > e["t"]]
    assert moves, "this scenario moves staff"
    e = moves[0]
    assert any(e["staff"] in line for line in m2.pending(e["t"], most=99)["resource_allocation"])
    assert not any(e["staff"] in line for line in m2.pending(e["arrives"], most=99)["resource_allocation"])


def test_comparison_and_seed_context(m2, b0):
    assert list(R.compare_outcomes(m2, b0).columns) == ["Outcome", "M2", "B0", "Difference"]
    runs = pd.DataFrame({"theme": "urban_mci", "stress": "high", "seed": [1, 1, 2, 2, 3, 3], "policy": ["M2", "B0"] * 3,
                         "weighted_survival": [0.60, 0.50, 0.55, 0.56, 0.80, 0.60]})
    ctx = R.seed_context(runs, "urban_mci", "high", 3, "M2", "B0")
    assert ctx["seeds"] == 3 and ctx["rank"] == 3 and ctx["ahead_share"] == pytest.approx(2 / 3)
    assert "rank" not in R.seed_context(runs, "urban_mci", "high", 99, "M2", "B0")
    assert R.seed_context(runs, "urban_mci", "low", 3, "M2", "B0") is None


def test_figures_build_in_both_themes(m2, b0):
    for mode in ("light", "dark"):
        tok = F.TOKENS[mode]
        names = {d.name for d in F.map_figure(m2.map_layers(60), m2.scenario.meta.city_km, tok).data}
        assert {"RED patient waiting", "Hospital (ER bays used / open)", "Ambulance, ALS", "Responder team"} <= names
        assert len(F.resource_figure(m2.resources(60), tok).data) == 3
        lines = F.series_figure([("M2", m2.series.victims_in_care, tok["selected"]),
                                 ("B0", b0.series.victims_in_care, tok["compare"])], 60, tok)
        assert [d.name for d in lines.data if d.showlegend is not False] == ["M2", "B0"]
    assert F.on_fill("#fab219") == "#0b0b0b" and F.on_fill("#d03b3b") == "#ffffff"


def test_the_app_renders_all_five_panels():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120)
    app.run()
    assert not app.exception
    app.session_state["minute"] = 90
    app.run()
    assert not app.exception
    heads = [s.value for s in app.subheader]
    for panel in ("Incident map", "Resources", "Agent activity", "Negotiation history", "Performance over the whole run"):
        assert panel in heads
    assert len(app.metric) == 6
