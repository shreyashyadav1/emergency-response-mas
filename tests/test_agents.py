"""Agents talk in structured messages, negotiate within bounds, and help."""
from __future__ import annotations

from statistics import mean

import pytest

from emergency_mas import params as P
from emergency_mas.agents import AgentSystem, Message, MsgType
from emergency_mas.datagen import generate_scenario
from emergency_mas.eval import run_one
from emergency_mas.sim import Simulator

REPLIES = {MsgType.ADMISSION_ACCEPT, MsgType.ADMISSION_REJECT, MsgType.COUNTER_PROPOSAL}


def messages(result) -> list[Message]:
    return [Message.model_validate(m) for tick in result.ticks for m in tick["messages"]]


@pytest.fixture(scope="module")
def m2():
    return messages(Simulator(generate_scenario("urban_mci", "high", 2), AgentSystem(True)).run())


@pytest.fixture(scope="module")
def m1():
    return messages(Simulator(generate_scenario("urban_mci", "high", 2), AgentSystem(False)).run())


def test_messages_name_sender_recipient_type_incident_and_action(m2):
    assert len(m2) > 50
    for msg in m2:
        assert msg.sender and msg.recipient and msg.type and msg.proposed_action.kind
        if msg.type in (MsgType.ASSESSMENT, MsgType.ADMISSION_REQUEST, *REPLIES):
            assert msg.incident_id is not None
    ids = [m.msg_id for m in m2]
    assert ids == sorted(ids) and len(ids) == len(set(ids))


def test_every_hospital_reply_answers_a_request(m2):
    requests = {m.msg_id: m for m in m2 if m.type == MsgType.ADMISSION_REQUEST}
    replies = [m for m in m2 if m.type in REPLIES]
    assert replies
    for r in replies:
        req = requests[r.in_reply_to]
        assert r.sender == req.recipient and r.recipient == req.sender
        assert r.proposed_action.unit_id == req.proposed_action.unit_id


def test_negotiation_happens_and_stays_within_three_rounds(m2):
    kinds = {m.type for m in m2}
    assert {MsgType.CONFLICT_NOTICE, MsgType.ALLOCATION_DECISION, MsgType.COUNTER_PROPOSAL} <= kinds
    assert max(m.round for m in m2) <= P.MAX_NEGOTIATION_ROUNDS
    notices = {m.msg_id for m in m2 if m.type == MsgType.CONFLICT_NOTICE}
    settled = {m.in_reply_to for m in m2 if m.type == MsgType.ALLOCATION_DECISION and m.in_reply_to in notices}
    assert settled == notices                      # every conflict gets a decision


def test_without_negotiation_hospitals_only_accept_or_reject(m1):
    kinds = {m.type for m in m1}
    assert MsgType.ADMISSION_REQUEST in kinds
    assert not kinds & {MsgType.COUNTER_PROPOSAL, MsgType.CONFLICT_NOTICE, MsgType.STAFF_REQUEST, MsgType.ALLOCATION_DECISION}


def test_assessment_removes_caller_undercounting():
    rows = [run_one("urban_mci", stress, seed, "M2")[0] for stress in ("medium", "high") for seed in range(1, 9)]
    assert abs(mean(r["agent_victims_bias"] for r in rows)) < abs(mean(r["report_victims_bias"] for r in rows)) / 2
    assert mean(r["agent_victims_rmse"] for r in rows) < mean(r["report_victims_rmse"] for r in rows)


def test_agents_beat_both_baselines_on_average():
    scores = {p: [] for p in ("B0", "B1", "M1", "M2")}
    for seed in range(1, 7):
        for p in scores:
            scores[p].append(run_one("urban_mci", "medium", seed, p)[0]["weighted_survival"])
    best_baseline = max(mean(scores["B0"]), mean(scores["B1"]))
    assert mean(scores["M2"]) > best_baseline
    assert mean(scores["M1"]) > mean(scores["B0"])
