"""The OpenAI Agents SDK agents (M3) work end to end without a network: a scripted model stands in for the hosted
model, calling each agent's tool and answering from what the tool returned."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai import RateLimitError
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from emergency_mas.eval.runner import run_one
from emergency_mas.llm import backend
from emergency_mas.llm.agents import EstimateReply, build_agents
from emergency_mas.llm.backend import (GEMINI_OPENAI_URL, Backend, DailyQuotaExceeded, NoCredit, PacedModel, Settings,
                                       parse_reply)

SCENARIO = ("urban_mci", "medium", 2)
KEYS = ("weighted_survival", "victims_served", "red_survivors", "ramp_wait_min", "messages")


def _answer(instructions: str, data) -> dict:
    """What a careful model would say, given the tool's data."""
    if instructions.startswith("You are the Incident Assessment Agent"):
        return {**data, "reason": "the caller's words match the base rates"}
    if instructions.startswith("You are the Dispatch Agent"):
        order = sorted(data, key=lambda h: (-h["bays_expected_free_at_arrival"], h["minutes_to_arrive"]))
        return {"ask_order": [h["hospital"] for h in order], "reason": "most room at arrival first"}
    if instructions.startswith("You are the Hospital Coordination Agent"):
        return {"decision": "counter" if data["bays_expected_free_at_arrival"] >= 0 else "reject", "reason": "room soon"}
    if isinstance(data, list) and data and "bids" in data[0]:
        return {"awards": [{"unit": r["unit"], "incident": r["optimizer_pick"] or r["bids"][0]["incident"]} for r in data],
                "reason": "most lives saved overall"}
    return {"option": 0, "reason": "care starts soonest"}


class ScriptedModel(Model):
    def __init__(self, garbage: bool = False) -> None:
        self.requests, self.garbage = 0, garbage

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, **_):
        self.requests += 1
        items = input if isinstance(input, list) else [{"role": "user", "content": input}]
        prompt = next((i.get("content") for i in items if isinstance(i, dict) and i.get("role") == "user"), "")
        outputs = [i["output"] for i in items if isinstance(i, dict) and i.get("type") == "function_call_output"]
        if tools and not outputs:
            names = [t.name for t in tools]
            name = ("statistical_estimate" if "statistical_estimate" in names else
                    "settlement_options" if "settlement" in str(prompt) and "settlement_options" in names else names[0])
            call = ResponseFunctionToolCall(arguments="{}", call_id=f"call{self.requests}", name=name,
                                            type="function_call", id=f"fc{self.requests}", status="completed")
            return ModelResponse(output=[call], usage=Usage(requests=1, input_tokens=120, output_tokens=8, total_tokens=128),
                                 response_id=None)
        text = "I would rather not say." if self.garbage else json.dumps(_answer(system_instructions, json.loads(outputs[-1])))
        message = ResponseOutputMessage(id=f"msg{self.requests}", role="assistant", status="completed", type="message",
                                        content=[ResponseOutputText(text=text, type="output_text", annotations=[])])
        return ModelResponse(output=[message], usage=Usage(requests=1, input_tokens=200, output_tokens=30, total_tokens=230),
                             response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    cache = tmp_path_factory.mktemp("llm_cache")
    model = ScriptedModel()
    llm = Backend(Settings(cache_dir=cache), mode="live", model=model)
    row, result = run_one(*SCENARIO, "M3", record=True, llm=llm)
    return cache, model, llm.stats, row, result


def test_four_agents_with_their_own_tools():
    agents = build_agents(None)
    tools = {key: [t.name for t in a.tools] for key, a in agents.items()}
    assert tools == {"assessment": ["base_rates", "statistical_estimate"], "dispatch": ["hospital_options"],
                     "hospital": ["capacity_at_arrival"], "allocation": ["contested_units", "settlement_options"]}
    assert len({a.instructions for a in agents.values()}) == 4


def test_m3_runs_through_the_sdk(live_run):
    _, model, stats, row, result = live_run
    decided = sum(stats.decisions.values())
    assert stats.decisions["assessment"] == 12 and stats.decisions["conflict"] > 0 and stats.decisions["routing"] > 0
    assert stats.live == decided and stats.fallbacks == 0
    assert model.requests == stats.requests == 2 * decided          # every decision: one tool call, one answer
    assert 0 < row["weighted_survival"] < 1
    rationales = [m["rationale"] for tick in result.ticks for m in tick["messages"]]
    assert any("LLM" in r for r in rationales)


def test_cached_answers_replay_without_a_model(live_run):
    cache, _, stats, row, _ = live_run
    replay = Backend(Settings(cache_dir=cache), mode="cache")
    again, _ = run_one(*SCENARIO, "M3", llm=replay)
    assert replay.stats.cached == sum(stats.decisions.values()) and replay.stats.missing == 0 and replay.stats.live == 0
    assert all(again[k] == row[k] for k in KEYS)


def test_bad_answers_leave_the_rules_decisions(tmp_path):
    llm = Backend(Settings(cache_dir=tmp_path), mode="live", model=ScriptedModel(garbage=True))
    m3, _ = run_one(*SCENARIO, "M3", llm=llm)
    m2, _ = run_one(*SCENARIO, "M2")
    assert llm.stats.invalid == sum(llm.stats.decisions.values()) > 0
    assert all(m3[k] == m2[k] for k in KEYS if k != "messages")


def test_dry_run_sends_nothing(tmp_path):
    llm = Backend(Settings(cache_dir=tmp_path), mode="dry-run")
    m3, _ = run_one(*SCENARIO, "M3", llm=llm)
    m2, _ = run_one(*SCENARIO, "M2")
    assert llm.stats.requests == 0 and llm.stats.estimated_input_tokens > 0
    assert all(m3[k] == m2[k] for k in KEYS)
    assert not any(tmp_path.iterdir())


def test_the_model_name_picks_the_server(monkeypatch):
    monkeypatch.setattr(backend, "load_env", lambda: None)
    for name in ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    openai, gemini = Settings.from_env(), Settings.from_env(model="gemini-3.5-flash-lite")
    assert (openai.model, openai.base_url, openai.api_key) == ("gpt-4o-mini", None, "openai-key")
    assert (gemini.base_url, gemini.api_key) == (GEMINI_OPENAI_URL, "gemini-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "other-key")
    other = Settings.from_env(model="some-model")
    assert (other.base_url, other.api_key, other.key_name) == ("https://api.example.com/v1", "other-key", "LLM_API_KEY")


class RefusingModel(Model):
    """A server that answers every request with one rate-limit error."""

    def __init__(self, text: str) -> None:
        self.text, self.requests = text, 0

    async def get_response(self, *args, **kwargs):
        self.requests += 1
        # a stand-in for the HTTP response, so the test does not depend on which HTTP library the client uses
        response = SimpleNamespace(request=None, status_code=429, headers={})
        raise RateLimitError(self.text, response=response, body=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize("text, error", [
    ("Error code: 429 - {'error': {'type': 'insufficient_quota', 'code': 'insufficient_quota'}}", NoCredit),
    ("Error code: 429 - Rate limit reached for gpt-4o-mini on requests per day (RPD): Limit 10000", DailyQuotaExceeded),
    ("Error code: 429 - {'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '500'}",
     DailyQuotaExceeded)])
def test_quota_errors_stop_without_retrying(text, error):
    server = RefusingModel(text)
    with pytest.raises(error):
        asyncio.run(PacedModel(server, rpm=60).get_response())
    assert server.requests == 1


def test_replies_are_parsed_and_checked():
    reply = parse_reply('```json\n{"red": 1, "yellow": 2.5, "green": 3, "hazard": "fire", "trapped": false, '
                        '"reason": "smoke reported"}\n```', EstimateReply)
    assert reply.hazard == "fire" and reply.yellow == 2.5
    with pytest.raises(Exception):
        parse_reply('{"red": 1, "yellow": 2, "green": 3, "hazard": "flood", "trapped": false, "reason": "x"}', EstimateReply)
