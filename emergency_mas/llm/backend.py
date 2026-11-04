"""Run the agents' judgment calls on a language model through the OpenAI Agents SDK.

The SDK talks to any server that speaks the OpenAI chat completions API: OpenAI's own, or Google
Gemini's OpenAI-compatible endpoint, which has a free tier. The model name picks the server:
gemini-* models go to Gemini, the rest to OpenAI. Settings come from the environment or from a
.env file in the project root:

    OPENAI_API_KEY=...            for OpenAI models; needed for live calls only
    GEMINI_API_KEY=...            for Gemini models
    LLM_MODEL=gpt-4o-mini         any model the key can use (`python -m emergency_mas.llm models`)
    LLM_BASE_URL=...              optional: any other OpenAI-compatible server, with its key in LLM_API_KEY
    LLM_RPM=8                     model requests per minute, to stay under the account's limit

Three modes. "live" calls the model when an answer is not cached yet. "cache" only replays
cached answers. "dry-run" never calls anything: it counts the decisions a run would send and
estimates their tokens, and the rules decision stands, so it costs nothing. Every live answer is
cached on disk by a hash of the model, the agent, the prompt and the tool data, so a run that has
been done once replays offline with the same answers and no key.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.models.interface import Model
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[2]
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
SERVERS = {"gemini": (GEMINI_OPENAI_URL, "GEMINI_API_KEY"), "openai": (None, "OPENAI_API_KEY")}   # base URL (None: OpenAI's), key
MODES = ("live", "cache", "dry-run")
CHARS_PER_TOKEN = 4                 # rough rule for English text, used only by the dry run
REQUESTS_PER_DECISION = 2           # one to call the agent's tool, one to answer


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment, without overriding what is already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass
class Settings:
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    base_url: str | None = None              # None: OpenAI's own API
    key_name: str = "OPENAI_API_KEY"         # where the key comes from, for messages
    rpm: float = 8.0
    max_turns: int = 4
    cache_dir: Path = ROOT / "data" / "llm_cache"

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        load_env()
        env = os.environ
        model = overrides.pop("model", None) or env.get("LLM_MODEL", cls.model)
        base_url, key_name = SERVERS["gemini" if model.startswith("gemini") else "openai"]
        if env.get("LLM_BASE_URL"):                   # any other OpenAI-compatible server
            base_url, key_name = env["LLM_BASE_URL"], "LLM_API_KEY"
        values = {"api_key": env.get(key_name), "model": model, "base_url": base_url, "key_name": key_name,
                  "rpm": float(env.get("LLM_RPM", cls.rpm))}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


@dataclass
class Stats:
    """What the language model did during one run."""
    decisions: Counter = field(default_factory=Counter)   # by kind
    departed: Counter = field(default_factory=Counter)    # by kind: valid answers that differ from the rules agent's choice
    cached: int = 0
    live: int = 0
    missing: int = 0          # cache mode: no cached answer, the rules decision stood
    invalid: int = 0          # the answer did not fit the schema or the options, the rules decision stood
    errors: int = 0           # the call failed after retries, the rules decision stood
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_input_tokens: int = 0

    @property
    def fallbacks(self) -> int:
        return self.missing + self.invalid + self.errors

    def row(self) -> dict:
        return {"llm_decisions": sum(self.decisions.values()), "llm_cached": self.cached, "llm_live": self.live,
                "llm_fallbacks": self.fallbacks, "llm_invalid": self.invalid, "llm_errors": self.errors,
                "llm_missing": self.missing, "llm_requests": self.requests, "llm_input_tokens": self.input_tokens,
                "llm_output_tokens": self.output_tokens, "llm_estimated_input_tokens": self.estimated_input_tokens,
                **{f"llm_{k}": n for k, n in sorted(self.decisions.items())},
                **{f"llm_departed_{k}": self.departed[k] for k in sorted(self.decisions)}}


@dataclass
class Situation:
    """What one decision's tools return, by tool name."""
    data: dict[str, Any]


def parse_reply(text: str, schema: type[BaseModel]) -> BaseModel:
    """The JSON object in a model's reply, validated against `schema`. Tolerates code fences and stray prose."""
    found = re.search(r"\{.*\}", text or "", flags=re.S)
    if not found:
        raise ValueError("no JSON object in the reply")
    return schema.model_validate(json.loads(found.group(0)))


class DailyQuotaExceeded(RuntimeError):
    """The model's requests-per-day quota is used up. Retrying today will not help; the cache keeps every answer so far."""


class NoCredit(RuntimeError):
    """The account has no API credit left. Retrying will not help until credit is added; the cache keeps every answer so far."""


class PacedModel(Model):
    """Wraps an SDK model: waits between requests to respect a requests-per-minute budget, and retries
    when the server says it is busy or rate-limited for the minute. A per-day quota or no-credit error stops at once."""

    def __init__(self, inner: Model, rpm: float, retries: int = 5):
        self.inner, self.gap, self.retries = inner, 60.0 / max(rpm, 0.1), retries
        self._last = 0.0

    async def get_response(self, *args, **kwargs):
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        for attempt in range(self.retries + 1):
            wait = self._last + self.gap - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
            try:
                return await self.inner.get_response(*args, **kwargs)
            except RateLimitError as exc:
                text = str(exc)
                if "insufficient_quota" in text:
                    raise NoCredit("the account has no API credit left (add some under Billing on platform.openai.com)") from exc
                if "PerDay" in text or "per day" in text:
                    limit = re.search(r"'quotaValue': '(\d+)'", text)
                    raise DailyQuotaExceeded("the daily quota for this model is used up"
                                             + (f" ({limit.group(1)} requests per day)" if limit else "")) from exc
                if attempt == self.retries:
                    raise
                await asyncio.sleep(min(120, 15 * 2 ** attempt))
            except (APIConnectionError, APITimeoutError, InternalServerError):
                if attempt == self.retries:
                    raise
                await asyncio.sleep(min(120, 15 * 2 ** attempt))

    def stream_response(self, *args, **kwargs):
        return self.inner.stream_response(*args, **kwargs)


def make_model(settings: Settings):
    """The SDK model object for the configured OpenAI-compatible server."""
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled
    from openai import AsyncOpenAI

    set_tracing_disabled(True)              # traces would be uploaded to OpenAI's dashboard, which needs an OpenAI key
    if not settings.api_key:
        raise RuntimeError(f"No API key: put {settings.key_name}=... in the project's .env file (see README).")
    client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key, max_retries=0, timeout=90)
    return PacedModel(OpenAIChatCompletionsModel(model=settings.model, openai_client=client), settings.rpm)


class Backend:
    """Asks the SDK agents for decisions, with caching, pacing, validation and a dry-run count."""

    def __init__(self, settings: Settings | None = None, mode: str = "cache", model=None):
        from agents import set_tracing_disabled

        from .agents import build_agents

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        set_tracing_disabled(True)          # traces would be uploaded to OpenAI's dashboard, which needs an OpenAI key
        self.settings = settings or Settings.from_env()
        self.mode = mode
        if model is None and mode == "live":
            model = make_model(self.settings)
        self.model_name = self.settings.model
        self.agents = build_agents(model)
        self.stats = Stats()
        self.consecutive_errors = 0
        self.last_error = ""
        self.verbose = False                 # print one line per live call (the CLI turns it on)

    @classmethod
    def from_env(cls, mode: str = "cache", **overrides: Any) -> Backend:
        return cls(Settings.from_env(**overrides), mode)

    def new_run(self) -> None:
        self.stats = Stats()

    def _key(self, agent, prompt: str, situation: Situation, schema: type[BaseModel]) -> str:
        blob = json.dumps({"model": self.model_name, "agent": agent.name, "instructions": agent.instructions,
                           "tools": [t.name for t in agent.tools], "prompt": prompt, "data": situation.data,
                           "schema": schema.model_json_schema()}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.settings.cache_dir / key[:2] / f"{key}.json"

    def ask(self, kind: str, agent_key: str, prompt: str, situation: Situation, schema: type[BaseModel]) -> BaseModel | None:
        """The agent's answer to one decision, or None when the rules decision should stand."""
        agent = self.agents[agent_key]
        self.stats.decisions[kind] += 1
        key = self._key(agent, prompt, situation, schema)
        path = self._path(key)
        if path.exists():
            self.stats.cached += 1
            record = json.loads(path.read_text(encoding="utf-8"))
            return self._validated(record["output_text"], schema)
        if self.mode == "dry-run":
            chars = REQUESTS_PER_DECISION * (len(agent.instructions) + len(prompt)) + len(json.dumps(situation.data))
            self.stats.estimated_input_tokens += chars // CHARS_PER_TOKEN
            return None
        if self.mode == "cache":
            self.stats.missing += 1
            return None
        from agents import Runner

        try:
            result = Runner.run_sync(agent, prompt, context=situation, max_turns=self.settings.max_turns)
        except (DailyQuotaExceeded, NoCredit):
            raise
        except Exception as exc:                       # noqa: BLE001 - any failure falls back to the rules decision
            self.stats.errors += 1
            self.consecutive_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.verbose:
                print(f"    error on {kind}: {self.last_error[:200]}", flush=True)
            if self.consecutive_errors >= 3:
                raise RuntimeError(f"three model calls in a row failed; last error: {self.last_error}") from exc
            return None
        self.consecutive_errors = 0
        usage = result.context_wrapper.usage
        self.stats.live += 1
        self.stats.requests += usage.requests
        self.stats.input_tokens += usage.input_tokens
        self.stats.output_tokens += usage.output_tokens
        if self.verbose:
            print(f"    {self.stats.live:3d} live  {kind:10s} {usage.requests} requests, "
                  f"{usage.input_tokens} in / {usage.output_tokens} out tokens", flush=True)
        text = str(result.final_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": kind, "agent": agent.name, "model": self.model_name, "prompt": prompt,
                                    "data": situation.data, "output_text": text,
                                    "usage": {"requests": usage.requests, "input_tokens": usage.input_tokens,
                                              "output_tokens": usage.output_tokens}},
                                   indent=1, default=str), encoding="utf-8")
        return self._validated(text, schema)

    def _validated(self, text: str, schema: type[BaseModel]) -> BaseModel | None:
        try:
            return parse_reply(text, schema)
        except (ValueError, ValidationError):
            self.stats.invalid += 1
            return None

    def reject(self) -> None:
        """A parsed answer that named an option that does not exist: the rules decision stands."""
        self.stats.invalid += 1

    def compare(self, kind: str, differs: bool) -> None:
        """Record whether a valid answer departed from what the rules agent would have chosen."""
        self.stats.departed[kind] += bool(differs)


def list_models(settings: Settings) -> list[str]:
    from openai import OpenAI

    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    return sorted(m.id for m in client.models.list())
