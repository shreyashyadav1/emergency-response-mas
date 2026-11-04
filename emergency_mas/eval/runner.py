"""Run policies over scenarios and collect one row of metrics per run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from scipy.stats import spearmanr

from .. import params as P
from ..agents import AgentSystem
from ..datagen import generate_scenario
from ..policies import NearestFirst, PriorityGreedy
from ..sim import RunResult, Simulator

POLICIES: dict[str, Callable] = {
    "B0": NearestFirst,
    "B1": PriorityGreedy,
    "M1": lambda: AgentSystem(negotiation=False),
    "M2": lambda: AgentSystem(negotiation=True),
}
POLICY_LABEL = {
    "B0": "B0 nearest first",
    "B1": "B1 priority greedy",
    "M1": "M1 agents, no negotiation",
    "M2": "M2 agents with negotiation",
    "M3": "M3 LLM agents (OpenAI Agents SDK)",
}


def make_policy(policy: str, llm=None):
    """A fresh policy object. M3 is not in POLICIES, so a plain experiment run never calls a language model;
    without a backend it only replays cached answers."""
    if policy == "M3":
        from ..agents.llm_system import LLMAgentSystem
        from ..llm.backend import Backend

        return LLMAgentSystem(llm or Backend.from_env(mode="cache"))
    return POLICIES[policy]()


def assessment_quality(scenario, policy) -> dict:
    """The Incident Assessment Agent's first estimate against the raw caller report: bias and root-mean-square
    error of the victim and RED counts, and how well each ranks incidents by the oracle urgency."""
    first = getattr(getattr(policy, "assessor", None), "first", None)
    if not first:
        return {}
    rows = []
    for inc in scenario.incidents:
        a = first.get(inc.incident_id)
        if a is None:
            continue
        rows.append({"agent_v": a.victims - inc.n_victims, "report_v": inc.report.reported_victims - inc.n_victims,
                     "agent_r": a.red - inc.n_red, "report_r": inc.report.reported_critical - inc.n_red,
                     "agent_rank": a.priority, "report_rank": 10 * inc.report.reported_critical + inc.report.reported_victims,
                     "truth": inc.urgency_truth})
    if len(rows) < 3:
        return {}
    f = pd.DataFrame(rows)
    out = {}
    for who in ("agent", "report"):
        out[f"{who}_victims_bias"] = round(f[f"{who}_v"].mean(), 3)
        out[f"{who}_victims_rmse"] = round(float((f[f"{who}_v"] ** 2).mean() ** 0.5), 3)
        out[f"{who}_red_bias"] = round(f[f"{who}_r"].mean(), 3)
        out[f"{who}_red_rmse"] = round(float((f[f"{who}_r"] ** 2).mean() ** 0.5), 3)
        out[f"{who}_rank_corr"] = round(float(spearmanr(f[f"{who}_rank"], f.truth).statistic), 3)
    return out


def run_one(theme: str, stress: str, seed: int, policy: str, record: bool = False, llm=None) -> tuple[dict, RunResult]:
    scenario = generate_scenario(theme, stress, seed)
    agent = make_policy(policy, llm)
    result = Simulator(scenario, agent, record=record).run()
    row = {"theme": theme, "stress": stress, "seed": seed, "policy": policy, **result.metrics,
           **assessment_quality(scenario, agent)}
    return row, result


def run_grid(themes: Iterable[str], stresses: Iterable[str], seeds: Iterable[int],
             policies: Iterable[str] = tuple(POLICIES), progress: bool = False) -> pd.DataFrame:
    rows = []
    for theme in themes:
        for stress in stresses:
            for seed in seeds:
                for policy in policies:
                    rows.append(run_one(theme, stress, seed, policy)[0])
            if progress:
                print(f"  done {theme} / {stress}", flush=True)
    return pd.DataFrame(rows)


def run_log_header(result: RunResult, scenario_meta: dict) -> dict:
    """The first line of a run log: which scenario and policy, and the final metrics."""
    return {"header": True, "scenario": scenario_meta, "policy": result.policy,
            "metrics": result.metrics, "horizon": len(result.ticks),
            "survival_model": {k: {"window": P.GOLDEN_WINDOW_MIN[k], "half_life": P.DECAY_HALF_LIFE_MIN[k]}
                               for k in P.TRIAGE}}


def write_run_log(result: RunResult, scenario_meta: dict, path: str | Path) -> Path:
    """One JSON line per minute, preceded by a header line: what the dashboard replays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(run_log_header(result, scenario_meta)) + "\n")
        for tick in result.ticks:
            f.write(json.dumps(tick, separators=(",", ":")) + "\n")
    return path


def read_run_log(path: str | Path) -> tuple[dict, list[dict]]:
    """The header and the minute records of a log written by `write_run_log`."""
    with Path(path).open(encoding="utf-8") as f:
        header = json.loads(f.readline())
        if not header.get("header"):
            raise ValueError(f"{path} does not start with a run log header")
        ticks = [json.loads(line) for line in f if line.strip()]
    return header, ticks
