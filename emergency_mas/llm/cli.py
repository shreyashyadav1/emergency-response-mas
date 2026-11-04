"""Command line for the LLM agents (M3).

    python -m emergency_mas.llm models               # the models your key can use
    python -m emergency_mas.llm smoke                # one small tool-calling request, to check the setup
    python -m emergency_mas.llm run --dry-run        # count the decisions and tokens of the planned runs, no calls
    python -m emergency_mas.llm run                  # run M3, calling the model only where no answer is cached
    python -m emergency_mas.llm run --cache-only     # replay cached answers, no key needed
    python -m emergency_mas.llm report               # docs/LLM_RESULTS.md from results/llm_runs.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

try:
    import fcntl
except ImportError:                     # Windows: no file locks, so run one `run` at a time there
    fcntl = None

import pandas as pd
from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel
from scipy.stats import wilcoxon

from .. import params as P
from ..datagen import generate_scenario
from ..eval.runner import run_one, write_run_log
from ..eval.stats import bootstrap_ci
from .backend import (REQUESTS_PER_DECISION, ROOT, Backend, DailyQuotaExceeded, NoCredit, Settings, Situation, list_models,
                      make_model, parse_reply)

RUNS_CSV = ROOT / "results" / "llm_runs.csv"
DRY_CSV = ROOT / "results" / "llm_dry_run.csv"
REPORT = ROOT / "docs" / "LLM_RESULTS.md"
KINDS = {"assessment": "Assessment of a new call", "routing": "Order of hospitals to ask", "counter": "Hospital: counter-offer or refuse",
         "conflict": "Contested unit awards", "settlement": "Admission settlement"}


def cmd_models(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if not settings.api_key:
        print(f"No key found. Put {settings.key_name}=... in the project's .env file.")
        return 1
    for name in list_models(settings):
        print(name)
    return 0


class SmokeAnswer(BaseModel):
    hospital: str
    bays: int


@function_tool(strict_mode=False)
def free_bays(ctx: RunContextWrapper[Situation]) -> str:
    """Free ER bays at each hospital."""
    return json.dumps(ctx.context.data["free_bays"])


def cmd_smoke(args: argparse.Namespace) -> int:
    """One agent, one tool, one JSON answer: checks the key, the model name, tool calling and the reply format."""
    settings = Settings.from_env(model=args.model)
    agent = Agent(name="Smoke test", tools=[free_bays], model=make_model(settings), model_settings=ModelSettings(temperature=0),
                  instructions='Call free_bays, then answer with only a JSON object: {"hospital": "<id with the most free bays>", "bays": <number>}')
    started = time.time()
    try:
        result = Runner.run_sync(agent, "Which hospital has the most free ER bays?",
                                 context=Situation({"free_bays": {"H1": 3, "H2": 7, "H3": 1}}), max_turns=4)
    except (DailyQuotaExceeded, NoCredit) as exc:
        print(f"model {settings.model}: {exc}.")
        return 1
    usage = result.context_wrapper.usage
    tools_called = sum(1 for item in result.new_items if item.type == "tool_call_item")
    print(f"model {settings.model}: {usage.requests} requests, {usage.input_tokens} input and {usage.output_tokens} output "
          f"tokens, {tools_called} tool call(s), {time.time() - started:.1f} s")
    print("reply:", result.final_output)
    answer = parse_reply(str(result.final_output), SmokeAnswer)
    ok = answer.hospital == "H2" and answer.bays == 7 and tools_called >= 1
    print("OK: the SDK, the key, tool calling and the JSON reply all work." if ok else "The reply was not the expected one.")
    return 0 if ok else 1


def _upsert(path: Path, rows: list[dict], keys: list[str]) -> pd.DataFrame:
    """Replace or add rows by `keys`. Several runs can share the file: one writes at a time, and readers
    always see a whole file, the old one or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        new = pd.DataFrame(rows)
        if path.exists():
            old = pd.read_csv(path)
            merged = old.merge(new[keys], on=keys, how="left", indicator=True)
            old = old[(merged["_merge"] == "left_only").to_numpy()]
            new = pd.concat([old, new], ignore_index=True)
        tmp = path.with_name(path.name + ".tmp")
        new.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return new


def _recorded(theme: str, stress: str, seed: int, model: str) -> dict | None:
    """The run already in results/llm_runs.csv for this scenario and model, if any."""
    if not RUNS_CSV.exists():
        return None
    old = pd.read_csv(RUNS_CSV)
    hit = old[(old.theme == theme) & (old.stress == stress) & (old.seed == seed) & (old.model == model)]
    return hit.iloc[0].to_dict() if len(hit) else None


def cmd_run(args: argparse.Namespace) -> int:
    mode = "dry-run" if args.dry_run else "cache" if args.cache_only else "live"
    llm = Backend(Settings.from_env(model=args.model, rpm=args.rpm), mode=mode)
    llm.verbose = mode == "live"
    rows, keys, unmatched = [], ["theme", "stress", "seed", "model"], 0
    print(f"M3 on {args.theme}/{args.stress}, seeds {args.seeds}, mode {mode}, model {llm.model_name}")
    for seed in args.seeds:
        llm.new_run()
        started = time.time()
        try:
            row, result = run_one(args.theme, args.stress, seed, "M3", record=mode != "dry-run", llm=llm)
        except (DailyQuotaExceeded, NoCredit) as exc:
            later = ("once credit is added" if isinstance(exc, NoCredit) else
                     "once the quota resets (or with --model set to another model)")
            print(f"  seed {seed}: stopped after {llm.stats.live} new answers, {exc}. Every answer so far is cached, "
                  f"so the same command continues from here {later}.")
            break
        row.update(llm.stats.row(), mode=mode, model=llm.model_name, seconds=round(time.time() - started, 1))
        rows.append(row)
        s = llm.stats
        if mode == "dry-run":
            print(f"  seed {seed}: {sum(s.decisions.values())} decisions {dict(s.decisions)}, about "
                  f"{REQUESTS_PER_DECISION * (sum(s.decisions.values()) - s.cached)} requests and "
                  f"{s.estimated_input_tokens:,} input tokens still to send ({s.cached} already cached)")
            continue
        scenario = generate_scenario(args.theme, args.stress, seed)
        write_run_log(result, scenario.meta.model_dump(mode="json"), ROOT / "runs" / f"{scenario.meta.scenario_id}__M3.jsonl")
        print(f"  seed {seed}: weighted survival {row['weighted_survival']:.3f}; {sum(s.decisions.values())} decisions, "
              f"{s.live} live, {s.cached} cached, {s.fallbacks} fell back to rules; {s.requests} requests, "
              f"{s.input_tokens:,} in / {s.output_tokens:,} out tokens; {row['seconds']} s")
        if mode == "cache":                 # a replay checks the recorded run and leaves its live counts alone
            before = _recorded(args.theme, args.stress, seed, llm.model_name)
            same = before is not None and abs(before["weighted_survival"] - row["weighted_survival"]) < 1e-9
            unmatched += not same
            print("    same as the recorded run" if same else
                  "    NOT the same as the recorded run" if before is not None else "    no recorded run to compare")
        else:
            _upsert(RUNS_CSV, [row], keys)  # saved seed by seed, so a stopped run keeps the seeds it finished
    if mode == "dry-run":
        frame = pd.DataFrame(rows)
        frame.to_csv(DRY_CSV, index=False)
        decisions = int(frame.llm_decisions.sum() - frame.llm_cached.sum())
        minutes = REQUESTS_PER_DECISION * decisions / max(llm.settings.rpm, 0.1)
        print(f"total still to send: {decisions} decisions, about {REQUESTS_PER_DECISION * decisions} requests and "
              f"{int(frame.llm_estimated_input_tokens.sum()):,} input tokens; about {minutes:.0f} minutes at "
              f"{llm.settings.rpm:g} requests per minute. Nothing was sent.")
        return 0
    if not rows:
        return 1
    if mode == "cache":
        print(f"replayed {len(rows)} seeds from the cache: "
              + (f"{unmatched} differ from or are missing in {RUNS_CSV.name}" if unmatched else "all match the recorded runs"))
        return 1 if unmatched else 0
    build_report()
    print(f"wrote {RUNS_CSV.relative_to(ROOT)} and {REPORT.relative_to(ROOT)}")
    return 0


def build_report(runs_path: Path = RUNS_CSV, out: Path = REPORT) -> str:
    """M3 against M2 and the nearest-first baseline on the same seeds, plus what the model did."""
    llm = pd.read_csv(runs_path)
    llm = llm[llm["mode"].isin(["live", "cache"])]
    L = ["# LLM agents (M3)", "",
         "Generated by `python -m emergency_mas.llm report` from `results/llm_runs.csv`. Do not edit by hand.", "",
         "M3 is M2 with its judgment calls made by OpenAI Agents SDK agents on a language model: the assessment of each new "
         "call, the order of hospitals to ask, a hospital's choice between a counter-offer and a refusal, contested unit "
         "awards and admission settlements. Everything else, including every check on the agents' actions, is the same as "
         "M2. M2, M1, B1 and B0 are recomputed here on the same seeds.", ""]
    for (theme, stress, model), g in llm.groupby(["theme", "stress", "model"]):
        rows, diffs = [], {p: [] for p in ("M2", "M1", "B1", "B0")}
        for r in g.sort_values("seed").itertuples():
            other = {p: run_one(theme, stress, int(r.seed), p)[0] for p in diffs}
            for p, d in diffs.items():
                d.append(r.weighted_survival - other[p]["weighted_survival"])
            rows.append(f"| {int(r.seed)} | {r.weighted_survival:.3f} | {other['M2']['weighted_survival']:.3f} | "
                        f"{other['B0']['weighted_survival']:.3f} | {diffs['M2'][-1]:+.3f} | {diffs['B0'][-1]:+.3f} | "
                        f"{r.red_survivors:.2f} | {other['M2']['red_survivors']:.2f} | {r.victims_served:.0f} | "
                        f"{other['M2']['victims_served']:.0f} | {int(r.llm_decisions)} | {int(r.llm_fallbacks)} |")
        L += [f"## {theme} / {stress}, model `{model}`", "",
              "| Seed | M3 weighted survival | M2 | B0 | M3 minus M2 | M3 minus B0 | M3 RED survivors | M2 RED survivors | "
              "M3 victims reaching care | M2 | LLM decisions | Fell back to rules |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows, ""]
        for p, d in diffs.items():
            lo, hi = bootstrap_ci(d)
            better, equal = sum(x > 1e-12 for x in d), sum(abs(x) <= 1e-12 for x in d)
            try:
                pv = f"{wilcoxon(d).pvalue:.1g}" if better + equal < len(d) or better else "1"
            except ValueError:                  # every difference is zero
                pv = "1"
            L.append(f"- M3 minus {p}, weighted survival: mean {sum(d) / len(d):+.3f} [{lo:+.3f}, {hi:+.3f}]; M3 better "
                     f"on {better}, equal on {equal}, worse on {len(d) - better - equal} of {len(d)} seeds; Wilcoxon p = {pv}.")
        L.append("")
        L += ["| Decision | Asked | Departed from the rules agent's choice |", "|---|---|---|"]
        for kind, label in KINDS.items():
            asked = int(g.get(f"llm_{kind}", pd.Series([0])).sum())
            departed = int(g.get(f"llm_departed_{kind}", pd.Series([0])).sum())
            L.append(f"| {label} | {asked} | {departed} ({departed / asked:.0%}) |" if asked else f"| {label} | 0 | n/a |")
        L += ["", f"Model requests {int(g.llm_requests.sum())}, input tokens {int(g.llm_input_tokens.sum()):,}, output tokens "
              f"{int(g.llm_output_tokens.sum()):,} (live calls only; replays from the cache cost nothing). Answers that failed "
              f"validation or calls that failed: {int(g.llm_fallbacks.sum())}, each replaced by the rules decision.", ""]
    n = len(llm.seed.unique()) if len(llm) else 0
    L += ["## Reproduce", "", "```bash",
          *[f".venv/bin/python -m emergency_mas.llm run --cache-only --theme {theme} --stress {stress} --model {model} "
            f"--seeds {' '.join(str(int(s)) for s in sorted(g.seed.unique()))}"
            for (theme, stress, model), g in llm.groupby(["theme", "stress", "model"])],
          "```", "",
          "Every model answer is stored in `data/llm_cache/` with the prompt and tool data it answered, so these commands "
          "replay the runs exactly without a key and say whether each seed matches the run recorded here. A fresh live run "
          "can differ, because hosted models are not exactly repeatable even at temperature 0.", "",
          "## Limits", "",
          (f"- {n} seeds is a small sample, so the intervals above are wide; " if n < 10 else f"- {n} seeds per model; ")
          + "the 600-run comparison of the rules agents is in RESULTS.md.",
          "- The language model sees the same numbers the rules agents use, through its tools, and decides; it does not "
          "do the arithmetic itself.", ""]
    text = "\n".join(L)
    out.write_text(text, encoding="utf-8")
    return text


def cmd_report(args: argparse.Namespace) -> int:
    if not RUNS_CSV.exists():
        print("No LLM runs yet: run `python -m emergency_mas.llm run` first.")
        return 1
    build_report()
    print(f"wrote {REPORT.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m emergency_mas.llm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list the models your key can use").set_defaults(func=cmd_models)
    smoke = sub.add_parser("smoke", help="one small tool-calling request")
    smoke.add_argument("--model")
    smoke.set_defaults(func=cmd_smoke)
    run = sub.add_parser("run", help="run M3 on a few seeds")
    run.add_argument("--theme", default="urban_mci", choices=list(P.THEMES))
    run.add_argument("--stress", default="medium", choices=list(P.STRESS))
    run.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    run.add_argument("--model")
    run.add_argument("--rpm", type=float, help="model requests per minute (default LLM_RPM or 8)")
    group = run.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="count decisions and tokens, send nothing")
    group.add_argument("--cache-only", action="store_true", help="replay cached answers only")
    run.set_defaults(func=cmd_run)
    sub.add_parser("report", help="write docs/LLM_RESULTS.md").set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    return args.func(args)
