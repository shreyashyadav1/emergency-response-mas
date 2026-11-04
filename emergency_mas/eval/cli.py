"""Command line: python -m emergency_mas.eval <command>."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .. import params as P
from ..datagen import generate_scenario
from .report import build_report, figure
from .runner import POLICIES, run_grid, run_one, write_run_log

ROOT = Path(__file__).resolve().parents[2]


def cmd_run(args: argparse.Namespace) -> int:
    start = time.time()
    frames = [run_grid(args.themes, args.stress, range(1, args.seeds + 1), args.policies, progress=True)]
    if args.extra:
        frames.append(run_grid(["earthquake", "highway_pileup"], ["high"], range(1, args.seeds + 1), args.policies, progress=True))
    frame = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"{len(frame)} runs in {time.time() - start:.0f}s -> {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.runs)
    fig = Path(args.figure)
    figure(frame, fig)
    out = Path(args.out)
    out.write_text(build_report(frame, fig, figure_link=str(fig.relative_to(out.parent)) if fig.is_relative_to(out.parent)
                                else str(fig)), encoding="utf-8")
    print(f"wrote {out} and {fig}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    for policy in args.policies:
        scenario = generate_scenario(args.theme, args.stress, args.seed)
        _, result = run_one(args.theme, args.stress, args.seed, policy, record=True)
        path = write_run_log(result, scenario.meta.model_dump(mode="json"),
                             Path(args.out) / f"{scenario.meta.scenario_id}__{policy}.jsonl")
        print(f"{policy}: weighted survival {result.metrics['weighted_survival']}  -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m emergency_mas.eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run every policy on the same scenarios")
    run.add_argument("--themes", nargs="+", default=["urban_mci"], choices=list(P.THEMES))
    run.add_argument("--stress", nargs="+", default=["low", "medium", "high"], choices=list(P.STRESS))
    run.add_argument("--seeds", type=int, default=30)
    run.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    run.add_argument("--no-extra", dest="extra", action="store_false", help="skip the earthquake and pile-up check")
    run.add_argument("--out", default=str(ROOT / "results" / "runs.csv"))
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="write docs/RESULTS.md and the results figure")
    rep.add_argument("--runs", default=str(ROOT / "results" / "runs.csv"))
    rep.add_argument("--out", default=str(ROOT / "docs" / "RESULTS.md"))
    rep.add_argument("--figure", default=str(ROOT / "docs" / "figures" / "results.png"))
    rep.set_defaults(func=cmd_report)

    rp = sub.add_parser("replay", help="save minute-by-minute run logs for the dashboard")
    rp.add_argument("--theme", default="urban_mci", choices=list(P.THEMES))
    rp.add_argument("--stress", default="medium", choices=list(P.STRESS))
    rp.add_argument("--seed", type=int, default=78)
    rp.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    rp.add_argument("--out", default=str(ROOT / "runs"))
    rp.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
