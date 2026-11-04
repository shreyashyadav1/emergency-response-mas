"""Command line: python -m emergency_mas.datagen <command>."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .. import params as P
from .analysis import contention_report, pressure_indices, victim_counts
from .calibrate import calibration_frame, summarize
from .docs import assumptions_md, data_dictionary_md
from .export import scenario_hash, write_scenario
from .generator import generate_scenario
from .history import base_rates, generate_history
from .validate import validate_scenario


def _resolve(args: argparse.Namespace) -> tuple[str, str]:
    if args.preset:
        return P.PRESETS[args.preset]["theme"], P.PRESETS[args.preset]["stress"]
    return args.theme, args.stress


def cmd_generate(args: argparse.Namespace) -> int:
    theme, stress = _resolve(args)
    s = generate_scenario(theme, stress, args.seed)
    manifest = write_scenario(s, args.out)
    checks = validate_scenario(s)
    print(f"{s.meta.scenario_id}  sha256 {manifest['scenario_sha256'][:16]}...")
    print("counts   ", victim_counts(s))
    print("pressure ", pressure_indices(s))
    print("conflicts", {k: v for k, v in contention_report(s).items() if k != "ambulance_pairs"})
    failed = [c for c in checks if not c.passed]
    print(f"checks    {len(checks) - len(failed)}/{len(checks)} passed", *(f"\n  FAIL {c.name}: {c.detail}" for c in failed))
    print(f"written to {Path(args.out) / s.meta.scenario_id}")
    return 1 if failed else 0


def cmd_library(args: argparse.Namespace) -> int:
    count = 0
    for preset, spec in P.PRESETS.items():
        for seed in range(1, args.seeds + 1):
            write_scenario(generate_scenario(spec["theme"], spec["stress"], seed), args.out)
            count += 1
    print(f"wrote {count} scenarios to {args.out}")
    return 0


def verify_library(root: str | Path) -> tuple[int, int]:
    """Regenerate every stored scenario; return (how many match their manifest and file, how many in total)."""
    folders = sorted(p for p in Path(root).iterdir() if (p / "MANIFEST.json").exists())
    good = 0
    for folder in folders:
        manifest = json.loads((folder / "MANIFEST.json").read_text())
        again = scenario_hash(generate_scenario(manifest["theme"], manifest["stress"], manifest["seed"]))
        stored = hashlib.sha256((folder / "scenario.json").read_bytes()).hexdigest()
        if again == manifest["scenario_sha256"] == stored:
            good += 1
        else:
            print(f"MISMATCH {folder.name}")
    return good, len(folders)


def cmd_verify(args: argparse.Namespace) -> int:
    good, total = verify_library(args.root)
    print(f"{good}/{total} scenarios regenerate to identical bytes")
    return 0 if good == total else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    frame = calibration_frame(range(1, args.seeds + 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "stress_indices.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(out / "stress_summary.csv")
    print(summary.round(2).to_string())
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    history = generate_history(args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history.to_csv(out / "past_incidents.csv", index=False)
    print(base_rates(history).to_string())
    print(f"wrote {len(history)} past incidents to {out / 'past_incidents.csv'}")
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DATA_DICTIONARY.md").write_text(data_dictionary_md(), encoding="utf-8")
    (out / "ASSUMPTIONS.md").write_text(assumptions_md(), encoding="utf-8")
    print(f"wrote DATA_DICTIONARY.md and ASSUMPTIONS.md to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m emergency_mas.datagen", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write one scenario")
    g.add_argument("--preset", choices=list(P.PRESETS))
    g.add_argument("--theme", choices=list(P.THEMES), default="urban_mci")
    g.add_argument("--stress", choices=list(P.STRESS), default="medium")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data/scenarios")
    g.set_defaults(func=cmd_generate)

    lib = sub.add_parser("library", help="write the frozen library: every preset for seeds 1..N")
    lib.add_argument("--seeds", type=int, default=5)
    lib.add_argument("--out", default="data/scenarios")
    lib.set_defaults(func=cmd_library)

    ver = sub.add_parser("verify", help="regenerate each stored scenario and compare hashes")
    ver.add_argument("--root", default="data/scenarios")
    ver.set_defaults(func=cmd_verify)

    cal = sub.add_parser("calibrate", help="spread of the stress indices over many seeds")
    cal.add_argument("--seeds", type=int, default=200)
    cal.add_argument("--out", default="data/calibration")
    cal.set_defaults(func=cmd_calibrate)

    his = sub.add_parser("history", help="synthetic past incidents used as base rates")
    his.add_argument("--n", type=int, default=600)
    his.add_argument("--out", default="data/history")
    his.set_defaults(func=cmd_history)

    doc = sub.add_parser("docs", help="write DATA_DICTIONARY.md and ASSUMPTIONS.md")
    doc.add_argument("--out", default="docs")
    doc.set_defaults(func=cmd_docs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
