"""Figures and quoted numbers for the progress slides. Everything is computed from the generated data.

Run from the project root:  .venv/bin/python scripts/make_figures.py
Writes docs/figures/scenario_map.png and docs/figures/deck_stats.json.
"""
from __future__ import annotations

import json
import math
from collections import Counter
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergency_mas import params as P  # noqa: E402
from emergency_mas.datagen import generate_scenario, validate_scenario  # noqa: E402
from emergency_mas.datagen.analysis import contention_report, pressure_indices, travel_model, victim_counts  # noqa: E402
from emergency_mas.datagen.calibrate import calibration_frame  # noqa: E402
from emergency_mas.datagen.cli import verify_library  # noqa: E402
from emergency_mas.datagen.history import generate_history  # noqa: E402
from emergency_mas.eval import paired, run_one  # noqa: E402

OUT = ROOT / "docs" / "figures"
SHOWCASE = ("urban_mci", "medium", 78)

INK, SLATE, MUTED = "#0f2233", "#52606d", "#6b7785"
SURFACE, HAIR, LINK = "#ffffff", "#d5dce4", "#b4bfcb"
ZONE_FILL = {"core": "#e4e9ef", "inner": "#edf1f5", "outer": "#f6f8fa"}
HOSPITAL_BLUE = "#1c5cab"
TRIAGE_COLOR = {"RED": "#d03b3b", "YELLOW": "#fab219", "GREEN": "#0ca30c"}
ON_TRIAGE = {"RED": "#ffffff", "YELLOW": INK, "GREEN": INK}


def worst_triage(incident) -> str:
    return "RED" if incident.n_red else "YELLOW" if incident.n_yellow else "GREEN"


def nearest_station(s, incident):
    tm = travel_model(s)
    station = {st.station_id: st for st in s.stations}
    amb = min(s.ambulances, key=lambda a: (tm.minutes((station[a.base_station_id].x, station[a.base_station_id].y),
                                                       (incident.x, incident.y), a.speed_kmh, incident.report_time_min),
                                           a.ambulance_id))
    return station[amb.base_station_id]


FIG_IN = 5.6
IN_PER_KM = FIG_IN * 0.98 / 24.6      # the axes fill 98% of a square figure holding 24.6 km


class Labeler:
    """Greedy label placement: try a few spots around a marker and keep the first that is free."""

    def __init__(self, xlim: tuple[float, float], ylim: tuple[float, float]):
        self.boxes: list[tuple[float, float, float, float]] = []
        self.xlim, self.ylim = xlim, ylim

    @staticmethod
    def radius_km(marker_area_pt2: float) -> float:
        return math.sqrt(marker_area_pt2) / 2 / 72 / IN_PER_KM

    def block(self, x: float, y: float, r: float) -> None:
        self.boxes.append((x - r, y - r, x + r, y + r))

    def _overlap(self, a, b) -> float:
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    def put(self, ax, x: float, y: float, r: float, text: str, size: float, **style) -> None:
        lines = text.split("\n")
        w = max(len(line) for line in lines) * 0.5 * size / 72 / IN_PER_KM      # an average glyph is about half an em
        h = len(lines) * 1.2 * size / 72 / IN_PER_KM
        g, d = 0.2, 0.7 * r
        options = [  # (ha, va, anchor x, anchor y, box)
            ("left", "center", x + r + g, y, (x + r + g, y - h / 2, x + r + g + w, y + h / 2)),
            ("right", "center", x - r - g, y, (x - r - g - w, y - h / 2, x - r - g, y + h / 2)),
            ("center", "bottom", x, y + r + g, (x - w / 2, y + r + g, x + w / 2, y + r + g + h)),
            ("center", "top", x, y - r - g, (x - w / 2, y - r - g - h, x + w / 2, y - r - g)),
            ("left", "bottom", x + d, y + d, (x + d, y + d, x + d + w, y + d + h)),
            ("left", "top", x + d, y - d, (x + d, y - d - h, x + d + w, y - d)),
            ("right", "bottom", x - d, y + d, (x - d - w, y + d, x - d, y + d + h)),
            ("right", "top", x - d, y - d, (x - d - w, y - d - h, x - d, y - d)),
        ]

        def cost(box) -> float:
            outside = (box[0] < self.xlim[0]) + (box[2] > self.xlim[1]) + (box[1] < self.ylim[0]) + (box[3] > self.ylim[1])
            return outside * 100 + sum(self._overlap(box, b) for b in self.boxes)

        ha, va, ax_, ay_, box = min(options, key=lambda o: cost(o[4]))
        self.boxes.append(box)
        ax.text(ax_, ay_, text, ha=ha, va=va, fontsize=size, zorder=11, linespacing=1.2, **style)


def station_label(s, st) -> str:
    units = [a for a in s.ambulances if a.base_station_id == st.station_id]
    types = {a.unit_type for a in units}
    ids = " ".join(a.ambulance_id for a in units)
    return f"{st.station_id}: {ids} ({types.pop()})" if len(types) == 1 else f"{st.station_id}: " + ", ".join(
        f"{a.ambulance_id} {a.unit_type}" for a in units)


def draw_map(s, path: Path) -> None:
    plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(FIG_IN, FIG_IN), dpi=250)
    fig.patch.set_facecolor(SURFACE)
    side, (cx, cy) = s.meta.city_km, P.CITY_CENTER
    xlim = ylim = (-0.3, side + 0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    lab = Labeler(xlim, ylim)

    ax.add_patch(Rectangle((0, 0), side, side, fc=ZONE_FILL["outer"], ec=HAIR, lw=0.8, zorder=0))
    ax.add_patch(Circle((cx, cy), P.ZONE_RADII_KM[1], fc=ZONE_FILL["inner"], ec=HAIR, lw=0.8, zorder=1))
    ax.add_patch(Circle((cx, cy), P.ZONE_RADII_KM[0], fc=ZONE_FILL["core"], ec=HAIR, lw=0.8, zorder=2))
    for name, radius in (("core", P.ZONE_RADII_KM[0]), ("inner zone", P.ZONE_RADII_KM[1]), ("outer zone", side / 2)):
        y_label = cy + radius - 0.55
        ax.text(cx, y_label, name, color=MUTED, fontsize=10, ha="center", va="center", zorder=3)
        lab.boxes.append((cx - 1.6, y_label - 0.5, cx + 1.6, y_label + 0.5))

    # every marker first, so labels know what to avoid
    stations = {st.station_id: st for st in s.stations}
    incident_area = {i.incident_id: max(190, 40 * i.n_victims) for i in s.incidents}
    for inc in s.incidents:
        lab.block(inc.x, inc.y, lab.radius_km(incident_area[inc.incident_id]))
    for st in s.stations:
        lab.block(st.x, st.y, lab.radius_km(62))
    for h in s.hospitals:
        lab.block(h.x, h.y, lab.radius_km(34 * h.er_bays_total))

    for inc in s.incidents:
        st = nearest_station(s, inc)
        ax.plot([inc.x, st.x], [inc.y, st.y], color=LINK, lw=0.7, zorder=4, solid_capstyle="round")

    for st in s.stations:
        ax.scatter([st.x], [st.y], marker="^", s=62, c=SLATE, edgecolors=SURFACE, linewidths=1.2, zorder=6)
        lab.put(ax, st.x, st.y, lab.radius_km(62), station_label(s, st), 9.6, color=SLATE)

    short = {"H1": "Harborview", "H2": "Eastgate", "H3": "Northfield"}
    for h in s.hospitals:
        area = 34 * h.er_bays_total
        ax.scatter([h.x], [h.y], marker="s", s=area, c=HOSPITAL_BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=7)
        ax.text(h.x, h.y, h.hospital_id, color="white", fontsize=9.6, fontweight="bold", ha="center", va="center", zorder=8)
        lab.put(ax, h.x, h.y, lab.radius_km(area), f"{short.get(h.hospital_id, h.name)}\n{h.er_bays_free} of {h.er_bays_total} bays free",
                10.4, color=INK)

    for inc in sorted(s.incidents, key=lambda i: -incident_area[i.incident_id]):     # small circles end up on top
        kind = worst_triage(inc)
        ax.scatter([inc.x], [inc.y], s=incident_area[inc.incident_id], c=TRIAGE_COLOR[kind], edgecolors=SURFACE,
                   linewidths=1.5, zorder=9)
        ax.text(inc.x, inc.y, str(int(inc.incident_id[-2:])), color=ON_TRIAGE[kind], fontsize=9.8, fontweight="bold",
                ha="center", va="center", zorder=10)

    ax.plot([side - 6.0, side - 1.0], [0.9, 0.9], color=SLATE, lw=1.2, solid_capstyle="butt", zorder=6)
    ax.text(side - 3.5, 1.4, "5 km", color=SLATE, fontsize=10, ha="center", va="bottom", zorder=6)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def typical_seed(frame, theme: str, stress: str) -> tuple[int, int]:
    """The seed whose indices are closest to the median of all seeds, and how many seeds were compared."""
    cols = ["victims", "red", "yellow", "ambulance_pressure", "er_bay_pressure", "specialist_pressure", "ambulance_pairs"]
    d = frame[(frame.theme == theme) & (frame.stress == stress)].copy()
    z = ((d[cols] - d[cols].median()) / d[cols].std()) ** 2
    d["distance"] = z.sum(axis=1) ** 0.5
    return int(d.sort_values("distance").iloc[0].seed), len(d)


RESULT_METRICS = ("weighted_survival", "expected_survivors", "red_survivors", "victims_served",
                  "mean_first_response_min", "ramp_wait_min", "unserved_critical_incidents", "ambulance_utilization")


def _diff(row) -> dict:
    return {"diff": round(float(row["diff"]), 3), "ci_low": round(float(row.ci_low), 3), "ci_high": round(float(row.ci_high), 3),
            "p": float(row.p_value), "better_share": round(float(row.a_better_share), 3)}


def results_stats(frame: pd.DataFrame) -> dict:
    """Headline numbers for the results slide, straight from results/runs.csv."""
    urban = frame[frame.theme == "urban_mci"]
    out = {"runs": len(frame), "seeds": int(frame.seed.nunique()),
           "settings": int(frame.groupby(["theme", "stress"]).ngroups), "by_stress": {}, "other_themes": {}}
    for stress in P.STRESS:
        g = urban[urban.stress == stress]
        means = {p: {m: round(float(g[g.policy == p][m].mean()), 3) for m in RESULT_METRICS} for p in ("B0", "B1", "M1", "M2")}
        best = "B1" if means["B1"]["weighted_survival"] >= means["B0"]["weighted_survival"] else "B0"
        out["by_stress"][stress] = {
            "means": means, "best_baseline": best,
            "m2_vs_best": _diff(paired(g, "weighted_survival", "M2", best, by=("stress",)).iloc[0]),
            "m2_vs_best_survivors": _diff(paired(g, "expected_survivors", "M2", best, by=("stress",)).iloc[0]),
            "m2_vs_b0_survivors": _diff(paired(g, "expected_survivors", "M2", "B0", by=("stress",)).iloc[0]),
            "negotiation": _diff(paired(g, "weighted_survival", "M2", "M1", by=("stress",)).iloc[0]),
            "gain_vs_b0": {p: round(means[p]["expected_survivors"] - means["B0"]["expected_survivors"], 2) for p in ("B1", "M1", "M2")},
        }
    for (theme, stress), g in frame[frame.theme != "urban_mci"].groupby(["theme", "stress"]):
        out["other_themes"][f"{theme}/{stress}"] = {p: round(float(g[g.policy == p]["weighted_survival"].mean()), 3)
                                                     for p in ("B0", "B1", "M1", "M2")}
    m2 = frame[frame.policy == "M2"]
    out["negotiation_per_run"] = {k: round(float(m2[k].mean()), 1) for k in (
        "messages", "messages_conflict_notice", "messages_counter_proposal", "messages_staff_request",
        "messages_allocation_decision", "messages_admission_request", "staff_moves", "diversions", "surges_opened")}
    out["assessment"] = {k: round(float(m2[k].mean()), 3) for k in (
        "report_victims_bias", "agent_victims_bias", "report_victims_rmse", "agent_victims_rmse",
        "report_red_bias", "agent_red_bias", "report_rank_corr", "agent_rank_corr")}
    return out


def example_thread(theme: str, stress: str, seeds) -> dict:
    """The first admission settled by a staff loan: request, counter-offer, staff search, decisions."""
    for seed in seeds:
        _, result = run_one(theme, stress, seed, "M2", record=True)
        msgs = [m for tick in result.ticks for m in tick["messages"]]
        for m in msgs:
            cond = m["proposed_action"].get("condition") or ""
            if m["type"] != "counter_proposal" or not cond.startswith("staff:"):
                continue
            unit, t = m["proposed_action"]["unit_id"], m["t"]
            same = [x for x in msgs if x["t"] == t]
            request = next(x for x in same if x["msg_id"] == m["in_reply_to"])
            to_hospital = next((x for x in same if x["type"] == "allocation_decision"
                                and x["proposed_action"].get("unit_id") == unit and x["proposed_action"].get("staff_id")), None)
            to_dispatch = next((x for x in same if x["type"] == "allocation_decision" and x["recipient"] == "dispatch"
                                and x["proposed_action"].get("unit_id") == unit), None)
            if to_hospital is None or to_dispatch is None:
                continue
            asks = [x for x in same if x["type"] == "staff_request" and x["in_reply_to"] == request["msg_id"]]
            offers = [x for x in same if x["type"] == "staff_offer" and x["in_reply_to"] in {a["msg_id"] for a in asks}]
            rows = [request, m, *asks[:1], *offers[:1], to_hospital, to_dispatch]
            return {"seed": seed, "theme": theme, "stress": stress, "t": t, "unit": unit,
                    "rows": [{"sender": r["sender"], "recipient": r["recipient"], "type": r["type"], "round": r["round"],
                              "text": r["rationale"], "action": {k: v for k, v in r["proposed_action"].items()
                                                                 if k in ("kind", "unit_id", "hospital_id", "staff_id", "condition")}}
                             for r in rows]}
    return {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    theme, stress, seed = SHOWCASE
    frame = calibration_frame(range(1, 201))
    closest, n_seeds = typical_seed(frame, theme, stress)
    assert closest == seed, f"showcase seed {seed} is no longer the typical draw (now {closest})"

    s = generate_scenario(theme, stress, seed)
    draw_map(s, OUT / "scenario_map.png")

    counts, press, report = victim_counts(s), pressure_indices(s), contention_report(s)
    checks = validate_scenario(s)
    times = [i.report_time_min for i in s.incidents]
    urban = frame[frame.theme == "urban_mci"].groupby("stress")
    knob = {
        level: {col: {"p10": round(float(urban[col].quantile(0.10)[level]), 2),
                      "p50": round(float(urban[col].median()[level]), 2),
                      "p90": round(float(urban[col].quantile(0.90)[level]), 2)}
                for col in ("ambulance_pressure", "er_bay_pressure", "specialist_pressure")}
        for level in P.STRESS
    }
    good, total = verify_library(ROOT / "data" / "scenarios")
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts="], cwd=ROOT,
                               capture_output=True, text=True).stdout
    n_tests = sum("::" in line for line in collected.splitlines())
    load = report["nearest_capable_load"]
    history = generate_history(600)
    stats = {
        "showcase": {
            "scenario_id": s.meta.scenario_id, "seed": seed, "compared_seeds": n_seeds,
            "counts": counts, "first_call_min": min(times), "last_call_min": max(times),
            "pressure": press, "ambulance_pairs": report["n_ambulance_pairs"],
            "single_provider": report["single_provider"], "red_yellow_needing_single_provider": report["red_yellow_needing_single_provider"],
            "red_yellow_total": counts["red"] + counts["yellow"],
            "nearest_capable_load": load, "incident_types": sorted({i.incident_type for i in s.incidents}),
            "nearest_station_counts": dict(sorted(Counter(nearest_station(s, i).station_id for i in s.incidents).items())),
            "incident_zone_counts": dict(sorted(Counter(i.zone for i in s.incidents).items())),
            "map": {
                "city_km": s.meta.city_km, "center": list(P.CITY_CENTER), "zone_radii_km": list(P.ZONE_RADII_KM),
                "incidents": [{"id": i.incident_id, "x": i.x, "y": i.y, "victims": i.n_victims, "worst": worst_triage(i)}
                              for i in s.incidents],
                "hospitals": [{"id": h.hospital_id, "x": h.x, "y": h.y, "bays": h.er_bays_total} for h in s.hospitals],
            },
            "hospitals": [{"id": h.hospital_id, "name": h.name, "level": h.trauma_level, "er_free": h.er_bays_free,
                           "er_total": h.er_bays_total, "icu_free": h.icu_beds_free, "capabilities": list(h.capabilities)}
                          for h in s.hospitals],
            "ambulances": [{"id": a.ambulance_id, "type": a.unit_type, "station": a.base_station_id, "capacity": a.capacity,
                            "status": a.status} for a in s.ambulances],
            "staff": {"doctors": counts["doctors"], "nurses": counts["nurses"],
                      "pool": sum(m.home_hospital == "POOL" for m in s.staff)},
            "checks_passed": sum(c.passed for c in checks), "checks_total": len(checks),
        },
        "stress_knob_urban": knob,
        "caller_undercount_share": round(1 - float(history.reported_victims.sum() / history.true_victims.sum()), 3),
        "calibration": {"seeds_per_setting": 200, "settings": 9, "scenarios": len(frame),
                        "scenarios_with_a_failed_check": int((frame.checks_failed > 0).sum())},
        "library": {"verified": good, "total": total},
        "tests": n_tests,
        "travel_pairs": sum(1 for _ in (ROOT / "data" / "scenarios" / s.meta.scenario_id / "travel_times.csv").open()) - 1,
    }
    runs = ROOT / "results" / "runs.csv"
    if runs.exists():
        stats["results"] = results_stats(pd.read_csv(runs))
        stats["example_thread"] = example_thread(theme, stress, [seed, *range(1, 31)])
        stats["showcase_runs"] = {p: {k: run_one(theme, stress, seed, p)[0][k] for k in RESULT_METRICS} for p in ("B0", "M2")}
    (OUT / "deck_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: stats[k] for k in ("library", "tests", "calibration", "travel_pairs")}, indent=1))
    if "results" in stats:
        print(json.dumps({k: stats[k] for k in ("example_thread", "showcase_runs")}, indent=1))
        print(json.dumps({s_: {k: v for k, v in d.items() if k != "means"} for s_, d in stats["results"]["by_stress"].items()}, indent=1))
        print(json.dumps({k: stats["results"][k] for k in ("negotiation_per_run", "assessment", "other_themes")}, indent=1))
    print("showcase", s.meta.scenario_id, counts, "pairs", report["n_ambulance_pairs"])
    print("wrote", OUT / "scenario_map.png")


if __name__ == "__main__":
    main()
