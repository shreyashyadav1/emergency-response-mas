"""Emergency response dashboard: replay one run minute by minute, against the same scenario under another policy.

    .venv/bin/streamlit run app/dashboard.py

Five panels: incident map, resources, agent activity, negotiation history and performance. Runs
are computed live (about a quarter of a second each) or replayed from the logs that
`python -m emergency_mas.eval replay` saves in runs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # so it runs from anywhere without installing the package

import pandas as pd
import streamlit as st

from emergency_mas import params as P
from emergency_mas.dashboard import figures as F
from emergency_mas.dashboard import replay as R
from emergency_mas.eval.runner import POLICIES, POLICY_LABEL

RUNS_CSV = ROOT / "results" / "runs.csv"
LOG_DIR = ROOT / "runs"
DEFAULT = {"theme": "urban_mci", "stress": "medium", "seed": 28, "policy": "M2", "against": "B0"}
THEME_LABEL = {"urban_mci": "Urban mass-casualty day", "earthquake": "Earthquake", "highway_pileup": "Highway pile-up"}
AGENT_ORDER = {"incident_assessment": 0, "dispatch": 1, "resource_allocation": 2}
CHART = {"displayModeBar": False}

st.set_page_config(page_title="Emergency response MAS", page_icon=":material/emergency:", layout="wide")


@st.cache_resource(max_entries=32, show_spinner="Running the simulation...")
def live_run(theme: str, stress: str, seed: int, policy: str) -> R.Replay:
    return R.live(theme, stress, seed, policy)


@st.cache_resource(max_entries=32, show_spinner="Reading the run log...")
def saved_run(path: str, modified: float) -> R.Replay:
    return R.from_log(path)


@st.cache_data
def experiment(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path)


def tokens() -> dict:
    return F.TOKENS["dark" if getattr(st.context.theme, "type", None) == "dark" else "light"]


# ---------------------------------------------------------------------- sidebar: what to show
with st.sidebar:
    st.header("Scenario")
    logs = R.saved_logs(LOG_DIR) if LOG_DIR.exists() else {}
    source = st.radio("Source", ["Run live", "Saved run log"] if logs else ["Run live"], horizontal=True,
                      help="Live runs take about a quarter of a second. Saved logs come from "
                           "`python -m emergency_mas.eval replay` and replay exactly the same minutes.")
    if source == "Run live":
        theme = st.selectbox("Theme", list(P.THEMES), index=list(P.THEMES).index(DEFAULT["theme"]),
                             format_func=lambda k: THEME_LABEL.get(k, k))
        stress = st.segmented_control("Stress", list(P.STRESS), default=DEFAULT["stress"]) or DEFAULT["stress"]
        seed = int(st.number_input("Seed", min_value=1, max_value=99_999, value=DEFAULT["seed"], step=1,
                                   help="The same seed always builds the same emergency. Seeds 1 to 30 are the ones in the experiment."))
        policy = st.selectbox("Policy", list(POLICIES), index=list(POLICIES).index(DEFAULT["policy"]), format_func=POLICY_LABEL.get)
        others = [p for p in POLICIES if p != policy]
        against = st.selectbox("Compare with", others, index=others.index(DEFAULT["against"]) if DEFAULT["against"] in others else 0,
                               format_func=POLICY_LABEL.get)
        a, b = live_run(theme, stress, seed, policy), live_run(theme, stress, seed, against)
    else:
        scenario_id = st.selectbox("Scenario", list(logs))
        files = logs[scenario_id]
        keys = [p for p in POLICY_LABEL if p in files]          # saved logs may include M3, the LLM agents
        policy = st.selectbox("Policy", keys, index=keys.index("M2") if "M2" in keys else 0, format_func=POLICY_LABEL.get)
        others = [p for p in keys if p != policy] or [policy]
        against = st.selectbox("Compare with", others, format_func=POLICY_LABEL.get)
        a, b = (saved_run(str(files[p]), files[p].stat().st_mtime) for p in (policy, against))
        theme, stress, seed = a.scenario.meta.theme, a.scenario.meta.stress, a.scenario.meta.seed
    st.divider()
    st.caption("**Policies.** B0 sends the nearest free unit to the earliest call. B1 serves the most critical "
               "report first. M1 runs the four agents without negotiation, M2 with it. M3 is M2 with the judgment calls "
               "made by OpenAI Agents SDK agents on a hosted model (OpenAI or Gemini); it appears under Saved run log once "
               "`emergency_mas.llm run` has run.")
    st.caption("**Colors.** Blue is the policy you picked and orange the one it is compared with. Incident colors are "
               "triage: red, yellow and green for the worst patient still waiting at the scene.")
    st.caption("The world is synthetic and its parameters are illustrative (docs/ASSUMPTIONS.md), so compare "
               "policies with each other rather than reading the numbers as real outcomes.")

tok = tokens()
meta = a.scenario.meta
horizon = a.horizon
runs = experiment(str(RUNS_CSV), RUNS_CSV.stat().st_mtime) if RUNS_CSV.exists() else pd.DataFrame()

st.title("Emergency response coordination")
st.caption(f"{THEME_LABEL.get(theme, theme)}, {stress} stress, seed {seed}: {len(a.scenario.incidents)} incidents, "
           f"{len(a.scenario.victims)} victims, {len(a.scenario.hospitals)} hospitals, {len(a.scenario.ambulances)} ambulances "
           f"and {len(a.scenario.responders)} responder teams over {horizon} minutes. "
           f"Showing **{a.label}** against **{b.label}**.")

st.session_state.setdefault("minute", 0)
st.session_state.setdefault("playing", False)
st.session_state.minute = min(st.session_state.minute, horizon - 1)


def step(delta: int) -> None:
    st.session_state.minute = min(max(st.session_state.minute + delta, 0), horizon - 1)


# ---------------------------------------------------------------------- panels
def controls() -> None:
    playing = st.session_state.playing
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button("Pause" if playing else "Play", icon=":material/pause:" if playing else ":material/play_arrow:",
                     type="primary", key="play"):
            if not playing and st.session_state.minute >= horizon - 1:
                st.session_state.minute = 0
            st.session_state.playing = not playing
            st.rerun()                   # a full rerun turns the timer on or off
        for label, delta in (("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10)):
            st.button(label, on_click=step, args=(delta,), key=f"step{delta}", help=f"{delta:+d} minutes")
        st.selectbox("Speed", [1, 2, 5, 10], index=1, key="speed", format_func=lambda s: f"{s} min per step",
                     label_visibility="collapsed", width=170)
    st.slider("Minute", 0, horizon - 1, key="minute")


def kpis(t: int) -> None:
    now, then = a.series.loc[t], b.series.loc[t]
    st.subheader(f"Performance at minute {t}")
    tiles = [("Victims in care", "victims_in_care", "normal", "{:.0f}", "Victims whose treatment has started."),
             ("RED patients in care", "red_in_care", "normal", "{:.0f}", "Critical patients whose treatment has started."),
             ("Critical incidents waiting", "critical_waiting", "inverse", "{:.0f}",
              "Reported incidents where at least one RED patient is not in care yet."),
             ("First unit on scene (min)", "first_response", "inverse", "{:.1f}",
              "Average time from the 911 call to the first unit on scene, over the incidents reached so far."),
             ("ER door waits (min)", "door_minutes", "inverse", "{:.0f}",
              "Ambulance minutes spent waiting at an ER door with no free bay, so far.")]
    cols = [col for _ in range(2) for col in st.columns(3)]
    for col, (label, key, good, fmt, why) in zip(cols, tiles):
        x, y = now[key], then[key]
        value = "n/a" if pd.isna(x) else fmt.format(x)
        if pd.isna(x) or pd.isna(y) or a.policy == b.policy:
            col.metric(label, value, border=True, help=why)
            continue
        d = round(float(x - y), 1) + 0.0
        col.metric(label, value, delta=("+" if d > 0 else "") + fmt.format(d), delta_color=good if d else "off",
                   delta_arrow="auto" if d else "off", delta_description=f"vs {b.policy}", border=True, help=why)
    cols[-1].metric("Unit conflicts settled", f"{now.conflicts:.0f}", border=True,
                    help="Contested ambulances and teams the Resource Allocation Agent has settled so far. Only M2 negotiates.")


def map_panel(r: R.Replay, t: int, key: str) -> None:
    layers = r.map_layers(t)
    st.plotly_chart(F.map_figure(layers, r.scenario.meta.city_km, tok), key=key, config=CHART, theme=None)
    with st.expander("Map as a table"):
        st.dataframe(layers["incidents"][["id", "type", "victims", "waiting", "RED", "YELLOW", "GREEN", "reached", "needs", "teams"]],
                     hide_index=True)
        st.dataframe(layers["units"][["id", "kind", "type", "doing", "onboard", "task"]], hide_index=True)


def resources_panel(r: R.Replay, t: int, key: str) -> None:
    st.plotly_chart(F.resource_figure(r.resources(t), tok), key=key, config=CHART, theme=None)
    h = r.map_layers(t)["hospitals"]
    table = pd.DataFrame({"Hospital": h.id + " " + h.short, "ER bays": h.bays_used.astype(str) + " / " + h.bays_total.astype(str),
                          "At door": h.door, "Corridor": h.corridor, "ICU free": h.icu_free, "Surge": h.surge,
                          "Doctor slots free": h.doctors_free, "Nurse slots free": h.nurses_free, "Staff lent in": h.lent_in})
    st.dataframe(table, hide_index=True)


def agents_panel(t: int) -> None:
    st.subheader("Agent activity")
    cards = sorted(a.agents(t), key=lambda c: (AGENT_ORDER.get(c["id"], 3), c["id"]))
    if not cards:
        st.info(f"{a.label} has no agents and sends no messages. Pick M1 or M2 as the policy to see the agents at work.")
        return
    st.caption(f"Status lines and the latest messages of {a.label} at minute {t}.")
    for row in range(0, len(cards), 3):
        for col, card in zip(st.columns(3), cards[row:row + 3]):
            with col.container(border=True):
                st.markdown(f"**{card['name']}**")
                st.caption(card["job"])
                st.markdown(card["status"])
                st.caption(f"Sent {card['sent']}, received {card['received']} so far")
                for m in card["recent"][:2]:
                    st.caption(f"min {m['t']}, {m['direction']} {m['other']}. {m['type']}: {m['text']}")


def negotiation_panel(t: int) -> None:
    st.subheader("Negotiation history")
    every = a.threads_until(t)
    if not every:
        st.info(f"No negotiations by minute {t}." + ("" if a.agents(t) else f" {a.label} does not negotiate."))
        return
    conflicts = sum(th.kind == "unit conflict" for th in every)
    counters = sum(any(m["type"] == "counter_proposal" for m in th.messages) for th in every)
    loans = sum(any(m["type"] == "staff_offer" for m in th.messages) for th in every)
    st.caption(f"By minute {t}: {conflicts} unit conflicts and {len(every) - conflicts} admission requests, "
               f"{counters} answered with a counter-offer and {loans} with a staff offer. Messages are linked by `in_reply_to`.")
    left, right = st.columns([6, 5])
    with left:
        only = st.toggle("Only conflicts, counter-offers, refusals and staff searches", value=True)
        shown = list(reversed(a.threads_until(t, negotiated_only=only)))
        if not shown:
            st.info("Nothing but plain acceptances so far.")
            return
        st.dataframe(pd.DataFrame([{"Minute": th.t, "Kind": th.kind, "Unit": th.unit, "Outcome": th.outcome,
                                    "Incident": th.incident, "Rounds": th.rounds, "Messages": len(th.messages)} for th in shown]),
                     hide_index=True, height=320, column_config={"Outcome": st.column_config.TextColumn(width="large")})

    def ident(th: R.Thread) -> str:
        return f"{th.kind}-{th.messages[0]['msg_id']}"

    labels = {ident(th): f"Minute {th.t}, {th.kind}, {th.unit or ''} {th.incident or ''}".strip() for th in shown}
    with right:
        if st.toggle("Follow the latest negotiation", value=True):
            pick = shown[0]
        else:
            chosen = st.selectbox("Read one negotiation", list(labels), format_func=labels.get)
            pick = next(th for th in shown if ident(th) == chosen)
        with st.container(border=True):
            st.markdown(f"**{labels[ident(pick)]}**")
            for row in pick.table().itertuples():
                st.markdown(f"Round {row.Round}. {row.From} to {row.To}, *{row.Type.lower()}*: {row.Message}")


def performance_panel(t: int) -> None:
    st.subheader("Performance over the whole run")
    cols = st.columns(3)
    for col, (title, key) in zip(cols, (("Victims in care", "victims_in_care"), ("RED patients in care", "red_in_care"),
                                        ("Ambulance minutes at full ER doors", "door_minutes"))):
        with col:
            st.markdown(f"**{title}**")
            st.plotly_chart(F.series_figure([(a.policy, a.series[key], tok["selected"]), (b.policy, b.series[key], tok["compare"])],
                                            t, tok), key=f"series-{key}", config=CHART, theme=None)
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**After {horizon} minutes**")
        st.dataframe(R.compare_outcomes(a, b), hide_index=True)
    with right:
        st.markdown("**Is this run typical?**")
        ctx = R.seed_context(runs, theme, stress, seed, a.policy, b.policy) if a.policy != b.policy else None
        if ctx is None:
            st.caption("The experiment in results/runs.csv has no runs to compare for this setting.")
        else:
            share = f"{a.policy} is ahead on {ctx['ahead_share']:.0%} of them"
            if "rank" in ctx:
                st.markdown(f"On seed {seed}, {a.policy} minus {b.policy} weighted survival is **{ctx['diff']:+.3f}**. "
                            f"Among the experiment's {ctx['seeds']} seeds at this setting that gain ranks {ctx['rank']} of "
                            f"{ctx['seeds']} (1 is the smallest). The median is {ctx['median_diff']:+.3f} and {share}.")
            else:
                st.markdown(f"Seed {seed} is not one of the experiment's {ctx['seeds']} seeds at this setting. Across them the "
                            f"median {a.policy} minus {b.policy} weighted survival is {ctx['median_diff']:+.3f}, and {share}.")
    with st.expander("Lines as a table"):
        cols = ["victims_in_care", "red_in_care", "critical_waiting", "first_response", "door_minutes", "messages", "conflicts"]
        st.dataframe(a.series[cols].add_prefix(f"{a.policy} ").join(b.series[cols].add_prefix(f"{b.policy} ")))


# ---------------------------------------------------------------------- the board, on a timer while playing
@st.fragment(run_every=0.8 if st.session_state.playing else None)
def board() -> None:
    if st.session_state.playing:
        nxt = st.session_state.minute + st.session_state.get("speed", 2)
        if nxt >= horizon - 1:
            st.session_state.minute, st.session_state.playing = horizon - 1, False
            st.rerun()
        st.session_state.minute = nxt
    controls()
    t = st.session_state.minute
    kpis(t)
    side = st.toggle(f"Show {b.policy} beside it", value=False, help="Draw the comparison run's map and resources next to these.")
    if side:
        for col, r in zip(st.columns(2), (a, b)):
            with col:
                st.subheader(f"Incident map, {r.policy}")
                map_panel(r, t, f"map-{r.policy}")
                st.subheader(f"Resources, {r.policy}")
                resources_panel(r, t, f"resources-{r.policy}")
    else:
        left, right = st.columns([7, 4])
        with left:
            st.subheader("Incident map")
            map_panel(a, t, "map")
        with right:
            st.subheader("Resources")
            st.caption(f"In use, free or unavailable at minute {t}, for {a.label}.")
            resources_panel(a, t, "resources")
    agents_panel(t)
    negotiation_panel(t)
    performance_panel(t)


board()
