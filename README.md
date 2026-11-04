# Emergency response multi-agent system

Midterm project, CSYE7380. Four kinds of AI agents coordinate ambulances, responders, hospitals and staff during a
mass-casualty event, and negotiate when their goals conflict. The plan is in [PLAN.md](PLAN.md), the results in
[docs/RESULTS.md](docs/RESULTS.md), and the LLM agents' (M3) results in [docs/LLM_RESULTS.md](docs/LLM_RESULTS.md).

**Headline results** (weighted survival, paired by seed, 30 seeds per setting):

- The agents with negotiation (M2) beat the best baseline at every stress level, by +0.056 to +0.067 (p < 0.001), and
  negotiation alone adds +0.025 to +0.039 over the same agents without it (M1).
- **The OpenAI Agents SDK version (M3, `gpt-4o-mini`) beats both baselines but is slightly worse than M2.** At medium
  stress it gains +0.053 over B1 and +0.072 over B0, and trails M2 by 0.003 (p = 0.02).

**Status:** data layer, simulator, two baselines, the four agents with negotiation (rules backend), the experiment, the
dashboard and the OpenAI Agents SDK agents (M3) are done. M3 runs on OpenAI (`gpt-4o-mini`) or on Gemini's free tier.

## Set up

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python -m pytest
```

## Run

```bash
.venv/bin/python -m emergency_mas.eval run --seeds 30      # 4 policies on the same 150 scenarios: 600 runs, about a minute
.venv/bin/python -m emergency_mas.eval report              # docs/RESULTS.md and docs/figures/results.png
.venv/bin/python -m emergency_mas.eval replay --seed 78    # minute-by-minute logs in runs/ for the dashboard
```

Policies: `B0` nearest first, `B1` priority greedy, `M1` agents without negotiation, `M2` agents with negotiation.

## Dashboard

```bash
.venv/bin/streamlit run app/dashboard.py
```

It replays one run minute by minute next to the same scenario under another policy, with five panels: incident map,
resources, agent activity, negotiation history and performance. Pick the theme, stress, seed and the two policies in the
sidebar; runs are computed live, or read from the logs `eval replay` saves in `runs/`. Play steps through the 240 minutes,
and "Show B0 beside it" draws both maps side by side. The default, seed 28 at medium stress, is the median of the 30
experiment seeds for M2's gain over B0, and the page says where any seed you pick ranks.

## LLM agents (M3) on the OpenAI Agents SDK

M3 is M2 with its judgment calls made by four OpenAI Agents SDK agents (`emergency_mas/llm/agents.py`), each with its
own instructions, function tools and reply schema:

- the Incident Assessment Agent estimates victims from the caller's own words and the base rates;
- the Dispatch Agent decides the order in which to ask hospitals to take a load;
- each Hospital Coordination Agent chooses between a counter-offer and a refusal;
- the Resource Allocation Agent awards contested units and settles countered admissions.

The SDK runs them on OpenAI's API, or on Google Gemini through Gemini's OpenAI-compatible endpoint; the model name
picks the server (`gemini-*` models go to Gemini, the rest to OpenAI). The tools return the numbers the rules agents
compute; the model decides. An answer that is malformed or names an option that was not offered leaves the rules
decision in place and is counted, and the simulator still checks every action.

1. Copy `.env.example` to `.env` and put a key in it: `OPENAI_API_KEY` from
   [platform.openai.com](https://platform.openai.com/api-keys) (the account needs prepaid credit, $5 minimum), or a free
   `GEMINI_API_KEY` from [aistudio.google.com](https://aistudio.google.com) with `LLM_MODEL=gemini-3.5-flash-lite`.
   `.env` is ignored by git.
2. Check the setup and measure before spending quota:

```bash
.venv/bin/python -m emergency_mas.llm smoke            # one small request: key, model, tool calling, reply format
.venv/bin/python -m emergency_mas.llm models           # the model names your key can use (set LLM_MODEL in .env)
.venv/bin/python -m emergency_mas.llm run --dry-run    # decisions, requests and tokens for seeds 1-5; nothing is sent
.venv/bin/python -m emergency_mas.llm run              # about 140 requests per seed at medium stress, paced to LLM_RPM
.venv/bin/python -m emergency_mas.llm report           # docs/LLM_RESULTS.md: M3 against M2 and B0 on the same seeds
```

A run saves each seed's row in `results/llm_runs.csv` as soon as the seed finishes. If it meets a daily limit (Gemini's
free tier allows 500 requests a day for `gemini-3.5-flash-lite` and only 20 for `gemini-3.6-flash`) or the OpenAI
account runs out of credit, it stops cleanly and the same command continues later.

Every answer is cached in `data/llm_cache/` with the prompt and tool data it answered, so `run --cache-only` with the
same `--model` and `--seeds` replays finished runs exactly without a key and checks each seed against the recorded run
(`docs/LLM_RESULTS.md` lists the exact commands). The run writes logs to `runs/`, so M3 also appears in the dashboard
under "Saved run log".

## Data

```bash
.venv/bin/python -m emergency_mas.datagen generate --preset baseline --seed 78
.venv/bin/python -m emergency_mas.datagen verify
.venv/bin/python -m emergency_mas.datagen calibrate --seeds 200
.venv/bin/python -m emergency_mas.datagen history --n 600
.venv/bin/python -m emergency_mas.datagen docs
```

Presets: `baseline` (urban, medium), `calm` (urban, low), `crunch` (urban, high), `surge` (earthquake, high), `pileup` (highway, high).
Each scenario folder in `data/scenarios/` has CSV tables, `scenario.json` and a `MANIFEST.json` with SHA-256 hashes.

## Slides

The progress deck (`slides/`) is not in the public repository, because its title slide carries a student ID. Every
number on it comes from `docs/figures/deck_stats.json`, which this script regenerates:

```bash
.venv/bin/python scripts/make_figures.py     # scenario map and the numbers quoted on the slides
```

## Layout

| Path | What |
|---|---|
| `emergency_mas/params.py` | Every modeling assumption and simulation rule in one file |
| `emergency_mas/schema.py` | Typed records. Field descriptions become the data dictionary |
| `emergency_mas/domain.py` | Travel time and survival functions |
| `emergency_mas/datagen/` | Generator, validation, stress analysis, export, history, CLI |
| `emergency_mas/sim/` | Simulator: state, actions, what policies may see, metrics |
| `emergency_mas/policies/` | Baselines B0 and B1 and shared helpers |
| `emergency_mas/agents/` | Messages, the four agents, and the policy that runs them together; `llm_system.py` is M3 |
| `emergency_mas/llm/` | The four agents as OpenAI Agents SDK agents, the OpenAI and Gemini connection, cache, pacing, CLI |
| `emergency_mas/eval/` | Experiment runner, paired statistics, report |
| `emergency_mas/dashboard/` | What each dashboard panel shows at a given minute, and its Plotly figures |
| `app/dashboard.py` | The Streamlit dashboard |
| `docs/` | `ASSUMPTIONS.md`, `DATA_DICTIONARY.md`, `RESULTS.md`, `LLM_RESULTS.md`, `WRITEUP.md` (the submission text), figures |
| `data/` | Frozen scenarios, calibration over 200 seeds, 600 synthetic past incidents |
| `results/runs.csv` | One row of metrics per run |
| `tests/` | Data, simulator, agent, dashboard and SDK-agent tests (the SDK tests use a scripted model, no network) |
| `scripts/` | Figure and slide-number generation |
