# Plan: multi-agent emergency response coordination

Midterm project for CSYE7380 (Professor Subrata Das, Section 08, Fall 2026). Due **Wednesday Nov 4, 2026, 6:00 pm** (text entry box).
Written Oct 21, updated Nov 4. Done: data layer, simulator, two baselines, the four agents with negotiation (rules
backend), a 600-run experiment, the updated progress deck, the dashboard, and the four agents on the OpenAI Agents SDK
(M3), run on 30 seeds on OpenAI's `gpt-4o-mini` (seeds 1 to 3 also on Gemini's free tier). Next: write-up, recording.

## 1. What has to be shown

| The assignment asks for | Where it lives | Status |
|---|---|---|
| Synthetic world: incidents, hospitals, ambulances, staff, responders, dispatch | `emergency_mas/datagen/` | done |
| Documented assumptions, reproducible runs | `docs/ASSUMPTIONS.md`, seeds, SHA-256 manifests, `verify` command | done |
| At least four agents with different jobs, tools and policies | `emergency_mas/agents/`, `emergency_mas/llm/` | done: rules backend, and OpenAI Agents SDK agents (on OpenAI or Gemini) |
| Structured messages: sender, recipient, type, incident id, proposed action | `emergency_mas/agents/messages.py` | done |
| Negotiation when requests conflict | `agents/allocation.py`, `agents/hospital.py` | done |
| Measurable improvement from coordination | `emergency_mas/eval/`, `docs/RESULTS.md` | done for the rules backend |
| Interactive dashboard with five panels | `app/dashboard.py`, `emergency_mas/dashboard/` (Streamlit + Plotly) | done |

## 2. Architecture

```
 seed --> datagen --> scenario files
                           |
                           v
              +---------------------------+        run log (JSONL)
              | Simulator, 1-minute ticks | ------------------------> Streamlit dashboard
              | world state, rule checks  |                           (replay and live)
              +------^------------+-------+
                     | actions    | observations
              +------+------------v-------+
              |        Message bus        |   every message is typed, logged and replayable
              +--+--------+--------+---+--+
                 |        |        |   |
          Incident   Dispatch   Hospital    Resource
          Assessment  Agent     Coordination Allocation
          Agent                 Agents (x3)  Agent
```

The simulator is the single source of truth. Agents never change state directly. They send messages, and the
simulator checks every requested action (capability, free bay, staff slot, ALS for RED patients, one RED patient
per ambulance load) and answers accepted or rejected with a reason.

## 3. Data layer (done Oct 21, revised Oct 26)

One seed builds the whole world. Each kind of entity has its own random stream, so changing one part never shifts another.

- **Entities:** 3 hospitals, doctors and nurses with skills, shifts and workload limits plus a regional on-call pool,
  5 ambulances, 6 responder teams, 12 incidents, about 50 victims, travel times.
- **Ground truth and report are separate.** Agents get a noisy caller report plus base rates from 600 synthetic past incidents.
- **Stress knob:** low, medium, high, and three themes. Ambulance pressure rises 0.9, 1.4, 1.9 for the urban theme.
- **Generator 0.2.0 (Oct 26):** callers now misjudge who is critical (they miss some RED patients and call some YELLOW
  ones critical). Before, rounding made the critical count almost exact, so there was nothing to assess.

## 4. Simulator (done Oct 26)

- 1-minute ticks over 240 minutes. Units drive, scene teams work, hospitals admit and discharge, then the policy decides.
- Scene gating: hazard scenes open 10 minutes after the matching team starts work; trapped victims wait for extrication.
- Every rule applies to every policy. Choices made while building, each documented in `docs/ASSUMPTIONS.md`:
  - an ambulance waits at most 30 minutes at a full ER door, then hands patients over in the corridor;
  - YELLOW patients may be taken by an emergency-trained doctor when no specialist is free; RED patients need the specialist;
  - an ALS crew starting to load a RED patient stabilizes them like a triage team does.
- Output: one JSON line per minute with the state, events, actions and agent messages, for the dashboard.

## 5. Agents (rules backend done Oct 26; SDK backend next)

| Agent | Job | Decides |
|---|---|---|
| Incident Assessment | Turn the noisy report into an estimate | expected victims and RED count from a fit on past incidents, likely hazard, scene tasks, urgency on the oracle scale; replaced by the scene count once a unit arrives |
| Dispatch | Bid for ambulances and teams for each incident, find a hospital for each load | values a unit by the expected survival it buys per minute of its time; never parks an ambulance at a closed scene; keeps fire engines for fires and extrications |
| Hospital Coordination (one per hospital) | Guard its bays, staff and capabilities | projects its state to the arrival time, keeps bays for RED, accepts, rejects, or counters with "yes in N minutes" or "yes if you lend me a nurse"; opens surge bays |
| Resource Allocation | Referee | awards contested units by an assignment solve and tells each loser what it gets instead; settles counter-offers by when care would start; arranges staff loans; places the on-call pool; diverts ambulances stuck at full doors |

Message fields: `msg_id, t, sender, recipient, type, incident_id, proposed_action, rationale, in_reply_to, round`.
Every negotiation ends with a decision within 3 rounds. Every reply, including the referee's final word to Dispatch,
points back through `in_reply_to` to the request it answers, so the dashboard can show each negotiation whole.

**SDK backend (M3, built Nov 3).** Each agent is an `agents.Agent` with its own instructions, function tools that return
the rules agent's numbers, and a Pydantic reply schema (`emergency_mas/llm/agents.py`); `emergency_mas/agents/llm_system.py`
hands it these decisions:

| Agent | Decision it makes in M3 | Tools |
|---|---|---|
| Incident Assessment | victims, RED count, hazard and trapped, from the caller's words | `base_rates`, `statistical_estimate` |
| Dispatch | the order in which to ask hospitals to take a load | `hospital_options` |
| Hospital Coordination | counter-offer or refuse, when it cannot simply accept | `capacity_at_arrival` |
| Resource Allocation | who gets each contested unit; how to settle a countered admission | `contested_units`, `settlement_options` |

The SDK runs on OpenAI (`gpt-4o-mini`) or on Gemini through its OpenAI-compatible endpoint; the model name picks the server. Seeds 1 to 3 ran first on Gemini's free tier (`gemini-3.5-flash-lite`); its limit of 500 requests a day stopped seed 4, so on Nov 4 M3 moved to OpenAI, which needs prepaid credit but costs under $1 for 30 seeds. A daily-limit or no-credit error stops a run cleanly, each finished seed is saved as it ends, and the cache lets the same command continue later. Replies are JSON text
validated against the schema, because Gemini does not accept a response format and tools in one request, so both servers get the same agents. Calls are
paced to `LLM_RPM`, retried when rate-limited, and cached by a hash of model, agent, prompt and tool data, so finished
runs replay offline. A dry run counts decisions and tokens first: seeds 1 to 5 at medium stress are 347 decisions,
about 694 requests and 276k input tokens. Invalid answers fall back to the rules decision and are counted, and the
report shows how often the model departed from the rules agent's choice.

## 6. Evaluation (first full run Oct 26, see `docs/RESULTS.md`)

Same scenarios for every condition, paired by seed: B0 nearest first, B1 priority greedy, M1 agents without
negotiation, M2 agents with negotiation. 30 seeds at each stress level, plus earthquake and pile-up at high stress.

- M2 beats the best baseline on weighted survival by +0.067, +0.056 and +0.064 at low, medium and high stress
  (all p < 0.001), and on expected survivors by +1.2, +1.3 and +2.0 per scenario. RED survivors are 1.5 to 2.1 times
  the best baseline's, and 2 to 3.5 times B0's.
- Negotiation alone (M2 minus M1) adds +0.034, +0.025 and +0.039. It helps most at high stress, but the gap does not
  grow steadily with stress, so the original success wording ("grows with stress") is only partly met.
- The assessment agent removes the callers' under-count (bias -0.42 to +0.01 victims) and lowers the count error, but
  ranks incidents no better than the raw critical count (rank correlation 0.81 for both).

- M3, the same agents with their judgment calls made by `gpt-4o-mini` through the OpenAI Agents SDK (30 seeds at
  medium stress, Nov 4): better than both baselines but slightly worse than M2, at +0.053 [+0.035, +0.070] against B1,
  +0.072 [+0.052, +0.093] against B0 and -0.003 [-0.007, -0.000] against M2 (p = 0.02). The model departed from the rules agents on 139 of 2,051 decisions (7%), mostly the order of hospitals
  to ask, and no answer failed validation. Details in `docs/LLM_RESULTS.md`.

Still to do: ablations (no assessment, no staff lending, one round).

## 7. Dashboard (done Nov 3)

Streamlit and Plotly (`.venv/bin/streamlit run app/dashboard.py`). It replays a run minute by minute, with the same
scenario under a second policy beside it. Runs are computed live (about 0.3 s each) or read from `eval replay` logs.
A play button, step buttons and a minute slider drive every panel at once.

1. **Incident map:** incidents sized by victims still waiting and colored by the worst triage among them, ambulances and teams with lines to where they are headed, hospitals with an ER-bay meter. Optionally both policies' maps side by side.
2. **Resources:** ambulances, teams, ER bays, ICU beds and doctor and nurse slots as in use, free or unavailable, plus a per-hospital table.
3. **Agent activity:** one card per agent with its status line, what it is still waiting on, message counts and latest messages.
4. **Negotiation history:** every unit conflict and admission request with its replies, followed through `in_reply_to`, and one negotiation read in full.
5. **Performance:** tiles at the current minute with the difference from the other policy, running totals over the run, the final outcomes side by side, and where the seed ranks among the experiment's 30.

The running totals at the last minute equal the run's final metrics (tested), so the dashboard and `docs/RESULTS.md`
never disagree.

## 8. Schedule

| Date | Work |
|---|---|
| Wed Oct 21 | Plan, data generator, tests, progress slides (done) |
| Mon Oct 26 | Simulator, baselines, four agents, negotiation, 600-run experiment (done) |
| Wed Oct 28 | Progress deck updated with the agents and the results (done) |
| Tue Nov 3 | Dashboard with the five panels (done); OpenAI Agents SDK agents on Gemini, tested offline (done) |
| Tue Nov 3 - Wed Nov 4 | M3 seeds 1 to 3 on Gemini, then 30 seeds on OpenAI (done); write-up, short screen recording, clean rerun from scratch |
| Wed Nov 4, by 6 pm | Submit |

## 9. Risks

| Risk | Mitigation |
|---|---|
| LLM calls are slow, costly and not exactly repeatable | Decision points only, temperature 0, cached responses, tokens measured before the full set |
| LLM agents lose to the rules agents | Give the LLM the scoring tools instead of asking it to do arithmetic. Report honestly either way |
| Simulation choices drive the results | Every rule applies to every policy; choices are listed in section 4 and in `docs/ASSUMPTIONS.md` |
| NumPy streams can change between major versions | Versions pinned in `requirements-lock.txt`, hashes stored in every manifest |
| API limits and cost | Gemini's free tier stopped seed 4 at its daily limit, so M3 moved to OpenAI (cents per seed); a dry run counts requests first, calls are paced and cached, and each finished seed is saved as it ends |

## 10. Submission checklist

- Repository link and a README with the run commands
- `docs/ASSUMPTIONS.md`, `docs/DATA_DICTIONARY.md`, `docs/RESULTS.md`
- Dashboard screenshots or a short screen recording
- Progress slides

## 11. Open questions

- Where the code should live for the text entry, for example a link to your own GitHub repository.
