# Intelligent Agents and Multi-Agent Systems for Emergency Response Coordination

**Shreyash Yadav**
CSYE7380 20134 Theory & Prac App AI Gen Model SEC 08 Fall 2026 [VTL-1-OL], Professor Subrata Das
Midterm project, November 4, 2026

**Code:** https://github.com/shreyashyadav1/emergency-response-mas

## Summary

I built a simulated city where a mass-casualty event plays out over four hours, and four kinds of AI agents coordinate
ambulances, responder teams, ER bays and hospital staff to serve the victims. The agents only talk through structured
messages. When their goals clash, they negotiate: hospitals make counter-offers, a referee agent decides who gets a
contested ambulance, and hospitals lend each other nurses.

To check that the coordination itself helps, I ran every policy on the same 150 scenarios (600 runs):

- The agents with negotiation (M2) raise severity-weighted survival by 0.056 to 0.067 over the best simple baseline at
  every stress level (p < 0.001 each), which is 1.2 to 2.0 more expected survivors per scenario.
- Negotiation alone, compared with the same agents without it (M1), adds 0.025 to 0.039.
- When the agents' judgment calls are made by OpenAI Agents SDK agents on `gpt-4o-mini` (M3), the system keeps almost
  all of that: +0.053 over the best baseline (better on 26 of 30 seeds, p < 0.001), +0.021 over the agents without
  negotiation, and 0.003 below the rules version.

## 1. The simulated world

Each scenario is built from one seed. The default (urban mass-casualty theme) has:

- **12 incidents** arriving over 90 minutes: vehicle accidents, building fires, collapses, hazmat releases and crowd
  incidents, with about 50 victims in total at medium stress. Each incident has a location, a victim count, a
  RED/YELLOW/GREEN mix,
  hazards and sometimes trapped people.
- **3 hospitals** with different capabilities (trauma, neurosurgery, burns, cardiac, decontamination, critical care),
  6 to 14 ER bays, ICU beds, and doctors and nurses with skills, shifts and workload limits. A regional on-call pool of
  3 doctors and 6 nurses comes on duty 15 to 45 minutes in, and all three hospitals want it.
- **5 ambulances**, alternating ALS and BLS. Only ALS units can carry a RED patient, one per load.
- **6 responder teams**: fire engines, heavy rescue, hazmat and paramedic quick-response cars. At fires, collapses and
  hazmat scenes, victims can't be reached until the right team has worked for 10 minutes, and trapped victims wait
  for extrication.
- **Travel times** from road distance, three congestion zones and extra congestion in the first 90 minutes.

A victim's chance of survival stays flat while care starts inside their golden window (25 to 60 minutes for RED) and
then halves every half-life. My main metric, **severity-weighted survival**, averages every victim's survival chance
with weights 10 for RED, 4 for YELLOW and 1 for GREEN.

The agents don't see the truth at first. They get a noisy caller report (callers under-count and can't triage) and
base rates learned from 600 synthetic past incidents. Three stress levels (low, medium, high) change the number of
victims, how full the hospitals already are, how many units are busy and the congestion. Two more themes, an
earthquake and a highway pile-up, change the incident mix.

**How the data is generated.** I used seeded NumPy random streams, Faker for names and caller narratives, and Pydantic
for the schema, rather than an LLM, so the same seed always gives the same world. Each kind of entity has its own random
stream, every saved scenario folder has a SHA-256 manifest, and a `verify` command regenerates the saved scenarios and
compares the hashes. A calibration run over 1,800 scenarios (200 seeds for each theme and stress level) failed none of
the validity checks. Every assumption is listed in `docs/ASSUMPTIONS.md`, which is generated from the single parameter
file the code reads, and every field is described in `docs/DATA_DICTIONARY.md`.

**The simulator** runs in 1-minute ticks for 240 minutes and is the only thing that changes the world. Agents ask for
actions, and the simulator checks each one (capability, a free bay, a staff slot, ALS for RED patients, one RED patient
per load) and accepts or rejects it with a reason. The same rules apply to every policy, so the differences in results
come from decisions only.

## 2. The four agents

| Agent | What it wants | What it decides |
|---|---|---|
| Incident Assessment | An accurate picture of each incident | Turns the caller's report into an estimate of victims, RED patients, hazard and scene tasks using the base rates, plus an urgency score. Switches to the real count once a unit is on scene. |
| Dispatch | Get help to victims fast | Bids for ambulances and teams for each incident, valuing a unit by the survival it buys per minute of its time. Chooses which hospitals to ask for each load. |
| Hospital Coordination (one per hospital) | Protect its own capacity | Projects its bays and staff to the ambulance's arrival time and keeps bays back for RED patients. Accepts, refuses or counter-offers ("yes in N minutes", "yes if you lend me a nurse"). Opens surge bays. |
| Resource Allocation | The best outcome for the whole system | Awards contested units with an assignment solve and tells each losing incident what it gets instead. Settles counter-offers by when care would actually start. Arranges staff loans, places the on-call pool and diverts ambulances stuck at full ER doors. |

These goals really do conflict. A hospital wants to hold beds for critical patients who may still come, Dispatch wants
every load placed now, and two incidents often want the same ambulance.

**Messages.** Every message has a `sender`, `recipient`, `type`, `incident_id` and `proposed_action`, plus an id, the
time, a `rationale`, `in_reply_to` and the negotiation `round`. Every reply points back to the request it answers, so a
negotiation can be read as one thread, and every negotiation ends with a decision within 3 rounds. At medium stress, one
run has about 240 messages, 26 unit conflicts, 16 hospital counter-offers, 18 staff requests and 9 staff moves.

**One real negotiation** (seed 2, minute 26, M2):

| Round | From | To | Message |
|---|---|---|---|
| 1 | Dispatch | Hospital H1 | Admission request: ambulance A5 bringing a RED and a GREEN patient, arriving in 32 minutes |
| 1 | H1 | Dispatch | Counter-proposal: a bay is free, but no nurse for trauma surgery |
| 2 | Resource Allocation | Hospital H2 | Staff request: H1 needs a nurse who can handle trauma surgery |
| 2 | H2 | Resource Allocation | Staff offer: can lend nurse N17 |
| 3 | Resource Allocation | H1 and Dispatch | Decision: H1 accepts if nurse N32 comes from the regional pool, and A5 goes to H1 |

## 3. Where the OpenAI Agents SDK fits

Each of the four agents also exists as an OpenAI Agents SDK `Agent`, with its own instructions, function tools and a
Pydantic reply schema (`emergency_mas/llm/agents.py`). In M3, these SDK agents make the judgment calls inside the same
coordination loop:

| SDK agent | Decision it makes | Its tools |
|---|---|---|
| Incident Assessment | Victims, RED count, hazard and trapped people, from the caller's own words | `base_rates`, `statistical_estimate` |
| Dispatch | The order in which to ask hospitals to take a load | `hospital_options` |
| Hospital Coordination | Counter-offer or refuse, when it can't simply accept | `capacity_at_arrival` |
| Resource Allocation | Who gets each contested unit, and how to settle a countered admission | `contested_units`, `settlement_options` |

The tools return the same numbers the rules agents compute, such as expected survival, capacity at arrival time and
when care would start, and the model decides. I kept the arithmetic in the tools on purpose, because I wanted the model
to weigh options rather than add up survival curves. Each answer is checked against its schema and against the options
it was offered. If it fails, the rules decision stands and the failure is counted, and the simulator still checks every
action either way. Every answer is also cached with the prompt and tool data it answered, so the runs replay exactly
without an API key.

The runs use `gpt-4o-mini` at temperature 0 through OpenAI's API. I first ran three seeds on Google Gemini through its
OpenAI-compatible endpoint, which the SDK supports, but its free tier only allows 500 requests a day, so the full run
moved to OpenAI. Thirty seeds took 2,051 decisions and 4,105 model requests (2.2 million input tokens), about 20
minutes in six parallel processes, and cost about 38 cents.

## 4. Evaluation

Every policy runs on the same scenarios, and differences are paired by seed: 30 seeds at each stress level for the urban
theme, plus the earthquake and highway pile-up at high stress, so 150 scenarios and 600 runs. Brackets are 95% bootstrap
intervals, and p-values come from Wilcoxon signed-rank tests on the paired differences.

- **B0, nearest first:** the nearest free unit goes to the earliest call.
- **B1, priority greedy:** the most critical report is served first.
- **M1:** the four agents, without negotiation.
- **M2:** the four agents with negotiation.
- **M3:** M2 with its judgment calls made by the SDK agents on `gpt-4o-mini` (urban theme, medium stress, the same 30
  seeds).

**Headline.** M2 against the best baseline, which was B1 at every stress level:

| Stress | Weighted survival, M2 minus B1 | Seeds where M2 is better | p | Extra expected survivors | Negotiation effect (M2 minus M1) | p |
|---|---|---|---|---|---|---|
| Low | +0.067 [+0.045, +0.089] | 80% | 6e-06 | +1.17 | +0.034 [+0.020, +0.050] | 0.0002 |
| Medium | +0.056 [+0.038, +0.074] | 87% | 5e-06 | +1.34 | +0.025 [+0.013, +0.037] | 0.0005 |
| High | +0.064 [+0.049, +0.078] | 97% | 1e-08 | +2.00 | +0.039 [+0.028, +0.049] | 8e-08 |

Against nearest-first (B0), M2 gains +0.075 to +0.098 in the urban theme, +0.047 in the earthquake and +0.097 in the
highway pile-up, and it's better on 90% to 100% of seeds in each setting.

**Every indicator at medium stress** (means over the same 30 seeds; the intervals are in `docs/RESULTS.md` and
`docs/LLM_RESULTS.md`):

| Indicator | B0 | B1 | M1 | M2 | M3 (SDK) |
|---|---|---|---|---|---|
| Weighted survival | 0.524 | 0.543 | 0.574 | 0.599 | 0.596 |
| Expected survivors | 41.0 | 41.4 | 42.0 | 42.7 | 42.7 |
| RED survivors | 1.08 | 1.49 | 1.97 | 2.27 | 2.24 |
| Victims reaching care | 23.3 | 22.1 | 27.4 | 31.2 | 30.7 |
| Critical incidents with a RED patient never treated | 4.30 | 4.40 | 3.27 | 2.97 | 3.07 |
| First unit on scene (minutes) | 46.9 | 47.4 | 45.3 | 36.0 | 36.0 |
| Ambulance utilization | 0.99 | 0.99 | 0.98 | 0.98 | 0.98 |
| ER bay utilization | 0.74 | 0.75 | 0.68 | 0.70 | 0.69 |
| Ambulance minutes waiting at full ER doors | 163 | 90 | 73 | 70 | 68 |

What I take from this:

- **Critical patients gain the most.** M2 has 1.5 to 2.1 times as many RED survivors as the best baseline, and 2.1 to
  3.5 times as many as nearest-first.
- **Negotiation is what gets units out faster.** The first unit reaches a scene about 11 minutes sooner under M2 than
  under B0. Without negotiation (M1) the agents are only 1 to 3 minutes faster, so most of that comes from negotiation,
  where contested units are settled together as one assignment.
- **Ambulances stop queuing at full ERs.** They wait 70 minutes per run in total under M2, against 163 under B0,
  because the agents send loads where a bay will actually be free.
- **ER bay use is a little lower with the agents** (0.70 against 0.74), partly because hospitals hold bays back for RED
  patients. That's by design.

**The SDK agents (M3).** On the same 30 seeds, M3 beats the best baseline (B1) by +0.053 [+0.035, +0.070], better on
26 of 30 seeds (p = 9e-06), and nearest-first by +0.072 [+0.052, +0.093], better on 27 of 30. It also beats the agents
without negotiation (M1) by +0.021 [+0.009, +0.034] (p = 0.003), so negotiation still pays off when the model makes the
judgment calls. Against the rules agents it's -0.003 [-0.007, -0.000] (p = 0.02), a small but real loss: equal on 9
seeds, better on 5 and worse on 16. The model departed
from the rules agent's choice on 139 of 2,051 decisions (7%): 17% of hospital orderings, 4% of assessments, 4% of
counter-or-refuse choices, 2% of contested unit awards and 1 of 34 settlements. No answer failed validation. The
hospital-order changes don't track the small gap. The largest single loss, seed 13 (-0.047), is also the only seed
where the model overruled the referee's settlement of an admission. A replay from the cache reproduces all 30 runs
exactly. So with these tools, the LLM agents match the hand-built agents but don't beat them.

## 5. Dashboard

The dashboard is built with Streamlit and Plotly (`streamlit run app/dashboard.py`). It replays one run minute by
minute, with the same scenario under a second policy beside it, and a play button, step buttons and a minute slider move
all five panels together:

1. **Incident map:** incidents sized by victims still waiting and colored by the worst triage among them, ambulances and
   teams with lines to where they're headed, and hospitals with an ER-bay meter. Both policies' maps can be shown side by
   side.
2. **Resource status:** ambulances, teams, ER bays, ICU beds, and doctor and nurse slots as in use, free or unavailable,
   plus a table per hospital.
3. **Agent activity:** one card per agent with its status, what it's still waiting on (calls not yet confirmed on scene,
   victims with no ambulance, ambulances inbound or stuck at a door, staff on the way), message counts and its latest
   messages.
4. **Negotiation history:** every unit conflict and admission request with its replies, and one negotiation read in
   full.
5. **Performance:** indicators at the current minute with the difference from the other policy, running totals, the
   final outcomes side by side, and where the seed ranks among the 30.

The running totals at the last minute equal the run's final metrics (there's a test for this), so the dashboard and the
results tables never disagree.

## 6. Limits and next steps

- The world is synthetic and the parameters are illustrative, not clinical. The comparison between policies is the
  result, not the survival numbers themselves.
- Negotiation helps at every stress level, but its effect doesn't grow steadily with stress (+0.034, +0.025, +0.039),
  so my original expectation that it would is only partly met.
- The Incident Assessment Agent removes the callers' under-count (victim count bias goes from -0.42 to +0.01), but it
  ranks incidents no better than the raw report (rank correlation 0.81 for both).
- M3 ran on one setting (urban, medium). Because its tools give it the rules agents' numbers, it mostly agrees with them.
  Letting it reason over less pre-computed information is the obvious next experiment.
- Ablations I planned but haven't run: no assessment agent, no staff lending, and a single negotiation round.

## 7. Running it

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/python -m pytest                                 # 60 tests
.venv/bin/python -m emergency_mas.eval run --seeds 30      # the 600 runs, about a minute
.venv/bin/python -m emergency_mas.eval report              # docs/RESULTS.md
.venv/bin/streamlit run app/dashboard.py                   # the dashboard
.venv/bin/python -m emergency_mas.llm run --cache-only --model gpt-4o-mini --seeds $(seq 1 30)
```

The last command replays the 30 M3 runs from the cached answers, so it needs no key. A live M3 run needs
`OPENAI_API_KEY` in `.env`; the README has the details.
