# HyperOp

Hyperparameter optimization (HPO) for genetic programming symbolic regression on 13
Feynman benchmark equations ([specification](data/Feynman/README.md)). It compares
classic HPO methods with an LLM agent that runs on small local models (Qwen3.5 2B/4B/9B
via Ollama). Fifteen further Feynman equations (`--help` lists them) are reserved for pilots
that choose settings without touching the evaluation tasks.

## Requirements

- .NET 10 SDK
- Docker (Linux containers) for irace and the local model server
- PowerShell 7 for batch runs

## Quick start

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --algorithm RandomSearch --instance I.6.20 --configurations 30 --evaluations 30000 --output ./results/demo
```

`--help` lists all options and benchmark keys. A repetition's budget is T × E candidate
evaluations (`--configurations` T, `--evaluations` E; the study uses T = 30, E = 30,000).
`--noise 0.1 --rows 200` is the noisy subset: Gaussian noise (10% of the target's standard
deviation, fixed seed) on the training and validation targets, and 200 evenly spaced training
and 200 validation rows, split into 100 for selection and 100 for the generalization channel
(test unchanged, scored against the clean and the noisy targets); job names contain
`__noise=0.1__rows=200`.
Every new GP candidate gets Levenberg-Marquardt constant fitting (5 iterations on 250
training rows; `--fitting-*` options), which is not counted as evaluations.

| Method | Description |
| --- | --- |
| `RandomSearch`, `GridSearch` | Uniform sampling; a fixed 5,000-point grid in seeded random order |
| `HillClimbingSearch` | Steepest descent on the grid with random restarts |
| `IRaceSearch` | irace 4.4.0 in a Docker container (image builds automatically; at least 6 trials; at most T trials: its post-selection race spends a remainder of two or more runs, a single remaining run stays unused) |
| `DefaultConfigurationSearch` | The default configuration, rerun T times |
| `RuleControlledSearch` | Random configurations from a shared pool, rule-based early stopping |
| `RandomRunLengthSearch` | Random configurations from a shared pool with random run lengths, no stopping |
| `AshaSearch` | Stop-only asynchronous successive halving: random configurations, rungs at E/10·3^k, best third continues (none before 3 runs reached a rung) |
| `AgentSearch` | LLM configurator and run controller sharing a pool of T × E evaluations |

The first five run T trials of E evaluations (grid search and hill climbing are reported in the
supplement only). The last four draw trials from a shared
pool of T × E: each run gets between E/10 and 4E evaluations (so at most 10T runs; there is no
separate trial cap), and a
stopped run returns its unused evaluations to the pool.

## Agent with local models

Start Ollama and pull the study models (`qwen3.5:{2b,4b,9b}-q4_K_M`):

```powershell
docker compose -f docker/agent.compose.yml up -d ollama-models
# NVIDIA GPU: add -f docker/agent.gpu.compose.yml
```

Run one model size at a time: with the compose settings (`OLLAMA_KEEP_ALIVE=-1`,
`OLLAMA_MAX_LOADED_MODELS=1`) jobs of different sizes block each other, and waiting controller
requests time out and fall back to Continue. The compose file sets a 131,072-token context and
`OLLAMA_NUM_PARALLEL=1`: Ollama 0.40.3 serves Qwen3.5 one request at a time, so concurrent agent
jobs queue their requests and overlap only in GP.

Run the agent from the host:

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --algorithm AgentSearch --model qwen3.5:4b-q4_K_M --feedback progress --instance I.6.20 --configurations 30 --evaluations 30000 --output ./results/agent
```

Both roles are stateless: each decision is one request with the state as text and a JSON reply.
The LLM configurator sees the trial table and its own plan and proposes a configuration and a
run length for each trial until the shared pool is spent, as every baseline does. After every
E/k candidate evaluations (`--checkpoints k`, default 10), the controller decides whether to
stop or continue the run; it sees only that run's feedback and what stopping or continuing
leaves for other configurations. `--controller-prompt` selects a variant of its instructions
(the study uses p8, p9 is its sensitivity variant; see `ControllerPrompts.cs`). For the RQ1
controls, `--controller rule` (stop below 1% improvement) or `--controller random --stop-hazard
p1,p2,...` (stop probability per decision, from `scripts/controller_stops.py --hazard`) replace
the LLM controller while the LLM configurator stays. `--feedback` sets the feedback channels:

| Condition | Feedback |
| --- | --- |
| `final` | No checkpoints; final results only |
| `progress` (default) | Generation, evaluations, best training NMSE and its improvement |
| `progress+curve` | Plus the recent best and median NMSE curve |
| `progress+diversity` | Plus structural and phenotypic diversity |
| `progress+size` | Plus tree length and depth statistics |
| `progress+generalization` | Plus validation NMSE of the current best |
| `all` | Progress and all channels |
| `--placebo` with `all` | Same fields, values from another run of the search (length-matched placebo) |

Channels combine (e.g. `progress+curve+size`). `--configurator-checkpoints hidden` shows
checkpoints to the controller only. OpenRouter is available with `--provider openrouter`
(API key `OpenRouterAPIKey` in user secrets).

## Batch runs

```powershell
./scripts/Run-Experiments.ps1 -Algorithms RandomSearch,IRaceSearch -Instances I.6.20,I.9.18 -Repetitions 10 -Output ./results/campaign01
./scripts/Run-Experiments.ps1 -Algorithms AgentSearch -Feedback final,progress,all -Instances I.6.20 -Repetitions 5 -Output ./results/agents
```

CPU jobs run in parallel (`-MaxProcesses`); agent jobs run afterwards (`-MaxAgentProcesses`,
default 4), in batches of one model size, with loaded models unloaded before each batch. The
script writes `campaign-manifest-*.json` (arguments, Ollama version and settings) and
`fallbacks.csv` (controller decisions that fell back to Continue, per job); no further agent
jobs start once a job's fallback rate exceeds `-MaxFallbackRate` (default 5%). A `/hidden`
suffix makes a feedback condition controller-only (`-Feedback final,progress/hidden,progress`),
and a `/placebo` suffix gives the placebo (`-Feedback all,all/placebo`). `-Controller rule` or
`-Controller random -StopHazard 0.1,0.2` applies an RQ1 control to all agent jobs of the call
(with `-Feedback progress/hidden`).
Cached results are reused only from the same clean commit (`--allow-mixed-code` for
development). `-PlanOnly`
previews the jobs; `-Instances` also accepts the development equations.

Long studies run with `./scripts/Start-Study.ps1 -Plan repilot|campaign` (the plans defined in
the script) and stop with `./scripts/Stop-Study.ps1` (running jobs finish; `-Now` ends them).
Starting again continues with the missing jobs; this needs the same code (by content
fingerprint; notes and the paper may change), the same models (by digest), Ollama version and
plan, recorded in `study-identity.json` at the first start. The code must be committed.
At the end, `scripts/repilot_check.py` checks the results against `expected-jobs.txt`. Progress,
a per-minute system monitor (GPU, CPU, RAM) and the Ollama server log go to the output
directory; `-Plan smoke` tries the cycle with 8 short 2B jobs.

## Results

Each job writes `<job>.result.json` and a `<job>__attempt=<id>.protocol.jsonl` event log
(GP runs, model requests with HTTP-level timing and tokens, controller decisions, timing
breakdown). Completed jobs are skipped on rerun; use a new output directory after
changing code or settings. A job whose protocol stays silent for 60 minutes
(`--stall-minutes`) is recorded as stalled and the process exits with code 3. Every result
records its provenance: a content fingerprint of the code (`scripts/code_fingerprint.py`
recomputes it for any copy, e.g. a public release with other commits), constant-fitting
implementation and settings, CPU vector support, core count, OS and runtime; a cached result of
other code or fitting is refused. No host name, git commit or local time is recorded: all times
are UTC. Agent results
also record how the search ended (`Search`: exit, evaluations left, rejected replies, clamped
proposals); job names over 200 characters carry a hash of the variant instead of the variant.
`--aggregate --output <dir>` combines results into `all_results.json`.

Every tuning run records the current best expression every E/10 evaluations and at its end
(validation and test NMSE, length). The selected expression is the snapshot with the lowest
validation NMSE; values below 1e-10 count as solved and equal, ties go to the shortest, then
the earliest expression. Test NMSE is stored for analysis only and never used by a method.
No method stops at an exact fit. The nominated configuration (the trial of the selected
expression; for irace its native first elite) is rerun with K fresh seeds (`--assessment-runs`,
default 5) at E and 4E.

## Analysis

```powershell
python scripts/analyse_outcomes.py ./results/campaign01 --out ./analysis/campaign01
python scripts/analyse_outcomes.py --self-test
```

The script (standard library only) resolves failed and retried attempts by the declared policy
(`scripts/attempts.py`), requires every expected job and one code and model identity (also
across several result directories analysed together), and
replays the selection over cumulative evaluations. It writes per-job outcomes (with
reliability and per-role request counts and tokens), per method and task the median log10
test NMSE (floored at 1e-10, capped at 100; failures score the cap), solve rate and expected
running time with bootstrap intervals, the ECDF of reached (task, target, repetition) triples,
and anytime curves. `--compare "RQ1:A~B,C~D//A~B"` adds task-weighted contrasts and
interactions per family (exact Wilcoxon signed-rank with Pratt zeros over the per-task means,
an interval with the tasks fixed and repetitions resampled, Holm within the family).

## Checks

```powershell
dotnet run --project HyperOp.Checks -c Release
```

The checks need neither Docker nor a model server.
