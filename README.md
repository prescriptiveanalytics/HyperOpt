# HyperOp

HyperOp compares hyperparameter optimization strategies for genetic-programming
symbolic regression. All strategies use the same HEAL.HeuristicLib GP evaluator
and benchmark data. The optimizers choose configurations; `AgentSearch` also uses
LLM feedback to control how long each GP run continues.

## Repository structure

| Location | Responsibility |
| --- | --- |
| `HyperOp.Experiments/Program.cs` | CLI options, benchmark selection, repetitions, optimizer construction, assessment and result saving. |
| `HyperOp.Experiments/ExperimentResultStore.cs` | JSON persistence, validation, cache loading and explicit aggregation. |
| `HyperOp.Algorithms/HyperParameterOptimization/HyperParameterOptimizationAlgorithm.cs` | Shared GP setup, exact candidate-evaluation accounting, optional controller checkpoints, configuration nomination and assessment. |
| `HyperOp.Algorithms/HyperParameterOptimization/` | Agent, random, grid, hill-climbing and irace implementations. |
| `HyperOp.Algorithms/HyperParameterOptimization/RunControl.cs` | Checkpoint observations and Continue/Stop decisions, including the next observation delay. |
| `HyperOp.Algorithms/HyperParameterOptimization/AgentComponents/AgentParameter.cs` | LLM provider, model, temperature, conversation turn limit and controller timeout. |
| `HyperOp.Algorithms/Resources/systemPrompt.md` | Embedded configuration-agent instructions. Controller instructions live in `AgentSearch.cs`. |
| `HyperOp.Algorithms/Resources/irace.R` | R-side racing and communication with the .NET evaluator. |
| `HyperOp.Algorithms/AlgorithmParameter.cs` | GP hyperparameters, supported mutators and valid ranges. |
| `HyperOp.Algorithms/Benchmarks/` | Regression descriptors, frozen data partitions and Steel Industry CSV loading. |
| `HyperOp.Algorithms/Feynman/` | Synthetic benchmark definitions and data generation. |
| `HyperOp.Algorithms/ResultDTO.cs` / `AggregatedResultDTO.cs` | Individual GP-run results and complete experiment results. |
| `data/SteelIndustry/` | Dataset and dataset documentation. |
| `docker/irace.Dockerfile` | R/irace runtime image. |

## Experiment workflow

1. `Program.cs` selects benchmarks and freezes their training, validation and test
   data before optimization. The same prepared data is reused across methods and
   repetitions. The current instance seed is 42; the HPO seed is `42 + 100 * repetition`.
2. The optimizer proposes a configuration and calls the shared GP evaluator.
   GP evolves expression trees using training NMSE. Linear scaling is fitted on
   training data, and the resulting model is scored on validation data.
3. Each trial returns `BestFitness`, which is **validation NMSE**: lower is better,
   zero is ideal, and nonfinite predictions are represented by positive infinity.
   Training fitness and population summaries are stored separately.
4. After tuning, the runner selects the best observed validation trial and obtains
   the optimizer's recommended configuration. These selections are frozen before
   test data are scored. Most optimizers nominate the best validation trial;
   irace nominates its final elite configuration.
5. The selected trial is scored on test data. The recommended configuration also
   receives **five fresh-seed assessment runs**, currently configured in `Program.cs`.
   These use the fixed `--evaluations` allowance, without LLM run control, and are
   separate from the tuning budget. Their validation and test scores measure the
   behavior of the configuration across new runs.
6. The runner writes one result file per completed experiment. Combining those
   files into `all_results.json` is a separate operation.

The evaluator counts actual candidate evaluations, including initialization, and
limits the final batch to avoid exceeding the allowance. Evaluation counts are
not estimates based on population size and generation count.

## Optimization strategies and budgets

| Algorithm | Search behavior | Meaning of the budget arguments |
| --- | --- | --- |
| `RandomSearch` | Samples parameters from their declared ranges. | Runs `--configurations` trials with `--evaluations` each. |
| `GridSearch` | Visits a seeded shuffle of a fixed discrete grid. | Up to `--configurations` trials, each with `--evaluations`; may exhaust the grid. |
| `HillClimbingSearch` | Evaluates unseen coordinate neighbors and moves to an improving neighbor. | Up to `--configurations` trials, each with `--evaluations`; may stop at a local optimum. |
| `IRaceSearch` | Uses irace to propose and race configurations across seeds. | `--configurations` caps target-runner calls, including repeated configurations; each receives `--evaluations`. Minimum four calls. |
| `AgentSearch` | Chooses distinct configurations, observation timing and run termination using LLM feedback. | At most `--configurations` configurations share `--configurations * --evaluations` candidate evaluations. |

For example, AgentSearch with `--configurations 100 --evaluations 1000` has a shared
budget of 100,000 evaluations. A run can exceed 1,000 evaluations. There are no
exploration reservations, fixed per-run caps or minimum configuration counts.
Unused budget can remain when the agent finishes early. Assessment runs remain
outside this shared tuning budget.

Both budget arguments must be supplied together. The common budget validation
requires `--evaluations` to be at least 100, so all legal population sizes fit.
`--budget-sweep` uses these five pairs instead of the default single budget:
`(10, 1000000)`, `(100, 100000)`, `(1000, 10000)`, `(10000, 1000)`, `(100000, 100)`.

The GP search space is:

| Parameter | Allowed values |
| --- | --- |
| PopulationSize | Integers 10–100 |
| MutationRate | 0–1 |
| MaxTreeDepth / MaxTreeLength | Integers 3–100 |
| MutatorType | NodeReplacement, Subtree, LocalPerturbation, Combined |

Grid and hill climbing use the discrete subset defined in `Util/ParameterSpace.cs`.
Tournament selection currently uses a fixed tournament size of five; selection
pressure is not exposed as a tunable parameter.

## AgentSearch feedback loop

Two LLM roles cooperate within `AgentSearch.cs`:

- **Configuration agent:** chooses hyperparameters and `firstObservationGeneration`,
  then invokes `RunSymbolicRegression`. It can inspect `GetExperimentHistory` and
  `GetRemainingEvaluations`, or end the overall search with `FinishSearch(reason)`.
- **Run controller:** receives training progress, previous checkpoints, overall
  budget and summaries of completed configurations. It returns Continue or Stop
  with a short reason. Continue also specifies a positive
  `NextObservationInGenerations`; Stop returns that field as null.

The SymReg loop pauses at the agent-selected generation to request a decision.
Checkpoint observations include training NMSE, improvement since the previous
checkpoint, structural diversity and median tree length. Diversity is the fraction
of unique expression strings, not semantic diversity. There is no configured
observation interval. If the evaluation budget is exhausted before a chosen
checkpoint, no further decision is requested.

A controller Stop ends only the current GP run. Its observations and reasons are
returned to the configuration agent, which can adapt the next configuration.
Identical hyperparameters return the previous result without another evaluation.
The prompts emphasize comparing alternatives, considering non-round parameter
values and keeping reasons to one sentence of at most 20 words. These are LLM
instructions rather than enforced search quotas or output-length limits.

C# enforces the total evaluation and configuration caps. The outer search loop also
ends on zero validation fitness or an explicit `FinishSearch` call. The finish
reason is logged, not stored in aggregated results. A text summary alone does not
finish the search: C# resubmits the current state so the agent can run another
configuration or explicitly finish.

`AgentParameter.Turns` limits each conversation, not the entire search. There is no
consecutive-no-progress cutoff; an agent that neither runs a configuration nor
finishes can keep making requests. Provider errors reported by the configuration
agent fail the search. Controller failures or timeouts instead fall back to Continue
with the last agent-chosen observation delay, recorded as `UsedFallback`.

## Build and run

Use the .NET 10 SDK. From the repository root:

```powershell
dotnet build HyperOp.slnx
dotnet run --project HyperOp.Experiments -- --help
```

A baseline run needs neither Docker nor an LLM service:

```powershell
dotnet run --project HyperOp.Experiments -- --algorithm RandomSearch --instance I.6.20 --configurations 100 --evaluations 1000 --run-id baseline
```

The current defaults in `Program.cs` are `IRaceSearch`, benchmark `I.6.20`, one
repetition and budget `(100, 1000)`. CLI arguments override them. In Visual Studio,
select `HyperOp.Experiments` as the startup project and its **HyperOp.Experiments**
launch profile to run on the host.

`--instance` accepts the keys registered in `Program.cs`, case-insensitively, and
supports comma-separated keys, for example `I.6.20,I.9.18,SteelIndustry`. Class names
and descriptor names are not CLI aliases. The help output lists all registered keys.
Use `--repetitions` and `--repetition-start` to select repeated jobs and their seeds.
`--algorithm all` runs every optimizer and requires both irace and agent dependencies.

### Agent provider setup

`Program.cs` currently constructs `new AgentParameter()`, selecting OpenRouter with
model `google/gemma-4-26b-a4b-it`. Configure its key in the algorithms project's user secrets:

```powershell
dotnet user-secrets set "OpenRouterAPIKey" "YOUR_KEY" --project HyperOp.Algorithms
dotnet run --project HyperOp.Experiments -- --algorithm AgentSearch --instance I.6.20 --configurations 100 --evaluations 1000 --run-id agent01
```

`--model` overrides the model name. Provider, temperature, controller timeout and
conversation turn limit are set through `AgentParameter`. Current defaults include
temperature 0.05, 50 turns per conversation and a 30-second controller timeout.

To use Ollama, change the construction in `Program.cs` to
`AgentSearch.CreateOllamaParameters()`. This selects the custom provider, `qwen3:8b`,
`http://localhost:11434` and non-streaming responses. Run an Ollama server with the
chosen model available; `--model` and `--ollama-url` can override those settings.
`--ollama-url` alone does **not** switch the provider from OpenRouter. The CLI help's
Ollama default text does not reflect the current OpenRouter construction.

Prompts are embedded in the algorithms assembly. Rebuild and restart the experiment
after editing them; a running experiment keeps its existing instructions.

### irace and Docker

Start Docker Desktop with Linux containers enabled. No host R installation is needed.
`IRaceSearch` checks for the `hyperop-irace` image and builds it from the copied
`docker/irace.Dockerfile` if absent. The first build downloads and installs dependencies.

For each optimization call, .NET starts a temporary R container, mounts its temporary
runner files and exchanges proposals and costs over standard input/output. GP
execution stays in .NET on the host. The image is reused; the container is removed
on normal exit. Do not launch a separate irace server or container manually.

```powershell
dotnet run --project HyperOp.Experiments -- --algorithm IRaceSearch --instance I.6.20 --configurations 100 --evaluations 1000 --run-id irace01
```

irace minimizes validation NMSE directly; nonfinite fitness receives a large penalty.
It supplies GP trial seeds and may repeat configuration IDs during racing. The final
elite determines the recommended configuration. Rebuild the image after changing
its Dockerfile:

```powershell
docker build -f docker/irace.Dockerfile -t hyperop-irace .
```

## Results and caching

Results go directly to `--output` (default `./results`), or its `--run-id` subdirectory
when supplied. Each `*.result.json` filename identifies the algorithm, model for
AgentSearch, benchmark, repetition, budget and assessment count.

`ConfigurationResults` stores the tuning runs: hyperparameters, GP seeds, validation
and training scores, evaluation counts, expression/scaling data, population snapshots,
quality curves and any controller observations. `ConfigurationAssessmentResults`
stores the separate fresh-seed runs. The aggregate also records selection and
recommendation metadata and partition row counts. It does not store exploration
reservations, a budget-policy field or the search stop reason.

Existing matching result files are validated and reused. Filenames do not encode
prompts or every agent setting: use a fresh `--run-id` after changing search behavior
if you want a new experiment rather than a cached result.

Aggregate a completed campaign explicitly:

```powershell
dotnet run --project HyperOp.Experiments -- --aggregate --output ./results --run-id agent01
```

This reads completed `*.result.json` files and writes `all_results.json` in that
campaign directory without running GP. Failed or incomplete experiments do not get
a completed result file; keep console output when diagnosing them.
