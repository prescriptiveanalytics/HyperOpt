# HyperOp

Hyperparameter optimization for genetic programming symbolic regression on Feynman
benchmarks and Steel Industry energy data.

## Requirements
`ResultDTO` snapshots. No R bindings or R-specific .NET dependencies are required.
- .NET 10 SDK.
- PowerShell 7 for batch runs.
- Docker Desktop with Linux containers for irace (the image builds automatically).
- Ollama with the requested model available for agent runs.
and press **F5**. This profile runs the .NET experiment on the host.
## Run
defaults directly in code or override them with command-line arguments.
From the repository root:

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --algorithm RandomSearch --instance I.6.20 --configurations 10 --evaluations 1000 --output ./results/demo
```
variables or manual Docker build/start commands are required. The initial build
Methods: `RandomSearch`, `GridSearch`, `HillClimbingSearch`, `IRaceSearch`,
`AgentSearch`. Use `--help` for all options and benchmark names.
The pipeline automatically starts an R container for each optimization run and removes
`--configurations` caps GP trials, including repeats. `--evaluations` caps candidate
evaluations per trial, including initialization; the minimum is 100. Supply both
when overriding budgets. irace requires at least four trials.
The image runs Rscript as its entrypoint. Docker must be running and able to bind
For a local agent:
container, with standard input/output carrying proposals and costs. The experiment
itself stays on the host. The image is reused across calls; each `Execute` call
dotnet run --project HyperOp.Experiments -c Release -- --algorithm AgentSearch --instance I.6.20 --model qwen3:8b --ollama-url http://localhost:11434 --output ./results/agent-demo

After changing the Dockerfile, rebuild the image manually from the repository root:
Agent runs request Continue/Stop decisions every 100 offspring generations.
Change `DecisionIntervalGenerations` and `DecisionTimeout` in
[AgentParameter.cs](HyperOp.Algorithms/HyperParameterOptimization/AgentComponents/AgentParameter.cs).
Short runs may finish before a checkpoint. Controller failures fall back to Continue.

## Batch runs
```powershell
docker build -f docker/irace.Dockerfile -t hyperop-irace .
./scripts/Run-Experiments.ps1 -Algorithms RandomSearch,IRaceSearch -Instances I.6.20,SteelIndustry -Repetitions 10 -Output ./results/campaign01

Run a short experiment:
The launcher builds once, runs CPU jobs in parallel, then agent jobs sequentially,
and combines completed results into `all_results.json`. Use `-MaxProcesses` to set
CPU workers, `-PlanOnly` to preview jobs, and `-RepetitionStart` to assign distinct
repetition ranges across machines. Agent batches accept `-Models` and `-OllamaUrl`.

## Results

Each completed experiment produces a `*.result.json` file; batch runs also save
stdout/stderr logs. Completed jobs are reused. Interrupted experiments restart.
Use a new output directory after changing code, data, prompts or settings.

Tuning uses validation NMSE (lower is better). Test scores are saved for the selected
expression and for the nominated configuration rerun with five fresh GP seeds.
These assessment runs are additional to the tuning budget. Agent results also
record checkpoint decisions and inference times.

Rebuild the aggregate manually:

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --aggregate --output ./results/campaign01
```

## Checks and references
five budget variations, and `--algorithm all` to run every optimizer. The default is
```powershell
dotnet run --project HyperOp.Checks -c Release
```
The search space comes from `AlgorithmParameter` range attributes and `MutatorType`.
The checks do not need Docker or a model service.
See the [Feynman specification](data/Feynman/README.md),
[Steel dataset notes](data/SteelIndustry/README.md) and
[paper sources](Paper/Paper/sigconf.tex) for research details.

For irace, `numConfigurations` / `--configurations` means the **maximum total target
evaluations**, including repeated trials of a configuration. A minimum of four is
required. Returned results include every actual trial, and `ConfigurationIndex`
stores irace's configuration ID, so repeated IDs are expected. irace can stop before
the budget is exhausted. Small budgets use fewer racing iterations (one for ten
trials); larger budgets use up to five iterations. GA stopping happens at the
library's candidate-evaluation checkpoints; `EvaluationsUsed` uses the same
generation-count approximation as the existing searches.

Adjust `--configurations` to change irace's total trial budget and `--evaluations` to
change the GA work per trial. Both options must be supplied together. Results are
written under `results/<run-id>/`, including per-experiment JSON and `all_results.json`.
