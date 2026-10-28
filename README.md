# HyperOp

Hyperparameter optimization for genetic programming symbolic regression on Feynman
benchmarks and Steel Industry energy data.

## Requirements

- .NET 10 SDK.
- PowerShell 7 for batch runs.
- Docker Desktop with Linux containers for irace (the image builds automatically).
- Ollama with the requested model available for agent runs.

## Run

From the repository root:

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --algorithm RandomSearch --instance I.6.20 --configurations 10 --evaluations 1000 --output ./results/demo
```

Methods: `RandomSearch`, `GridSearch`, `HillClimbingSearch`, `IRaceSearch`,
`AgentSearch`. Use `--help` for all options and benchmark names.

`--configurations` caps GP trials, including repeats. `--evaluations` caps candidate
evaluations per trial, including initialization; the minimum is 100. Supply both
when overriding budgets. irace requires at least four trials.

For a local agent:

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --algorithm AgentSearch --instance I.6.20 --model qwen3:8b --ollama-url http://localhost:11434 --output ./results/agent-demo
```

Agent runs request Continue/Stop decisions every 100 offspring generations.
Change `DecisionIntervalGenerations` and `DecisionTimeout` in
[AgentParameter.cs](HyperOp.Algorithms/HyperParameterOptimization/AgentComponents/AgentParameter.cs).
Short runs may finish before a checkpoint. Controller failures fall back to Continue.

## Batch runs

```powershell
./scripts/Run-Experiments.ps1 -Algorithms RandomSearch,IRaceSearch -Instances I.6.20,SteelIndustry -Repetitions 10 -Output ./results/campaign01
```

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

```powershell
dotnet run --project HyperOp.Checks -c Release
```

The checks do not need Docker or a model service.
See the [Feynman specification](data/Feynman/README.md),
[Steel dataset notes](data/SteelIndustry/README.md) and
[paper sources](Paper/Paper/sigconf.tex) for research details.
