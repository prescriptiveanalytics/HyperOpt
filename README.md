# HyperOp

Hyperparameter optimization (HPO) for genetic programming symbolic regression on 13
Feynman benchmark equations ([specification](data/Feynman/README.md)). It compares
classic HPO methods with an LLM agent that runs on small local models (Qwen3.5 2B/4B/9B
via Ollama).

## Requirements

- .NET 10 SDK
- Docker (Linux containers) for irace and the local model server
- PowerShell 7 for batch runs

## Quick start

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --algorithm RandomSearch --instance I.6.20 --configurations 10 --evaluations 1000 --output ./results/demo
```

`--help` lists all options and benchmark keys. `--configurations` (T) caps GP trials
and `--evaluations` (E) caps candidate evaluations per trial.

| Method | Description |
| --- | --- |
| `RandomSearch`, `GridSearch` | Uniform sampling; a fixed 5,000-point grid in seeded random order |
| `HillClimbingSearch` | Steepest descent on the grid with random restarts |
| `IRaceSearch` | irace 4.4.0 in a Docker container (image builds automatically; at least 6 trials) |
| `DefaultConfigurationSearch` | The default configuration, rerun T times |
| `RuleControlledSearch` | Agent setting without an LLM: random configurations, rule-based early stopping |
| `AgentSearch` | LLM configurator and run controller sharing a pool of T × E evaluations |

## Agent with local models

Start Ollama and pull the study models (`qwen3.5:{2b,4b,9b}-q4_K_M`):

```powershell
docker compose -f docker/agent.compose.yml up -d ollama-models
# NVIDIA GPU: add -f docker/agent.gpu.compose.yml
```

Run the agent from the host:

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --algorithm AgentSearch --model qwen3.5:4b-q4_K_M --feedback progress --instance I.6.20 --configurations 10 --evaluations 1000 --output ./results/agent
```

The configurator proposes a configuration and an evaluation allowance for each trial. Every
`--checkpoint-interval` generations (default 10), the controller decides whether to stop
or continue the run; a stopped run returns its unused evaluations to the pool.
`--feedback` sets what the controller sees at each checkpoint:

| Condition | Feedback |
| --- | --- |
| `final` | No checkpoints; final results only |
| `progress` (default) | Generation, evaluations, best training NMSE and its improvement |
| `progress+curve` | Plus the recent best and median NMSE curve |
| `progress+diversity` | Plus structural and phenotypic diversity |
| `progress+size` | Plus tree length and depth statistics |
| `progress+generalization` | Plus validation NMSE of the current best |
| `all` | Progress and all channels |

Channels combine (e.g. `progress+curve+size`). `--configurator-checkpoints hidden` shows
checkpoints to the controller only. OpenRouter is available with `--provider openrouter`
(API key `OpenRouterAPIKey` in user secrets).

## Batch runs

```powershell
./scripts/Run-Experiments.ps1 -Algorithms RandomSearch,IRaceSearch -Instances I.6.20,I.9.18 -Repetitions 10 -Output ./results/campaign01
./scripts/Run-Experiments.ps1 -Algorithms AgentSearch -Feedback final,progress,all -Instances I.6.20 -Repetitions 5 -Output ./results/agents
```

CPU jobs run in parallel (`-MaxProcesses`); agent jobs run afterwards (`-MaxAgentProcesses`,
default 1). `-PlanOnly` previews the jobs.

## Results

Each job writes `<job>.result.json` and a `<job>__attempt=<id>.protocol.jsonl` event log
(GP runs, model requests with HTTP-level timing and tokens, controller decisions, timing
breakdown). Completed jobs are skipped on rerun; use a new output directory after
changing code or settings. `--aggregate --output <dir>` combines results into
`all_results.json`.

Selection uses validation NMSE; the selected expression and five fresh reruns of the
nominated configuration are scored on the test partition.

## Checks

```powershell
dotnet run --project HyperOp.Checks -c Release
```

The checks need neither Docker nor a model server.
