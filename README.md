# HyperOp

## irace optimization

`IRaceSearch` implements `HyperParameterOptimizationAlgorithm.Execute` with the
[irace R package](https://cran.r-project.org/package=irace). R proposes configurations
and performs racing; the .NET process evaluates each proposal using the base class's
`PrepareAlgorithm`, the requested candidate evaluation limit, and the standard
`ResultDTO` snapshots. No R bindings or R-specific .NET dependencies are required.

Start Docker Desktop with Linux containers enabled. Set `HyperOp.Experiments` as the
Visual Studio startup project, select its **HyperOp.Experiments** launch profile,
and press **F5**. This profile runs the .NET experiment on the host.

The settings near the top of `Program.cs` default to `IRaceSearch`, benchmark
`I.6.20`, 100 trials, 1,000 GA evaluations per trial, and one repetition. Change those
defaults directly in code or override them with command-line arguments.

`IRaceSearch` uses the hardcoded image name `hyperop-irace`. On the first `Execute`
call it checks Docker and builds the image automatically if absent. MSBuild copies
`docker/irace.Dockerfile` into the application output and publish directories, so
this works independently of Visual Studio's working directory. No environment
variables or manual Docker build/start commands are required. The initial build
downloads and compiles R packages and may take a few minutes.

The pipeline automatically starts an R container for each optimization run and removes
it on normal exit; no separate `docker run` or server process is needed. The container
communicates with the running .NET experiment, so do not launch it by itself.

The image runs Rscript as its entrypoint. Docker must be running and able to bind
mount the host temporary directory. Each invocation gets an isolated directory and
container, with standard input/output carrying proposals and costs. The experiment
itself stays on the host. The image is reused across calls; each `Execute` call
starts a fresh R container with that call's seed and budget.

After changing the Dockerfile, rebuild the image manually from the repository root:

```powershell
docker build -f docker/irace.Dockerfile -t hyperop-irace .
```

Run a short experiment:

```powershell
dotnet run --project HyperOp.Experiments -- --algorithm IRaceSearch --instance I.6.20 --configurations 100 --evaluations 1000 --repetitions 1 --output ./results
```

Omit arguments to use the same defaults as F5. Use `--budget-sweep` for the original
five budget variations, and `--algorithm all` to run every optimizer. The default is
one repetition. Instance selection accepts the benchmark aliases
in `Program.cs`, descriptor names, and class names such as `Feynman1`.

The search space comes from `AlgorithmParameter` range attributes and `MutatorType`.
irace minimizes costs, so the runner returns negative best fitness and penalizes
nonfinite fitness. It uses irace's trial seeds for the GA and the supplied search
seed for irace. The benchmark descriptor controls dataset generation as usual.

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
