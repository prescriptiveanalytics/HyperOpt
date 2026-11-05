using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Execution;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Operators;
using HEAL.HeuristicLib.Operators.Evaluators;
using HEAL.HeuristicLib.Operators.Interceptors;
using HEAL.HeuristicLib.Operators.Refiners;
using HEAL.HeuristicLib.Operators.Terminators;
using HEAL.HeuristicLib.Problems;
using HEAL.HeuristicLib.Problems.MachineLearning;
using HEAL.HeuristicLib.Random;
using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.Logging;
using Parlot.Fluent;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public abstract class HyperParameterOptimizationAlgorithm
    {
        public abstract Task<List<ResultDTO>> Execute(
            RegressionDescriptor feynmanInstance,
            int seed,
            int numConfigurations,
            int evaluationsPerConfiguration
            );

        public virtual string RecommendationRule => "minimum_validation_nmse";

        // One-time setup (e.g. building a container image), run before a job's clock starts so it is not charged to tuning.
        public virtual Task PrepareAsync() => Task.CompletedTask;

        public IProtocolLog Log { get; set; } = NullProtocolLog.Instance;

        // Methods that share a pool of T x E evaluations have no trial cap of their own: every run uses at least its shortest
        // stoppable length (the first checkpoint or E/10, whichever is shorter), except possibly the last one, which may get
        // the remainder. The pool therefore bounds the number of runs, and this limit cannot end a search before the pool
        // is spent: a policy that stops every run at its first checkpoint still spends the whole budget, as short probes.
        // (A cap of 3T ended such searches at 24-55% of the pool in the development pilot.)
        public static int MaximumSharedPoolTrials(int numConfigurations, int evaluationsPerConfiguration, int shortestStoppableRun) =>
            checked((int)((long)numConfigurations * evaluationsPerConfiguration / Math.Max(1, shortestStoppableRun)) + 1);

        // Methods that choose each run's length (agent variants, rule baseline, random run length) grant every run between
        // E/MinimumRunDivisor and MaximumRunFactor x E evaluations (3k-120k at E = 30k), never more than the pool has left.
        // The bound lets short probes and long runs coexist while limiting the advantage of long runs over the fixed-length
        // baselines; all nominees are assessed at the same lengths (AssessmentRunLengths), whatever their tuning runs used.
        public const int MinimumRunDivisor = 10;
        public const int MaximumRunFactor = 4;
        public static int MinimumRunEvaluations(int evaluationsPerConfiguration) =>
            (evaluationsPerConfiguration + MinimumRunDivisor - 1) / MinimumRunDivisor;
        public static int MaximumRunEvaluations(int evaluationsPerConfiguration) => checked(evaluationsPerConfiguration * MaximumRunFactor);
        public static int BoundRunAllowance(int requested, int evaluationsPerConfiguration, int remaining) =>
            Math.Min(remaining, Math.Clamp(requested, MinimumRunEvaluations(evaluationsPerConfiguration), MaximumRunEvaluations(evaluationsPerConfiguration)));

        // Fresh-seed assessment of every method's nominee at the reference length E and at the longest run length 4E;
        // the same seeds at both lengths, so the effect of length is paired.
        public static int[] AssessmentRunLengths(int evaluationsPerConfiguration) =>
            [evaluationsPerConfiguration, MaximumRunEvaluations(evaluationsPerConfiguration)];

        // Constant fitting for every GP run of every method (tuning and assessment).
        public static ConstantFittingSettings ConstantFitting { get; set; } = new();

        // Settings that distinguish otherwise identical jobs of this method (feedback condition, checkpoint interval).
        public virtual string? Variant => null;
        private string _phase = "tuning";

        // Model requests made by the optimizer itself; empty for methods without an LLM.
        public virtual IReadOnlyList<LlmCallRecord> LlmCalls => [];

        // The nominee is the trial whose snapshot was selected, so the reassessed configuration produced the selected expression.
        public virtual ResultDTO NominateConfiguration(IReadOnlyList<ResultDTO> tuningResults) =>
            SelectExpression(tuningResults) is { } selection ? tuningResults[selection.TrialIndex] : BestValidationTrial(tuningResults);

        // Validation NMSE below this counts as solved: exact recoveries end between 1e-20 and 1e-32 in floating point,
        // good approximations stay above about 1e-8. Selection treats all values below it as equal.
        public const double SolvedNmse = 1e-10;

        // Trial with the lowest final validation NMSE, values below SolvedNmse counting as equal, then the shortest final
        // expression, then the earliest trial.
        public static ResultDTO BestValidationTrial(IReadOnlyList<ResultDTO> tuningResults) =>
            tuningResults.Select((result, index) => (result, index)).Where(entry => double.IsFinite(entry.result.BestFitness))
                .OrderBy(entry => Math.Max(entry.result.BestFitness, SolvedNmse))
                .ThenBy(entry => entry.result.SelectedExpressionLength ?? int.MaxValue).ThenBy(entry => entry.index)
                .Select(entry => entry.result).FirstOrDefault()
            ?? throw new InvalidOperationException("No finite validation result is available for nomination.");

        // Primary selection over the snapshots of all tuning runs: lowest validation NMSE, values below SolvedNmse counting
        // as equal, then the shortest expression, then the earliest trial and evaluation count. Null without snapshots.
        public static ExpressionSelection? SelectExpression(IReadOnlyList<ResultDTO> tuningResults) =>
            tuningResults.SelectMany((result, index) => (result.Snapshots ?? []).Select(snapshot => (index, result, snapshot)))
                .Where(entry => double.IsFinite(entry.snapshot.ValidationNmse))
                .OrderBy(entry => Math.Max(entry.snapshot.ValidationNmse, SolvedNmse)).ThenBy(entry => entry.snapshot.Length)
                .ThenBy(entry => entry.index).ThenBy(entry => entry.snapshot.Evaluations)
                .Select(entry => new ExpressionSelection(entry.index, entry.result.ConfigurationIndex, entry.snapshot.Evaluations,
                    entry.snapshot.ValidationNmse, entry.snapshot.TestNmse, entry.snapshot.Length, entry.snapshot.Expression)
                    { NoisyTestNmse = entry.snapshot.NoisyTestNmse })
                .FirstOrDefault();

        public Task<List<ResultDTO>> AssessConfiguration(RegressionDescriptor instance,
            AlgorithmParameter configuration, IReadOnlyList<ResultDTO> tuningResults,
            int seed, int repetitions, int evaluationsPerRun) =>
            AssessConfiguration(instance, configuration, tuningResults, seed, repetitions, [evaluationsPerRun]);

        // Runs every seed at every length, lengths in the given order; ConfigurationIndex is the seed's index.
        public async Task<List<ResultDTO>> AssessConfiguration(RegressionDescriptor instance,
            AlgorithmParameter configuration, IReadOnlyList<ResultDTO> tuningResults,
            int seed, int repetitions, IReadOnlyList<int> runLengths)
        {
            configuration = configuration with { };
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(repetitions);
            if (runLengths.Count == 0 || runLengths.Any(length => length < configuration.PopulationSize))
                throw new ArgumentOutOfRangeException(nameof(runLengths), "Every assessment length must fit the initial population.");
            var usedSeeds = tuningResults.Select(result => result.GPSeed
                ?? throw new InvalidOperationException("Tuning seeds must be recorded before assessment."))
                .ToHashSet();
            var random = new Random(seed ^ int.MinValue);
            var assessmentResults = new List<ResultDTO>();
            _phase = "assessment";
            try
            {
                // A separate deterministic seed stream, shared across methods, without a data-seed option.
                // Negative GP seeds also separate the usual positive tuning/racing seeds.
                var gpSeeds = new int[repetitions];
                for (int repetition = 0; repetition < repetitions; repetition++)
                {
                    int gpSeed;
                    do { gpSeed = random.Next(int.MinValue, 0); } while (!usedSeeds.Add(gpSeed));
                    gpSeeds[repetition] = gpSeed;
                }
                foreach (int length in runLengths)
                    for (int repetition = 0; repetition < repetitions; repetition++)
                    {
                        var result = await EvaluateConfiguration(instance, configuration, repetition, gpSeeds[repetition], length);
                        result.EvaluateTest(instance.PreparedData);
                        assessmentResults.Add(result);
                    }
            }
            finally { _phase = "tuning"; }
            return assessmentResults;
        }

        // GP seed of tuning trial i in an HPO run: a SplitMix64 hash of (HPO seed, i). Methods share it for the same
        // repetition and trial, while repetitions do not reuse each other's seeds. Non-negative, unlike assessment seeds.
        public static int TrialSeed(int hpoSeed, int trialIndex)
        {
            ulong z = (((ulong)(uint)hpoSeed << 32) | (uint)trialIndex) + 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return (int)((z ^ (z >> 31)) & int.MaxValue);
        }

        // Snapshot spacing of tuning runs (E/SnapshotsPerConfiguration evaluations); set by ValidateBudget, 0 records none.
        public const int SnapshotsPerConfiguration = 10;
        private int _snapshotEvaluations;

        protected void ValidateBudget(int numConfigurations, int evaluationsPerConfiguration)
        {
            _snapshotEvaluations = CheckpointSchedule.Interval(evaluationsPerConfiguration, SnapshotsPerConfiguration);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numConfigurations);
            // Every legal population must fit without silently narrowing the search space.
            int largestPopulation = Convert.ToInt32(Util.ParameterSpace.Range(nameof(AlgorithmParameter.PopulationSize)).Maximum);
            if (evaluationsPerConfiguration < largestPopulation)
                throw new ArgumentOutOfRangeException(nameof(evaluationsPerConfiguration),
                    $"At least {largestPopulation} candidate evaluations per trial are required for the full population-size domain.");
        }

        protected virtual async Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor instance,
           AlgorithmParameter parameter, int configurationId, int seed, int evaluationsLimit)
        {
            return await EvaluateConfiguration(instance, parameter, configurationId, seed, evaluationsLimit, decideAsync: null);
        }
        protected virtual async Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor instance,
            AlgorithmParameter parameter, int configurationId, int seed, int evaluationsLimit,
            Func<RunProgress, CancellationToken, Task<RunDecision>>? decideAsync,
            int? firstObservationEvaluations = null, CancellationToken cancellationToken = default,
            FeedbackChannels channels = FeedbackChannels.Progress)
        {
            if (decideAsync is not null && firstObservationEvaluations is not > 0)
                throw new ArgumentOutOfRangeException(nameof(firstObservationEvaluations), "An agent-controlled run requires a positive first checkpoint.");
            parameter = parameter with { };
            var trialClock = Stopwatch.StartNew();
            var cpuAtStart = Process.GetCurrentProcess().TotalProcessorTime;
            var (algorithm, problem) = PrepareAlgorithm(instance, parameter);
            // Only candidates reaching the real evaluator are counted. The limiter assigns
            // worst fitness to unscored candidates in the last, possibly partial, generation.
            var countedEvaluator = algorithm.Evaluator.CountEvaluatedCandidates(out var counter);
            // Constant fitting is part of GP time; its share and the number of fitted candidates are recorded separately.
            var fittingDuration = new ObservationDuration();
            var fittedCandidates = new ObservationCounter();
            var fittingStatistics = new ConstantFittingStatistics();
            if (algorithm.Refiner is ScaledConstantFittingRefiner scaledFitting)
                algorithm = algorithm with { Refiner = scaledFitting with { Statistics = fittingStatistics } };

            algorithm = algorithm with {
                Evaluator = countedEvaluator.LimitEvaluations(evaluationsLimit, enforceLimitWithinBatch: true),
                Refiner = algorithm.Refiner?.CountRefinedCandidates(fittedCandidates).MeasureRefinerDuration(fittingDuration) is { } fitting
                    // Each new candidate is fitted with this probability, drawn from the run's seeded generator.
                    ? ConstantFitting.Probability >= 1 ? fitting : fitting.WithRate(ConstantFitting.Probability)
                    : null,
                Terminator = AfterOperatorCountTerminator.For(problem, counter, evaluationsLimit),
            };

            var analyzer = Analyzer.BestMedianWorst(algorithm);
            
            var stopwatch = Stopwatch.StartNew();
            var run = algorithm.CreateRun(problem, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(seed))
                .WithAnalyzer(analyzer);

            PopulationState<ExpressionTree>? lastState = null;
            bool stoppedByController = false;
            // Tuning runs record the current best expression at every multiple of the snapshot spacing, at the generation
            // boundary that reaches it; like feedback, this is not GP time and reads validation and test rows only for the record.
            bool recordSnapshots = _phase == "tuning" && _snapshotEvaluations > 0;
            var snapshots = new List<ResultDTO.RunSnapshot>();
            long nextSnapshot = _snapshotEvaluations;
            void RecordSnapshot(int generation, PopulationState<ExpressionTree> state)
            {
                if (!recordSnapshots || counter.CurrentCount < nextSnapshot) return;
                snapshots.Add(Snapshot(instance, state, counter.CurrentCount, generation));
                nextSnapshot = (counter.CurrentCount / _snapshotEvaluations + 1L) * _snapshotEvaluations;
            }
            if (decideAsync is not null)
            {
                int generation = -1;
                long nextObservationEvaluations = firstObservationEvaluations!.Value;
                var feedback = new RunFeedbackBuilder(instance.PreparedData, parameter, channels | FeedbackChannels.Progress);
                foreach (var state in run.Stream())
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    lastState = state;
                    generation++;
                    // Feedback computation and the controller's inference are not GP time; they count as trial overhead.
                    stopwatch.Stop();
                    RecordSnapshot(generation, state);
                    feedback.Observe(generation, state);
                    // Checkpoints follow the controller's schedule in evaluations, at generation boundaries and never at
                    // initialization; the evaluation budget remains the hard limit.
                    if (generation < 1 || counter.CurrentCount < nextObservationEvaluations || counter.CurrentCount >= evaluationsLimit)
                    {
                        stopwatch.Start();
                        continue;
                    }
                    var progress = feedback.Checkpoint(generation, counter.CurrentCount, evaluationsLimit - counter.CurrentCount, state);
                    var decision = await decideAsync(progress, cancellationToken);
                    stopwatch.Start();
                    if (decision.Action == RunAction.Stop)
                    {
                        stoppedByController = true;
                        break;
                    }
                    if (decision.Action != RunAction.Continue || decision.NextObservationInEvaluations is not > 0)
                        throw new InvalidOperationException("Continue requires a positive NextObservationInEvaluations.");
                    nextObservationEvaluations = (long)counter.CurrentCount + decision.NextObservationInEvaluations.Value;
                }
            }
            else if (recordSnapshots)
            {
                // Streaming yields the same states as CompleteAsync (checked), so recording does not change the run.
                int generation = -1;
                foreach (var state in run.Stream())
                {
                    lastState = state;
                    generation++;
                    stopwatch.Stop();
                    RecordSnapshot(generation, state);
                    stopwatch.Start();
                }
            }
            else
            {
                lastState = await run.CompleteAsync();
            }
            if (lastState is null) throw new InvalidOperationException("No population state was produced during the run.");

            stopwatch.Stop();
            var curve = run.GetResult(analyzer);
            var objectives = lastState.Population.Select(individual => individual.ObjectiveVector[0]).ToArray();
            var best = lastState.Population.Where(individual => double.IsFinite(individual.ObjectiveVector[0]))
                .OrderBy(individual => individual.ObjectiveVector[0]).FirstOrDefault()
                ?? throw new InvalidOperationException("GP produced no finite training objective.");
            var model = new SymbolicRegressor(best.Candidate, "prediction")
                .FitLinearScaling(instance.PreparedData.Training);
            double validationFitness = instance.PreparedData.ValidationFitness(model);
            trialClock.Stop();
            var result = new ResultDTO {
                TerminationReason = stoppedByController ? "controller_stop" : "evaluation_limit",
                Configuration = parameter,
                ConfigurationIndex = configurationId,
                GPSeed = seed,
                BestFitness = validationFitness,
                FitnessMetric = "validation_nmse",
                TrainingBestFitness = best.ObjectiveVector[0],
                SelectedModel = model,
                LinearScalingSlope = model.Parameters.Slope,
                LinearScalingIntercept = model.Parameters.Intercept,
                SelectedExpression = best.Candidate.ToInfixString(),
                SelectedExpressionLength = best.Candidate.Length,
                MeanPopulationFitness = objectives.Average(),
                WorstPopulationFitness = objectives.Max(),
                ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                ConstantFittingMs = fittingDuration.CurrentDuration.TotalMilliseconds,
                ConstantFittingCandidates = fittedCandidates.CurrentCount,
                ConstantFittingTrialPoints = fittingStatistics.TrialPoints,
                ConstantFitting = ConstantFitting,
                TrialWallTimeMs = trialClock.Elapsed.TotalMilliseconds,
                TrialCpuTimeMs = (Process.GetCurrentProcess().TotalProcessorTime - cpuAtStart).TotalMilliseconds,
                EvaluationsUsed = counter.CurrentCount,
                EvaluationsLimit = evaluationsLimit,
                // Entry zero is initialization, not an offspring generation.
                GenerationsCompleted = curve.Count - 1,
                PopulationSize = parameter.PopulationSize,
                BestIndividualsSnapshot = lastState.Population.Select(individual => new ResultDTO.IndividualSnapshot {
                    ObjectiveValue = individual.ObjectiveVector[0],
                    Depth = individual.Candidate.Depth,
                    Length = individual.Candidate.Length,
                    Complexity = individual.Candidate.Complexity,
                    InfixRepresentation = individual.Candidate.ToInfixString()
                }).ToList(),
                QualityCurve = curve.Select((generation, index) => new ResultDTO.GenerationQualitySnapshot {
                    GenerationNumber = index,
                    BestQuality = generation.Best.ObjectiveVector[0],
                    MedianQuality = generation.Median.ObjectiveVector[0],
                    WorstQuality = generation.Worst.ObjectiveVector[0]
                }).ToList()
            };
            if (recordSnapshots)
            {
                // The final state closes the record; it replaces a periodic snapshot taken at the same count.
                var final = new ResultDTO.RunSnapshot {
                    Evaluations = counter.CurrentCount, Generation = curve.Count - 1, ValidationNmse = validationFitness,
                    TestNmse = instance.PreparedData.TestFitness(model), NoisyTestNmse = instance.PreparedData.NoisyTestFitness(model),
                    Length = best.Candidate.Length,
                    Expression = result.SelectedExpression!, Final = true
                };
                if (snapshots.Count > 0 && snapshots[^1].Evaluations == final.Evaluations) snapshots.RemoveAt(snapshots.Count - 1);
                snapshots.Add(final);
                result.Snapshots = snapshots;
            }
            Log.Write("gp_run", new {
                Phase = _phase, result.ConfigurationIndex, result.GPSeed, result.Configuration, result.EvaluationsLimit,
                result.EvaluationsUsed, result.GenerationsCompleted, result.TerminationReason,
                ValidationNmse = result.BestFitness, TrainingNmse = result.TrainingBestFitness,
                GpMs = result.ExecutionTimeMs, result.ConstantFittingMs, result.ConstantFittingCandidates, result.ConstantFittingTrialPoints, result.ConstantFitting, result.TrialWallTimeMs
            });
            // One line per GP run, so long jobs show progress.
            Console.WriteLine($"[{DateTime.UtcNow:HH:mm:ss}Z]   {_phase} run {configurationId + 1}: {result.GenerationsCompleted} generations, " +
                $"{result.EvaluationsUsed} evaluations, validation NMSE {result.BestFitness:G4}, GP {result.ExecutionTimeMs / 1000:F1} s " +
                $"(constant fitting {result.ConstantFittingMs / 1000:F1} s)");
            return result;
        }

        private static (GeneticAlgorithm<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>>, SymbolicRegressionProblem) PrepareAlgorithm(
            RegressionDescriptor feynmanInstance,
            AlgorithmParameter parameter)
        {
            Validator.ValidateObject(parameter, new ValidationContext(parameter), validateAllProperties: true);
            var regressionData = feynmanInstance.PreparedData.Training;

            var searchSpace = new ExpressionTreeSearchSpace(
                maximumLength: parameter.MaxTreeLength,
                maximumDepth: parameter.MaxTreeDepth,
                operations: Symbols.DefaultOperations,
                variables: feynmanInstance.AllowedInputVariables
            );

            var problem = new SymbolicRegressionProblem(
                regressionData,
                Metrics.NMSE,
                searchSpace,
                useLinearScaling: true
                );

            var mutator = CreateMutator(parameter.MutatorType);
            var algorithm = GeneticAlgorithm.Create(
                new RampedHalfAndHalfTreeCreator(),
                new SubtreeCrossover(),
                mutator,
                selector: TournamentSelector.For(problem, tournamentSize: 5),
                refiner: ConstantFitting.Iterations == 0 ? null : new ScaledConstantFittingRefiner {
                    MaximumIterations = ConstantFitting.Iterations,
                    FittingData = ConstantFitting.Rows > 0 ? feynmanInstance.PreparedData.FittingSample(ConstantFitting.Rows) : null
                },
                populationSize: parameter.PopulationSize,
                maximumGenerations: int.MaxValue,  // Let evaluationLimit be the stopping criterion
                mutationRate: parameter.MutationRate
                );
            return (algorithm, problem);
        }

        // The population's best training expression, linearly scaled on training rows as at the end of a run.
        private static ResultDTO.RunSnapshot Snapshot(RegressionDescriptor instance, PopulationState<ExpressionTree> state, int evaluations, int generation)
        {
            var best = state.Population.Where(individual => double.IsFinite(individual.ObjectiveVector[0]))
                .OrderBy(individual => individual.ObjectiveVector[0]).FirstOrDefault();
            if (best is null)
                return new ResultDTO.RunSnapshot { Evaluations = evaluations, Generation = generation, ValidationNmse = double.PositiveInfinity,
                    TestNmse = double.PositiveInfinity, Length = 0, Expression = "" };
            var model = new SymbolicRegressor(best.Candidate, "prediction").FitLinearScaling(instance.PreparedData.Training);
            return new ResultDTO.RunSnapshot { Evaluations = evaluations, Generation = generation,
                ValidationNmse = instance.PreparedData.ValidationFitness(model), TestNmse = instance.PreparedData.TestFitness(model),
                NoisyTestNmse = instance.PreparedData.NoisyTestFitness(model),
                Length = best.Candidate.Length, Expression = best.Candidate.ToInfixString() };
        }

        private static IMutator<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>> CreateMutator(MutatorType mutatorType)
        {
            return mutatorType switch {
                MutatorType.NodeReplacement => new NodeReplacementMutator(),
                MutatorType.Subtree => new SubtreeMutator(),
                MutatorType.LocalPerturbation => new LocalPerturbationMutator(),
                MutatorType.Combined => ChooseOneMutator.Create(
                    new NodeReplacementMutator(),
                    new SubtreeMutator(),
                    new LocalPerturbationMutator()),
                _ => throw new ArgumentException($"Unknown mutator type: {mutatorType}")
            };
        }

    }
}
