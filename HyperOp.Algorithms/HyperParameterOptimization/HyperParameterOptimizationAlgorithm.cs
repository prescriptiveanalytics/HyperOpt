using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Execution;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Operators;
using HEAL.HeuristicLib.Operators.Evaluators;
using HEAL.HeuristicLib.Operators.Interceptors;
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

        // Settings that distinguish otherwise identical jobs of this method (feedback condition, checkpoint interval).
        public virtual string? Variant => null;
        private string _phase = "tuning";

        // Model requests made by the optimizer itself; empty for methods without an LLM.
        public virtual IReadOnlyList<LlmCallRecord> LlmCalls => [];

        public virtual ResultDTO NominateConfiguration(IReadOnlyList<ResultDTO> tuningResults) => BestValidationTrial(tuningResults);

        // Per-run allowance for fresh-seed assessment; methods with fixed per-trial work use E.
        public virtual int AssessmentEvaluations(ResultDTO recommendation, int evaluationsPerConfiguration) => evaluationsPerConfiguration;

        public static ResultDTO BestValidationTrial(IReadOnlyList<ResultDTO> tuningResults) =>
            tuningResults.Where(result => double.IsFinite(result.BestFitness))
                .OrderBy(result => result.BestFitness).FirstOrDefault()
            ?? throw new InvalidOperationException("No finite validation result is available for nomination.");

        public async Task<List<ResultDTO>> AssessConfiguration(RegressionDescriptor instance,
            AlgorithmParameter configuration, IReadOnlyList<ResultDTO> tuningResults,
            int seed, int repetitions, int evaluationsPerRun)
        {
            configuration = configuration with { };
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(repetitions);
            // An agent-stopped nominee may have used fewer than 100 evaluations; its initial population must still fit.
            ArgumentOutOfRangeException.ThrowIfLessThan(evaluationsPerRun, configuration.PopulationSize);
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
                for (int repetition = 0; repetition < repetitions; repetition++)
                {
                    int gpSeed;
                    do { gpSeed = random.Next(int.MinValue, 0); } while (!usedSeeds.Add(gpSeed));
                    var result = await EvaluateConfiguration(instance, configuration, repetition, gpSeed, evaluationsPerRun);
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

        protected static void ValidateBudget(int numConfigurations, int evaluationsPerConfiguration)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numConfigurations);
            // Every legal population must fit without silently narrowing the search space.
            if (evaluationsPerConfiguration < 100)
                throw new ArgumentOutOfRangeException(nameof(evaluationsPerConfiguration),
                    "At least 100 candidate evaluations per trial are required for the full population-size domain.");
        }

        protected virtual async Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor instance,
           AlgorithmParameter parameter, int configurationId, int seed, int evaluationsLimit)
        {
            return await EvaluateConfiguration(instance, parameter, configurationId, seed, evaluationsLimit, decideAsync: null);
        }
        protected virtual async Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor instance,
            AlgorithmParameter parameter, int configurationId, int seed, int evaluationsLimit,
            Func<RunProgress, CancellationToken, Task<RunDecision>>? decideAsync,
            int? firstObservationGeneration = null, CancellationToken cancellationToken = default,
            FeedbackChannels channels = FeedbackChannels.Progress)
        {
            if (decideAsync is not null && firstObservationGeneration is not > 0)
                throw new ArgumentOutOfRangeException(nameof(firstObservationGeneration), "An agent-controlled run requires a positive first checkpoint.");
            parameter = parameter with { };
            var trialClock = Stopwatch.StartNew();
            var cpuAtStart = Process.GetCurrentProcess().TotalProcessorTime;
            var (algorithm, problem) = PrepareAlgorithm(instance, parameter);
            // Only candidates reaching the real evaluator are counted. The limiter assigns
            // worst fitness to unscored candidates in the last, possibly partial, generation.
            var countedEvaluator = algorithm.Evaluator.CountEvaluatedCandidates(out var counter);

            algorithm = algorithm with {
                Evaluator = countedEvaluator.LimitEvaluations(evaluationsLimit, enforceLimitWithinBatch: true),
                Terminator = AfterOperatorCountTerminator.For(problem, counter, evaluationsLimit),
            };

            var analyzer = Analyzer.BestMedianWorst(algorithm);
            
            var stopwatch = Stopwatch.StartNew();
            var run = algorithm.CreateRun(problem, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(seed))
                .WithAnalyzer(analyzer);

            PopulationState<ExpressionTree>? lastState = null;
            bool stoppedByController = false;
            if (decideAsync is not null)
            {
                int generation = -1;
                long nextObservationGeneration = firstObservationGeneration!.Value;
                var feedback = new RunFeedbackBuilder(instance.PreparedData, parameter, channels | FeedbackChannels.Progress);
                foreach (var state in run.Stream())
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    lastState = state;
                    generation++;
                    // Feedback computation and the controller's inference are not GP time; they count as trial overhead.
                    stopwatch.Stop();
                    feedback.Observe(generation, state);
                    // Checkpoints follow the controller's schedule; the evaluation budget remains the hard limit.
                    if (generation < nextObservationGeneration || counter.CurrentCount >= evaluationsLimit)
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
                    if (decision.Action != RunAction.Continue || decision.NextObservationInGenerations is not > 0)
                        throw new InvalidOperationException("Continue requires a positive NextObservationInGenerations.");
                    nextObservationGeneration = (long)generation + decision.NextObservationInGenerations.Value;
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
                MeanPopulationFitness = objectives.Average(),
                WorstPopulationFitness = objectives.Max(),
                ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
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
            Log.Write("gp_run", new {
                Phase = _phase, result.ConfigurationIndex, result.GPSeed, result.Configuration, result.EvaluationsLimit,
                result.EvaluationsUsed, result.GenerationsCompleted, result.TerminationReason,
                ValidationNmse = result.BestFitness, TrainingNmse = result.TrainingBestFitness,
                GpMs = result.ExecutionTimeMs, result.TrialWallTimeMs
            });
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
                populationSize: parameter.PopulationSize,
                maximumGenerations: int.MaxValue,  // Let evaluationLimit be the stopping criterion
                mutationRate: parameter.MutationRate
                );
            return (algorithm, problem);
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
