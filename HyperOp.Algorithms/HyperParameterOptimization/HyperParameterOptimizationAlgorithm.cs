using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Operators;
using HEAL.HeuristicLib.Operators.Evaluators;
using HEAL.HeuristicLib.Problems;
using HEAL.HeuristicLib.Problems.MachineLearning;
using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;
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

        public virtual ResultDTO NominateConfiguration(IReadOnlyList<ResultDTO> tuningResults) =>
            tuningResults.Where(result => double.IsFinite(result.BestFitness))
                .OrderBy(result => result.BestFitness).FirstOrDefault()
            ?? throw new InvalidOperationException("No finite validation result is available for nomination.");

        public async Task<List<ResultDTO>> AssessConfiguration(RegressionDescriptor instance,
            AlgorithmParameter configuration, IReadOnlyList<ResultDTO> tuningResults,
            int seed, int repetitions, int evaluationsPerRun)
        {
            configuration = configuration with { };
            ValidateBudget(repetitions, evaluationsPerRun);
            var usedSeeds = tuningResults.Select(result => result.GPSeed
                ?? throw new InvalidOperationException("Tuning seeds must be recorded before assessment."))
                .ToHashSet();
            var random = new Random(seed ^ int.MinValue);
            var assessmentResults = new List<ResultDTO>();
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
            return assessmentResults;
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
            parameter = parameter with { };
            var (algorithm, problem) = PrepareAlgorithm(instance, parameter);
            // Only candidates reaching the real evaluator are counted. The limiter assigns
            // worst fitness to unscored candidates in the last, possibly partial, generation.
            var countedEvaluator = algorithm.Evaluator.CountEvaluatedCandidates(out var counter);
            algorithm = algorithm with {
                Evaluator = countedEvaluator.LimitEvaluations(evaluationsLimit, enforceLimitWithinBatch: true),
                Terminator = AfterOperatorCountTerminator.For(problem, counter, evaluationsLimit)
            };
            var analyzer = Analyzer.BestMedianWorst(algorithm);
            var run = algorithm.CreateRun(problem, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(seed))
                .WithAnalyzer(analyzer);
            var stopwatch = Stopwatch.StartNew();
            var finalState = await run.CompleteAsync();
            stopwatch.Stop();
            var (generations, best, mean, worst, snapshots, curve) = this.EvaluateFinalState(finalState, run.GetResult(analyzer));
            var result = new ResultDTO {
                Configuration = parameter, ConfigurationIndex = configurationId, GPSeed = seed,
                BestFitness = best, MeanPopulationFitness = mean, WorstPopulationFitness = worst,
                ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                EvaluationsUsed = counter.CurrentCount, EvaluationsLimit = evaluationsLimit,
                GenerationsCompleted = generations, PopulationSize = parameter.PopulationSize,
                BestIndividualsSnapshot = snapshots, QualityCurve = curve
            };
            CompleteResult(instance, result, finalState);
            return result;
        }

        protected (GeneticAlgorithm<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>>, SymbolicRegressionProblem) PrepareAlgorithm(
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

        private IMutator<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>> CreateMutator(MutatorType mutatorType)
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

        protected static void CompleteResult(RegressionDescriptor descriptor, ResultDTO result,
            PopulationState<ExpressionTree> finalState)
        {
            var best = finalState.Population.Where(individual => double.IsFinite(individual.ObjectiveVector[0]))
                .OrderBy(individual => individual.ObjectiveVector[0]).FirstOrDefault()
                ?? throw new InvalidOperationException("GP produced no finite training objective.");
            result.TrainingBestFitness = best.ObjectiveVector[0];
            var model = new SymbolicRegressor(best.Candidate, "prediction")
                .FitLinearScaling(descriptor.PreparedData.Training);
            result.SelectedModel = model;
            result.LinearScalingSlope = model.Parameters.Slope;
            result.LinearScalingIntercept = model.Parameters.Intercept;
            result.SelectedExpression = best.Candidate.ToInfixString();
            result.BestFitness = descriptor.PreparedData.ValidationFitness(model);
            result.FitnessMetric = "validation_nmse";
        }
    }
}
