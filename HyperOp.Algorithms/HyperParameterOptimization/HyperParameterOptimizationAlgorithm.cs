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
using Parlot.Fluent;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    struct InterceptorFeedback
    {
        //Quality curve (best, median, worst) of the current run.
        //Population state of the current run.
    }

    public abstract class HyperParameterOptimizationAlgorithm
    {
        public abstract Task<List<ResultDTO>> Execute(
            RegressionDescriptor feynmanInstance,
            int seed,
            int numConfigurations,
            int evaluationsPerConfiguration
            );

        public virtual string RecommendationRule => "minimum_validation_nmse";

        public virtual ResultDTO NominateConfiguration(IReadOnlyList<ResultDTO> tuningResults) => BestValidationTrial(tuningResults);

        public static ResultDTO BestValidationTrial(IReadOnlyList<ResultDTO> tuningResults) =>
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

        //private record MyTerm(Func<bool> myfunc) : StatelessTerminator<ExpressionTree,ISearchState>// ISearchState -> PopulationState<ExpressionTree>
        //{
        //    public override bool IsTerminalState(ISearchState state) => myfunc();
        //}

        protected virtual async Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor instance,
           AlgorithmParameter parameter, int configurationId, int seed, int evaluationsLimit)
        {
            return await EvaluateConfiguration(instance, parameter, configurationId, seed, evaluationsLimit, interceptRun: null);
        }
        protected virtual async Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor instance,
            AlgorithmParameter parameter, int configurationId, int seed, int evaluationsLimit, Func<AlgorithmRun<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>, PopulationState<ExpressionTree>>, bool>? interceptRun = null)
        {
            parameter = parameter with { };
            var (algorithm, problem) = PrepareAlgorithm(instance, parameter);
            // Only candidates reaching the real evaluator are counted. The limiter assigns
            // worst fitness to unscored candidates in the last, possibly partial, generation.
            var countedEvaluator = algorithm.Evaluator.CountEvaluatedCandidates(out var counter);

            //TODO: CHeck if both are necessary (if time is left, otherwise ignore)
            algorithm = algorithm with {
                Evaluator = countedEvaluator.LimitEvaluations(evaluationsLimit, enforceLimitWithinBatch: true),
                Terminator = AfterOperatorCountTerminator.For(problem, counter, evaluationsLimit),
            };

            //TODO: Extend
            //var bmwanalyzer = Analyzer.BestMedianWorst(algorithm);
            //var divanalyzer = 

            var analyzer = Analyzer.BestMedianWorst(algorithm);
            
            var stopwatch = Stopwatch.StartNew();
            var run = algorithm.CreateRun(problem, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(seed))
                .WithAnalyzer(analyzer);

            PopulationState<ExpressionTree>? lastState = null;
            if(interceptRun is not null)
            {
                foreach (var res in run.Stream())
                {
                    lastState = res;
                    var stop = interceptRun(run);
                    
                    if (stop)
                    {
                        break;
                    }

                }
            } else
            {
                lastState = await run.CompleteAsync();
            }
            

            if(lastState == null) throw new InvalidOperationException("No population state was produced during the run.");  
            //var 2
            //var a1 = algorithm.WithTerminator(new MyTerm(()=> true));
            //var run2 =a1.Complete(problem, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(seed));

            stopwatch.Stop();
            var curve = run.GetResult(analyzer);
            var objectives = lastState.Population.Select(individual => individual.ObjectiveVector[0]).ToArray();
            var best = lastState.Population.Where(individual => double.IsFinite(individual.ObjectiveVector[0]))
                .OrderBy(individual => individual.ObjectiveVector[0]).FirstOrDefault()
                ?? throw new InvalidOperationException("GP produced no finite training objective.");
            var model = new SymbolicRegressor(best.Candidate, "prediction")
                .FitLinearScaling(instance.PreparedData.Training);
            return new ResultDTO {
                Configuration = parameter,
                ConfigurationIndex = configurationId,
                GPSeed = seed,
                BestFitness = instance.PreparedData.ValidationFitness(model),
                FitnessMetric = "validation_nmse",
                TrainingBestFitness = best.ObjectiveVector[0],
                SelectedModel = model,
                LinearScalingSlope = model.Parameters.Slope,
                LinearScalingIntercept = model.Parameters.Intercept,
                SelectedExpression = best.Candidate.ToInfixString(),
                MeanPopulationFitness = objectives.Average(),
                WorstPopulationFitness = objectives.Max(),
                ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
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
