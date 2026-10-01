using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Random;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.HyperParameterOptimization.Util;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Reflection;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class GridSearch : HyperParameterOptimizationAlgorithm
    {
        // Generates a grid of hyperparameter combinations based on how many discretization steps are specified
        // Grid size is product of parameters Default = 5.000 Configurations (gets really large fast!)
        private List<AlgorithmParameter> GenerateGrid(int evaluations, int populationSize = 10, int mutationRate = 5, int maxTreeDepth = 5, int maxTreeLength = 5)
        {
            var popRange = typeof(AlgorithmParameter)
                .GetProperty(nameof(AlgorithmParameter.PopulationSize))
                ?.GetCustomAttribute<RangeAttribute>();

            var mutationRange = typeof(AlgorithmParameter)
                .GetProperty(nameof(AlgorithmParameter.MutationRate))
                ?.GetCustomAttribute<RangeAttribute>();

            var maxTreeDepthRange = typeof(AlgorithmParameter)
                .GetProperty(nameof(AlgorithmParameter.MaxTreeDepth))
                ?.GetCustomAttribute<RangeAttribute>();

            var maxTreeLengthRange = typeof(AlgorithmParameter)
                .GetProperty(nameof(AlgorithmParameter.MaxTreeLength))
                ?.GetCustomAttribute<RangeAttribute>();

            var mutatorTypes = Enum.GetValues<MutatorType>();

            var populationValues = this.Discretize(popRange!, populationSize)
                .Select(x => (int)Math.Round(x))
                .Distinct(); // Distinct necesssary due to potential rounding issues when discretizing

            var mutationValues = this.Discretize(mutationRange!, mutationRate);

            var maxTreeDepthValues = this.Discretize(maxTreeDepthRange!, maxTreeDepth)
                .Select(x => (int)Math.Round(x))
                .Distinct();

            var maxTreeLengthValues = this.Discretize(maxTreeLengthRange!, maxTreeLength)
                .Select(x => (int)Math.Round(x))
                .Distinct();

            return (
                from popSize in populationValues
                from mutRate in mutationValues
                from maxDepth in maxTreeDepthValues
                from maxLenght in maxTreeLengthValues
                from mutatorType in mutatorTypes
                select new AlgorithmParameter
                {
                    PopulationSize = popSize,
                    MutationRate = mutRate,
                    MaxTreeDepth = maxDepth,
                    MaxTreeLength = maxLenght,
                    MutatorType = mutatorType,
                    Evaluations = evaluations
                }
            ).ToList();
        }
        

        public override async Task<List<ResultDTO>> Execute(FeynmanDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            var results = new List<ResultDTO>();

            // TODO: How to deal with num Configurations?
            var configurations = GenerateGrid(evaluationsPerConfiguration);
            int configIndex = 0;
            // now try all
            foreach (var config in configurations)
            {
                Console.WriteLine($"Running symbolic regression with the following parameters:{config}");
                var (alg, problem) = PrepareAlgorithm(feynmanInstance, config, evaluationsPerConfiguration);
                var algorithm = alg.WithMaxEvaluatedCandidates(alg.Evaluator, evaluationsPerConfiguration);
                // Execute with timing
                var qualityAnalyzer = Analyzer.BestMedianWorst(algorithm);
                var run = algorithm.CreateRun(problem, RandomNumberGenerator.Create(seed + configIndex)).WithAnalyzer(qualityAnalyzer);

                var stopwatch = Stopwatch.StartNew();
                var finalState = await run.CompleteAsync();
                stopwatch.Stop();

                var qualityCurve = run.GetResult(qualityAnalyzer);
                var (generationCount, bestFitness, meanFitness, worstFitness, snapshots, qualitySnapshots) = this.EvaluateFinalState(finalState, qualityCurve);
                
                // Create result record
                var result = new ResultDTO
                {
                    Configuration = config,
                    ConfigurationIndex = configIndex,
                    BestFitness = bestFitness,
                    MeanPopulationFitness = meanFitness,
                    WorstPopulationFitness = worstFitness,
                    ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                    EvaluationsUsed = generationCount * config.PopulationSize,  // Use an approximation
                    EvaluationsLimit = evaluationsPerConfiguration,
                    GenerationsCompleted = generationCount,
                    PopulationSize = config.PopulationSize,
                    BestIndividualsSnapshot = snapshots,
                    QualityCurve = qualitySnapshots
                };
                results.Add(result);
                configIndex++;
            }

            return results;
        }
    }
}
