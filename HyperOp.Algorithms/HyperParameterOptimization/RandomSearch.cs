using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Random;
using HyperOp.Algorithms.Feynman;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Reflection;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class RandomSearch : HyperParameterOptimizationAlgorithm
    {
        public override async Task<List<ResultDTO>> Execute(
            FeynmanDescriptor feynmanInstance,
            int seed,
            int numConfigurations,
            int evaluationsPerConfiguration)
        {
            var results = new List<ResultDTO>();
            var random = new Random(seed);

            // Extract parameter ranges from AlgorithmParameter attributes
            var popRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.PopulationSize))
                ?.GetCustomAttribute<RangeAttribute>();
            var mutationRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MutationRate))
                ?.GetCustomAttribute<RangeAttribute>();
            var maxTreeDepthRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MaxTreeDepth))
                ?.GetCustomAttribute<RangeAttribute>();
            var maxTreeLengthRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MaxTreeLength))
                ?.GetCustomAttribute<RangeAttribute>();

            var mutatorTypes = Enum.GetValues(typeof(MutatorType)).Cast<MutatorType>().ToArray();

            // Evaluate exactly numConfigurations hyperparameter configurations
            for (int configIndex = 0; configIndex < numConfigurations; configIndex++)
            {
                // Sample a random hyperparameter configuration
                var parameter = new AlgorithmParameter {
                    PopulationSize = random.Next((int)popRange!.Minimum!, (int)popRange.Maximum!),
                    MutationRate = random.NextDouble() * ((double)mutationRange!.Maximum! - (double)mutationRange.Minimum!) + (double)mutationRange.Minimum!,
                    MaxTreeDepth = random.Next((int)maxTreeDepthRange!.Minimum!, (int)maxTreeDepthRange.Maximum!),
                    MaxTreeLength = random.Next((int)maxTreeLengthRange!.Minimum!, (int)maxTreeLengthRange.Maximum!),
                    MutatorType = mutatorTypes[random.Next(mutatorTypes.Length)]
                };

                // Prepare algorithm with this configuration
                var (algorithm, problem) = PrepareAlgorithm(feynmanInstance, parameter, evaluationsPerConfiguration);

                // Execute with timing
                var qualityAnalyzer = Analyzer.BestMedianWorst(algorithm);
                var run = algorithm.CreateRun(problem, RandomNumberGenerator.Create(seed + configIndex)).WithAnalyzer(qualityAnalyzer);

                var stopwatch = Stopwatch.StartNew();
                var finalState = await run.CompleteAsync();
                stopwatch.Stop();

                var qualityCurve = run.GetResult(qualityAnalyzer);
                int generationCount = qualityCurve.Count;
                var worstOfFirstGeneration = qualityCurve.First().Worst.ObjectiveVector[0];

                // Extract metrics from population using API's ObjectiveVector
                var objectives = finalState.Population.Select(ind => ind.ObjectiveVector[0]).ToList();
                var bestFitness = objectives.Max();
                var meanFitness = objectives.Average();
                var worstFitness = objectives.Min();

                // Create individual snapshots for the best individuals
                var snapshots = finalState.Population
                    .Select(ind => new ResultDTO.IndividualSnapshot {
                        ObjectiveValue = ind.ObjectiveVector[0],
                        Depth = ind.Candidate.Depth,
                        Length = ind.Candidate.Length,
                        Complexity = ind.Candidate.Complexity,
                        InfixRepresentation = ind.Candidate.ToInfixString()
                    })
                    .ToList();

                // Create per-generation quality curve snapshots
                var qualitySnapshots = qualityCurve
                    .Select((gen, genIndex) => new ResultDTO.GenerationQualitySnapshot
                    {
                        GenerationNumber = genIndex,
                        BestQuality = gen.Best.ObjectiveVector[0],
                        MedianQuality = gen.Median.ObjectiveVector[0],
                        WorstQuality = gen.Worst.ObjectiveVector[0]
                    })
                    .ToList();

                // Create result record
                var result = new ResultDTO {
                    Configuration = parameter,
                    ConfigurationIndex = configIndex,
                    BestFitness = bestFitness,
                    MeanPopulationFitness = meanFitness,
                    WorstPopulationFitness = worstFitness,
                    ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                    EvaluationsUsed = generationCount * parameter.PopulationSize,  // Use an approximation
                    EvaluationsLimit = evaluationsPerConfiguration,
                    GenerationsCompleted = generationCount,
                    PopulationSize = parameter.PopulationSize,
                    BestIndividualsSnapshot = snapshots,
                    QualityCurve = qualitySnapshots
                };

                results.Add(result);
            }

            return results;
        }
    }
}
