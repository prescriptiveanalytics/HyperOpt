using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Random;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.HyperParameterOptimization.Util;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Reflection;
using System.Reflection.Metadata;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class HillClimbingSearch : HyperParameterOptimizationAlgorithm
    {
        // Creates a copy of the current configuration while changing
        // exactly one parameter.
        private AlgorithmParameter CreateConfiguration(
            AlgorithmParameter current,
            int? populationSize = null,
            double? mutationRate = null,
            int? maxTreeDepth = null,
            int? maxTreeLength = null,
            MutatorType? mutatorType = null)
        {
            return new AlgorithmParameter
            {
                PopulationSize = populationSize ?? current.PopulationSize,
                MutationRate = mutationRate ?? current.MutationRate,
                MaxTreeDepth = maxTreeDepth ?? current.MaxTreeDepth,
                MaxTreeLength = maxTreeLength ?? current.MaxTreeLength,
                MutatorType = mutatorType ?? current.MutatorType,
                Evaluations = current.Evaluations
            };
        }

        private AlgorithmParameter CreateInitialConfiguration(
        (
            List<int> PopulationSizes,
            List<double> MutationRates,
            List<int> MaxTreeDepths,
            List<int> MaxTreeLengths,
            MutatorType[] MutatorTypes
        ) searchSpace,
        int evaluations)
        {
            return new AlgorithmParameter
            {
                PopulationSize = this.GetMiddleValue(searchSpace.PopulationSizes),
                MutationRate = this.GetMiddleValue(searchSpace.MutationRates),
                MaxTreeDepth = this.GetMiddleValue(searchSpace.MaxTreeDepths),
                MaxTreeLength = this.GetMiddleValue(searchSpace.MaxTreeLengths),
                MutatorType = searchSpace.MutatorTypes[0],
                Evaluations = evaluations
            };
        }

        private (List<int> PopulationSizes,List<double> MutationRates,List<int> MaxTreeDepths,List<int> MaxTreeLengths,MutatorType[] MutatorTypes) GenerateSearchSpace(
        int populationSize = 10,
        int mutationRate = 5,
        int maxTreeDepth = 5,
        int maxTreeLength = 5)
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

            var populationValues = this.Discretize(popRange!, populationSize)
                .Select(x => (int)Math.Round(x))
                .Distinct()
                .ToList();

            var mutationValues = this.Discretize(mutationRange!, mutationRate)
                .ToList();

            var maxTreeDepthValues = this.Discretize(maxTreeDepthRange!, maxTreeDepth)
                .Select(x => (int)Math.Round(x))
                .Distinct()
                .ToList();

            var maxTreeLengthValues = this.Discretize(maxTreeLengthRange!, maxTreeLength)
                .Select(x => (int)Math.Round(x))
                .Distinct()
                .ToList();

            var mutatorTypes = Enum.GetValues<MutatorType>();

            return (
                populationValues,
                mutationValues,
                maxTreeDepthValues,
                maxTreeLengthValues,
                mutatorTypes
            );
        }

        // Generates all configurations that differ from the current
        // configuration in exactly one hyperparameter.
        private List<AlgorithmParameter> GenerateNeighbors(
            AlgorithmParameter current,
            (
                List<int> PopulationSizes,
                List<double> MutationRates,
                List<int> MaxTreeDepths,
                List<int> MaxTreeLengths,
                MutatorType[] MutatorTypes
            ) searchSpace)
        {
            var neighbors = new List<AlgorithmParameter>();

            // PopulationSize neighbors
            foreach (var value in searchSpace.PopulationSizes)
            {
                if (value == current.PopulationSize)
                    continue;

                neighbors.Add(CreateConfiguration(
                    current,
                    populationSize: value));
            }

            // MutationRate neighbors
            foreach (var value in searchSpace.MutationRates)
            {
                if (value == current.MutationRate)
                    continue;

                neighbors.Add(CreateConfiguration(
                    current,
                    mutationRate: value));
            }

            // MaxTreeDepth neighbors
            foreach (var value in searchSpace.MaxTreeDepths)
            {
                if (value == current.MaxTreeDepth)
                    continue;

                neighbors.Add(CreateConfiguration(
                    current,
                    maxTreeDepth: value));
            }

            // MaxTreeLength neighbors
            foreach (var value in searchSpace.MaxTreeLengths)
            {
                if (value == current.MaxTreeLength)
                    continue;

                neighbors.Add(CreateConfiguration(
                    current,
                    maxTreeLength: value));
            }

            // MutatorType neighbors
            foreach (var value in searchSpace.MutatorTypes)
            {
                if (value == current.MutatorType)
                    continue;

                neighbors.Add(CreateConfiguration(
                    current,
                    mutatorType: value));
            }

            return neighbors;
        }

        // Determines whether two configurations represent the same
        // hyperparameter configuration.
        private bool SameConfiguration(
            AlgorithmParameter a,
            AlgorithmParameter b)
        {
            return
                a.PopulationSize == b.PopulationSize &&
                a.MutationRate == b.MutationRate &&
                a.MaxTreeDepth == b.MaxTreeDepth &&
                a.MaxTreeLength == b.MaxTreeLength &&
                a.MutatorType == b.MutatorType;
        }

        public override async Task<List<ResultDTO>> Execute(FeynmanDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            var results = new List<ResultDTO>();

            var searchSpace = this.GenerateSearchSpace();

            var currentConfiguration = CreateInitialConfiguration(
                searchSpace,
                evaluationsPerConfiguration);

            var evaluatedConfigurations = new List<AlgorithmParameter>();

            int configIndex = 0;

            Console.WriteLine($"Running symbolic regression with parameters: {currentConfiguration}");
            var (alg, problem) = PrepareAlgorithm(feynmanInstance!, currentConfiguration, evaluationsPerConfiguration);
            var algorithm = alg.WithMaxEvaluatedCandidates(alg.Evaluator, evaluationsPerConfiguration);

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
                Configuration = currentConfiguration,
                ConfigurationIndex = -1, // ??? count somehow everytime the same configuration is used?
                BestFitness = bestFitness,
                MeanPopulationFitness = meanFitness,
                WorstPopulationFitness = worstFitness,
                ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                EvaluationsUsed = generationCount * currentConfiguration.PopulationSize,  // Use an approximation
                EvaluationsLimit = evaluationsPerConfiguration,
                GenerationsCompleted = generationCount,
                PopulationSize = currentConfiguration.PopulationSize,
                BestIndividualsSnapshot = snapshots,
                QualityCurve = qualitySnapshots
            };
            var currentBestResult = result;

            results.Add(result);

            evaluatedConfigurations.Add(currentConfiguration);
            configIndex++;

            // Start hill climbing search
            while(results.Count < numConfigurations)
            {
                var neighbors = GenerateNeighbors(currentConfiguration, searchSpace);
                // remove duplicates
                neighbors = neighbors.Where(n => !evaluatedConfigurations.Any(e => SameConfiguration(n, e))).ToList();
                if (neighbors.Count == 0) break;

                int remainingConfigurations =numConfigurations - results.Count;

                neighbors = neighbors.Take(remainingConfigurations).ToList();

                var neighborResults = new List<ResultDTO>();
                // Evaluate each neighbor configuration
                foreach(var neighbor in neighbors)
                {
                    Console.WriteLine($"Running symbolic regression with parameters: {neighbor}");
                    var (algNeighbor, problemNeighbor) = PrepareAlgorithm(feynmanInstance!, neighbor, evaluationsPerConfiguration);
                    var algorithmNeighbor = algNeighbor.WithMaxEvaluatedCandidates(algNeighbor.Evaluator, evaluationsPerConfiguration);
                    var qualityAnalyzerNeighbor = Analyzer.BestMedianWorst(algorithmNeighbor);
                    var runNeighbor = algorithmNeighbor.CreateRun(problemNeighbor, RandomNumberGenerator.Create(seed + configIndex)).WithAnalyzer(qualityAnalyzerNeighbor);
                    stopwatch.Restart();
                    var finalStateNeighbor = await runNeighbor.CompleteAsync();
                    stopwatch.Stop();

                    var qualityCurveNeighbor = runNeighbor.GetResult(qualityAnalyzerNeighbor);
                    var (generationCountNeighbor, bestFitnessNeighbor, meanFitnessNeighbor, worstFitnessNeighbor, snapshotsNeighbor, qualitySnapshotsNeighbor) = this.EvaluateFinalState(finalStateNeighbor, qualityCurveNeighbor);
                    // Create result record
                    var resultNeighbor = new ResultDTO
                    {
                        Configuration = neighbor,
                        ConfigurationIndex = -1,
                        BestFitness = bestFitnessNeighbor,
                        MeanPopulationFitness = meanFitnessNeighbor,
                        WorstPopulationFitness = worstFitnessNeighbor,
                        ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                        EvaluationsUsed = generationCountNeighbor * neighbor.PopulationSize,
                        EvaluationsLimit = evaluationsPerConfiguration,
                        GenerationsCompleted = generationCountNeighbor,
                        PopulationSize = neighbor.PopulationSize,
                        BestIndividualsSnapshot = snapshotsNeighbor,
                        QualityCurve = qualitySnapshotsNeighbor
                    };
                    neighborResults.Add(resultNeighbor);
                    evaluatedConfigurations.Add(neighbor);
                    configIndex++;
                }

                if (neighborResults.Count == 0) break;
                var bestNeighborResult = neighborResults.OrderByDescending(r => r.BestFitness).First();
                // End if no improvement
                if ((bestNeighborResult.BestFitness <= currentBestResult.BestFitness)) break;
                currentConfiguration = bestNeighborResult.Configuration;
                currentBestResult = bestNeighborResult;
            }

            return results;
        }
    }
}
