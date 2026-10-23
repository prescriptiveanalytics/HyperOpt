using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
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
                MutatorType = mutatorType ?? current.MutatorType
            };
        }

        private AlgorithmParameter CreateInitialConfiguration(
        (
            List<int> PopulationSizes,
            List<double> MutationRates,
            List<int> MaxTreeDepths,
            List<int> MaxTreeLengths,
            MutatorType[] MutatorTypes
        ) searchSpace)
        {
            return new AlgorithmParameter
            {
                PopulationSize = this.GetMiddleValue(searchSpace.PopulationSizes),
                MutationRate = this.GetMiddleValue(searchSpace.MutationRates),
                MaxTreeDepth = this.GetMiddleValue(searchSpace.MaxTreeDepths),
                MaxTreeLength = this.GetMiddleValue(searchSpace.MaxTreeLengths),
                MutatorType = searchSpace.MutatorTypes[0]
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

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            var searchSpace = GenerateSearchSpace();
            var currentConfiguration = CreateInitialConfiguration(searchSpace);
            var evaluatedConfigurations = new List<AlgorithmParameter>();
            var currentBestResult = await EvaluateConfiguration(feynmanInstance, currentConfiguration,
                0, seed, evaluationsPerConfiguration);
            results.Add(currentBestResult);
            evaluatedConfigurations.Add(currentConfiguration);

            while (results.Count < numConfigurations)
            {
                var neighbors = GenerateNeighbors(currentConfiguration, searchSpace)
                    .Where(n => !evaluatedConfigurations.Any(e => SameConfiguration(n, e)))
                    .Take(numConfigurations - results.Count).ToList();
                var neighborResults = new List<ResultDTO>();
                foreach (var neighbor in neighbors)
                {
                    int trialIndex = results.Count;
                    var result = await EvaluateConfiguration(feynmanInstance, neighbor,
                        trialIndex, seed + trialIndex, evaluationsPerConfiguration);
                    // Every executed neighbor consumes a trial, including unsuccessful ones.
                    results.Add(result);
                    neighborResults.Add(result);
                    evaluatedConfigurations.Add(neighbor);
                }
                var bestNeighbor = neighborResults.OrderBy(r => r.BestFitness).FirstOrDefault();
                if (bestNeighbor is null || bestNeighbor.BestFitness >= currentBestResult.BestFitness)
                    break;
                currentConfiguration = bestNeighbor.Configuration;
                currentBestResult = bestNeighbor;
            }
            return results;
        }
    }
}
