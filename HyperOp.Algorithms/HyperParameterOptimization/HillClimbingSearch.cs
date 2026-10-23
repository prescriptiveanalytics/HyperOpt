using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class HillClimbingSearch : HyperParameterOptimizationAlgorithm
    {
        // Preserve coordinate order: truncated neighborhoods depend on it.
        private static IEnumerable<AlgorithmParameter> GenerateNeighbors(AlgorithmParameter current)
        {
            foreach (int value in ParameterSpace.PopulationSizes)
                if (value != current.PopulationSize) yield return current with { PopulationSize = value };
            foreach (double value in ParameterSpace.MutationRates)
                if (value != current.MutationRate) yield return current with { MutationRate = value };
            foreach (int value in ParameterSpace.MaxTreeDepths)
                if (value != current.MaxTreeDepth) yield return current with { MaxTreeDepth = value };
            foreach (int value in ParameterSpace.MaxTreeLengths)
                if (value != current.MaxTreeLength) yield return current with { MaxTreeLength = value };
            foreach (var value in ParameterSpace.MutatorTypes)
                if (value != current.MutatorType) yield return current with { MutatorType = value };
        }

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            var currentConfiguration = new AlgorithmParameter {
                PopulationSize = ParameterSpace.PopulationSizes[ParameterSpace.PopulationSizes.Length / 2],
                MutationRate = ParameterSpace.MutationRates[ParameterSpace.MutationRates.Length / 2],
                MaxTreeDepth = ParameterSpace.MaxTreeDepths[ParameterSpace.MaxTreeDepths.Length / 2],
                MaxTreeLength = ParameterSpace.MaxTreeLengths[ParameterSpace.MaxTreeLengths.Length / 2],
                MutatorType = ParameterSpace.MutatorTypes[0]
            };
            var evaluatedConfigurations = new HashSet<AlgorithmParameter> { currentConfiguration };
            var currentBestResult = await EvaluateConfiguration(feynmanInstance, currentConfiguration,
                0, seed, evaluationsPerConfiguration);
            results.Add(currentBestResult);

            while (results.Count < numConfigurations)
            {
                var neighbors = GenerateNeighbors(currentConfiguration)
                    .Where(n => !evaluatedConfigurations.Contains(n))
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
