using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class GridSearch : HyperParameterOptimizationAlgorithm
    {
        // 10 population sizes × 5 rates × 5 depths × 5 lengths × 4 mutators.
        private static AlgorithmParameter[] GenerateGrid()
        {
            return (
                from popSize in ParameterSpace.PopulationSizes
                from mutRate in ParameterSpace.MutationRates
                from maxDepth in ParameterSpace.MaxTreeDepths
                from maxLength in ParameterSpace.MaxTreeLengths
                from mutatorType in ParameterSpace.MutatorTypes
                select new AlgorithmParameter
                {
                    PopulationSize = popSize,
                    MutationRate = mutRate,
                    MaxTreeDepth = maxDepth,
                    MaxTreeLength = maxLength,
                    MutatorType = mutatorType
                }
            ).ToArray();
        }
        

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            var configurations = GenerateGrid();
            // A seeded permutation avoids favoring the lexicographically first grid corner
            // when the trial cap is smaller than the fixed grid. Exhaustion may leave budget unused.
            new Random(seed).Shuffle(configurations);
            foreach (var config in configurations.Take(numConfigurations))
            {
                int trialIndex = results.Count;
                results.Add(await EvaluateConfiguration(feynmanInstance, config,
                    trialIndex, seed + trialIndex, evaluationsPerConfiguration));
            }
            return results;
        }
    }
}
