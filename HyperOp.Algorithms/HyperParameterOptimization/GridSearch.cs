using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class GridSearch : HyperParameterOptimizationAlgorithm
    {
        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            var configurations = ParameterSpace.Grid();
            // A seeded permutation avoids favoring the lexicographically first grid corner
            // when the trial cap is smaller than the fixed grid. Exhaustion may leave budget unused.
            new Random(seed).Shuffle(configurations);
            foreach (var config in configurations.Take(numConfigurations))
            {
                int trialIndex = results.Count;
                results.Add(await EvaluateConfiguration(feynmanInstance, config,
                    trialIndex, TrialSeed(seed, trialIndex), evaluationsPerConfiguration));
            }
            return results;
        }
    }
}
