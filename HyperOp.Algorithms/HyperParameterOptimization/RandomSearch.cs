using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class RandomSearch : HyperParameterOptimizationAlgorithm
    {
        public override async Task<List<ResultDTO>> Execute(
            RegressionDescriptor feynmanInstance,
            int seed,
            int numConfigurations,
            int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            var random = new Random(seed);

            // Evaluate exactly numConfigurations hyperparameter configurations
            for (int configIndex = 0; configIndex < numConfigurations; configIndex++)
            {
                results.Add(await EvaluateConfiguration(feynmanInstance, ParameterSpace.Sample(random),
                    configIndex, TrialSeed(seed, configIndex), evaluationsPerConfiguration));
            }

            return results;
        }
    }
}
