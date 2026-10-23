using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;
using System.ComponentModel.DataAnnotations;
using System.Reflection;

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
                    PopulationSize = random.Next((int)popRange!.Minimum!, (int)popRange.Maximum! + 1),
                    MutationRate = random.NextDouble() * ((double)mutationRange!.Maximum! - (double)mutationRange.Minimum!) + (double)mutationRange.Minimum!,
                    MaxTreeDepth = random.Next((int)maxTreeDepthRange!.Minimum!, (int)maxTreeDepthRange.Maximum! + 1),
                    MaxTreeLength = random.Next((int)maxTreeLengthRange!.Minimum!, (int)maxTreeLengthRange.Maximum! + 1),
                    MutatorType = mutatorTypes[random.Next(mutatorTypes.Length)]
                };

                results.Add(await EvaluateConfiguration(feynmanInstance, parameter,
                    configIndex, seed + configIndex, evaluationsPerConfiguration));
            }

            return results;
        }
    }
}
