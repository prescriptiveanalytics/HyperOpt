using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;
using System.ComponentModel.DataAnnotations;
using System.Reflection;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class GridSearch : HyperParameterOptimizationAlgorithm
    {
        // Generates a grid of hyperparameter combinations based on how many discretization steps are specified
        // Grid size is product of parameters Default = 5.000 Configurations (gets really large fast!)
        private List<AlgorithmParameter> GenerateGrid(int populationSize = 10, int mutationRate = 5, int maxTreeDepth = 5, int maxTreeLength = 5)
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
                    MutatorType = mutatorType
                }
            ).ToList();
        }
        

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            var configurations = GenerateGrid().ToArray();
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
