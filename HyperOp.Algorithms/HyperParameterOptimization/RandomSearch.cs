using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Random;
using HyperOp.Algorithms.Feynman;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Reflection;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class RandomSearch : HyperParameterOptimizationAlgorithm
    {
        public override async Task<List<(AlgorithmParameter, Population<ExpressionTree>)>> Execute(FeynmanDescriptor feynmanInstance, CancellationToken token)
        {
            var ret = new List<(AlgorithmParameter, Population<ExpressionTree>)>();
            Random r = new Random(42);

            var popRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.PopulationSize))?.GetCustomAttribute<RangeAttribute>();
            var mutationRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MutationRate))?.GetCustomAttribute<RangeAttribute>();
            var generationsRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.Generations))?.GetCustomAttribute<RangeAttribute>();
            var maxTreeDepthRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MaxTreeDepth))?.GetCustomAttribute<RangeAttribute>();
            var maxTreeLengthRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MaxTreeLength))?.GetCustomAttribute<RangeAttribute>();

            var mutatorTypes = Enum.GetValues(typeof(MutatorType)).Cast<MutatorType>().ToArray();

            for (int i = 0; i < 10; i++)
            {
                if (token.IsCancellationRequested) { break; }
                var parameter = new AlgorithmParameter {
                    PopulationSize = r.Next((int)popRange.Minimum, (int)popRange.Maximum),
                    MutationRate = r.NextDouble() * ((double)mutationRange.Maximum - (double)mutationRange.Minimum) + (double)mutationRange.Minimum,
                    Generations = r.Next((int)generationsRange.Minimum, (int)generationsRange.Maximum),
                    MaxTreeDepth = r.Next((int)maxTreeDepthRange.Minimum, (int)maxTreeDepthRange.Maximum),
                    MaxTreeLength = r.Next((int)maxTreeLengthRange.Minimum, (int)maxTreeLengthRange.Maximum),
                    MutatorType = mutatorTypes[r.Next(mutatorTypes.Length)]
                };

                var (algorithm, problem) = PrepareAlgorithm(feynmanInstance, parameter);
                var populationState = await algorithm.CompleteAsync(problem, RandomNumberGenerator.Create(42), ct: token);

                ret.Add((parameter, populationState.Population));
            }

            return ret;
        }
    }
}
