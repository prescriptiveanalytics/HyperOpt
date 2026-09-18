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
        public override async Task<List<(AlgorithmParameter, Population<ExpressionTree>)>> Execute(FeynmanDescriptor feynmanInstance, int seed, int evaluations)
        {
            var ret = new List<(AlgorithmParameter, Population<ExpressionTree>)>();
            Random r = new Random(seed);

            var popRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.PopulationSize))?.GetCustomAttribute<RangeAttribute>();
            var mutationRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MutationRate))?.GetCustomAttribute<RangeAttribute>();
            var maxTreeDepthRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MaxTreeDepth))?.GetCustomAttribute<RangeAttribute>();
            var maxTreeLengthRange = typeof(AlgorithmParameter).GetProperty(nameof(AlgorithmParameter.MaxTreeLength))?.GetCustomAttribute<RangeAttribute>();

            var mutatorTypes = Enum.GetValues(typeof(MutatorType)).Cast<MutatorType>().ToArray();

            //we simply keep on trying until the budget is exhausted
            int remainingEvaluations = evaluations;
            int divider = 10;
            while (remainingEvaluations > 0)
            {
                int currentEvaluations = Math.Min(remainingEvaluations, evaluations / divider);
                remainingEvaluations = remainingEvaluations - currentEvaluations;

                var parameter = new AlgorithmParameter {
                    PopulationSize = r.Next((int)popRange.Minimum, (int)popRange.Maximum),
                    MutationRate = r.NextDouble() * ((double)mutationRange.Maximum - (double)mutationRange.Minimum) + (double)mutationRange.Minimum,
                    MaxTreeDepth = r.Next((int)maxTreeDepthRange.Minimum, (int)maxTreeDepthRange.Maximum),
                    MaxTreeLength = r.Next((int)maxTreeLengthRange.Minimum, (int)maxTreeLengthRange.Maximum),
                    MutatorType = mutatorTypes[r.Next(mutatorTypes.Length)],
                    Evaluations = currentEvaluations
                };

                var (algorithm, problem) = PrepareAlgorithm(feynmanInstance, parameter);
                var populationState = await algorithm.CompleteAsync(problem, RandomNumberGenerator.Create(seed));

                ret.Add((parameter, populationState.Population));
            }

            return ret;
        }
    }
}
