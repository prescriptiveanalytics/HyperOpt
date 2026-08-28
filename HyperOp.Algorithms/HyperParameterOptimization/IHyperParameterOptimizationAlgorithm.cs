using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using MathNet.Numerics.Statistics;
using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public interface IHyperParameterOptimizationAlgorithm
    {
        //We simply store the final population of each hyperparameter configuration, so we can analyze them later. Better to have the raw results than to later realize we missed something
        public List<(AlgorithmParameter, Population<ExpressionTree>)> Execute(Feynman.FeynmanDescriptor feynmanInstance, CancellationToken token);

        //Two trivial ideas for calculating the fitness of a population, which can be used to compare different hyperparameter configurations
        public double CalculateMeanFitness(Population<ExpressionTree> population) => population.Select(v => v.ObjectiveVector[0]).Mean();
        public double CalculateTopFitness(Population<ExpressionTree> population) => population.Max(v => v.ObjectiveVector[0]);
    }
}
