using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms.Feynman;
using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    internal class RandomSearch : IHyperParameterOptimizationAlgorithm
    {
        List<(AlgorithmParameter, Population<ExpressionTree>)> IHyperParameterOptimizationAlgorithm.Execute(FeynmanDescriptor feynmanInstance, CancellationToken token)
        {
            throw new NotImplementedException();
        }
    }
}
