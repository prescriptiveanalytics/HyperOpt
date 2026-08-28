using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms.Feynman;
using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    internal class IRaceSearch : IHyperParameterOptimizationAlgorithm
    {
        public List<(AlgorithmParameter, Population<ExpressionTree>)> Execute(FeynmanDescriptor feynmanInstance, CancellationToken token)
        {
            throw new NotImplementedException();
        }
    }
}
