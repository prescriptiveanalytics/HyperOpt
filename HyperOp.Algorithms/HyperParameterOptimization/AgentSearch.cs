using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms.Feynman;
using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class AgentSearch : HyperParameterOptimizationAlgorithm
    {
        public override async Task<List<(AlgorithmParameter, Population<ExpressionTree>)>> Execute(FeynmanDescriptor feynmanInstance, int seed, int evaluations)
        {
            throw new NotImplementedException();
        }
    }
}
