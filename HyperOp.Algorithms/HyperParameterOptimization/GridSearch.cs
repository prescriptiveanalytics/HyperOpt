using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms.Feynman;
using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class GridSearch : HyperParameterOptimizationAlgorithm
    {
        public override async Task<List<ResultDTO>> Execute(FeynmanDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            throw new NotImplementedException();
        }
    }
}
