using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms
{
    public class AlgorithmParameter
    {
        public int PopulationSize { get; set; } = 100;
        public double MutationRate { get; set; } = 0.1;
        public int Generations { get; set; } = 10;   
        public int MaxTreeDepth { get; set; } = 10;
        public int MaxTreeLength { get; set; } = 20;
    }
}
