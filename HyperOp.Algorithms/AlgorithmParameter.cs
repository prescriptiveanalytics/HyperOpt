using System.ComponentModel.DataAnnotations;

namespace HyperOp.Algorithms
{
    public record AlgorithmParameter
    {
        [Range(1, 1_000_000)]
        public int PopulationSize { get; set; } = 100;
        [Range(0.0, 1.0)]
        public double MutationRate { get; set; } = 0.1;
        [Range(1, 1_000_000)]
        public int Generations { get; set; } = 10;
        [Range(1,1_000)]
        public int MaxTreeDepth { get; set; } = 10;
        [Range(1,10_000)]
        public int MaxTreeLength { get; set; } = 20;
    }
}
