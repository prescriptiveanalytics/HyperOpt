using System.ComponentModel.DataAnnotations;

namespace HyperOp.Algorithms
{
    public enum MutatorType
    {
        NodeReplacement,
        Subtree,
        LocalPerturbation,
        Combined
    }

    public record AlgorithmParameter
    {
        [Range(1, 100)]
        public int PopulationSize { get; set; } = 100;
        [Range(0.0, 1.0)]
        public double MutationRate { get; set; } = 0.2;
        [Range(1, 100)]
        public int Generations { get; set; } = 100;
        [Range(1, 100)]
        public int MaxTreeDepth { get; set; } = 100;
        [Range(1, 100)]
        public int MaxTreeLength { get; set; } = 100;
        public MutatorType MutatorType { get; set; } = MutatorType.Combined;
    }
}
