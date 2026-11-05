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
        // Ranges and default follow GP symbolic regression practice with constant optimization (proposal, under review):
        // population 100–1000 (SRBench grids; Operon and Haider et al. use 1000), mutation up to 0.5 (Haider 0.15,
        // Operon 0.25), depth 5–20 (Operon 10, Haider 20), length 10–100 (Haider 10–50, Operon 50).
        [Range(100, 1000)]
        public int PopulationSize { get; set; } = 1000;
        [Range(0.0, 0.5)]
        public double MutationRate { get; set; } = 0.15;
        [Range(5, 20)]
        public int MaxTreeDepth { get; set; } = 20;
        [Range(10, 100)]
        public int MaxTreeLength { get; set; } = 50;
        public MutatorType MutatorType { get; set; } = MutatorType.Combined;
    }
}
