using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms
{
    public class ResultDTO
    {
        public required AlgorithmParameter Configuration { get; set; }

        public required int ConfigurationIndex { get; set; }

        public required double BestFitness { get; set; }

        public required double MeanPopulationFitness { get; set; }

        public required double WorstPopulationFitness { get; set; }

        public required double ExecutionTimeMs { get; set; }

        public required int EvaluationsUsed { get; set; }

        public required int EvaluationsLimit { get; set; }

        public required int GenerationsCompleted { get; set; }

        public required int PopulationSize { get; set; }

        public required List<IndividualSnapshot> BestIndividualsSnapshot { get; set; }

        public List<GenerationQualitySnapshot>? QualityCurve { get; set; }

        public class IndividualSnapshot
        {
            public required double ObjectiveValue { get; set; }
            public required int Depth { get; set; }
            public required int Length { get; set; }
            public required double Complexity { get; set; }
            public required string InfixRepresentation { get; set; }
        }

        public class GenerationQualitySnapshot
        {
            public required int GenerationNumber { get; set; }
            public required double BestQuality { get; set; }
            public required double MedianQuality { get; set; }
            public required double WorstQuality { get; set; }
        }
    }
}
