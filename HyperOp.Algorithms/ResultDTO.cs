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

        // BestFitness is validation Pearson R²; population statistics remain training metrics.
        public string FitnessMetric { get; set; } = "training_pearson_r2";
        public double? TrainingBestFitness { get; set; }
        public double? TestFitness { get; set; }
        public string? SelectedExpression { get; set; }

        [System.Text.Json.Serialization.JsonIgnore]
        internal HEAL.HeuristicLib.Encodings.SymbolicExpressions.ExpressionTree? SelectedModel { get; set; }

        public void EvaluateTest(Benchmarks.PreparedRegressionData data)
        {
            if (SelectedModel is null) throw new InvalidOperationException("No selected GP expression is available.");
            TestFitness = data.TestFitness(SelectedModel);
        }

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
