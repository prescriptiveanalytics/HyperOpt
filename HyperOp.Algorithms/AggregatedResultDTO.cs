using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms
{
    public class AggregatedResultDTO
    {
        public required string AlgorithmName { get; set; }
        public string? ModelName { get; set; }
        public required string InstanceName { get; set; }
        public required int Repetition { get; set; }
        public required int Seed { get; set; }
        public required int NumConfigurations { get; set; }
        public required int EvaluationsPerConfiguration { get; set; }
        public required int TotalConfigurationsEvaluated { get; set; }
        public required List<ResultDTO> ConfigurationResults { get; set; }
        public int TrainingRows { get; set; }
        public int ValidationRows { get; set; }
        public int TestRows { get; set; }
        public int? SelectedTrialIndex { get; set; }
        public AlgorithmParameter? RecommendedConfiguration { get; set; }
        public int? RecommendedConfigurationIndex { get; set; }
        public string? RecommendationRule { get; set; }
        // Post-tuning runs of the fixed recommendation, separate from the HPO trial budget.
        public List<ResultDTO>? ConfigurationAssessmentResults { get; set; }
    }
}
