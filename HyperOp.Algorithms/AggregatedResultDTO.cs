using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms
{
    public class AggregatedResultDTO
    {
        public required string AlgorithmName { get; set; }
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
    }
}
