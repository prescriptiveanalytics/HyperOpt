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
        public required int TotalConfigurationsEvaluated { get; set; }
        public required List<ResultDTO> ConfigurationResults { get; set; }
    }
}
