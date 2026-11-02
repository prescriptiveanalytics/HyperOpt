using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms
{
    public class AggregatedResultDTO
    {
        public required string AlgorithmName { get; set; }
        public string? ModelName { get; set; }
        public string? ModelProvider { get; set; }
        // Method settings that distinguish jobs, e.g. "feedback=progress+curve__interval=10"; part of the file name.
        public string? Variant { get; set; }
        // Ollama model identity: digest of the exact weights, size class and quantization, and server version.
        public HyperParameterOptimization.AgentComponents.OllamaModelInfo? ModelInfo { get; set; }
        // Event protocol of the attempt that produced this result, in the same directory.
        public string? ProtocolFile { get; set; }
        public Logging.TimingBreakdown? Timing { get; set; }
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
        // Per-run allowance of the assessment runs; E unless the method chose the nominee's run length.
        public int? AssessmentEvaluationsPerRun { get; set; }
    }
}
