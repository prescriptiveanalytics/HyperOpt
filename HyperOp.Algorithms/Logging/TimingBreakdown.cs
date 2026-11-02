namespace HyperOp.Algorithms.Logging
{
    // One model request, measured on the client. Duration includes queueing at the inference server.
    public sealed record LlmCallRecord(string Role, double StartMs, double DurationMs, string Outcome,
        int? InputTokens, int? OutputTokens, int? ReasoningTokens, string? Error);

    // Wall-clock decomposition of one experiment job, in milliseconds.
    // TuningMs = TuningTrialWallMs + OptimizerMs; each tuning trial's wall time is its GP time,
    // the controller's inference while GP waits, and trial overhead (setup, calibration and validation scoring).
    public sealed class TimingBreakdown
    {
        public double TotalJobMs { get; set; }
        public double ModelWarmupMs { get; set; }
        public double TuningMs { get; set; }
        public double TuningTrialWallMs { get; set; }
        public double TuningGpMs { get; set; }
        public double TuningTrialOverheadMs { get; set; }
        // Optimizer time outside GP trials: configurator inference and agent framework for AgentSearch,
        // the R process for irace, proposal logic for the other methods.
        public double OptimizerMs { get; set; }
        public double ConfiguratorInferenceMs { get; set; }
        public int ConfiguratorRequests { get; set; }
        public double ControllerInferenceMs { get; set; }
        public int ControllerRequests { get; set; }
        public int ControllerFallbacks { get; set; }
        public int InputTokens { get; set; }
        public int OutputTokens { get; set; }
        public int ReasoningTokens { get; set; }
        // Runtime context window of the served model and the requests whose prompt filled at least 90% of it
        // (the server truncates silently beyond the window).
        public int? ContextLength { get; set; }
        public int RequestsNearContextLimit { get; set; }
        // Process CPU time of the tuning trials; unlike wall time, it does not grow when parallel jobs compete for cores.
        public double TuningTrialCpuMs { get; set; }
        public double SelectionAndTestScoringMs { get; set; }
        public double AssessmentMs { get; set; }
        public double AssessmentGpMs { get; set; }
        public double LlmInferenceMs => ConfiguratorInferenceMs + ControllerInferenceMs;
        public double LlmShareOfTuning => TuningMs > 0 ? LlmInferenceMs / TuningMs : 0;
    }
}
