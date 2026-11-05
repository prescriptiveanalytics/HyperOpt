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
        // Constant fitting within TuningGpMs.
        public double TuningConstantFittingMs { get; set; }
        public double TuningTrialOverheadMs { get; set; }
        // Optimizer time outside GP trials: configurator inference and agent framework for AgentSearch,
        // the R process for irace, proposal logic for the other methods.
        public double OptimizerMs { get; set; }
        public double ConfiguratorInferenceMs { get; set; }
        public int ConfiguratorRequests { get; set; }
        public double ControllerInferenceMs { get; set; }
        // Controller decisions (made by the LLM, a rule or a random draw) and those that fell back to Continue. Checkpoints kept
        // without a decision (less than one checkpoint left) are not decisions, so they cannot dilute the fallback rate;
        // ControllerModelRequests counts the controller's HTTP requests to the model (cost).
        public int ControllerRequests { get; set; }
        public int ControllerFallbacks { get; set; }
        public int ControllerObservationsWithoutDecision { get; set; }
        public int ControllerModelRequests { get; set; }
        public int InputTokens { get; set; }
        public int OutputTokens { get; set; }
        public int ReasoningTokens { get; set; }
        // Runtime context window of the served model and the requests whose prompt and reply filled at least 90% of it
        // (the server truncates silently beyond the window, so a request at the limit may have lost its start).
        public int? ContextLength { get; set; }
        public int RequestsNearContextLimit { get; set; }
        // Memory of the served model at the start of the job (total and in GPU memory, bytes) and the job process's peak
        // working set (MB). A model only partly in GPU memory runs layers on the CPU and slows every request.
        public long? ModelMemoryBytes { get; set; }
        public long? ModelGpuMemoryBytes { get; set; }
        public double PeakWorkingSetMb { get; set; }
        // Largest single request per role (prompt tokens, and prompt plus reply): the context the job actually needed.
        public int MaxConfiguratorInputTokens { get; set; }
        public int MaxControllerInputTokens { get; set; }
        public int MaxRequestTokens { get; set; }

        public void RecordControllerDecisions(IEnumerable<HyperOp.Algorithms.HyperParameterOptimization.RunControlObservation> observations,
            IReadOnlyList<LlmCallRecord> calls)
        {
            var all = observations.ToArray();
            var decisions = all.Where(o => o.Decision.Reason != HyperOp.Algorithms.HyperParameterOptimization.AgentSearch.NoDecisionReason).ToArray();
            ControllerRequests = decisions.Length;
            ControllerFallbacks = decisions.Count(o => o.UsedFallback);
            ControllerObservationsWithoutDecision = all.Length - decisions.Length;
            ControllerModelRequests = calls.Count(call => call.Role == "controller");
        }

        public void RecordContextUsage(IReadOnlyList<LlmCallRecord> calls)
        {
            static int RequestTokens(LlmCallRecord call) => (call.InputTokens ?? 0) + (call.OutputTokens ?? 0);
            int MaxInput(string role) => calls.Where(call => call.Role == role).Select(call => call.InputTokens ?? 0).DefaultIfEmpty().Max();
            MaxConfiguratorInputTokens = MaxInput("configurator");
            MaxControllerInputTokens = MaxInput("controller");
            MaxRequestTokens = calls.Select(RequestTokens).DefaultIfEmpty().Max();
            RequestsNearContextLimit = ContextLength is int context ? calls.Count(call => RequestTokens(call) >= 0.9 * context) : 0;
        }
        // Process CPU time of the tuning trials; unlike wall time, it does not grow when parallel jobs compete for cores.
        public double TuningTrialCpuMs { get; set; }
        public double SelectionAndTestScoringMs { get; set; }
        public double AssessmentMs { get; set; }
        public double AssessmentGpMs { get; set; }
        public double LlmInferenceMs => ConfiguratorInferenceMs + ControllerInferenceMs;
        public double LlmShareOfTuning => TuningMs > 0 ? LlmInferenceMs / TuningMs : 0;
    }
}
