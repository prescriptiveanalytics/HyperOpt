using LlmTornado.Code;
using HyperOp.Algorithms.HyperParameterOptimization;

namespace HyperOp.Algorithms.HyperParameterOptimization.AgentComponents
{
    public record AgentParameter
    {
        public string ModelName { get; set; } = "google/gemma-4-26b-a4b-it";
        public LLmProviders Provider { get; set; } = LLmProviders.OpenRouter;
        // Use the Ollama service name when both containers share a Docker network.
        public string OllamaBaseUrl { get; set; } = "http://ollama:11434";
        public ChatRequestServiceTiers ServiceTiers { get; set; } = ChatRequestServiceTiers.Flex;
        public double Temperature { get; set; } = 0.05;
        // Upper bound on generated tokens per request; proposals and controller decisions are far shorter.
        public int MaxOutputTokens { get; set; } = 1024;
        public bool Streaming { get; set; } = true;
        public string SystemPrompt { get; set; } = "systemPrompt";
        // Information the run controller receives at checkpoints; None runs without checkpoints (final result only).
        public FeedbackChannels Feedback { get; set; } = FeedbackChannels.Progress;
        // Controller checkpoints per per-configuration budget E: one every E/Checkpoints candidate evaluations.
        public int Checkpoints { get; set; } = CheckpointSchedule.DefaultCheckpoints;
        // False: the controller still acts at checkpoints, but the configurator receives only final results.
        public bool ConfiguratorSeesCheckpoints { get; set; } = true;
        // Placebo: the channel sections (curve, diversity, size, generalization) shown to both roles come from another
        // run of the job; progress, the comparison with earlier runs and the budget stay real. Same prompt, no information.
        public bool ChannelPlacebo { get; set; }
        // Pilot switches against over-stopping (4 November 2026): the controller's comparison with earlier runs at the same
        // evaluation count, and the checkpoint at which the controller is first asked (1 = after E/10).
        public bool ControllerComparison { get; set; } = true;
        public int FirstControllerCheckpoint { get; set; } = 1;
        // Controller instructions (see ControllerPrompts); the study uses ControllerPrompts.Study, the others are pilot variants.
        public string ControllerPrompt { get; set; } = ControllerPrompts.Study;
        // Who decides at the checkpoints (RQ1 controls): "llm" (the LLM controller), "rule" (the stagnation part of the
        // rule-controlled baseline: stop when the best training NMSE improved by less than 1% since the previous checkpoint),
        // or "random" (stop with probability StopHazard[k-1] at the k-th decision, the last value repeating). The LLM
        // configurator is the same in all three.
        public string Controller { get; set; } = "llm";
        public double[]? StopHazard { get; set; }
        // GP pauses while the controller decides, so this only guards against hangs. A short limit would turn
        // slow inference into automatic Continue decisions and confound model comparisons with hardware speed.
        public TimeSpan DecisionTimeout { get; set; } = TimeSpan.FromMinutes(5);
    }
}
