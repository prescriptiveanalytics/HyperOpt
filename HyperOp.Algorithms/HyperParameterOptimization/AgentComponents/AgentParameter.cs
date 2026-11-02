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
        // Upper bound on generated tokens per request; tool calls and controller decisions are far shorter.
        public int MaxOutputTokens { get; set; } = 1024;
        public bool Streaming { get; set; } = true;
        // Turn cap per conversation; the agent explicitly decides when to finish the search.
        public int Turns { get; set; } = 50;
        public string SystemPrompt { get; set; } = "systemPrompt";
        // Information the run controller receives at checkpoints; None runs without checkpoints (final result only).
        public FeedbackChannels Feedback { get; set; } = FeedbackChannels.Progress;
        // Generations between controller checkpoints.
        public int CheckpointInterval { get; set; } = 10;
        // False: the controller still acts at checkpoints, but the configurator receives only final results.
        public bool ConfiguratorSeesCheckpoints { get; set; } = true;
        // GP pauses while the controller decides, so this only guards against hangs. A short limit would turn
        // slow inference into automatic Continue decisions and confound model comparisons with hardware speed.
        public TimeSpan DecisionTimeout { get; set; } = TimeSpan.FromMinutes(5);
    }
}
