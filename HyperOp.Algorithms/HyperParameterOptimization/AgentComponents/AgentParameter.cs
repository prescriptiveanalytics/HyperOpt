using LlmTornado.Code;

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
        public bool Streaming { get; set; } = true;
        // Interaction cap, separate from the GP trial cap.
        public int Turns { get; set; } = 50;
        public string SystemPrompt { get; set; } = "systemPrompt";
    }
}
