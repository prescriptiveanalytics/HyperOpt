using LlmTornado.Code;

namespace HyperOp.Algorithms.HyperParameterOptimization.AgentComponents
{
    public record AgentParameter
    {
        public string ModelName { get; set; } = "google/gemma-4-26b-a4b-it";
        public LLmProviders Provider { get; set; } = LlmTornado.Code.LLmProviders.OpenRouter;
        // Use the Ollama service name when both containers share a Docker network.
        public string OllamaBaseUrl { get; set; } = "http://ollama:11434";
        public ChatRequestServiceTiers ServiceTiers { get; set; } = ChatRequestServiceTiers.Flex;
        public double Temperature { get; set; } = 0.05;
        public bool Streaming { get; set; } = true;
        public int Turns { get; set; } = 50; // TODO: Make this dependent on evaluations -> maybe exclude from here and compute in Execute Function of AgentSearch

        public string SytstemPrompt { get; set; } = "systemPrompt";
        public string Instruction { get; set; } = string.Empty;

        //public TimeSpan TimeLimit { get; set; } = TimeSpan.FromMinutes(5); // Replaced by evaluation limit
    }
}
