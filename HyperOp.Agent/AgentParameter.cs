using LlmTornado.Code;


namespace HyperOp.Agent
{
    public record AgentParameter
    {
        public string ModelName{ get; set; } = "google/gemma-4-26b-a4b-it";
        public LLmProviders  Provider { get; set; } = LlmTornado.Code.LLmProviders.OpenRouter;
        public ChatRequestServiceTiers ServiceTiers { get; set; } = ChatRequestServiceTiers.Flex;
        public double Temperature { get; set; } = 0.05;
        public bool Streaming { get; set; } = true;
        public int Turns { get; set; } = 50;

        public string SytstemPrompt { get; set; } = string.Empty;
        public string Instruction { get; set; } = string.Empty;

        public TimeSpan TimeLimit { get; set; } = TimeSpan.FromMinutes(5);
    }
}
