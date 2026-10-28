using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.AgentComponents;
using LlmTornado;
using LlmTornado.Agents;
using LlmTornado.Chat.Models;
using Microsoft.Extensions.Configuration;
using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public record Experiment
    {
        public required AlgorithmParameter HyperParameters { get; init; }
        [System.Text.Json.Serialization.JsonNumberHandling(System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals)]
        public required double BestFitness { get; init; }
        public required double Improvement { get; init; }
        public required bool IsNewBest { get; init; }
        public int EvaluationsUsed { get; init; }
        public int RemainingEvaluations { get; init; }
    }

    public record EvaluationBudget
    {
        public required int TotalEvaluations { get; init; }
        public required int UsedEvaluations { get; init; }
        public required int RemainingEvaluations { get; init; }
    }

    public class AgentSearch : HyperParameterOptimizationAlgorithm
    {
        private readonly TornadoAgent _agent;
        private readonly int _maxTurns;
        private int _totalEvaluations;
        private int _evaluationsPerConfiguration;
        private int _numConfigurations;
        private readonly SemaphoreSlim _trialLock = new(1, 1);
        private int UsedEvaluations => Results.Sum(result => result.EvaluationsUsed);
        private int RemainingEvaluations => _totalEvaluations - UsedEvaluations;
        private readonly List<Experiment> _experimentHistory = [];
        private readonly bool useOLlama;

        public RegressionDescriptor? FeynmanInstance { get; set; }
        public int Seed { get; set; }
        public List<ResultDTO> Results { get; set; } = new List<ResultDTO>();

        public AgentSearch(bool useOLlama = true)
            : this(useOLlama ? CreateOllamaParameters() : new AgentParameter())
        {
            this.useOLlama = useOLlama;
        }

        public static AgentParameter CreateOllamaParameters()
        {
            return new AgentParameter {
                Provider = LlmTornado.Code.LLmProviders.Custom,
                ModelName = "qwen3:8b",
                OllamaBaseUrl = "http://localhost:11434",
                Streaming = false
            };
        }

        public AgentSearch(AgentParameter agentParameters)
        {
            var api = agentParameters.Provider switch {
                LlmTornado.Code.LLmProviders.OpenRouter => new TornadoApi(agentParameters.Provider, GetAPIKeyFromConfiguration()),
                LlmTornado.Code.LLmProviders.Custom => new TornadoApi(new Uri(agentParameters.OllamaBaseUrl)),
                _ => throw new NotSupportedException($"AgentSearch does not support provider {agentParameters.Provider}.")
            };
            var assembly = Assembly.GetExecutingAssembly();
            using var stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.Resources.{agentParameters.SystemPrompt}.md")
                ?? throw new FileNotFoundException($"Embedded prompt '{agentParameters.SystemPrompt}' was not found.");
            using var reader = new StreamReader(stream);
            _agent = new TornadoAgent(
                    client: api,
                    model: new ChatModel(agentParameters.ModelName, agentParameters.Provider),
                    instructions: reader.ReadToEnd(),
                    tools: [(Func<int, double, int, int, MutatorType, Task<Experiment>>)RunSymbolicRegression, GetExperimentHistory, GetRemainingEvaluations],
                    streaming: agentParameters.Streaming
                    );
            _agent.Options.Temperature = agentParameters.Temperature;
            if (agentParameters.Provider == LlmTornado.Code.LLmProviders.OpenRouter)
                _agent.Options.ServiceTier = agentParameters.ServiceTiers;
            else
                _agent.Options.ReasoningEffort = LlmTornado.Code.ChatReasoningEfforts.None;
            _maxTurns = agentParameters.Turns;
        }

        private static string GetAPIKeyFromConfiguration()
        {
            var configuration = new ConfigurationBuilder()
                .AddUserSecrets<AgentSearch>()
                .Build();
            var openRouterAPIKey = configuration["OpenRouterAPIKey"];
            if (string.IsNullOrEmpty(openRouterAPIKey)) throw new InvalidOperationException("Set OpenRouterAPIKey in user secrets for HyperOp.Algorithms.");

            return openRouterAPIKey;
        }

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            int totalEvaluations = checked(numConfigurations * evaluationsPerConfiguration);

            Results = new List<ResultDTO>();
            _experimentHistory.Clear();

            FeynmanInstance = feynmanInstance;
            Seed = seed;

            _numConfigurations = numConfigurations;
            _evaluationsPerConfiguration = evaluationsPerConfiguration;
            _totalEvaluations = totalEvaluations;
            var sw = Stopwatch.StartNew();

            await _agent.Run(
                $"Optimize symbolic regression. Total budget: {_totalEvaluations} evaluations. " +
                $"Evaluate at most {numConfigurations} configurations. " +
                $"C# assigns {evaluationsPerConfiguration} evaluations to each run. " +
                "Try improvements while evaluation budget remains.",
                maxTurns: _maxTurns
            );

            sw.Stop();
            Console.WriteLine($"Agent run duration (including regression evaluations): {sw.Elapsed.TotalSeconds:F2} s");
            var exitReason = _agent.LastRunExitReason;
            Console.WriteLine($"Exit reason: {exitReason?.GetType().Name}: {exitReason?.Message}");
            if (Results.Count == 0)
                throw new InvalidOperationException(
                    $"AgentSearch evaluated no configurations. Exit reason: {exitReason?.GetType().Name}: {exitReason?.Message}",
                    exitReason?.Exception);
            return Results;
        }

        // ------------- Tool Calls --------------
        [Description(
            "Runs symbolic regression on the current problem. " +
            "Supply only hyperparameters; C# controls evaluation limits and the number of configurations. " +
            "PopulationSize must not exceed the configured evaluation limit. " +
            "Returns BestFitness (validation NMSE; lower is better, zero is ideal), Improvement, " +
            "IsNewBest, EvaluationsUsed and RemainingEvaluations."
        )]
        private Task<Experiment> RunSymbolicRegression(
            int populationSize, double mutationRate, int maxTreeDepth, int maxTreeLength, MutatorType mutatorType)
        {
            return RunSymbolicRegression(new AlgorithmParameter {
                PopulationSize = populationSize,
                MutationRate = mutationRate,
                MaxTreeDepth = maxTreeDepth,
                MaxTreeLength = maxTreeLength,
                MutatorType = mutatorType
            });
        }

        public async Task<Experiment> RunSymbolicRegression(AlgorithmParameter algorithmParameter)
        {
            try
            {
                if (useOLlama) { await _trialLock.WaitAsync(); }

                if (Results.Count >= _numConfigurations)
                    throw new InvalidOperationException("The configured number of regression runs has been reached.");

                Console.WriteLine($"Running symbolic regression with the following parameters:{algorithmParameter}");
                int trialIndex = Results.Count;
                var result = await EvaluateConfiguration(FeynmanInstance!, algorithmParameter,
                    trialIndex, Seed + trialIndex, _evaluationsPerConfiguration);

                // Compute if improvement to previous best
                double previousBest = Results.Count > 0 ? Results.Min(r => r.BestFitness) : double.PositiveInfinity;
                // Positive means an error reduction. The first trial has no measured improvement.
                double improvement = double.IsFinite(previousBest) && double.IsFinite(result.BestFitness)
                    ? previousBest - result.BestFitness : 0;
                bool isNewBest = result.BestFitness < previousBest;

                Results.Add(result);
                Console.WriteLine($"Remaining Evaluations: {RemainingEvaluations}");
                var experiment = new Experiment {
                    HyperParameters = result.Configuration,
                    BestFitness = result.BestFitness,
                    Improvement = improvement,
                    IsNewBest = isNewBest,
                    EvaluationsUsed = result.EvaluationsUsed,
                    RemainingEvaluations = RemainingEvaluations,
                };
                _experimentHistory.Add(experiment);
                return experiment;

            }
            finally { if (useOLlama) { _trialLock.Release(); } }
        }

        [Description(
            "Use GetExperimentHistory when you need to review previous experiments or compare multiple configurations."
        )]
        public List<Experiment> GetExperimentHistory()
        {
            return _experimentHistory;
        }

        [Description(
            "Returns the total evaluation budget, actual candidate evaluations as UsedEvaluations, " +
            "and the remaining unallocated budget. Each completed run consumes its full configured allocation."
        )]
        public EvaluationBudget GetRemainingEvaluations()
        {
            return new EvaluationBudget {
                TotalEvaluations = _totalEvaluations,
                UsedEvaluations = UsedEvaluations,
                RemainingEvaluations = RemainingEvaluations
            };
        }
    }
}
