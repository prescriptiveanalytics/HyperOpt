using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.AgentComponents;
using HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.Util;
using HyperOp.Algorithms.HyperParameterOptimization.Util;
using LlmTornado;
using LlmTornado.Agents;
using LlmTornado.Chat;
using LlmTornado.Chat.Models;
using Microsoft.Extensions.Configuration;
using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;

// version without GPU
// docker run -d --name ollama -p 127.0.0.1:11434:11434 -v ollama-data:/root/.ollama ollama/ollama
// docker exec ollama ollama pull qwen3:8b

// version with GPU
// docker run -d --name ollama --gpus all -p 127.0.0.1:11434:11434 -v ollama-data:/root/.ollama ollama/ollama
// docker exec ollama ollama pull qwen3:8b

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
        //TODO: Habs adapt according to the numer of evaluations / number of configurations
        // Add all agent relavent parameters here, otherwise we cannot refere to the prepare algorithm function

        // Initialize the respective LLM Tornado components
        private TornadoApi? _api;
        private ChatModel? _model;
        private TornadoAgent? _agent;
        private int _maxTurns;
        private string? _systemPrompt;
        private int _currentEvaluations;
        private int _totalEvaluations;
        private int _evaluationsPerConfiguration;
        private int _numConfigurations;
        private readonly SemaphoreSlim _trialLock = new(1, 1);
        private int RemainingEvaluations => _totalEvaluations - _currentEvaluations;

        public List<Experiment> _experimentHistory = new List<Experiment>();

        // Hold feynman instance seperately and make available for changes
        public RegressionDescriptor? FeynmanInstance { get; set; }
        public int Seed { get; set; }
        public List<ResultDTO> Results { get; set; } = new List<ResultDTO>();

        public AgentSearch(bool useOLlama = true)
            : this(useOLlama ? CreateOllamaParameters() : new AgentParameter())
        {
        }

        public static AgentParameter CreateOllamaParameters()
        {
            return new AgentParameter
            {
                Provider = LlmTornado.Code.LLmProviders.Custom,
                ModelName = "qwen3:8b",
                OllamaBaseUrl = "http://localhost:11434", //"http://ollama:11434",
                Streaming = false
            };
        }

        public AgentSearch(AgentParameter agentParameters)
        {
            Initialize(agentParameters);
        }

        public void Initialize(AgentParameter agentParameters)
        {
            switch (agentParameters.Provider)
            {
                case LlmTornado.Code.LLmProviders.OpenRouter:
                    InitializeOpenRouter(agentParameters);
                    break;
                case LlmTornado.Code.LLmProviders.Custom:
                    InitializeOllama(agentParameters);
                    break;
                default:
                    throw new NotSupportedException($"AgentSearch does not support provider {agentParameters.Provider}.");
            }
        }

        public void InitializeOpenRouter(AgentParameter agentParameters)
        {
            string openRouterAPIKey = GetAPIKeyFromConfiguration();
            _api = new TornadoApi(LlmTornado.Code.LLmProviders.OpenRouter, openRouterAPIKey);
            _model = new ChatModel(agentParameters.ModelName, LlmTornado.Code.LLmProviders.OpenRouter);

            InitializeAgent(agentParameters);
            _agent!.Options.ServiceTier = agentParameters.ServiceTiers;
        }

        public void InitializeOllama(AgentParameter agentParameters)
        {
            _api = new TornadoApi(new Uri(agentParameters.OllamaBaseUrl));
            _model = new ChatModel(agentParameters.ModelName, LlmTornado.Code.LLmProviders.Custom);

            InitializeAgent(agentParameters);
            _agent!.Options.ReasoningEffort = LlmTornado.Code.ChatReasoningEfforts.None; // Disable reasoning locally
        }

        private void InitializeAgent(AgentParameter agentParameters)
        {
            // Taken from aisra Code
            _systemPrompt = Assembly.GetExecutingAssembly()
                    .Let(asm => asm.ReadEmbeddedTextFile($"{asm.GetName().Name}.Resources.{agentParameters.SytstemPrompt}.md"));

            _agent = new TornadoAgent(
                    client: _api!,
                    model: _model!,
                    instructions: _systemPrompt,
                    tools: [(Func<int, double, int, int, MutatorType, Task<Experiment>>)RunSymbolicRegression, GetExperimentHistory, GetRemainingEvaluations],
                    streaming: agentParameters.Streaming
                    );
            _agent.Options.Temperature = agentParameters.Temperature;
            _maxTurns = agentParameters.Turns;
        }

        private string GetAPIKeyFromConfiguration()
        {
            // Im HyperOptAgent Project
            // dotnet user-secrets init
            // dotnet user-secrets set "OpenRouterAPIKey" "sk-..."
            IConfiguration configuration = new ConfigurationBuilder()
                .AddUserSecrets<AgentSearch>() // Loads user secrets for the specified assembly
                .Build();
            var openRouterAPIKey = configuration["OpenRouterAPIKey"];
            if (string.IsNullOrEmpty(openRouterAPIKey)) throw new Exception("No open router api has been configured in user secrets for OptimizerSystem.csproj");

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

            // The maximum number of turns (eg. evaluations of hyperparamter configurations) that the agent is allowed to take
            //_maxTurns = numConfigurations * evaluationsPerConfiguration;  // This is a simplification, we could also let the agent decide how many evaluations to use per configuration
            _numConfigurations = numConfigurations;
            _evaluationsPerConfiguration = evaluationsPerConfiguration;
            _totalEvaluations = totalEvaluations;
            _currentEvaluations = 0;
            var sw = Stopwatch.StartNew();

            Conversation result = await _agent!.Run(
                $"Optimize symbolic regression. Total budget: {_totalEvaluations} evaluations. " +
                $"Evaluate at most {numConfigurations} configurations. " +
                $"C# assigns {evaluationsPerConfiguration} evaluations to each run. " +
                //"First call RunSymbolicRegression with baseline algorithmParameter=" +
                //System.Text.Json.JsonSerializer.Serialize(new AlgorithmParameter
                //{
                //    PopulationSize = Math.Min(100, evaluationsPerConfiguration)
                //}) +
                //". " +
                "Try improvements while evaluation budget remains.",
                maxTurns: _maxTurns
                // Think about cancelation token
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
            return RunSymbolicRegression(new AlgorithmParameter
            {
                PopulationSize = populationSize,
                MutationRate = mutationRate,
                MaxTreeDepth = maxTreeDepth,
                MaxTreeLength = maxTreeLength,
                MutatorType = mutatorType
            });
        }

        public async Task<Experiment> RunSymbolicRegression(AlgorithmParameter algorithmParameter)
        {
            await _trialLock.WaitAsync();
            try
            {
                if (Results.Count >= _numConfigurations)
                    throw new InvalidOperationException("The configured number of regression runs has been reached.");
                int evaluationsPerConfiguration = _evaluationsPerConfiguration;

                Console.WriteLine($"Running symbolic regression with the following parameters:{algorithmParameter}");
                int trialIndex = Results.Count;
                var result = await EvaluateConfiguration(FeynmanInstance!, algorithmParameter,
                    trialIndex, Seed + trialIndex, evaluationsPerConfiguration);
                _currentEvaluations += result.EvaluationsUsed;

                // Compute if improvement to previous best
                double previousBest = Results.Count > 0 ? Results.Min(r => r.BestFitness) : double.PositiveInfinity;
                // Positive means an error reduction. The first trial has no measured improvement.
                double improvement = double.IsFinite(previousBest) && double.IsFinite(result.BestFitness)
                    ? previousBest - result.BestFitness : 0;
                bool isNewBest = result.BestFitness < previousBest;

                Results.Add(result);
                Console.WriteLine($"Remaining Evaluations: {RemainingEvaluations}");
                var experiment = new Experiment { HyperParameters = result.Configuration,
                    BestFitness = result.BestFitness,
                    Improvement = improvement,
                    IsNewBest = isNewBest,
                    EvaluationsUsed = result.EvaluationsUsed,
                    RemainingEvaluations = RemainingEvaluations,
                };
                _experimentHistory.Add(experiment);
                return experiment;

            }
            finally { _trialLock.Release(); }
        }

        [Description(
            "Use GetExperimentHistory when you need to review previous experiments or compare multiple configurations."
        )]
        public List<Experiment> GetExperimentHistory()
        {
            return _experimentHistory;
        }

        [Description(
            "Returns the best hyperparameter configuration evaluated so far and its fitness."
        )]
        public Experiment? GetBestExperiment()
        {
            return _experimentHistory.OrderBy(e => e.BestFitness).FirstOrDefault();
        }

        [Description(
            "Returns the total evaluation budget, actual candidate evaluations as UsedEvaluations, " +
            "and the remaining unallocated budget. Each completed run consumes its full configured allocation."
        )]
        public EvaluationBudget GetRemainingEvaluations()
        {
            return new EvaluationBudget {
                TotalEvaluations = _totalEvaluations,
                UsedEvaluations = _currentEvaluations,
                RemainingEvaluations = RemainingEvaluations
            };
        }
    }
}
