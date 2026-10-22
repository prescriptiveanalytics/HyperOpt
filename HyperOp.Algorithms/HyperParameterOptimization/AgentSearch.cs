using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Random;
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
            ArgumentNullException.ThrowIfNull(feynmanInstance);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numConfigurations);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(evaluationsPerConfiguration);
            if (evaluationsPerConfiguration < 10)
                throw new ArgumentOutOfRangeException(nameof(evaluationsPerConfiguration),
                    "At least 10 evaluations are required for the minimum population size.");
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
            "Returns BestFitness (higher is better), Improvement, IsNewBest, EvaluationsUsed and RemainingEvaluations."
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
            if (FeynmanInstance is null || _evaluationsPerConfiguration <= 0)
                throw new InvalidOperationException("Execute must initialize the regression problem and evaluation limits first.");
            if (Results.Count >= _numConfigurations)
                throw new InvalidOperationException("The configured number of regression runs has been reached.");
            int evaluationsPerConfiguration = _evaluationsPerConfiguration;
            ArgumentNullException.ThrowIfNull(algorithmParameter);
            System.ComponentModel.DataAnnotations.Validator.ValidateObject(
                algorithmParameter,
                new System.ComponentModel.DataAnnotations.ValidationContext(algorithmParameter),
                validateAllProperties: true);
            if (!Enum.IsDefined(algorithmParameter.MutatorType))
                throw new ArgumentException("Unknown mutator type.", nameof(algorithmParameter));
            if (evaluationsPerConfiguration <= 0 || evaluationsPerConfiguration < algorithmParameter.PopulationSize)
                throw new ArgumentOutOfRangeException(nameof(evaluationsPerConfiguration),
                    "The evaluation limit must be positive and at least the population size.");
            if (evaluationsPerConfiguration > RemainingEvaluations)
                throw new InvalidOperationException(
                    $"Requested {evaluationsPerConfiguration} evaluations, " +
                    $"but only {RemainingEvaluations} remain.");

            Console.WriteLine($"Running symbolic regression with the following parameters:{algorithmParameter}");
            // Prepare algorithm with this configuration
            var (alg, problem) = PrepareAlgorithm(FeynmanInstance!, algorithmParameter, evaluationsPerConfiguration);
            var algorithm = alg.WithMaxEvaluatedCandidates(alg.Evaluator, evaluationsPerConfiguration);

            var qualityAnalyzer = Analyzer.BestMedianWorst(algorithm);
            var run = algorithm.CreateRun(problem, RandomNumberGenerator.Create(Seed)).WithAnalyzer(qualityAnalyzer);

            var stopwatch = Stopwatch.StartNew();
            var finalState = await run.CompleteAsync();
            stopwatch.Stop();

            var qualityCurve = run.GetResult(qualityAnalyzer);
            var (generationCount, bestFitness, meanFitness, worstFitness, snapshots, qualitySnapshots) = this.EvaluateFinalState(finalState, qualityCurve);

            // Create result record
            var result = new ResultDTO
            {
                Configuration = algorithmParameter,
                ConfigurationIndex = -1, // ??? count somehow everytime the same configuration is used?
                BestFitness = bestFitness,
                MeanPopulationFitness = meanFitness,
                WorstPopulationFitness = worstFitness,
                ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds,
                EvaluationsUsed = generationCount * algorithmParameter.PopulationSize,  // Use an approximation
                EvaluationsLimit = evaluationsPerConfiguration,
                GenerationsCompleted = generationCount,
                PopulationSize = algorithmParameter.PopulationSize,
                BestIndividualsSnapshot = snapshots,
                QualityCurve = qualitySnapshots
            };

            // Charge the full allocation so approximate metrics cannot permit extra runs.
            _currentEvaluations += evaluationsPerConfiguration;

            // Compute if improvement to previous best
            double previousBest = Results.Count > 0 ? Results.Max(r => r.BestFitness) : -1;
            double improvement = previousBest >= 0 ? result.BestFitness - previousBest : 100.00;
            bool isNewBest = result.BestFitness > previousBest;

            // What to provide to LLM to assert how good a configuration was? Simply best fitness?
            Results.Add(result);
            // Agent would benefit from a return type that is a bit simpler
            Console.WriteLine($"Remaining Evaluations: {RemainingEvaluations}");
            var experiment = new Experiment { HyperParameters = algorithmParameter, 
                BestFitness = bestFitness,
                Improvement = improvement,
                IsNewBest = isNewBest,
                EvaluationsUsed = result.EvaluationsUsed,
                RemainingEvaluations = RemainingEvaluations,
            };
            _experimentHistory.Add(experiment);
            return experiment;

            //return new Experiment { HyperParameters = algorithmParameter, QualityMetrik = 0.0 };
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
            "Returns the total evaluation budget, allocations charged to completed runs as UsedEvaluations, " +
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
