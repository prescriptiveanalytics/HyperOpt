using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Random;
using HyperOp.Algorithms.Feynman;
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

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public record Experiment
    {
        public required AlgorithmParameter HyperParameters { get; init; }
        public required double BestFitness { get; init; }
        public required double Improvement { get; init; }
        public required bool IsNewBest { get; init; }
        public int EvaluationsUsed { get; init; }
    }

    public record EvaluationBudget
    {
        public required int TotalEvaluations { get; init; }
        public required int UsedEvaluations { get; init; }
        public required int RemainingEvaluations { get; init; }
    }

    public class AgentSearch : HyperParameterOptimizationAlgorithm
    {
        // Add all agent relavent parameters here, otherwise we cannot refere to the prepare algorithm function

        // Initialize the respective LLM Tornado components
        private TornadoApi? _api;
        private ChatModel? _model;
        private TornadoAgent? _agent;
        private int _maxTurns;
        private string? _systemPrompt;
        private int _currentEvaluations;
        private int _totalEvaluations;
        private int RemainingEvaluations => _totalEvaluations - _currentEvaluations;

        public List<Experiment> _experimentHistory = new List<Experiment>();

        // Hold feynman instance seperately and make available for changes
        public FeynmanDescriptor? FeynmanInstance { get; set; }
        public int Seed { get; set; }
        public List<ResultDTO> Results { get; set; } = new List<ResultDTO>();

        public AgentSearch()
        {
            // use default agent parameters
            Initialize(new AgentParameter());
        }

        public void Initialize(AgentParameter agentParameters)
        {
            string openRouterAPIKey = GetAPIKeyFromConfiguration();
            _api = new TornadoApi(LlmTornado.Code.LLmProviders.OpenRouter, openRouterAPIKey);
            _model = new ChatModel(agentParameters.ModelName, LlmTornado.Code.LLmProviders.OpenRouter);

            // Taken from aisra Code
            _systemPrompt = Assembly.GetExecutingAssembly()
                    .Let(asm => asm.ReadEmbeddedTextFile($"{asm.GetName().Name}.Resources.{agentParameters.SytstemPrompt}.md"));

            _agent = new TornadoAgent(
                    client: _api,
                    model: _model,
                    instructions: _systemPrompt,
                    tools: [RunSymbolicRegression, GetExperimentHistory], //, GetBestExperiment, GetRemainingEvaluations],
                    streaming: agentParameters.Streaming
                    );
            _agent.Options.Temperature = agentParameters.Temperature;
            _agent.Options.ServiceTier = agentParameters.ServiceTiers;
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

        public override async Task<List<ResultDTO>> Execute(FeynmanDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            Results = new List<ResultDTO>();
            var random = new Random(seed);

            FeynmanInstance = feynmanInstance;
            Seed = seed;

            // The maximum number of turns (eg. evaluations of hyperparamter configurations) that the agent is allowed to take
            //_maxTurns = numConfigurations * evaluationsPerConfiguration;  // This is a simplification, we could also let the agent decide how many evaluations to use per configuration
            _totalEvaluations = numConfigurations * evaluationsPerConfiguration;
            _currentEvaluations = 0;
            // Now run Agent, tell him that it must balance the number of configurations of hyperparameters and the number of evaluations per configuration OR do this in Code?
            var sw = Stopwatch.StartNew();

            Conversation result = await _agent!.Run(
                "", // TODO: Think about agent input
                maxTurns: _maxTurns
                // Think about cancelation token
            );

            var responseTime = sw.Elapsed;
            sw.Restart();

            return Results;
        }

        // ------------- Tool Calls --------------
        [Description(
            "Runs a symbolic regression algorithm with algorithm parameters and maximum evaluations per configruation on the current available problem data." +
            "Maximum Evaluations determines how many evaluation actions the GA-SR algorithm can take for the given configuration and must be set to a reasonable number with respect to " +
            "generation count and population size, since for each individual in each generation there is at least one evaluation. Eg. with 10 generations and 100 population size we need at least 1000 evaluations." +
            "It must be lower or equal to the RemainingEvaluations available." +
            "Returns an Experiment record with the hyperparameters and the best fitness achieved."
        )]
        public async Task<Experiment> RunSymbolicRegression(AlgorithmParameter algorithmParameter, int evaluationsPerConfiguration)
        {
            if (evaluationsPerConfiguration > RemainingEvaluations)
                throw new InvalidOperationException(
                    $"Requested {evaluationsPerConfiguration} evaluations, " +
                    $"but only {RemainingEvaluations} remain.");

            Console.WriteLine($"Running symbolic regression with the following parameters:{algorithmParameter}");
            // Prepare algorithm with this configuration
            // TODO: Think about how many evaluations per configuration should be used, currently agent decides this let the agent decide this?
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

            _currentEvaluations += result.EvaluationsUsed;

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
            "Returns the current evaluation budget, including total, used, and remaining evaluations."
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
