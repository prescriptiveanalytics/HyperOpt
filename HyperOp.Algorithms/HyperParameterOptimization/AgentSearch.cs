using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Random;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.HyperParameterOptimization.AgentComponents;
using HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.Util;
using LlmTornado;
using LlmTornado.Agents;
using LlmTornado.Chat;
using LlmTornado.Chat.Models;
using LlmTornado.FineTuning;
using Microsoft.Extensions.Configuration;
using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public record Experiment
    {
        public required AlgorithmParameter HyperParameters { get; init; }
        public required double QualityMetrik { get; init; }
        //public string? Expression { get; init; } = string.Empty;
        public int RemainingEvaluations { get; init; }
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
        private int _currentEvaluations = 0;
        private int _remainingEvaluations;

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
                    tools: [RunSymbolicRegression],
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
            _remainingEvaluations = numConfigurations * evaluationsPerConfiguration;

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

        private (int generationCount, 
                double bestFitness, 
                double meanFitness, 
                double worstFitness, 
                List<ResultDTO.IndividualSnapshot> snapshots, 
                List<ResultDTO.GenerationQualitySnapshot> qualitySnapshots) 
            EvaluateFinalState(PopulationState<ExpressionTree> finalState, List<BestMedianWorstEntry<ExpressionTree>> qualityCurve)
        {
            int generationCount = qualityCurve.Count;
            var worstOfFirstGeneration = qualityCurve.First().Worst.ObjectiveVector[0];

            // Extract metrics from population using API's ObjectiveVector
            var objectives = finalState.Population.Select(ind => ind.ObjectiveVector[0]).ToList();
            var bestFitness = objectives.Max();
            var meanFitness = objectives.Average();
            var worstFitness = objectives.Min();

            // Create individual snapshots for the best individuals
            var snapshots = finalState.Population
                .Select(ind => new ResultDTO.IndividualSnapshot
                {
                    ObjectiveValue = ind.ObjectiveVector[0],
                    Depth = ind.Candidate.Depth,
                    Length = ind.Candidate.Length,
                    Complexity = ind.Candidate.Complexity,
                    InfixRepresentation = ind.Candidate.ToInfixString()
                })
                .ToList();

            // Create per-generation quality curve snapshots
            List<ResultDTO.GenerationQualitySnapshot> qualitySnapshots = qualityCurve
                .Select((gen, genIndex) => new ResultDTO.GenerationQualitySnapshot
                {
                    GenerationNumber = genIndex,
                    BestQuality = gen.Best.ObjectiveVector[0],
                    MedianQuality = gen.Median.ObjectiveVector[0],
                    WorstQuality = gen.Worst.ObjectiveVector[0]
                })
                .ToList();

            return (generationCount,bestFitness, meanFitness, worstFitness, snapshots, qualitySnapshots);
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
            // keep track of number of evaluations already used and stop if we reach the maximum allowed evaluations
            _currentEvaluations += evaluationsPerConfiguration;

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
            var (generationCount, bestFitness, meanFitness, worstFitness, snapshots, qualitySnapshots) = EvaluateFinalState(finalState, qualityCurve);

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

            _remainingEvaluations -= result.EvaluationsUsed;

            // What to provide to LLM to assert how good a configuration was? Simply best fitness?
            Results.Add(result);
            // Agent would benefit from a return type that is a bit simpler
            Console.WriteLine($"Remaining Evaluations: {_remainingEvaluations}");
            return new Experiment { HyperParameters = algorithmParameter, QualityMetrik = bestFitness, RemainingEvaluations = _remainingEvaluations };
            //return new Experiment { HyperParameters = algorithmParameter, QualityMetrik = 0.0 };
        }
    }
}
