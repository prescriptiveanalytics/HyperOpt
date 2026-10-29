using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Execution;
using HEAL.HeuristicLib.Problems;
using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.AgentComponents;
using LlmTornado;
using LlmTornado.Agents;
using LlmTornado.Chat.Models;
using Microsoft.Extensions.Configuration;
using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

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
        public string TerminationReason { get; init; } = "evaluation_limit";
        public bool ReusedResult { get; init; }
        // Ordered training checkpoints and controller explanations for choosing the next configuration.
        public IReadOnlyList<RunControlObservation> ControlObservations { get; init; } = [];
    }

    public record EvaluationBudget
    {
        public required int TotalEvaluations { get; init; }
        public required int UsedEvaluations { get; init; }
        public required int RemainingEvaluations { get; init; }
        public int ConfigurationsCompleted { get; init; }
    }

    public class AgentSearch : HyperParameterOptimizationAlgorithm
    {
        private readonly TornadoAgent _agent;
        private readonly int _maxTurns;
        private int _totalEvaluations;
        private int _numConfigurations;
        public string? SearchStopReason { get; private set; }
        private readonly SemaphoreSlim _trialLock = new(1, 1);
        private int UsedEvaluations => Results.Sum(result => result.EvaluationsUsed) + Volatile.Read(ref _activeEvaluations);
        private int RemainingEvaluations => _totalEvaluations - UsedEvaluations;
        private readonly List<Experiment> _experimentHistory = [];
        private readonly TornadoAgent _controller;
        private readonly TimeSpan _decisionTimeout;
        private int _activeEvaluations;
        private static readonly JsonSerializerOptions DecisionJson = new() {
            PropertyNameCaseInsensitive = true,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            Converters = { new JsonStringEnumConverter(allowIntegerValues: false) }
        };

        public RegressionDescriptor? FeynmanInstance { get; set; }
        public int Seed { get; set; }
        public List<ResultDTO> Results { get; set; } = new List<ResultDTO>();

        public AgentSearch(bool useOLlama = true)
            : this(useOLlama ? CreateOllamaParameters() : new AgentParameter())
        {

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
                    tools: [(Func<int, double, int, int, MutatorType, int, Task<Experiment>>)RunSymbolicRegression, GetExperimentHistory, GetRemainingEvaluations, FinishSearch],
                    streaming: agentParameters.Streaming
                    );
            _agent.Options.Temperature = agentParameters.Temperature;
            if (agentParameters.Provider == LlmTornado.Code.LLmProviders.OpenRouter)
                _agent.Options.ServiceTier = agentParameters.ServiceTiers;
            else
                _agent.Options.ReasoningEffort = LlmTornado.Code.ChatReasoningEfforts.None;
            if (agentParameters.DecisionTimeout <= TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(agentParameters.DecisionTimeout));
            _decisionTimeout = agentParameters.DecisionTimeout;
            _controller = new TornadoAgent(client: api,
                model: new ChatModel(agentParameters.ModelName, agentParameters.Provider),
                outputSchema: typeof(ControllerReply),
                instructions: "Your objective is hyperparameter optimization by comparing multiple distinct configurations. " +
                    "This GP run is one experiment. Stop it when another configuration would be a better use of the remaining budget. " +
                    "Use SearchContext and PreviousExperiments: with few alternatives tested, learning from new configurations has high value. " +
                    "Small improvements alone do not justify continuing. Stagnation, declining diversity or growing trees without useful gains " +
                    "are reasons to switch. Explain why another block of generations is more valuable than trying an alternative when choosing Continue. " +
                    "Compare checkpoints: diversity is unique expression strings, not semantic diversity. " +
                    "Do not compare training NMSE directly with validation scores; UsedFallback is a request failure, not a search judgment. " +
                    "Return JSON with Action (Continue or Stop), Reason, and NextObservationInGenerations. " +
                    "Reason: keywords, not sentences. Report improvement numerically." +
                    "No speculation, repetition, or restating the next checkpoint; its timing is already a separate field. " +
                    "For Continue, choose a positive number of additional generations before observing again; for Stop, use null. " +
                    "Consider every positive integer delay, including non-round values. Do not default to multiples of five or ten or other human-preferred numbers; " +
                    "base timing on observed changes and budget, without preferring unusual numbers merely for their appearance. " +
                    "Choose the next checkpoint soon enough to reassess and leave meaningful budget for other configurations. " +
                    "A distant checkpoint can exhaust the budget before you can stop the run.",
                streaming: false);
            _controller.Options.Temperature = agentParameters.Temperature;
            if (agentParameters.Provider == LlmTornado.Code.LLmProviders.OpenRouter)
                _controller.Options.ServiceTier = agentParameters.ServiceTiers;
            else _controller.Options.ReasoningEffort = LlmTornado.Code.ChatReasoningEfforts.None;
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(agentParameters.Turns);
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
            _totalEvaluations = totalEvaluations;
            SearchStopReason = null;
            var sw = Stopwatch.StartNew();

            string searchRequest = $"Test and compare distinct hyperparameter configurations using {_totalEvaluations} shared evaluations across all configurations " +
                $"and at most {numConfigurations} configurations. Allocate work to learn from alternatives and adapt each next experiment.";

            // A summary alone is not an explicit search-stop decision.
            while (SearchStopReason is null && Results.Count < _numConfigurations && RemainingEvaluations >= 10 &&
                !Results.Any(result => result.BestFitness == 0))
            {
                await _agent.Run(searchRequest, maxTurns: _maxTurns);
                var conversationExit = _agent.LastRunExitReason;
                Console.WriteLine($"Configuration agent conversation ended: {conversationExit?.GetType().Name}: {conversationExit?.Message}");
                if (conversationExit?.Exception is Exception failure)
                    throw new InvalidOperationException("The configuration agent failed before the search completed.", failure);

                searchRequest = "Use the run feedback to test a different configuration. FinishSearch only if the evidence supports ending the overall comparison.\n" +
                    JsonSerializer.Serialize(new {
                        Budget = GetRemainingEvaluations(),
                        MaximumConfigurations = _numConfigurations,
                        NextRunEvaluationAllowance = RemainingEvaluations,
                        Experiments = _experimentHistory
                    }, DecisionJson);
            }

            sw.Stop();
            Console.WriteLine($"Agent run duration (including regression evaluations): {sw.Elapsed.TotalSeconds:F2} s");
            var exitReason = _agent.LastRunExitReason;
            string searchExit = SearchStopReason is not null ? $"agent_stop: {SearchStopReason}" : Results.Any(result => result.BestFitness == 0) ? "zero_validation_fitness"
                : Results.Count >= _numConfigurations ? "configuration_limit" : "insufficient_evaluation_budget";
            Console.WriteLine($"Search completed: {searchExit}; {Results.Count} configurations, {RemainingEvaluations} evaluations remaining.");
            if (Results.Count == 0)
                throw new InvalidOperationException(
                    $"AgentSearch evaluated no configurations. Exit reason: {exitReason?.GetType().Name}: {exitReason?.Message}",
                    exitReason?.Exception);
            return Results;
        }

        // ------------- Tool Calls --------------
        [Description(
            "Runs symbolic regression on the current problem. " +
            "Supply hyperparameters and firstObservationGeneration (a positive generation number for the first checkpoint). " +
            "The controller chooses subsequent checkpoint timing; C# enforces the total budget and configuration cap. " +
            "PopulationSize must not exceed the remaining shared evaluation budget. " +
            "Identical hyperparameters reuse a previous result (ReusedResult=true); choose a different configuration next. " +
            "Returns BestFitness (validation NMSE; lower is better, zero is ideal), Improvement, " +
            "IsNewBest, EvaluationsUsed, RemainingEvaluations, TerminationReason and ControlObservations " +
            "(ordered training checkpoints, controller decisions and reasons). Use this feedback to choose the next configuration."
        )]
        private Task<Experiment> RunSymbolicRegression(
            int populationSize, double mutationRate, int maxTreeDepth, int maxTreeLength, MutatorType mutatorType,
            int firstObservationGeneration)
        {
            return RunSymbolicRegression(new AlgorithmParameter {
                PopulationSize = populationSize,
                MutationRate = mutationRate,
                MaxTreeDepth = maxTreeDepth,
                MaxTreeLength = maxTreeLength,
                MutatorType = mutatorType
            }, firstObservationGeneration);
        }

        public async Task<Experiment> RunSymbolicRegression(AlgorithmParameter algorithmParameter, int firstObservationGeneration)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(firstObservationGeneration);
            await _trialLock.WaitAsync();
            try
            {
                if (SearchStopReason is not null)
                    throw new InvalidOperationException("The agent has finished the search.");
                algorithmParameter = algorithmParameter with { };
                var previous = _experimentHistory.FirstOrDefault(experiment => experiment.HyperParameters == algorithmParameter);
                if (previous is not null)
                    return previous with {
                        HyperParameters = previous.HyperParameters with { },
                        ReusedResult = true,
                        Improvement = 0,
                        IsNewBest = false,
                        RemainingEvaluations = RemainingEvaluations
                    };

                if (Results.Count >= _numConfigurations)
                    throw new InvalidOperationException("The configured number of regression runs has been reached.");
                int runBudget = RemainingEvaluations;
                if (runBudget < algorithmParameter.PopulationSize)
                    throw new InvalidOperationException("Insufficient remaining evaluation budget to initialize this population.");

                var observations = new List<RunControlObservation>();
                int lastChosenObservationDelay = firstObservationGeneration;
                async Task<RunDecision> DecideAsync(RunProgress progress, CancellationToken token)
                {
                    Volatile.Write(ref _activeEvaluations, progress.EvaluationsUsed);
                    var timer = Stopwatch.StartNew();
                    bool fallback = false;
                    RunDecision decision;
                    try
                    {
                        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token);
                        timeout.CancelAfter(_decisionTimeout);
                        decision = await RequestRunDecisionAsync(algorithmParameter, progress, observations.ToArray(), timeout.Token).WaitAsync(timeout.Token);
                        if (!Enum.IsDefined(decision.Action) || string.IsNullOrWhiteSpace(decision.Reason) ||
                            (decision.Action == RunAction.Continue && decision.NextObservationInGenerations is not > 0))
                            throw new JsonException("Invalid controller decision.");
                    }
                    catch (Exception ex) when (!token.IsCancellationRequested)
                    {
                        fallback = true;
                        decision = new RunDecision(RunAction.Continue, $"Controller unavailable: {ex.GetType().Name}; reusing the last agent-chosen observation delay.",
                            lastChosenObservationDelay);
                    }
                    if (decision.Action == RunAction.Continue)
                        lastChosenObservationDelay = decision.NextObservationInGenerations!.Value;
                    observations.Add(new RunControlObservation(progress, decision, timer.Elapsed.TotalMilliseconds, fallback));
                    Console.WriteLine($"Run decision at generation {progress.Generation} ({progress.EvaluationsUsed} evaluations): " +
                        $"{decision.Action} — {decision.Reason}" +
                        (decision.Action == RunAction.Continue ? $" Next observation in {decision.NextObservationInGenerations} generations." : ""));
                    return decision;
                }

                Console.WriteLine($"Running symbolic regression with the following parameters:{algorithmParameter}");
                int trialIndex = Results.Count;
                var result = await EvaluateConfiguration(FeynmanInstance!, algorithmParameter,
                    trialIndex, Seed + trialIndex, runBudget, DecideAsync, firstObservationGeneration);
                result.ControlObservations = observations;
                // Compute if improvement to previous best
                double previousBest = Results.Count > 0 ? Results.Min(r => r.BestFitness) : double.PositiveInfinity;
                // Positive means an error reduction. The first trial has no measured improvement.
                double improvement = double.IsFinite(previousBest) && double.IsFinite(result.BestFitness)
                    ? previousBest - result.BestFitness : 0;
                bool isNewBest = result.BestFitness < previousBest;

                Volatile.Write(ref _activeEvaluations, 0);
                Results.Add(result);
                Console.WriteLine($"Remaining Evaluations: {RemainingEvaluations}");
                var experiment = new Experiment {
                    TerminationReason = result.TerminationReason,
                    HyperParameters = result.Configuration,
                    BestFitness = result.BestFitness,
                    Improvement = improvement,
                    IsNewBest = isNewBest,
                    EvaluationsUsed = result.EvaluationsUsed,
                    RemainingEvaluations = RemainingEvaluations,
                    ControlObservations = observations.ToArray(),
                };
                _experimentHistory.Add(experiment);
                return experiment;

            }
            finally { Volatile.Write(ref _activeEvaluations, 0); _trialLock.Release(); }
        }

        public sealed record ControllerReply(
            [property: JsonRequired] string Action, [property: JsonRequired] string Reason,
            [property: JsonRequired] int? NextObservationInGenerations);

        [Description("Finish the entire hyperparameter search when further experiments are not worthwhile. Supply a short reason based on observed results.")]
        public async Task<string> FinishSearch(string reason)
        {
            await _trialLock.WaitAsync();
            try
            {
                if (Results.Count == 0)
                    throw new InvalidOperationException("Evaluate a configuration before recommending that the search finish.");
                ArgumentException.ThrowIfNullOrWhiteSpace(reason);
                SearchStopReason ??= reason.Trim();
                return $"Search finished: {SearchStopReason}";
            }
            finally { _trialLock.Release(); }
        }

        // A fresh, tool-free conversation avoids re-entering the active HPO agent.
        protected virtual async Task<RunDecision> RequestRunDecisionAsync(AlgorithmParameter configuration,
            RunProgress progress, IReadOnlyList<RunControlObservation> previousObservations, CancellationToken cancellationToken)
        {
            var response = await _controller.Run(JsonSerializer.Serialize(new {
                Configuration = configuration,
                Progress = progress,
                PreviousObservations = previousObservations,
                SearchContext = new {
                    Budget = GetRemainingEvaluations(),
                    MaximumConfigurations = _numConfigurations,
                    CurrentRunEvaluationsUsed = progress.EvaluationsUsed,
                    CurrentRunEvaluationsRemaining = progress.EvaluationsRemaining,
                    CurrentRunAllowance = progress.EvaluationsUsed + progress.EvaluationsRemaining
                },
                PreviousExperiments = _experimentHistory.Select(experiment => new {
                    experiment.HyperParameters, experiment.BestFitness, experiment.Improvement,
                    experiment.EvaluationsUsed, experiment.TerminationReason,
                    FirstObservation = experiment.ControlObservations.FirstOrDefault(),
                    LastObservation = experiment.ControlObservations.LastOrDefault()
                }).ToArray()
            }, DecisionJson),
                maxTurns: 1, cancellationToken: cancellationToken).WaitAsync(cancellationToken);
            string content = response.Messages.Last().Content ?? throw new JsonException("Controller returned no content.");
            var decision = JsonSerializer.Deserialize<ControllerReply>(content, DecisionJson);
            if (decision is null || !Enum.TryParse<RunAction>(decision.Action, true, out var action) || !Enum.IsDefined(action) || string.IsNullOrWhiteSpace(decision.Reason))
                throw new JsonException("Expected a Continue/Stop action and a nonempty reason.");
            if (action == RunAction.Continue && decision.NextObservationInGenerations is not > 0)
                throw new JsonException("Continue requires a positive NextObservationInGenerations.");
            return new RunDecision(action, decision.Reason, action == RunAction.Continue ? decision.NextObservationInGenerations : null);
        }

        [Description(
            "Review previous configurations, validation fitness and ControlObservations, including training trends " +
            "and controller reasons for continuing or stopping. Use this evidence to adapt the next configuration."
        )]
        public List<Experiment> GetExperimentHistory()
        {
            return _experimentHistory;
        }

        [Description(
            "Returns the total evaluation budget, actual candidate evaluations as UsedEvaluations, " +
            "and remaining budget, including the active run at its latest checkpoint, and completed configurations."
        )]
        public EvaluationBudget GetRemainingEvaluations()
        {
            return new EvaluationBudget {
                TotalEvaluations = _totalEvaluations,
                UsedEvaluations = UsedEvaluations,
                RemainingEvaluations = RemainingEvaluations,
                ConfigurationsCompleted = Results.Count
            };
        }
    }
}
