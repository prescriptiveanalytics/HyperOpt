using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Execution;
using HEAL.HeuristicLib.Problems;
using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.AgentComponents;
using HyperOp.Algorithms.Logging;
using LlmTornado;
using LlmTornado.Agents;
using LlmTornado.Agents.DataModels;
using LlmTornado.Chat;
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
        // Ordered training checkpoints and controller explanations for choosing the next configuration.
        public IReadOnlyList<CheckpointView> ControlObservations { get; init; } = [];
        public int CheckpointCount { get; init; }
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
        // Serializes this job's tool calls so parallel tool calls cannot overspend the shared budget.
        // It is per instance and never spans jobs, so it does not limit GPU use across concurrent agent jobs.
        private readonly SemaphoreSlim _trialLock = new(1, 1);
        // Problem facts given to both roles: input count and partition sizes, but no names or data.
        public sealed record ProblemFacts(int InputVariables, int TrainingRows, int ValidationRows);
        private ProblemFacts? _problem;
        // Consecutive configurator conversations without a new trial before the search ends as idle.
        public const int MaxIdleConversations = 3;
        // Earlier checkpoints of the current run included in a controller request.
        public const int ControllerObservationWindow = 5;
        // Smallest legal population: below this many remaining evaluations no further trial can run.
        private const int MinimumPopulation = 10;
        private bool SearchCannotContinue => Results.Count >= _numConfigurations || RemainingEvaluations < MinimumPopulation;
        // Set when the tool ends the configurator conversation because no further trial can run.
        private bool _searchExhausted;
        // Consecutive proposals rejected by the tool (invalid values or allowance); the conversation ends at the limit.
        public const int MaxConsecutiveRejections = 5;
        private int _consecutiveRejections;
        private bool _conversationRejected;
        private int UsedEvaluations => Results.Sum(result => result.EvaluationsUsed) + Volatile.Read(ref _activeEvaluations);
        private int RemainingEvaluations => _totalEvaluations - UsedEvaluations;
        private readonly List<Experiment> _experimentHistory = [];
        private readonly TornadoAgent _controller;
        private readonly TimeSpan _decisionTimeout;
        private int _activeEvaluations;
        private readonly List<LlmCallRecord> _llmCalls = [];
        private readonly Stopwatch _searchClock = new();
        private readonly LlmRole _configuratorCalls;
        private readonly LlmRole _controllerCalls;
        public override IReadOnlyList<LlmCallRecord> LlmCalls { get { lock (_llmCalls) return _llmCalls.ToArray(); } }
        private static readonly JsonSerializerOptions DecisionJson = new() {
            PropertyNameCaseInsensitive = true,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            // Channels that are not requested stay out of the prompts entirely.
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter(allowIntegerValues: false) }
        };

        // Information the run controller receives; None means runs are not observed (final result only).
        public FeedbackChannels Feedback { get; }
        // Generations between controller checkpoints.
        public int CheckpointInterval { get; }
        // Whether the configurator sees checkpoint details; hidden leaves it final results while the controller still acts.
        public bool ConfiguratorSeesCheckpoints { get; }
        public override string Variant => $"feedback={FeedbackChannelNames.Format(Feedback)}" +
            (Feedback == FeedbackChannels.None ? "" : $"__interval={CheckpointInterval}") +
            (Feedback != FeedbackChannels.None && !ConfiguratorSeesCheckpoints ? "__checkpoints=hidden" : "");

        // The configurator's view of a trial: without checkpoint details when they are hidden from it.
        private Experiment ForConfigurator(Experiment experiment) =>
            ConfiguratorSeesCheckpoints ? experiment : experiment with { ControlObservations = [], CheckpointCount = 0 };

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
                ModelName = OllamaModels.StudyModels[^1],
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
            _agent.Options.MaxTokens = agentParameters.MaxOutputTokens;
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
                instructions: ControllerInstructions(agentParameters.Feedback, agentParameters.CheckpointInterval),
                streaming: false);
            _controller.Options.Temperature = agentParameters.Temperature;
            _controller.Options.MaxTokens = agentParameters.MaxOutputTokens;
            if (agentParameters.Provider == LlmTornado.Code.LLmProviders.OpenRouter)
                _controller.Options.ServiceTier = agentParameters.ServiceTiers;
            else _controller.Options.ReasoningEffort = LlmTornado.Code.ChatReasoningEfforts.None;
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(agentParameters.Turns);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(agentParameters.CheckpointInterval);
            _maxTurns = agentParameters.Turns;
            Feedback = agentParameters.Feedback;
            CheckpointInterval = agentParameters.CheckpointInterval;
            ConfiguratorSeesCheckpoints = agentParameters.ConfiguratorSeesCheckpoints;
            _configuratorCalls = new LlmRole(this, "configurator");
            _controllerCalls = new LlmRole(this, "controller");
        }

        // One LLM role of this search: records its HTTP requests (timing, tokens, outcome) and logs its tool use.
        private sealed class LlmRole(AgentSearch owner, string role) : ILlmHttpObserver
        {
            public double Now() => owner._searchClock.Elapsed.TotalMilliseconds;

            public void Completed(double startMs, int? statusCode, string? body, Exception? error)
            {
                string outcome = error is not null ? $"failed:{error.GetType().Name}" : $"http_{statusCode}";
                int? input = null, output = null, reasoning = null;
                if (error is null && statusCode is >= 200 and < 300 && body is not null)
                {
                    try
                    {
                        using var document = JsonDocument.Parse(body);
                        var root = document.RootElement;
                        if (root.TryGetProperty("usage", out var usage) && usage.ValueKind == JsonValueKind.Object)
                        {
                            input = usage.TryGetProperty("prompt_tokens", out var p) ? p.GetInt32() : null;
                            output = usage.TryGetProperty("completion_tokens", out var c) ? c.GetInt32() : null;
                            reasoning = usage.TryGetProperty("completion_tokens_details", out var d) && d.ValueKind == JsonValueKind.Object &&
                                d.TryGetProperty("reasoning_tokens", out var r) ? r.GetInt32() : null;
                        }
                        var message = root.GetProperty("choices")[0].GetProperty("message");
                        outcome = message.TryGetProperty("tool_calls", out var calls) && calls.ValueKind == JsonValueKind.Array && calls.GetArrayLength() > 0
                            ? "tool_call:" + string.Join(",", calls.EnumerateArray().Select(call => call.GetProperty("function").GetProperty("name").GetString()))
                            : "response";
                    }
                    catch (Exception parseError) when (parseError is JsonException or KeyNotFoundException or InvalidOperationException or IndexOutOfRangeException)
                    {
                        outcome = "unparsed_response";
                    }
                }
                var record = new LlmCallRecord(role, startMs, Now() - startMs, outcome, input, output, reasoning,
                    error?.Message ?? (statusCode is >= 200 and < 300 ? null : Truncate(body, 2000)));
                lock (owner._llmCalls) owner._llmCalls.Add(record);
                owner.Log.Write("llm_call", record);
            }

            public ValueTask OnEvent(AgentRunnerEvents runnerEvent)
            {
                switch (runnerEvent)
                {
                    case AgentRunnerToolInvokedEvent invoked:
                        owner.Log.Write("llm_tool_call", new { Role = role, invoked.ToolCalled?.Name, invoked.ToolCalled?.Arguments });
                        break;
                    case AgentRunnerToolCompletedEvent completed:
                        owner.Log.Write("llm_tool_result", new { Role = role, completed.ToolCall?.Name,
                            Result = Truncate(completed.ToolResult?.Content?.ToString(), 4000) });
                        break;
                    case AgentRunnerErrorEvent error:
                        owner.Log.Write("llm_error", new { Role = role, error.ErrorMessage, Exception = error.Exception?.ToString() });
                        break;
                    case AgentRunnerCompletedEvent or AgentRunnerCancelledEvent or AgentRunnerMaxTurnsReachedEvent or AgentRunnerMaxTokensReachedEvent:
                        owner.Log.Write("llm_runner_event", new { Role = role, runnerEvent.EventType });
                        break;
                }
                return ValueTask.CompletedTask;
            }
        }

        private static string? Truncate(string? text, int length) =>
            text is null || text.Length <= length ? text : text[..length] + "…";

        private static object DescribeConversation(Conversation conversation) => conversation.Messages.Select(message => new {
            message.Role, message.Content, Reasoning = message.ReasoningContent ?? message.Reasoning,
            ToolCalls = message.ToolCalls?.Select(call => new { call.Id, call.FunctionCall?.Name, call.FunctionCall?.Arguments }),
            message.ToolCallId
        }).ToArray();

        // Instructions describe only the channels the controller actually receives.
        private static string ControllerInstructions(FeedbackChannels channels, int interval)
        {
            var text = new System.Text.StringBuilder(
                "Your objective is hyperparameter optimization by comparing multiple distinct configurations. " +
                $"This GP run is one experiment, observed every {interval} generations. Decide whether to continue it until the next " +
                "checkpoint or stop it now, so that its remaining evaluations go to other configurations. " +
                "Use SearchContext and PreviousExperiments: with few alternatives tested, learning from new configurations has high value. " +
                "Small improvements alone do not justify continuing. " +
                "Progress reports training NMSE (lower is better): BestTrainingNmse and RecentImprovement, its reduction since the previous checkpoint. ");
            if (channels.HasFlag(FeedbackChannels.Curve))
                text.Append("Curve lists best and median training NMSE at recent generations; a flat curve means stagnation. ");
            if (channels.HasFlag(FeedbackChannels.Diversity))
                text.Append("Diversity: UniqueStructures is the fraction of distinct expressions; PhenotypicDiversity is one minus the mean " +
                    "squared correlation between individuals' outputs (0 when all behave alike up to offset and scale); InvalidOutputs is the " +
                    "fraction with non-finite outputs; TrainingNmseSpread is the interquartile range of training NMSE over its median. " +
                    "Low diversity means the population has converged. ");
            if (channels.HasFlag(FeedbackChannels.Size))
                text.Append("Size: median and maximum expression length, median depth, the fractions of trees at the length and depth limits, " +
                    "and the change of median length since the previous checkpoint; many trees at a limit mean the limit constrains the search, " +
                    "growing trees without improvement indicate bloat. ");
            if (channels.HasFlag(FeedbackChannels.Generalization))
                text.Append("Generalization: validation NMSE of the current best expression and its gap to the best training NMSE; " +
                    "a growing gap indicates overfitting. ");
            text.Append("PreviousExperiments report validation NMSE; do not compare it directly with training NMSE. " +

                "Return JSON with Action (Continue or Stop) and Reason. Reason: keywords, not sentences; report numbers. No speculation or repetition.");
            return text.ToString();
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
            // Sampling seed per HPO repetition, so a repetition's LLM decisions can be replayed with the same model and server.
            _agent.Options.Seed = seed;
            _controller.Options.Seed = seed;

            _numConfigurations = numConfigurations;
            _totalEvaluations = totalEvaluations;
            SearchStopReason = null;
            _searchExhausted = false;
            lock (_llmCalls) _llmCalls.Clear();
            _searchClock.Restart();
            LlmHttpTiming.Observer = _configuratorCalls;
            var sw = Stopwatch.StartNew();

            // The same problem facts in every feedback condition; variable names and data are never shown.
            _problem = new ProblemFacts(feynmanInstance.AllowedInputVariables.Length,
                feynmanInstance.PreparedData.TrainingRows.Count, feynmanInstance.PreparedData.ValidationRows.Count);
            string searchRequest = $"Test and compare distinct hyperparameter configurations using {_totalEvaluations} shared evaluations across all configurations " +
                $"and at most {numConfigurations} configurations. Allocate work to learn from alternatives and adapt each next experiment.\n" +
                "Problem: " + JsonSerializer.Serialize(_problem, DecisionJson);

            // A summary alone is not an explicit search-stop decision.
            int conversation = 0, idleConversations = 0;
            while (SearchStopReason is null && !SearchCannotContinue &&
                !Results.Any(result => result.BestFitness == 0) && idleConversations < MaxIdleConversations)
            {
                int trialsBefore = Results.Count;
                _consecutiveRejections = 0;
                _conversationRejected = false;
                Log.Write("configurator_request", new { Conversation = conversation, Input = searchRequest });
                Conversation? messages = null;
                try { messages = await _agent.Run(searchRequest, maxTurns: _maxTurns, onAgentRunnerEvent: _configuratorCalls.OnEvent); }
                catch (OperationCanceledException) when (_searchExhausted || _conversationRejected) { }
                var conversationExit = _agent.LastRunExitReason;
                Log.Write("configurator_conversation", new {
                    Conversation = conversation++, Exit = _searchExhausted ? "SearchExhausted" : _conversationRejected ? "TooManyRejectedProposals"
                        : conversationExit?.GetType().Name, conversationExit?.Message,
                    Messages = messages is null ? null : DescribeConversation(messages)
                });
                Console.WriteLine($"Configuration agent conversation ended: {(_searchExhausted ? "search exhausted" : _conversationRejected
                    ? $"{MaxConsecutiveRejections} consecutive rejected proposals" : conversationExit?.GetType().Name)}: {conversationExit?.Message}");
                if (!_searchExhausted && !_conversationRejected && conversationExit?.Exception is Exception failure)
                    throw new InvalidOperationException("The configuration agent failed before the search completed.", failure);
                // A conversation that neither ran a trial nor finished the search would otherwise repeat indefinitely.
                idleConversations = Results.Count > trialsBefore ? 0 : idleConversations + 1;

                searchRequest = "Use the run feedback to test a different configuration. FinishSearch only if the evidence supports ending the overall comparison.\n" +
                    JsonSerializer.Serialize(new {
                        Problem = _problem,
                        Budget = GetRemainingEvaluations(),
                        MaximumConfigurations = _numConfigurations,
                        NextRunEvaluationAllowance = RemainingEvaluations,
                        Experiments = _experimentHistory.Select(ForConfigurator)
                    }, DecisionJson);
            }

            sw.Stop();
            Console.WriteLine($"Agent run duration (including regression evaluations): {sw.Elapsed.TotalSeconds:F2} s");
            var exitReason = _agent.LastRunExitReason;
            string searchExit = SearchStopReason is not null ? $"agent_stop: {SearchStopReason}" : Results.Any(result => result.BestFitness == 0) ? "zero_validation_fitness"
                : idleConversations >= MaxIdleConversations ? "configurator_idle"
                : Results.Count >= _numConfigurations ? "configuration_limit" : "insufficient_evaluation_budget";
            Console.WriteLine($"Search completed: {searchExit}; {Results.Count} configurations, {RemainingEvaluations} evaluations remaining.");
            Log.Write("search_completed", new { Exit = searchExit, Trials = Results.Count, RemainingEvaluations, Conversations = conversation });
            if (Results.Count == 0)
                throw new InvalidOperationException(
                    $"AgentSearch evaluated no configurations. Exit reason: {exitReason?.GetType().Name}: {exitReason?.Message}",
                    exitReason?.Exception);
            return Results;
        }

        // ------------- Tool Calls --------------
        [Description(
            "Runs symbolic regression on the current problem with the given hyperparameters. " +
            "evaluations is the maximum number of candidate evaluations for this run: at least populationSize and at most the remaining shared budget. " +
            "If runs are observed, a controller may stop the run early and return its unused evaluations to the shared budget. " +
            "Repeating identical hyperparameters runs a new trial with a new GP seed and consumes budget. " +
            "Returns BestFitness (validation NMSE; lower is better, zero is ideal), Improvement, " +
            "IsNewBest, EvaluationsUsed, RemainingEvaluations, TerminationReason and ControlObservations " +
            "(checkpoints of observed runs: training values, controller decisions and reasons). Use this feedback to choose the next configuration."
        )]
        private Task<Experiment> RunSymbolicRegression(
            int populationSize, double mutationRate, int maxTreeDepth, int maxTreeLength, MutatorType mutatorType,
            int evaluations)
        {
            return RunSymbolicRegression(new AlgorithmParameter {
                PopulationSize = populationSize,
                MutationRate = mutationRate,
                MaxTreeDepth = maxTreeDepth,
                MaxTreeLength = maxTreeLength,
                MutatorType = mutatorType
            }, evaluations);
        }

        public async Task<Experiment> RunSymbolicRegression(AlgorithmParameter algorithmParameter, int evaluations)
        {
            await _trialLock.WaitAsync();
            try
            {
                if (SearchStopReason is not null)
                    throw new InvalidOperationException("The agent has finished the search.");
                algorithmParameter = algorithmParameter with { };
                // Repeated configurations are new trials with new GP seeds, charged like any other trial.
                if (SearchCannotContinue)
                {
                    EndExhaustedConversation();
                    throw new InvalidOperationException("The trial cap or the shared evaluation budget is exhausted; the search is over.");
                }
                if (RemainingEvaluations < algorithmParameter.PopulationSize)
                    throw new InvalidOperationException($"Insufficient remaining evaluation budget ({RemainingEvaluations}) to initialize this population.");
                if (evaluations < algorithmParameter.PopulationSize)
                    throw new ArgumentOutOfRangeException(nameof(evaluations),
                        $"evaluations must be at least populationSize ({algorithmParameter.PopulationSize}); {RemainingEvaluations} evaluations remain.");
                // A request above the remaining pool receives the remainder; small models often repeat the initial pool size.
                int requestedEvaluations = evaluations;
                evaluations = Math.Min(evaluations, RemainingEvaluations);
                if (evaluations < requestedEvaluations)
                    Log.Write("allowance_clamped", new { Trial = Results.Count, Requested = requestedEvaluations, Granted = evaluations });

                var observations = new List<RunControlObservation>();
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
                        if (!Enum.IsDefined(decision.Action) || string.IsNullOrWhiteSpace(decision.Reason))
                            throw new JsonException("Invalid controller decision.");
                    }
                    catch (Exception ex) when (!token.IsCancellationRequested)
                    {
                        fallback = true;
                        decision = new RunDecision(RunAction.Continue, $"Controller unavailable: {ex.GetType().Name}; continuing to the next checkpoint.");
                    }
                    // Checkpoints follow a fixed interval; the controller decides only whether the run continues.
                    if (decision.Action == RunAction.Continue)
                        decision = decision with { NextObservationInGenerations = CheckpointInterval };
                    var observation = new RunControlObservation(progress, decision, timer.Elapsed.TotalMilliseconds, fallback);
                    observations.Add(observation);
                    Log.Write("checkpoint", new { Trial = Results.Count, Observation = observation });
                    Console.WriteLine($"Run decision at generation {progress.Generation} ({progress.EvaluationsUsed} evaluations): " +
                        $"{decision.Action} — {decision.Reason}");
                    return decision;
                }

                Console.WriteLine($"Running symbolic regression with the following parameters:{algorithmParameter}, evaluations {evaluations}");
                int trialIndex = Results.Count;
                // Without feedback the run is not observed: it uses its full allowance and only the final result is reported.
                var result = Feedback == FeedbackChannels.None
                    ? await EvaluateConfiguration(FeynmanInstance!, algorithmParameter, trialIndex, TrialSeed(Seed, trialIndex), evaluations)
                    : await EvaluateConfiguration(FeynmanInstance!, algorithmParameter, trialIndex, TrialSeed(Seed, trialIndex), evaluations,
                        DecideAsync, CheckpointInterval, channels: Feedback);
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
                    CheckpointCount = observations.Count,
                    ControlObservations = SummarizeObservations(observations),
                };
                _experimentHistory.Add(experiment);
                Log.Write("agent_trial", new { Trial = trialIndex, result.GPSeed, RequestedEvaluations = requestedEvaluations,
                    GrantedEvaluations = evaluations, Experiment = experiment });
                _consecutiveRejections = 0;
                if (SearchCannotContinue) EndExhaustedConversation();
                return ForConfigurator(experiment);

            }
            catch (Exception) when (!_searchExhausted)
            {
                // Without a limit, a model that keeps proposing invalid trials spends its whole turn budget on rejections.
                if (++_consecutiveRejections >= MaxConsecutiveRejections && !_conversationRejected)
                {
                    _conversationRejected = true;
                    Log.Write("conversation_rejected", new { Trial = Results.Count, Rejections = _consecutiveRejections });
                    _agent.Cancel();
                }
                throw;
            }
            finally { Volatile.Write(ref _activeEvaluations, 0); _trialLock.Release(); }
        }

        // The configurator sees the first and the last few checkpoints of a run, so its history stays bounded.
        private static CheckpointView[] SummarizeObservations(IReadOnlyList<RunControlObservation> observations) =>
            (observations.Count <= ControllerObservationWindow ? observations
                : [observations[0], .. observations.TakeLast(ControllerObservationWindow - 1)]).Select(CheckpointView.From).ToArray();

        // When no further trial can run, the conversation has nothing left to decide; without this, a model can keep
        // proposing trials that are rejected until its turn limit, which only spends inference time.
        private void EndExhaustedConversation()
        {
            if (_searchExhausted) return;
            _searchExhausted = true;
            Log.Write("search_exhausted", new { Trials = Results.Count, RemainingEvaluations });
            _agent.Cancel();
        }

        // A run's length is part of what the agent chose, so its nominee is reassessed at the work that trial actually used.
        public override int AssessmentEvaluations(ResultDTO recommendation, int evaluationsPerConfiguration) => recommendation.EvaluationsUsed;

        public sealed record ControllerReply(
            [property: JsonRequired] string Action, [property: JsonRequired] string Reason);

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
            string request = JsonSerializer.Serialize(new {
                Problem = _problem,
                Configuration = configuration,
                Progress = progress,
                // The most recent checkpoints only, so the prompt does not grow with the run's length.
                PreviousObservations = previousObservations.TakeLast(ControllerObservationWindow).Select(CheckpointView.From).ToArray(),
                EarlierObservationsOmitted = Math.Max(0, previousObservations.Count - ControllerObservationWindow),
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
            }, DecisionJson);
            Log.Write("controller_request", new { Trial = Results.Count, Input = request });
            // Scoped to this call: the configurator's own requests resume its role afterwards.
            LlmHttpTiming.Observer = _controllerCalls;
            var response = await _controller.Run(request, maxTurns: 1, onAgentRunnerEvent: _controllerCalls.OnEvent,
                cancellationToken: cancellationToken).WaitAsync(cancellationToken);
            var last = response.Messages.Last();
            Log.Write("controller_response", new { Trial = Results.Count, last.Content, Reasoning = last.ReasoningContent ?? last.Reasoning });
            string content = last.Content ?? throw new JsonException("Controller returned no content.");
            var decision = JsonSerializer.Deserialize<ControllerReply>(content, DecisionJson);
            if (decision is null || !Enum.TryParse<RunAction>(decision.Action, true, out var action) || !Enum.IsDefined(action) || string.IsNullOrWhiteSpace(decision.Reason))
                throw new JsonException("Expected a Continue/Stop action and a nonempty reason.");
            return new RunDecision(action, decision.Reason);
        }

        [Description(
            "Review previous configurations, validation fitness and ControlObservations, including training trends " +
            "and controller reasons for continuing or stopping. Use this evidence to adapt the next configuration."
        )]
        public List<Experiment> GetExperimentHistory()
        {
            return _experimentHistory.Select(ForConfigurator).ToList();
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
