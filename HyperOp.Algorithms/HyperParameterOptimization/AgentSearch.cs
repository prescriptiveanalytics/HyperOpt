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
using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public record Experiment
    {
        // Zero-based trial number within the search; prompts may omit trials, so gaps mark omitted ones.
        public int Trial { get; init; }
        public required AlgorithmParameter HyperParameters { get; init; }
        [System.Text.Json.Serialization.JsonNumberHandling(System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals)]
        public required double BestFitness { get; init; }
        public required double Improvement { get; init; }
        public required bool IsNewBest { get; init; }
        public int GrantedEvaluations { get; init; }
        public int EvaluationsUsed { get; init; }
        public int RemainingEvaluations { get; init; }
        public string TerminationReason { get; init; } = "evaluation_limit";
        // The configurator's analysis when it proposed this trial; shown beside the trial in later requests.
        public string? Reason { get; init; }
        // Kept checkpoints (the first and the most recent ones) with the controller's decisions.
        public IReadOnlyList<CheckpointView> ControlObservations { get; init; } = [];
        public int CheckpointCount { get; init; }
        // One-based checkpoint of the last improvement of best training NMSE (1 when there was none after the first).
        public int LastImprovementCheckpoint { get; init; }
    }

    public record EvaluationBudget
    {
        public required int TotalEvaluations { get; init; }
        public required int UsedEvaluations { get; init; }
        public required int RemainingEvaluations { get; init; }
        public int ConfigurationsCompleted { get; init; }
    }

    // Both LLM roles are stateless: each decision is one request that carries the whole relevant state as text, and the
    // reply is JSON under a schema. The configurator proposes one trial per request until the shared pool is spent, like
    // every baseline (an option to finish early was removed on 4 November 2026: in all pilots the configurator finished
    // at 3-20% of the budget, often on misread facts); the controller decides at each checkpoint whether the running
    // trial continues.
    public class AgentSearch : HyperParameterOptimizationAlgorithm
    {
        private readonly TornadoAgent _agent;
        private int _totalEvaluations;
        private int _numConfigurations;
        private int _evaluationsPerConfiguration;
        private int MinimumRun => MinimumRunEvaluations(_evaluationsPerConfiguration);
        private int MaximumRun => MaximumRunEvaluations(_evaluationsPerConfiguration);
        // Serializes trials so concurrent callers cannot overspend the shared budget; per instance, never across jobs.
        private readonly SemaphoreSlim _trialLock = new(1, 1);
        // Problem facts given to both roles: input count and partition sizes, but no names or data.
        public sealed record ProblemFacts(int InputVariables, int TrainingRows, int ValidationRows);
        private ProblemFacts? _problem;
        // Consecutive invalid configurator replies (unparsable or out of range) before the search
        // ends as configurator_invalid. Each retry states the rejection and uses another sampling seed, so it is not the
        // same request again: with a fixed seed an identical request returns an identical reply.
        public const int MaxInvalidReplies = 3;
        // Earlier checkpoints of the current run included in a controller request.
        public const int ControllerObservationWindow = 5;
        // Smallest legal population: below this many remaining evaluations no further trial can run.
        private static readonly int MinimumPopulation = Convert.ToInt32(Util.ParameterSpace.Range(nameof(AlgorithmParameter.PopulationSize)).Minimum);
        private static readonly int MaximumPopulation = Convert.ToInt32(Util.ParameterSpace.Range(nameof(AlgorithmParameter.PopulationSize)).Maximum);
        // The shared pool is the budget; the configuration limit follows from it (MaximumSharedPoolTrials) and cannot end a
        // search before the pool is spent.
        private int ConfigurationLimit => MaximumSharedPoolTrials(_numConfigurations, _evaluationsPerConfiguration,
            Feedback == FeedbackChannels.None ? MinimumRun : Math.Min(MinimumRun, CheckpointEvaluations));
        private bool SearchCannotContinue => Results.Count >= ConfigurationLimit || RemainingEvaluations < MinimumPopulation;
        // A GP run that failed after it started; its evaluations cannot be charged reliably, so the job fails, as for
        // every other method, instead of the failure counting as an invalid proposal.
        private Exception? _trialFailure;
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
        // Controller requests: the same, with numbers to three significant digits (see ConfiguratorState).
        private static readonly JsonSerializerOptions PromptJson = new(DecisionJson) { Converters = { new RoundedDoubleConverter() } };
        // Prompts p8 and later: the same, with UniqueStructures named DistinctExpressionShare (9B read 1 as "zero diversity").
        private static readonly JsonSerializerOptions NeutralPromptJson = new(PromptJson) { PropertyNamingPolicy = new RenamedFields() };

        private sealed class RenamedFields : JsonNamingPolicy
        {
            public override string ConvertName(string name) =>
                name == nameof(DiversityFeedback.UniqueStructures) ? ControllerPrompts.DistinctExpressionShare : name;
        }

        // Information the run controller receives; None means runs are not observed (final result only).
        public FeedbackChannels Feedback { get; }
        // Controller checkpoints per per-configuration budget E.
        public int Checkpoints { get; }
        // Candidate evaluations between checkpoints, E/Checkpoints for the current search's E.
        private int CheckpointEvaluations => CheckpointSchedule.Interval(Math.Max(1, _totalEvaluations / Math.Max(1, _numConfigurations)), Checkpoints);
        // Whether the configurator sees checkpoint details; hidden leaves it final results while the controller still acts.
        public bool ConfiguratorSeesCheckpoints { get; }
        // Channels in the configurator's view: none in final and in the controller-only (hidden) condition.
        private FeedbackChannels ConfiguratorChannels => ConfiguratorSeesCheckpoints ? Feedback : FeedbackChannels.None;
        // Length-matched placebo for the channels (see AgentParameter.ChannelPlacebo).
        public bool ChannelPlacebo { get; }
        // Pilot switches (see AgentParameter).
        public bool ControllerComparison { get; }
        public string ControllerPrompt { get; }
        public int FirstControllerCheckpoint { get; }
        // True checkpoint progress of the job's completed runs: the donors of placebo channel values.
        private readonly List<RunProgress> _placeboDonors = [];
        public override string Variant => $"feedback={FeedbackChannelNames.Format(Feedback)}" +
            (Feedback == FeedbackChannels.None ? "" : "__" + CheckpointSchedule.Name(Checkpoints)) +
            (Feedback != FeedbackChannels.None && !ConfiguratorSeesCheckpoints ? "__checkpoints=hidden" : "") +
            (ChannelPlacebo ? "__placebo" : "") +
            (ControllerComparison ? "" : "__compare=off") + (FirstControllerCheckpoint == 1 ? "" : $"__first-decision=cp{FirstControllerCheckpoint}") +
            (ControllerPrompt == ControllerPrompts.Study || Controller != "llm" ? "" : $"__cprompt={ControllerPrompt}") +
            (Controller == "llm" ? "" : $"__controller={Controller}") +
            (Controller == "random" ? $"__stop-hazard={string.Join("+", StopHazard!.Select(FormatHazard))}" : "");

        public static readonly string[] Controllers = ["llm", "rule", "random"];
        // Who decides at the checkpoints (see AgentParameter.Controller); the stop hazard of the random controller.
        public string Controller { get; }
        public double[]? StopHazard { get; }
        public static string FormatHazard(double value) => value.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
        // No decision is requested when less than one checkpoint of the allowance remains (p8 and later, and the rule and
        // random controllers, so all three are asked at the same checkpoints); the run continues to its allowance.
        public const string NoDecisionReason = "no decision requested: less than one checkpoint left";
        private bool SkipsForcedDecisions => Controller != "llm" || ControllerPrompts.UsesNeutralFacts(ControllerPrompt);

        // Channel sections of a randomly chosen checkpoint of an earlier run, drawn with a SplitMix64 hash of (job seed, trial,
        // checkpoint), so the choice is the same in every process (HashCode.Combine is randomized per process). For the first
        // run, which has no earlier run, the channels stay frozen at its own first checkpoint (the only true channel values
        // the placebo shows). Donor values are made consistent with the current run where a mismatch would give the placebo
        // away: curve generations end at the current generation, sizes never exceed the run's limits, and the generalization
        // gap is taken against the run's own best training NMSE.
        public RunProgress WithPlaceboChannels(RunProgress progress, RunProgress firstOfRun, int trial, int checkpoint,
            AlgorithmParameter? configuration = null)
        {
            RunProgress donor = firstOfRun;
            if (_placeboDonors.Count > 0)
                donor = _placeboDonors[(int)((uint)TrialSeed(TrialSeed(Seed, trial), checkpoint) % (uint)_placeboDonors.Count)];
            var curve = donor.Curve;
            if (curve is { Generations.Length: > 0 })
            {
                int shift = progress.Generation - curve.Generations[^1];
                curve = curve with { Generations = curve.Generations.Select(g => Math.Max(0, g + shift)).ToArray() };
            }
            var size = donor.Size;
            if (size is not null && configuration is not null)
                size = size with {
                    MaxLength = Math.Min(size.MaxLength, configuration.MaxTreeLength),
                    MedianLength = Math.Min(size.MedianLength, configuration.MaxTreeLength),
                    MedianDepth = Math.Min(size.MedianDepth, configuration.MaxTreeDepth)
                };
            var generalization = donor.Generalization;
            if (generalization is not null && progress.BestTrainingNmse is double training)
                generalization = generalization with { Gap = generalization.ValidationNmseOfBest - training };
            return progress with { Curve = curve, Diversity = donor.Diversity, Size = size, Generalization = generalization };
        }

        // The configurator's view of a trial: without checkpoint details when they are hidden from it.
        private Experiment ForConfigurator(Experiment experiment) =>
            ConfiguratorSeesCheckpoints ? experiment : experiment with { ControlObservations = [], CheckpointCount = 0, LastImprovementCheckpoint = 0 };

        // Trials as the configurator's table shows them (see ConfiguratorState.ShownTrials).
        public IReadOnlyList<Experiment> ShownTrials() => ConfiguratorState.ShownTrials(_experimentHistory.Select(ForConfigurator).ToList());

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
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(agentParameters.Checkpoints);
            Feedback = agentParameters.Feedback;
            Checkpoints = agentParameters.Checkpoints;
            ConfiguratorSeesCheckpoints = agentParameters.ConfiguratorSeesCheckpoints;
            ChannelPlacebo = agentParameters.ChannelPlacebo;
            if (!ControllerPrompts.Names.Contains(agentParameters.ControllerPrompt))
                throw new ArgumentException($"Unknown controller prompt '{agentParameters.ControllerPrompt}'.");
            ControllerPrompt = agentParameters.ControllerPrompt;
            if (!Controllers.Contains(agentParameters.Controller))
                throw new ArgumentException($"Unknown controller '{agentParameters.Controller}'; expected one of {string.Join(", ", Controllers)}.");
            Controller = agentParameters.Controller;
            if (Controller != "llm" && agentParameters.Feedback == FeedbackChannels.None)
                throw new ArgumentException("A rule or random controller needs checkpoints (feedback other than final).");
            if (Controller == "random")
            {
                if (agentParameters.StopHazard is not { Length: > 0 } hazard || hazard.Any(p => !(p >= 0 && p <= 1)))
                    throw new ArgumentException("The random controller needs stop probabilities in [0, 1] per decision (--stop-hazard).");
                // Rounded as in the job name, so the name states the hazard that is used.
                StopHazard = hazard.Select(p => double.Parse(FormatHazard(p), System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            }
            else if (agentParameters.StopHazard is not null)
                throw new ArgumentException("--stop-hazard applies to the random controller only.");
            ControllerComparison = agentParameters.ControllerComparison;
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(agentParameters.FirstControllerCheckpoint);
            FirstControllerCheckpoint = agentParameters.FirstControllerCheckpoint;
            if (ChannelPlacebo && (Feedback & ~FeedbackChannels.Progress) == FeedbackChannels.None)
                throw new ArgumentException("A channel placebo needs at least one channel besides progress.");
            var assembly = Assembly.GetExecutingAssembly();
            using var stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.Resources.{agentParameters.SystemPrompt}.md")
                ?? throw new FileNotFoundException($"Embedded prompt '{agentParameters.SystemPrompt}' was not found.");
            using var reader = new StreamReader(stream);
            _agent = new TornadoAgent(
                    client: api,
                    model: new ChatModel(agentParameters.ModelName, agentParameters.Provider),
                    outputSchema: typeof(ConfiguratorReply),
                    instructions: reader.ReadToEnd().TrimEnd() + "\n\n" + ConfiguratorState.Legend(ConfiguratorChannels, withReason: true),
                    streaming: false
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
                instructions: ControllerPrompt == "standard"
                    ? ControllerInstructions(agentParameters.Feedback, agentParameters.Checkpoints, agentParameters.ControllerComparison,
                        agentParameters.FirstControllerCheckpoint)
                    : ControllerPrompts.Instructions(ControllerPrompt, agentParameters.Feedback, agentParameters.Checkpoints),
                streaming: false);
            _controller.Options.Temperature = agentParameters.Temperature;
            _controller.Options.MaxTokens = agentParameters.MaxOutputTokens;
            if (agentParameters.Provider == LlmTornado.Code.LLmProviders.OpenRouter)
                _controller.Options.ServiceTier = agentParameters.ServiceTiers;
            else _controller.Options.ReasoningEffort = LlmTornado.Code.ChatReasoningEfforts.None;
            _configuratorCalls = new LlmRole(this, "configurator");
            _controllerCalls = new LlmRole(this, "controller");
        }

        // One LLM role of this search: records its HTTP requests (timing, tokens, outcome).
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
                        root.GetProperty("choices")[0].GetProperty("message");
                        outcome = "response";
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

        // Instructions describe only the channels the controller actually receives.
        private static string ControllerInstructions(FeedbackChannels channels, int checkpoints, bool comparison = true, int firstCheckpoint = 1)
        {
            var text = new StringBuilder(
                "Your objective is hyperparameter optimization by comparing multiple distinct configurations. " +
                $"This GP run is one experiment, observed after every {100.0 / checkpoints:0.#}% of the per-configuration budget in candidate evaluations" +
                (firstCheckpoint > 1 ? $", first after {100.0 * firstCheckpoint / checkpoints:0.#}%" : "") + ". Decide whether to continue it until the next " +
                "checkpoint or stop it now. Continuing spends more of this run's allowance on it; stopping returns the rest to the shared " +
                "budget for other configurations. Lower NMSE is better. " +
                // Neutral wording: the instructions describe the decision and the feedback but do not favour either action.
                "SearchContext describes the rest of the search. " +
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
            text.Append("SearchContext gives this run's evaluations used and remaining allowance, and the evaluations left for other " +
                "configurations if this run stops now or if it continues to the end of its allowance, also as a number of runs of the " +
                "reference length. ");
            if (comparison)
                text.Append("AtSameEvaluations compares this run's BestTrainingNmse with the best training NMSE of earlier runs after the same " +
                    "number of evaluations (runs that got that far): their best and median, and ThisRunRank, this run's position among them " +
                    "and itself (1 = lowest NMSE, i.e. better than all earlier runs; RunsCompared is the number of runs ranked). ");
            text.Append("Return JSON with Action (Continue or Stop) and Reason. Reason: keywords, not sentences; report numbers. No speculation or repetition.");
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

        // The configurator's reply. Analysis and plan come first, so a model without a reasoning phase writes its reasoning
        // before the numbers.
        public sealed record ConfiguratorReply(
            [property: JsonRequired, Description("Keywords: what the evidence shows and why this proposal; at most 120 characters")] string Analysis,
            [property: JsonRequired, Description("Your plan for the rest of the search, kept or revised; at most 200 characters")] string Plan,
            [property: JsonRequired, Description("100 to 1000")] int PopulationSize,
            [property: JsonRequired, Description("0 to 0.5")] double MutationRate,
            [property: JsonRequired, Description("5 to 20")] int MaxTreeDepth,
            [property: JsonRequired, Description("10 to 100")] int MaxTreeLength,
            [property: JsonRequired, Description("NodeReplacement, Subtree, LocalPerturbation or Combined")] string MutatorType,
            [property: JsonRequired, Description("Evaluations for this run, within the next run's minimum and maximum")] int Evaluations);

        // Random controller: stop at the k-th decision of a trial with probability StopHazard[k-1] (the last value repeats),
        // drawn from a SplitMix64 hash of (job seed, trial, k), so the draws do not depend on the process or on other trials.
        public RunDecision RandomStopDecision(int trial, int decision)
        {
            double hazard = StopHazard![Math.Min(decision, StopHazard.Length) - 1];
            double draw = new Random(TrialSeed(TrialSeed(Seed ^ 0x5EED, trial), decision)).NextDouble();
            return draw < hazard
                ? new RunDecision(RunAction.Stop, $"random stop (p = {FormatHazard(hazard)})")
                : new RunDecision(RunAction.Continue, $"random continue (p = {FormatHazard(hazard)})");
        }

        // How the search ended, for the result file: exit reason, trials, evaluations left unspent, configurator requests,
        // rejected replies, proposals moved into the search space, and allowances moved into the run bounds.
        public sealed record SearchSummary(string Exit, int Trials, int RemainingEvaluations, int ConfiguratorRequests,
            int InvalidReplies, int ClampedProposals, int ClampedAllowances);
        // Set at the end of Execute, also when the search evaluated no configuration and the job fails.
        public SearchSummary? Summary { get; private set; }
        private int _clampedAllowances;

        // An enum member by its exact name (case-insensitive). Enum.TryParse also accepts numbers ("1") and comma-separated
        // flag combinations ("Subtree, LocalPerturbation" would become Combined).
        public static T? ExactEnum<T>(string? value) where T : struct, Enum
        {
            string? name = Enum.GetNames<T>().FirstOrDefault(n => string.Equals(n, value?.Trim(), StringComparison.OrdinalIgnoreCase));
            return name is null ? null : Enum.Parse<T>(name);
        }

        // The proposal moved into the search space and, for the population, into what the pool has left; each change is
        // described for the protocol and the next request.
        public static (AlgorithmParameter Configuration, List<string> Changes) ClampToSearchSpace(ConfiguratorReply reply, MutatorType mutator,
            int remainingEvaluations)
        {
            var changes = new List<string>();
            int Integer(string name, int value, int? cap = null)
            {
                var range = Util.ParameterSpace.Range(name);
                int maximum = Math.Min(Convert.ToInt32(range.Maximum), cap ?? int.MaxValue);
                int clamped = Math.Clamp(value, Convert.ToInt32(range.Minimum), Math.Max(Convert.ToInt32(range.Minimum), maximum));
                if (clamped != value) changes.Add($"{name} {value} -> {clamped} (allowed {range.Minimum} to {maximum})");
                return clamped;
            }
            var mutation = Util.ParameterSpace.Range(nameof(AlgorithmParameter.MutationRate));
            double rate = Math.Clamp(reply.MutationRate, Convert.ToDouble(mutation.Minimum), Convert.ToDouble(mutation.Maximum));
            if (rate != reply.MutationRate) changes.Add($"MutationRate {reply.MutationRate} -> {rate} (allowed {mutation.Minimum} to {mutation.Maximum})");
            var configuration = new AlgorithmParameter {
                PopulationSize = Integer(nameof(AlgorithmParameter.PopulationSize), reply.PopulationSize, remainingEvaluations),
                MutationRate = rate,
                MaxTreeDepth = Integer(nameof(AlgorithmParameter.MaxTreeDepth), reply.MaxTreeDepth),
                MaxTreeLength = Integer(nameof(AlgorithmParameter.MaxTreeLength), reply.MaxTreeLength),
                MutatorType = mutator
            };
            return (configuration, changes);
        }

        // The state message of one configurator request; adjustment describes how the previous trial's proposal was moved
        // into the search space.
        public string ConfiguratorRequest(string? plan, int planTrial, string? rejection, string? adjustment = null)
        {
            var text = new StringBuilder("FACTS (computed by the system; lower NMSE is better)\n");
            int remaining = RemainingEvaluations;
            text.Append($"Shared evaluations: {_totalEvaluations} total, {UsedEvaluations} used, {remaining} remaining " +
                $"({ConfiguratorState.Percent((double)remaining / _totalEvaluations)}; " +
                $"{ConfiguratorState.Number((double)remaining / _evaluationsPerConfiguration)} reference runs of {_evaluationsPerConfiguration}). " +
                "Evaluations still unused when the search ends are lost.\n");
            text.Append($"Next run: between {Math.Min(MinimumRun, remaining)} and {Math.Min(MaximumRun, remaining)} evaluations");
            if (remaining < MaximumPopulation) text.Append($"; PopulationSize at most {remaining}");
            text.Append(".\n");
            var finite = _experimentHistory.Where(t => double.IsFinite(t.BestFitness)).ToList();
            if (_experimentHistory.Count == 0)
                text.Append("Trials so far: 0.\n");
            else if (finite.Count == 0)
                text.Append($"Trials so far: {_experimentHistory.Count}; none reached a finite validation NMSE.\n");
            else
            {
                var best = finite.OrderBy(t => t.BestFitness).ThenBy(t => t.Trial).First();
                var h = best.HyperParameters;
                int lastNewBest = _experimentHistory.Where(t => t.IsNewBest).Select(t => t.Trial).DefaultIfEmpty(best.Trial).Max();
                text.Append($"Trials so far: {_experimentHistory.Count}. Best: trial {best.Trial}, valNMSE {ConfiguratorState.Number(best.BestFitness)} " +
                    $"(pop {h.PopulationSize}, mut {ConfiguratorState.Number(h.MutationRate, 2)}, depth {h.MaxTreeDepth}, len {h.MaxTreeLength}, {h.MutatorType}); " +
                    $"{_experimentHistory.Count - 1 - lastNewBest} trials since the last new best.\n");
            }
            text.Append($"Problem: {_problem!.InputVariables} input variables, {_problem.TrainingRows} training rows, {_problem.ValidationRows} validation rows.\n");
            if (_experimentHistory.Count > 0)
            {
                text.Append("\nTRIALS (in trial order)\n");
                var view = _experimentHistory.Select(ForConfigurator).ToList();
                text.Append(ConfiguratorState.Table(view, ConfiguratorChannels, withReason: true)).Append('\n');
                if (ConfiguratorChannels != FeedbackChannels.None)
                {
                    var best = view.Where(t => double.IsFinite(t.BestFitness)).OrderBy(t => t.BestFitness).ThenBy(t => t.Trial).FirstOrDefault();
                    var latest = view[^1];
                    if (best is not null && best.Trial != latest.Trial)
                        text.Append('\n').Append(ConfiguratorState.Detail(best, ConfiguratorChannels, "BEST")).Append('\n');
                    text.Append('\n').Append(ConfiguratorState.Detail(latest, ConfiguratorChannels, "LATEST")).Append('\n');
                }
            }
            if (!string.IsNullOrWhiteSpace(plan))
                text.Append($"\nYOUR PLAN (written before trial {planTrial}): {ConfiguratorState.Clip(plan, ConfiguratorState.PlanLength)}\n");
            if (adjustment is not null)
                text.Append($"\nPREVIOUS PROPOSAL ADJUSTED to the valid range (trial {_experimentHistory.Count - 1} ran with the adjusted values): " +
                    $"{ConfiguratorState.Clip(adjustment, 300)}\n");
            if (rejection is not null)
                text.Append($"\nPREVIOUS REPLY REJECTED: {ConfiguratorState.Clip(rejection, 300)} Reply again with valid values.\n");
            text.Append($"\nPropose trial {_experimentHistory.Count}.");
            return text.ToString();
        }

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            int totalEvaluations = checked(numConfigurations * evaluationsPerConfiguration);

            Results = new List<ResultDTO>();
            _experimentHistory.Clear();
            _placeboDonors.Clear();
            _clampedAllowances = 0;

            FeynmanInstance = feynmanInstance;
            Seed = seed;
            // Sampling seed per HPO repetition, so a repetition's LLM decisions can be replayed with the same model and server.
            _controller.Options.Seed = seed;

            _numConfigurations = numConfigurations;
            _totalEvaluations = totalEvaluations;
            _evaluationsPerConfiguration = evaluationsPerConfiguration;
            _trialFailure = null;
            lock (_llmCalls) _llmCalls.Clear();
            _searchClock.Restart();
            var sw = Stopwatch.StartNew();

            // The same problem facts in every feedback condition; variable names and data are never shown.
            _problem = new ProblemFacts(feynmanInstance.AllowedInputVariables.Length,
                feynmanInstance.PreparedData.TrainingRows.Count, feynmanInstance.PreparedData.ValidationRows.Count);

            string? plan = null, rejection = null, adjustment = null, lastAdjustment = null;
            int planTrial = 0, attempt = 0, requests = 0, invalidReplies = 0, clampedProposals = 0;
            string searchExit = "insufficient_evaluation_budget";
            Summary = null;
            while (!SearchCannotContinue)
            {
                if (attempt >= MaxInvalidReplies) { searchExit = "configurator_invalid"; break; }
                string request = ConfiguratorRequest(plan, planTrial, rejection, lastAdjustment);
                int trial = Results.Count;
                Log.Write("configurator_request", new { Trial = trial, Attempt = attempt, Input = request });
                // Retries change the seed as well as the request, so a deterministic model is not asked the same thing twice.
                _agent.Options.Seed = seed + attempt;
                LlmHttpTiming.Observer = _configuratorCalls;
                requests++;
                int recordsBefore;
                lock (_llmCalls) recordsBefore = _llmCalls.Count;
                var response = await _agent.Run(request, maxTurns: 1, onAgentRunnerEvent: _configuratorCalls.OnEvent);
                // A failed request (server down, HTTP error) fails the job, as before; only replies the model actually gave
                // count as invalid. Only records of this request count, so a request that never reached the server is not
                // mistaken for the previous trial's reply.
                LlmCallRecord? call;
                lock (_llmCalls) call = _llmCalls.Skip(recordsBefore).LastOrDefault(c => c.Role == "configurator");
                if (call is null || call.Outcome is not ("response" or "unparsed_response"))
                    throw new InvalidOperationException($"The configurator request failed: {call?.Outcome ?? "no request recorded"} {call?.Error}",
                        _agent.LastRunExitReason?.Exception);
                string? content = response.Messages.LastOrDefault()?.Content;
                Log.Write("configurator_reply", new { Trial = trial, Attempt = attempt, Content = content });
                ConfiguratorReply? reply = null;
                string? error = null;
                try
                {
                    if (string.IsNullOrWhiteSpace(content))
                        throw new JsonException(_agent.LastRunExitReason?.Exception is Exception failure
                            ? $"no reply ({failure.Message})" : "no reply");
                    reply = JsonSerializer.Deserialize<ConfiguratorReply>(content, DecisionJson)
                        ?? throw new JsonException("empty reply");
                }
                catch (JsonException parseError) { error = $"unreadable reply: {parseError.Message}"; }
                if (reply is not null)
                {
                    if (ExactEnum<MutatorType>(reply.MutatorType) is not MutatorType mutator)
                        error = $"unknown mutatorType '{reply.MutatorType}'.";
                    else if (!double.IsFinite(reply.MutationRate))
                        error = $"mutationRate must be a number, got {reply.MutationRate}.";
                    else
                    {
                        // Values outside the search space are moved to the nearest bound, like the allowance, and the next
                        // request says so; rejecting them had ended searches (4B proposed population 1500 three times).
                        // The population also never exceeds what the pool has left, so a remainder below the usual population
                        // is spent like every baseline spends it instead of ending the search as configurator_invalid.
                        var (configuration, changes) = ClampToSearchSpace(reply, mutator, RemainingEvaluations);
                        if (changes.Count > 0)
                        {
                            clampedProposals++;
                            adjustment = string.Join("; ", changes);
                            Log.Write("configuration_clamped", new { Trial = trial, Attempt = attempt, Changes = changes });
                        }
                        try
                        {
                            await RunSymbolicRegression(configuration, reply.Evaluations, reply.Analysis);
                        }
                        catch (Exception rejected) when (_trialFailure is null && rejected is not OperationCanceledException)
                        {
                            error = rejected.Message;
                        }
                        if (_trialFailure is not null)
                            throw new InvalidOperationException($"A GP trial failed after it had started: {_trialFailure.Message}", _trialFailure);
                    }
                }
                if (error is not null)
                {
                    Log.Write("configurator_rejected", new { Trial = trial, Attempt = attempt, Error = error });
                    Console.WriteLine($"Configurator reply rejected (attempt {attempt + 1} of {MaxInvalidReplies}): {error}");
                    rejection = error;
                    adjustment = null;
                    invalidReplies++;
                    attempt++;
                    continue;
                }
                // A valid reply resets the retries; its plan replaces the earlier one (rejected replies never change it).
                rejection = null;
                attempt = 0;
                lastAdjustment = adjustment;
                adjustment = null;
                if (!string.IsNullOrWhiteSpace(reply!.Plan)) { plan = reply.Plan.Trim(); planTrial = trial; }
                Log.Write("configurator_plan", new { Trial = trial, Plan = plan });
            }

            sw.Stop();
            Console.WriteLine($"Agent run duration (including regression evaluations): {sw.Elapsed.TotalSeconds:F2} s");
            if (searchExit != "configurator_invalid" && Results.Count >= ConfigurationLimit) searchExit = "configuration_limit";
            Console.WriteLine($"Search completed: {searchExit}; {Results.Count} configurations, {RemainingEvaluations} evaluations remaining.");
            Summary = new SearchSummary(searchExit, Results.Count, RemainingEvaluations, requests, invalidReplies, clampedProposals, _clampedAllowances);
            Log.Write("search_completed", new { Exit = searchExit, Trials = Results.Count, RemainingEvaluations, ConfiguratorRequests = requests,
                InvalidReplies = invalidReplies, ClampedProposals = clampedProposals, ClampedAllowances = _clampedAllowances });
            if (Results.Count == 0)
                throw new InvalidOperationException($"AgentSearch evaluated no configurations ({searchExit}).");
            return Results;
        }

        public Task<Experiment> RunSymbolicRegression(AlgorithmParameter algorithmParameter, int evaluations) =>
            RunSymbolicRegression(algorithmParameter, evaluations, reason: null);

        public async Task<Experiment> RunSymbolicRegression(AlgorithmParameter algorithmParameter, int evaluations, string? reason)
        {
            await _trialLock.WaitAsync();
            try
            {
                algorithmParameter = algorithmParameter with { };
                // Invalid proposals are rejected here, before any GP work, so they count as invalid replies and not as
                // failed trials (which end the job).
                System.ComponentModel.DataAnnotations.Validator.ValidateObject(algorithmParameter,
                    new System.ComponentModel.DataAnnotations.ValidationContext(algorithmParameter), validateAllProperties: true);
                if (!Enum.IsDefined(algorithmParameter.MutatorType))
                    throw new ArgumentException($"Unknown mutator type: {algorithmParameter.MutatorType}");
                // Repeated configurations are new trials with new GP seeds, charged like any other trial.
                if (SearchCannotContinue)
                    throw new InvalidOperationException("The shared evaluation budget is exhausted; the search is over.");
                if (RemainingEvaluations < algorithmParameter.PopulationSize)
                    throw new InvalidOperationException($"Insufficient remaining evaluation budget ({RemainingEvaluations}) to initialize this population.");
                // A request outside the per-run bounds receives the nearest bound, and never more than the pool has left; small
                // models often repeat the initial pool size. Requests below the population are clamped the same way (at least
                // E/10, which is above every population when E >= 1000), not rejected.
                int requestedEvaluations = evaluations;
                evaluations = Math.Max(BoundRunAllowance(evaluations, _evaluationsPerConfiguration, RemainingEvaluations), algorithmParameter.PopulationSize);
                if (evaluations != requestedEvaluations)
                {
                    _clampedAllowances++;
                    Log.Write("allowance_clamped", new { Trial = Results.Count, Requested = requestedEvaluations, Granted = evaluations });
                }

                var observations = new List<RunControlObservation>();
                var trueProgress = new List<RunProgress>();
                async Task<RunDecision> DecideAsync(RunProgress progress, CancellationToken token)
                {
                    Volatile.Write(ref _activeEvaluations, progress.EvaluationsUsed);
                    // Placebo: both roles see another run's channel values; the true ones go to the protocol only.
                    var truth = progress;
                    trueProgress.Add(truth);
                    if (ChannelPlacebo) progress = WithPlaceboChannels(truth, trueProgress[0], Results.Count, trueProgress.Count - 1, algorithmParameter);
                    // Less than one checkpoint of the allowance left: stopping would save less than a checkpoint, so no decision
                    // is requested and the run continues to its allowance. The checkpoint is still kept as an observation, so
                    // the configurator sees it (a short run would otherwise show no checkpoint at all); its reason marks it, and
                    // the protocol records it as checkpoint_skipped, not as a controller decision.
                    if (SkipsForcedDecisions && progress.EvaluationsRemaining < CheckpointEvaluations)
                    {
                        var kept = new RunControlObservation(progress, new RunDecision(RunAction.Continue, NoDecisionReason, CheckpointEvaluations), 0, false);
                        observations.Add(kept);
                        Log.Write("checkpoint_skipped", new { Trial = Results.Count, progress.EvaluationsUsed, progress.EvaluationsRemaining, Observation = kept,
                            TrueProgress = ChannelPlacebo ? truth : null });
                        return kept.Decision;
                    }
                    var timer = Stopwatch.StartNew();
                    bool fallback = false;
                    RunDecision decision;
                    if (Controller == "rule")
                        decision = RuleControlledSearch.StagnationDecision(progress, observations)
                            ?? new RunDecision(RunAction.Continue, "rule: improvement at least 1%");
                    else if (Controller == "random")
                        decision = RandomStopDecision(Results.Count, observations.Count + 1);
                    else
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
                        // The decision records only the exception type (it is shown to the configurator); the message goes here.
                        Log.Write("controller_failed", new { Trial = Results.Count, progress.EvaluationsUsed, Error = ex.GetType().Name, ex.Message });
                        fallback = true;
                        decision = new RunDecision(RunAction.Continue, $"Controller unavailable: {ex.GetType().Name}; continuing to the next checkpoint.");
                    }
                    // Checkpoints follow a fixed interval; the controller decides only whether the run continues.
                    if (decision.Action == RunAction.Continue)
                        decision = decision with { NextObservationInEvaluations = CheckpointEvaluations };
                    var observation = new RunControlObservation(progress, decision, timer.Elapsed.TotalMilliseconds, fallback);
                    observations.Add(observation);
                    Log.Write("checkpoint", new { Trial = Results.Count, Observation = observation, TrueProgress = ChannelPlacebo ? truth : null });
                    Console.WriteLine($"Run decision at generation {progress.Generation} ({progress.EvaluationsUsed} evaluations): " +
                        $"{decision.Action} — {decision.Reason}");
                    return decision;
                }

                Console.WriteLine($"Running symbolic regression with the following parameters:{algorithmParameter}, evaluations {evaluations}");
                int trialIndex = Results.Count;
                // Without feedback the run is not observed: it uses its full allowance and only the final result is reported.
                ResultDTO result;
                try
                {
                    result = Feedback == FeedbackChannels.None
                        ? await EvaluateConfiguration(FeynmanInstance!, algorithmParameter, trialIndex, TrialSeed(Seed, trialIndex), evaluations)
                        : await EvaluateConfiguration(FeynmanInstance!, algorithmParameter, trialIndex, TrialSeed(Seed, trialIndex), evaluations,
                            DecideAsync, CheckpointEvaluations * FirstControllerCheckpoint, channels: Feedback);
                }
                catch (Exception failure) when (failure is not OperationCanceledException)
                {
                    _trialFailure = failure;
                    Log.Write("trial_failed", new { Trial = trialIndex, EvaluationsUsed = Volatile.Read(ref _activeEvaluations), failure.Message });
                    throw;
                }
                result.ControlObservations = observations;
                double previousBest = Results.Count > 0 ? Results.Min(r => r.BestFitness) : double.PositiveInfinity;
                // Positive means an error reduction. The first trial has no measured improvement.
                double improvement = double.IsFinite(previousBest) && double.IsFinite(result.BestFitness)
                    ? previousBest - result.BestFitness : 0;
                bool isNewBest = result.BestFitness < previousBest;

                Volatile.Write(ref _activeEvaluations, 0);
                Results.Add(result);
                // A completed run's true checkpoints become placebo donors for later runs.
                _placeboDonors.AddRange(trueProgress);
                Console.WriteLine($"Remaining Evaluations: {RemainingEvaluations}");
                var experiment = new Experiment {
                    Trial = trialIndex,
                    TerminationReason = result.TerminationReason,
                    HyperParameters = result.Configuration,
                    BestFitness = result.BestFitness,
                    Improvement = improvement,
                    IsNewBest = isNewBest,
                    GrantedEvaluations = evaluations,
                    EvaluationsUsed = result.EvaluationsUsed,
                    RemainingEvaluations = RemainingEvaluations,
                    Reason = reason is null ? null : ConfiguratorState.Clip(reason, ConfiguratorState.ReasonLength),
                    CheckpointCount = observations.Count,
                    ControlObservations = SummarizeObservations(observations),
                    LastImprovementCheckpoint = 1 + observations.Select((o, i) => (o, i))
                        .Where(x => x.o.Progress.RecentImprovement is > 0).Select(x => x.i).DefaultIfEmpty(0).Max(),
                };
                _experimentHistory.Add(experiment);
                Log.Write("agent_trial", new { Trial = trialIndex, result.GPSeed, RequestedEvaluations = requestedEvaluations,
                    GrantedEvaluations = evaluations, Experiment = experiment });
                return ForConfigurator(experiment);
            }
            finally { Volatile.Write(ref _activeEvaluations, 0); _trialLock.Release(); }
        }

        // A run's first checkpoint and its most recent ones, at most ConfiguratorState.DetailCheckpoints.
        private static CheckpointView[] SummarizeObservations(IReadOnlyList<RunControlObservation> observations) =>
            (observations.Count <= ConfiguratorState.DetailCheckpoints ? observations
                : [observations[0], .. observations.TakeLast(ConfiguratorState.DetailCheckpoints - 1)]).Select(CheckpointView.From).ToArray();

        public sealed record ControllerReply(
            [property: JsonRequired] string Action, [property: JsonRequired] string Reason);

        // The budget as the decision affects it, in fields that do not overlap. Earlier prompts showed the pool's total, used
        // and remaining evaluations (used including this run) beside the run's own counts, and models misread them, e.g.
        // "budget remaining below run allowance". Stopping returns the run's unused allowance to the other configurations.
        public record ControllerBudgetView(int ThisRunEvaluationsUsed, int ThisRunAllowanceRemaining,
            int EvaluationsForOtherConfigurationsIfStoppedNow, int EvaluationsForOtherConfigurationsIfContinuedToAllowance,
            double ReferenceRunsIfStoppedNow, double ReferenceRunsIfContinuedToAllowance);

        // Prompts p8 and later: the budget as facts without "if stopped now / if continued" counterfactuals.
        public record NeutralBudgetView(int ThisRunEvaluationsUsed, int ThisRunAllowance, int SharedPoolAfterThisRunsAllowance,
            double SharedPoolAfterThisRunsAllowanceInReferenceRuns);

        public NeutralBudgetView NeutralControllerBudget(RunProgress progress)
        {
            var budget = ControllerBudget(progress);
            return new NeutralBudgetView(progress.EvaluationsUsed, progress.EvaluationsUsed + progress.EvaluationsRemaining,
                budget.EvaluationsForOtherConfigurationsIfContinuedToAllowance, budget.ReferenceRunsIfContinuedToAllowance);
        }

        public ControllerBudgetView ControllerBudget(RunProgress progress)
        {
            int ifStopped = Math.Max(0, _totalEvaluations - Results.Sum(result => result.EvaluationsUsed) - progress.EvaluationsUsed);
            return new ControllerBudgetView(progress.EvaluationsUsed, progress.EvaluationsRemaining, ifStopped,
                Math.Max(0, ifStopped - progress.EvaluationsRemaining),
                Math.Round((double)ifStopped / _evaluationsPerConfiguration, 1),
                Math.Round((double)Math.Max(0, ifStopped - progress.EvaluationsRemaining) / _evaluationsPerConfiguration, 1));
        }

        // This run's best training NMSE against all completed runs at the same evaluation count, computed as for the rule
        // baseline's median rule; earlier runs report only final validation NMSE otherwise, which a model easily compares
        // with an early training value. Null before any completed run got this far.
        // ThisRunRank: 1 + the number of earlier runs with lower training NMSE (1 = best); RunsCompared counts this run too.
        public record SameEvaluationsComparison(int Evaluations, int EarlierRuns, double Best, double Median, int ThisRunRank, int RunsCompared);

        public SameEvaluationsComparison? CompareAtSameEvaluations(RunProgress progress)
        {
            var references = RuleControlledSearch.TrainingReferences(Results, progress.EvaluationsUsed);
            if (references.Length == 0) return null;
            return new SameEvaluationsComparison(progress.EvaluationsUsed, references.Length, references[0], RuleControlledSearch.Median(references),
                1 + (progress.BestTrainingNmse is double current ? references.Count(reference => reference < current) : references.Length),
                references.Length + 1);
        }

        // The controller's request: this run as JSON with numbers to three significant digits. Earlier trials enter only through
        // the comparison at the same evaluation count; with a table of their final validation NMSE, the 4B controller compared
        // early training NMSE with it and stopped most runs at the first checkpoint (pilot, 4 November 2026).
        public string ControllerRequest(AlgorithmParameter configuration, RunProgress progress, IReadOnlyList<RunControlObservation> previousObservations)
        {
            bool neutral = ControllerPrompts.UsesNeutralFacts(ControllerPrompt);
            string run = JsonSerializer.Serialize(new {
                Problem = _problem,
                Configuration = configuration,
                Progress = progress,
                // The most recent checkpoints only, so the prompt does not grow with the run's length.
                PreviousObservations = previousObservations.TakeLast(ControllerObservationWindow).Select(CheckpointView.From).ToArray(),
                EarlierObservationsOmitted = Math.Max(0, previousObservations.Count - ControllerObservationWindow),
                SearchContext = neutral ? NeutralControllerBudget(progress) : (object)ControllerBudget(progress),
                AtSameEvaluations = ControllerComparison && ControllerPrompts.ShowsComparison(ControllerPrompt) ? CompareAtSameEvaluations(progress) : null
            }, neutral ? NeutralPromptJson : PromptJson);
            return run;
        }

        // A fresh conversation per decision; the controller is stateless like the configurator.
        protected virtual async Task<RunDecision> RequestRunDecisionAsync(AlgorithmParameter configuration,
            RunProgress progress, IReadOnlyList<RunControlObservation> previousObservations, CancellationToken cancellationToken)
        {
            string request = ControllerRequest(configuration, progress, previousObservations);
            Log.Write("controller_request", new { Trial = Results.Count, Input = request });
            // Scoped to this call: the configurator's own requests set their role again.
            LlmHttpTiming.Observer = _controllerCalls;
            var response = await _controller.Run(request, maxTurns: 1, onAgentRunnerEvent: _controllerCalls.OnEvent,
                cancellationToken: cancellationToken).WaitAsync(cancellationToken);
            var last = response.Messages.Last();
            Log.Write("controller_response", new { Trial = Results.Count, last.Content, Reasoning = last.ReasoningContent ?? last.Reasoning });
            string content = last.Content ?? throw new JsonException("Controller returned no content.");
            var decision = JsonSerializer.Deserialize<ControllerReply>(content, DecisionJson);
            if (decision is null || ExactEnum<RunAction>(decision.Action) is not RunAction action || string.IsNullOrWhiteSpace(decision.Reason))
                throw new JsonException("Expected a Continue/Stop action and a nonempty reason.");
            return new RunDecision(action, decision.Reason);
        }

        public EvaluationBudget GetRemainingEvaluations()
        {
            return new EvaluationBudget {
                TotalEvaluations = _totalEvaluations,
                UsedEvaluations = UsedEvaluations,
                RemainingEvaluations = RemainingEvaluations,
                ConfigurationsCompleted = Results.Count
            };
        }

        // Writes doubles to three significant digits (integers from 1000 on); reading is unchanged.
        private sealed class RoundedDoubleConverter : JsonConverter<double>
        {
            public override double Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
                reader.TokenType == JsonTokenType.String ? double.Parse(reader.GetString()!, CultureInfo.InvariantCulture) : reader.GetDouble();

            public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
            {
                if (!double.IsFinite(value)) { writer.WriteStringValue(value.ToString(CultureInfo.InvariantCulture)); return; }
                writer.WriteRawValue(ConfiguratorState.Number(value));
            }
        }
    }
}
