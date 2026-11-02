using HyperOp.Algorithms;
using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.Feynman.Instances;
using HyperOp.Algorithms.HyperParameterOptimization;
using HyperOp.Algorithms.HyperParameterOptimization.AgentComponents;
using HyperOp.Algorithms.Logging;
using HyperOp.Experiments;
using System.Diagnostics;

var parsedArgs = ParseArgs(args);
const int instanceSeed = 42;
// Development setting; freeze the final assessment count after the pilot.
const int assessmentRepetitions = 5;
Dictionary<string, RegressionDescriptor> relevantBenchmarksets = new Dictionary<string, RegressionDescriptor>(StringComparer.OrdinalIgnoreCase)
{
        { "I.6.20",    new Feynman2(instanceSeed) },
        { "I.9.18",    new Feynman5(instanceSeed) },
        { "I.15.3.x",  new Feynman17(instanceSeed) },
        { "I.30.5",    new Feynman31(instanceSeed) },
        { "I.32.17",   new Feynman33(instanceSeed) },
        { "I.41.16",   new Feynman44(instanceSeed) },
        { "I.48.20",   new Feynman50(instanceSeed) },
        { "II.6.15a",  new Feynman56(instanceSeed) },
        { "II.11.27",  new Feynman64(instanceSeed) },
        { "II.11.28",  new Feynman65(instanceSeed) },
        { "II.35.21",  new Feynman81(instanceSeed) },
        { "III.9.52",  new Feynman90(instanceSeed) },
        { "III.10.19", new Feynman91(instanceSeed) }
};

// Defaults for F5 / launching without arguments. Command-line options override these.
const string defaultAlgorithm = nameof(IRaceSearch);
const string defaultInstance = "I.6.20";
int repetitions = 1;
int repetitionStart = 1;
List<(int, int)> defaultBudget = [(100, 1_000)];
List<(int, int)> budgetVariations = new List<(int, int)>()
{
    (10, 1_000_000),
    (100, 100_000),
    (1000, 10_000),
    (10000, 1_000),
    (100000, 100),
};

if (parsedArgs.TryGetValue("help", out _) || parsedArgs.TryGetValue("h", out _))
{
    PrintUsage();
    return;
}

if (!parsedArgs.ContainsKey("budget-sweep"))
    budgetVariations = defaultBudget;

if (parsedArgs.TryGetValue("repetitions", out var repetitionsValue))
    repetitions = ParsePositiveInt("repetitions", repetitionsValue);
if (parsedArgs.TryGetValue("repetition-start", out var repetitionStartValue))
    repetitionStart = ParsePositiveInt("repetition-start", repetitionStartValue);
if (parsedArgs.ContainsKey("configurations") || parsedArgs.ContainsKey("evaluations"))
{
    if (!parsedArgs.TryGetValue("configurations", out var configurationsValue) ||
        !parsedArgs.TryGetValue("evaluations", out var evaluationsValue))
        throw new ArgumentException("Specify both --configurations and --evaluations to override the budget sweep.");
    budgetVariations = [(ParsePositiveInt("configurations", configurationsValue), ParsePositiveInt("evaluations", evaluationsValue))];
}

var algorithmFilter = parsedArgs.TryGetValue("algorithm", out var algValue) ? algValue : defaultAlgorithm;
// Accept one or more comma-separated instance keys; fall back to the single default
var instanceFilters = parsedArgs.TryGetValue("instance", out var instValue)
    ? instValue.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList()
    : new List<string> { defaultInstance };
var outputRoot = parsedArgs.TryGetValue("output", out var outputValue) ? outputValue : "./results";
// --run-id remains an optional campaign subdirectory, rather than a per-process directory.
var runOutputDirectory = parsedArgs.TryGetValue("run-id", out var runIdValue)
    ? Path.Combine(outputRoot, Uri.EscapeDataString(runIdValue)) : outputRoot;

Directory.CreateDirectory(runOutputDirectory);

if (parsedArgs.ContainsKey("aggregate"))
{
    try
    {
        int count = ExperimentResultStore.Aggregate(runOutputDirectory);
        Console.WriteLine($"Aggregated {count} completed experiments into {Path.Combine(runOutputDirectory, "all_results.json")}");
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"Aggregation failed: {ex.Message}");
        Environment.ExitCode = 1;
    }
    return;
}

await ExecuteExperiments(runOutputDirectory, algorithmFilter, instanceFilters);

async Task ExecuteExperiments(string outputDirectory, string algorithmNameFilter, List<string> instanceNameFilters)
{
    var algorithmNames = GetAvailableAlgorithmNames().Where(name =>
        string.Equals(algorithmNameFilter, "all", StringComparison.OrdinalIgnoreCase) ||
        NormalizeKey(name) == NormalizeKey(algorithmNameFilter)).ToList();
    if (algorithmNames.Count == 0)
        throw new ArgumentException($"Unknown algorithm '{algorithmNameFilter}'. Available: {string.Join(", ", GetAvailableAlgorithmNames())}");
    // An Ollama URL is meaningless for OpenRouter, so it selects the local provider unless --provider says otherwise.
    // HYPEROP_PROVIDER and HYPEROP_OLLAMA_URL supply container defaults (docker/agent.compose.yml).
    string? ollamaUrl = parsedArgs.TryGetValue("ollama-url", out var urlValue) ? urlValue : Environment.GetEnvironmentVariable("HYPEROP_OLLAMA_URL");
    string provider = (parsedArgs.TryGetValue("provider", out var providerValue) ? providerValue
        : Environment.GetEnvironmentVariable("HYPEROP_PROVIDER") ?? "ollama").ToLowerInvariant();
    var agentParameters = provider switch {
        "ollama" => AgentSearch.CreateOllamaParameters(),
        "openrouter" => new AgentParameter(),
        _ => throw new ArgumentException("--provider must be 'ollama' or 'openrouter'.")
    };
    if (parsedArgs.TryGetValue("model", out var model)) agentParameters.ModelName = model;
    if (parsedArgs.ContainsKey("ollama-url") && provider != "ollama")
        throw new ArgumentException("--ollama-url requires --provider ollama.");
    if (provider == "ollama" && ollamaUrl is not null) agentParameters.OllamaBaseUrl = ollamaUrl;
    // Feedback condition of the agent's run controller and the checkpoint interval shared with the rule baseline.
    if (parsedArgs.TryGetValue("feedback", out var feedbackValue))
        agentParameters.Feedback = FeedbackChannelNames.Parse(feedbackValue);
    int checkpointInterval = parsedArgs.TryGetValue("checkpoint-interval", out var intervalValue)
        ? ParsePositiveInt("checkpoint-interval", intervalValue) : agentParameters.CheckpointInterval;
    agentParameters.CheckpointInterval = checkpointInterval;
    if (parsedArgs.TryGetValue("configurator-checkpoints", out var checkpointsValue))
        agentParameters.ConfiguratorSeesCheckpoints = checkpointsValue.ToLowerInvariant() switch {
            "shown" => true, "hidden" => false,
            _ => throw new ArgumentException("--configurator-checkpoints must be 'shown' or 'hidden'.")
        };
    if (parsedArgs.TryGetValue("decision-timeout", out var timeoutValue))
        agentParameters.DecisionTimeout = TimeSpan.FromSeconds(ParsePositiveInt("decision-timeout", timeoutValue));
    if (string.IsNullOrWhiteSpace(agentParameters.ModelName)) throw new ArgumentException("--model cannot be empty.");
    // Fail before any GP work when the requested local model is not available, and pin its exact weights.
    OllamaModelInfo? modelInfo = provider == "ollama" && algorithmNames.Contains(nameof(AgentSearch))
        ? await OllamaModels.ResolveAsync(agentParameters.OllamaBaseUrl, agentParameters.ModelName) : null;
    if (modelInfo is not null)
        Console.WriteLine($"Ollama {modelInfo.ServerVersion}: {modelInfo.Name} ({modelInfo.ParameterSize}, {modelInfo.QuantizationLevel}, {modelInfo.Digest[..12]})");
    List<HyperParameterOptimizationAlgorithm> hyperOpAlgs = algorithmNames.Select(name => name switch {
        nameof(AgentSearch) => new AgentSearch(agentParameters),
        nameof(IRaceSearch) => (HyperParameterOptimizationAlgorithm)new IRaceSearch(),
        nameof(DefaultConfigurationSearch) => new DefaultConfigurationSearch(),
        nameof(RuleControlledSearch) => new RuleControlledSearch(checkpointInterval),
        nameof(GridSearch) => new GridSearch(),
        nameof(HillClimbingSearch) => new HillClimbingSearch(),
        nameof(RandomSearch) => new RandomSearch(),
        _ => throw new ArgumentException($"Unknown algorithm '{name}'.")
    }).ToList();

    // Resolve which benchmark instances to run
    IEnumerable<KeyValuePair<string, RegressionDescriptor>> selectedInstances;
    if (instanceNameFilters.Count == 0)
        throw new ArgumentException("Specify at least one benchmark instance.");
    selectedInstances = instanceNameFilters
            .Select(filter =>
            {
                if (relevantBenchmarksets.TryGetValue(filter, out var descriptor))
                    return (key: relevantBenchmarksets.Keys.First(key => key.Equals(filter, StringComparison.OrdinalIgnoreCase)), descriptor);
                throw new ArgumentException($"Unknown benchmark instance '{filter}'. Available: {string.Join(", ", relevantBenchmarksets.Keys)}");
            })
            .DistinctBy(x => x.key)
            .Select(x => new KeyValuePair<string, RegressionDescriptor>(x.key, x.descriptor));

    // Freeze selected instances before any HPO call, then share them across all repetitions.
    var frozenInstances = selectedInstances.Select(entry =>
        (entry.Key, Descriptor: entry.Value, Data: entry.Value.PreparedData)).ToList();

    foreach (var algorithm in hyperOpAlgs)
    {
        foreach (var (benchmarkKey, regressionDescriptor, preparedData) in frozenInstances)
        {
            foreach (int repetition in Enumerable.Range(repetitionStart, repetitions))
            {
                foreach (var entry in budgetVariations)
                {
                    int numConfigurations = entry.Item1;
                    int evaluationsPerConfiguration = entry.Item2;
                    Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Executing {algorithm.GetType().Name} on {benchmarkKey} " +
                        $"(Repetition {repetition}, Configurations: {numConfigurations}, " +
                        (algorithm is AgentSearch or RuleControlledSearch
                            ? $"Shared evaluations: {checked(numConfigurations * evaluationsPerConfiguration)})"
                            : $"Evaluations per Configuration: {evaluationsPerConfiguration})"));

                    JsonlProtocolLog? protocol = null;
                    try
                    {
                        string? modelName = algorithm is AgentSearch ? agentParameters.ModelName : null;
                        string jobName = ExperimentResultStore.JobName(algorithm.GetType().Name, benchmarkKey,
                            repetition, numConfigurations, evaluationsPerConfiguration, assessmentRepetitions, modelName, algorithm.Variant);
                        string resultFilePath = Path.Combine(outputDirectory, jobName + ".result.json");
                        int hpoSeed = checked(42 + 100 * repetition);

                        if (File.Exists(resultFilePath))
                        {
                            Console.WriteLine("  → Result already exists. Loading from cache.");
                            var cached = ExperimentResultStore.Load(resultFilePath);
                            if (cached.RecommendationRule != algorithm.RecommendationRule ||
                                cached.TrainingRows != preparedData.TrainingRows.Count ||
                                cached.ValidationRows != preparedData.ValidationRows.Count ||
                                cached.TestRows != preparedData.TestRows.Count)
                                throw new InvalidDataException("Cached result predates the verified benchmark/evaluation protocol. Use a fresh run directory.");
                            continue;
                        }

                        // Each attempt has its own protocol; an interrupted attempt's protocol stays beside the new one.
                        string protocolFile = $"{jobName}__attempt={Guid.NewGuid().ToString("N")[..8]}.protocol.jsonl";
                        protocol = new JsonlProtocolLog(Path.Combine(outputDirectory, protocolFile));
                        algorithm.Log = protocol;
                        await algorithm.PrepareAsync();
                        var jobClock = Stopwatch.StartNew();
                        var timing = new TimingBreakdown();
                        if (algorithm is AgentSearch && modelInfo is not null)
                        {
                            timing.ModelWarmupMs = await OllamaModels.WarmUpAsync(agentParameters.OllamaBaseUrl, modelInfo.Name);
                            timing.ContextLength = await OllamaModels.RuntimeContextLengthAsync(agentParameters.OllamaBaseUrl, modelInfo.Name);
                        }
                        protocol.Write("job_started", new {
                            Algorithm = algorithm.GetType().Name, Instance = benchmarkKey, Repetition = repetition, Seed = hpoSeed,
                            T = numConfigurations, E = evaluationsPerConfiguration, K = assessmentRepetitions,
                            Model = modelName, Provider = algorithm is AgentSearch ? agentParameters.Provider.ToString() : null,
                            ModelInfo = algorithm is AgentSearch ? modelInfo : null,
                            Agent = algorithm is AgentSearch ? new { agentParameters.Temperature, agentParameters.Turns, DecisionTimeoutS = agentParameters.DecisionTimeout.TotalSeconds,
                                Feedback = FeedbackChannelNames.Format(agentParameters.Feedback), agentParameters.CheckpointInterval,
                                agentParameters.ConfiguratorSeesCheckpoints } : null,
                            algorithm.Variant,
                            timing.ModelWarmupMs, Host = new { Environment.MachineName, Environment.ProcessorCount, OS = Environment.OSVersion.ToString(),
                                Runtime = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription },
                            Arguments = args
                        });

                        var phaseClock = Stopwatch.StartNew();
                        List<ResultDTO> algRes = await algorithm.Execute(
                            regressionDescriptor,
                            hpoSeed,
                            numConfigurations,
                            evaluationsPerConfiguration
                            );
                        timing.TuningMs = phaseClock.Elapsed.TotalMilliseconds;

                        phaseClock.Restart();
                        var bestConfig = HyperParameterOptimizationAlgorithm.BestValidationTrial(algRes);
                        // Freeze both selections before any test data are scored.
                        var recommendation = algorithm.NominateConfiguration(algRes);
                        var nominatedConfiguration = recommendation.Configuration with { };
                        int assessmentEvaluations = algorithm.AssessmentEvaluations(recommendation, evaluationsPerConfiguration);
                        bestConfig.EvaluateTest(preparedData);
                        timing.SelectionAndTestScoringMs = phaseClock.Elapsed.TotalMilliseconds;
                        protocol.Write("selection", new { SelectedTrialIndex = algRes.IndexOf(bestConfig), bestConfig.TestFitness,
                            RecommendedConfiguration = nominatedConfiguration, RecommendedConfigurationIndex = recommendation.ConfigurationIndex,
                            algorithm.RecommendationRule, AssessmentEvaluations = assessmentEvaluations });

                        phaseClock.Restart();
                        var assessmentResults = await algorithm.AssessConfiguration(regressionDescriptor,
                            nominatedConfiguration, algRes, hpoSeed, assessmentRepetitions, assessmentEvaluations);
                        timing.AssessmentMs = phaseClock.Elapsed.TotalMilliseconds;
                        FillTiming(timing, algRes, assessmentResults, algorithm.LlmCalls);
                        timing.TotalJobMs = jobClock.Elapsed.TotalMilliseconds;
                        var allConfigsResult = new AggregatedResultDTO
                        {
                            AlgorithmName = algorithm.GetType().Name,
                            ModelName = modelName,
                            ModelProvider = algorithm is AgentSearch ? agentParameters.Provider.ToString() : null,
                            Variant = algorithm.Variant,
                            ModelInfo = algorithm is AgentSearch ? modelInfo : null,
                            ProtocolFile = protocolFile,
                            Timing = timing,
                            AssessmentEvaluationsPerRun = assessmentEvaluations,
                            InstanceName = benchmarkKey,
                            Repetition = repetition,
                            Seed = hpoSeed,
                            NumConfigurations = numConfigurations,
                            EvaluationsPerConfiguration = evaluationsPerConfiguration,
                            TotalConfigurationsEvaluated = algRes.Count,
                            ConfigurationResults = algRes,
                            TrainingRows = preparedData.TrainingRows.Count,
                            ValidationRows = preparedData.ValidationRows.Count,
                            TestRows = preparedData.TestRows.Count,
                            SelectedTrialIndex = algRes.IndexOf(bestConfig),
                            RecommendedConfiguration = nominatedConfiguration,
                            RecommendedConfigurationIndex = recommendation.ConfigurationIndex,
                            RecommendationRule = algorithm.RecommendationRule,
                            ConfigurationAssessmentResults = assessmentResults
                        };

                        ExperimentResultStore.Save(outputDirectory, allConfigsResult);
                        protocol.Write("job_completed", new { ResultFile = Path.GetFileName(resultFilePath), Timing = timing });

                        Console.WriteLine($"Evaluated {algRes.Count} configurations. Best validation NMSE: {bestConfig.BestFitness:G6}; " +
                            $"selected expression test NMSE: {bestConfig.TestFitness:G6}; " +
                            $"{assessmentResults.Count} fresh-seed assessment runs saved separately.");
                        Console.WriteLine($"Time: job {timing.TotalJobMs / 1000:F1} s; tuning {timing.TuningMs / 1000:F1} s = GP {timing.TuningGpMs / 1000:F1} s " +
                            $"+ trial overhead {timing.TuningTrialOverheadMs / 1000:F1} s + controller {timing.ControllerInferenceMs / 1000:F1} s " +
                            $"+ optimizer {timing.OptimizerMs / 1000:F1} s (configurator requests {timing.ConfiguratorInferenceMs / 1000:F1} s); " +
                            $"LLM share of tuning {timing.LlmShareOfTuning:P1}; assessment {timing.AssessmentMs / 1000:F1} s.");
                        if (timing.RequestsNearContextLimit > 0)
                            Console.WriteLine($"Warning: {timing.RequestsNearContextLimit} model requests filled at least 90% of the {timing.ContextLength}-token context; " +
                                "the server truncates longer prompts silently.");
                    }
                    catch (Exception ex)
                    {
                        Environment.ExitCode = 1;
                        Console.WriteLine($"Error: {ex.Message}");
                        protocol?.Write("job_failed", new { Error = ex.ToString() });
                    }
                    finally
                    {
                        algorithm.Log = NullProtocolLog.Instance;
                        protocol?.Dispose();
                    }
                }
            }
        }
    }

    Console.WriteLine($"Experiment complete. Results saved to {outputDirectory}");
}

// Decomposes tuning wall time; see TimingBreakdown for the identities between the fields.
static void FillTiming(TimingBreakdown timing, IReadOnlyList<ResultDTO> tuning, IReadOnlyList<ResultDTO> assessment,
    IReadOnlyList<LlmCallRecord> llmCalls)
{
    var observations = tuning.SelectMany(trial => trial.ControlObservations ?? []).ToArray();
    timing.TuningTrialWallMs = tuning.Sum(trial => trial.TrialWallTimeMs ?? trial.ExecutionTimeMs);
    timing.TuningGpMs = tuning.Sum(trial => trial.ExecutionTimeMs);
    timing.ControllerInferenceMs = observations.Sum(observation => observation.InferenceTimeMs);
    timing.ControllerRequests = observations.Length;
    timing.ControllerFallbacks = observations.Count(observation => observation.UsedFallback);
    timing.TuningTrialOverheadMs = timing.TuningTrialWallMs - timing.TuningGpMs - timing.ControllerInferenceMs;
    timing.OptimizerMs = timing.TuningMs - timing.TuningTrialWallMs;
    var configurator = llmCalls.Where(call => call.Role == "configurator").ToArray();
    timing.ConfiguratorRequests = configurator.Length;
    timing.ConfiguratorInferenceMs = configurator.Sum(call => call.DurationMs);
    timing.InputTokens = llmCalls.Sum(call => call.InputTokens ?? 0);
    timing.OutputTokens = llmCalls.Sum(call => call.OutputTokens ?? 0);
    timing.ReasoningTokens = llmCalls.Sum(call => call.ReasoningTokens ?? 0);
    if (timing.ContextLength is int context)
        timing.RequestsNearContextLimit = llmCalls.Count(call => call.InputTokens >= 0.9 * context);
    timing.TuningTrialCpuMs = tuning.Sum(trial => trial.TrialCpuTimeMs ?? 0);
    timing.AssessmentGpMs = assessment.Sum(run => run.ExecutionTimeMs);
}

Dictionary<string, string> ParseArgs(string[] cmdArgs)
{
    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    string[] known = ["help", "h", "algorithm", "instance", "output", "run-id", "repetitions", "repetition-start",
        "configurations", "evaluations", "budget-sweep", "model", "provider", "ollama-url", "decision-timeout", "feedback",
        "checkpoint-interval", "configurator-checkpoints", "aggregate"];

    for (int i = 0; i < cmdArgs.Length; i++)
    {
        var arg = cmdArgs[i];
        if (!arg.StartsWith("--")) throw new ArgumentException($"Unexpected argument: {arg}");

        var key = arg[2..];
        if (!known.Contains(key, StringComparer.OrdinalIgnoreCase) || result.ContainsKey(key))
            throw new ArgumentException($"Unknown or duplicate option: --{key}");
        if (i + 1 < cmdArgs.Length && !cmdArgs[i + 1].StartsWith("--"))
        {
            result[key] = cmdArgs[i + 1];
            i++;
        }
        else
        {
            result[key] = "true";
        }
    }

    return result;
}

int ParsePositiveInt(string name, string value)
{
    if (!int.TryParse(value, out var number) || number <= 0)
        throw new ArgumentException($"--{name} must be a positive integer.");
    return number;
}

IEnumerable<string> GetAvailableAlgorithmNames()
{
    return new[] { nameof(AgentSearch), nameof(DefaultConfigurationSearch), nameof(GridSearch), nameof(HillClimbingSearch),
        nameof(IRaceSearch), nameof(RandomSearch), nameof(RuleControlledSearch) };
}

string NormalizeKey(string value)
{
    var cleaned = value.Replace("_", string.Empty).Replace("-", string.Empty).Replace(" ", string.Empty);
    return cleaned.EndsWith("Search", StringComparison.OrdinalIgnoreCase)
        ? cleaned[..^"Search".Length].ToLowerInvariant()
        : cleaned.ToLowerInvariant();
}

void PrintUsage()
{
    Console.WriteLine("Usage:");
    Console.WriteLine("  HyperOp.Experiments [--algorithm <name>] [--instance <key>[,<key>...]] [--output <dir>] [--run-id <id>]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine($"  --algorithm   HPO algorithm to run, or 'all' (default: {defaultAlgorithm}). One of:");
    Console.WriteLine($"                  {string.Join(", ", GetAvailableAlgorithmNames())}");
    Console.WriteLine($"  --instance    One or more benchmark keys, comma-separated (default: {defaultInstance}). Available:");
    Console.WriteLine($"                  {string.Join(", ", relevantBenchmarksets.Keys)}");
    Console.WriteLine("  --output      Shared result directory (default: ./results)");
    Console.WriteLine("  --run-id      Optional campaign subdirectory (no subdirectory by default)");
    Console.WriteLine("  --repetitions Number of repetitions (default: 1)");
    Console.WriteLine("  --repetition-start First repetition identifier (default: 1; also determines seeds)");
    Console.WriteLine("  --provider    Agent provider: ollama (default; local models) or openrouter (google/gemma-4-26b-a4b-it, cloud)");
    Console.WriteLine($"                  (default {OllamaModels.StudyModels[^1]}; study models {string.Join(", ", OllamaModels.StudyModels)})");
    Console.WriteLine("  --model <name> --ollama-url <url>  Agent model / Ollama server (default http://localhost:11434; implies ollama)");
    Console.WriteLine("                  HYPEROP_PROVIDER and HYPEROP_OLLAMA_URL set the defaults when the options are omitted;");
    Console.WriteLine("                  agent jobs fail before any GP work if the local model is not available");
    Console.WriteLine("  --decision-timeout <s>  Run-controller timeout in seconds (default: 300); a timeout continues the run");
    Console.WriteLine("  --feedback    Agent run feedback: final (no checkpoints), progress (default), all, or progress plus any of");
    Console.WriteLine("                  curve, diversity, size, generalization, e.g. progress+curve+size");
    Console.WriteLine("  --checkpoint-interval <n>  Generations between checkpoints for AgentSearch and RuleControlledSearch (default: 10)");
    Console.WriteLine("  --configurator-checkpoints shown|hidden  hidden: the controller acts, the configurator sees final results only");
    Console.WriteLine("  --aggregate   Combine completed *.result.json files into all_results.json, without running GP");
    Console.WriteLine("  --configurations <n> --evaluations <n>  Override the default budget (default: 100 configs / 1000 evals)");
    Console.WriteLine("                  AgentSearch shares their product across configurations; --evaluations is not a per-run cap.");
    Console.WriteLine("  --budget-sweep Run the full five budget variations instead of the default single budget");
    Console.WriteLine("  --help        Show this message");
    Console.WriteLine();
    Console.WriteLine("Examples:");
    Console.WriteLine("  HyperOp.Experiments --algorithm RandomSearch --instance I.6.20");
    Console.WriteLine("  HyperOp.Experiments --algorithm RandomSearch --instance I.6.20,I.9.18,II.11.27");
    Console.WriteLine("  HyperOp.Experiments --instance I.6.20 --run-id campaign01 --output /data/results");
    Console.WriteLine("  HyperOp.Experiments --aggregate --output /data/results/campaign01");
}
