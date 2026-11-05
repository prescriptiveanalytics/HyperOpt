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
// Fresh-seed reruns of the nominated configuration; freeze the final count after the pilot.
int assessmentRepetitions = parsedArgs.TryGetValue("assessment-runs", out var assessmentRunsValue)
    ? int.TryParse(assessmentRunsValue, out var runs) && runs >= 0 ? runs : throw new ArgumentException("--assessment-runs must be a non-negative integer.")
    : 5;
// Development equations: AI Feynman equations outside the paper's benchmark, for pilots that choose budgets and settings
// without looking at the evaluation tasks. Near-duplicates of benchmark equations (I.6.20a/b, I.15.3t) are left out.
Dictionary<string, RegressionDescriptor> developmentSets = new(StringComparer.OrdinalIgnoreCase)
{
        { "I.8.14",    new Feynman4(instanceSeed) },
        { "I.13.12",   new Feynman14(instanceSeed) },
        { "I.16.6",    new Feynman20(instanceSeed) },
        { "I.18.4",    new Feynman21(instanceSeed) },
        { "I.27.6",    new Feynman27(instanceSeed) },
        { "I.29.16",   new Feynman29(instanceSeed) },
        { "I.34.14",   new Feynman36(instanceSeed) },
        { "I.40.1",    new Feynman43(instanceSeed) },
        { "II.6.11",   new Feynman55(instanceSeed) },
        { "II.11.3",   new Feynman61(instanceSeed) },
        { "II.24.17",  new Feynman72(instanceSeed) },
        { "II.35.18",  new Feynman80(instanceSeed) },
        { "II.36.38",  new Feynman82(instanceSeed) },
        { "III.14.14", new Feynman94(instanceSeed) },
        { "III.15.12", new Feynman95(instanceSeed) }
};
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
List<(int, int)> defaultBudget = [(30, 30_000)];
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
int ParseNonNegativeInt(string name, string value) => int.TryParse(value, out var parsed) && parsed >= 0
    ? parsed : throw new ArgumentException($"--{name} must be a non-negative integer.");
HyperParameterOptimizationAlgorithm.ConstantFitting = new ConstantFittingSettings {
    Iterations = parsedArgs.TryGetValue("fitting-iterations", out var fittingIterationsValue)
        ? ParseNonNegativeInt("fitting-iterations", fittingIterationsValue) : ConstantFittingSettings.DefaultIterations,
    Rows = parsedArgs.TryGetValue("fitting-rows", out var fittingRowsValue) ? ParseNonNegativeInt("fitting-rows", fittingRowsValue) : ConstantFittingSettings.DefaultRows,
    Probability = parsedArgs.TryGetValue("fitting-probability", out var fittingProbabilityValue)
        ? double.TryParse(fittingProbabilityValue, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var probability)
            && probability > 0 && probability <= 1 ? probability : throw new ArgumentException("--fitting-probability must be in (0, 1].")
        : 1
};

// Target noise for the noisy subset: standard deviation as a fraction of the target's, on training and validation only.
double targetNoise = parsedArgs.TryGetValue("noise", out var noiseValue)
    ? double.TryParse(noiseValue, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var noise) && noise >= 0
        ? noise : throw new ArgumentException("--noise must be a non-negative number.")
    : 0;
// Small-data variant: training and validation reduced to this many evenly spaced rows each (0: all rows).
int sampleRows = parsedArgs.TryGetValue("rows", out var rowsValue) ? ParseNonNegativeInt("rows", rowsValue) : 0;
// A job whose protocol stays silent this long is treated as hung: GP work cannot be interrupted, so the process records
// the stall and exits, and a rerun of the same command resumes after the completed jobs. Trials, checkpoints and model
// requests all write protocol records, far more often than this in normal operation.
int stallMinutes = parsedArgs.TryGetValue("stall-minutes", out var stallMinutesValue) ? ParseNonNegativeInt("stall-minutes", stallMinutesValue) : 60;
JsonlProtocolLog? activeProtocol = null;
using var stallWatchdog = new Timer(_ => {
    var current = Volatile.Read(ref activeProtocol);
    if (stallMinutes == 0 || current is null || DateTime.UtcNow - current.LastWriteUtc < TimeSpan.FromMinutes(stallMinutes)) return;
    Console.WriteLine($"Error: no progress for {stallMinutes} min; the job is treated as hung and the process exits.");
    current.Write("job_stalled", new { StallMinutes = stallMinutes });
    Environment.Exit(3);
}, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));

// Captured once per process: every job of this process runs the same code and fitting settings.
var provenance = RunProvenance.Capture(HyperParameterOptimizationAlgorithm.ConstantFitting);
if (parsedArgs.ContainsKey("code-fingerprint"))
{
    // For Start-Study.ps1 and release checks: the content fingerprint of the code that this build runs.
    Console.WriteLine(provenance.CodeFingerprint ?? throw new InvalidOperationException("The source files were not found next to the binaries."));
    return;
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
    // Feedback condition of the agent's run controller and the checkpoint schedule shared with the rule baseline.
    if (parsedArgs.TryGetValue("feedback", out var feedbackValue))
        agentParameters.Feedback = FeedbackChannelNames.Parse(feedbackValue);
    int checkpoints = parsedArgs.TryGetValue("checkpoints", out var checkpointsPerBudgetValue)
        ? ParsePositiveInt("checkpoints", checkpointsPerBudgetValue) : agentParameters.Checkpoints;
    agentParameters.Checkpoints = checkpoints;
    if (parsedArgs.TryGetValue("configurator-checkpoints", out var checkpointsValue))
        agentParameters.ConfiguratorSeesCheckpoints = checkpointsValue.ToLowerInvariant() switch {
            "shown" => true, "hidden" => false,
            _ => throw new ArgumentException("--configurator-checkpoints must be 'shown' or 'hidden'.")
        };
    // Length-matched placebo: channel values from another run of the job (RQ2 control for the prompt-length confound).
    agentParameters.ChannelPlacebo = parsedArgs.ContainsKey("placebo");
    // Pilot switches against over-stopping (see AgentParameter); each is part of the job name.
    if (parsedArgs.TryGetValue("controller-comparison", out var comparisonValue))
        agentParameters.ControllerComparison = comparisonValue.ToLowerInvariant() switch {
            "on" => true, "off" => false, _ => throw new ArgumentException("--controller-comparison must be 'on' or 'off'.") };
    if (parsedArgs.TryGetValue("first-decision", out var firstDecisionValue))
        agentParameters.FirstControllerCheckpoint = ParsePositiveInt("first-decision", firstDecisionValue);
    if (parsedArgs.TryGetValue("controller", out var controllerValue))
        agentParameters.Controller = controllerValue.ToLowerInvariant();
    if (parsedArgs.TryGetValue("stop-hazard", out var hazardValue))
        agentParameters.StopHazard = hazardValue.Split(new[] { ',', '+' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(v => double.TryParse(v, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var p)
                ? p : throw new ArgumentException($"--stop-hazard: '{v}' is not a number.")).ToArray();
    if (parsedArgs.TryGetValue("controller-prompt", out var controllerPromptValue))
        agentParameters.ControllerPrompt = controllerPromptValue.ToLowerInvariant();
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
        nameof(RuleControlledSearch) => new RuleControlledSearch(checkpoints),
        nameof(GridSearch) => new GridSearch(),
        nameof(HillClimbingSearch) => new HillClimbingSearch(),
        nameof(RandomSearch) => new RandomSearch(),
        nameof(RandomRunLengthSearch) => new RandomRunLengthSearch(),
        nameof(AshaSearch) => new AshaSearch(),
        _ => throw new ArgumentException($"Unknown algorithm '{name}'.")
    }).ToList();

    // Resolve which benchmark instances to run
    IEnumerable<KeyValuePair<string, RegressionDescriptor>> selectedInstances;
    if (instanceNameFilters.Count == 0)
        throw new ArgumentException("Specify at least one benchmark instance.");
    selectedInstances = instanceNameFilters
            .Select(filter =>
            {
                foreach (var table in new[] { relevantBenchmarksets, developmentSets })
                    if (table.TryGetValue(filter, out var descriptor))
                        return (key: table.Keys.First(key => key.Equals(filter, StringComparison.OrdinalIgnoreCase)), descriptor);
                throw new ArgumentException($"Unknown benchmark instance '{filter}'. Available: {string.Join(", ", relevantBenchmarksets.Keys)}; " +
                    $"development: {string.Join(", ", developmentSets.Keys)}");
            })
            .DistinctBy(x => x.key)
            .Select(x => new KeyValuePair<string, RegressionDescriptor>(x.key, x.descriptor));

    // Freeze selected instances before any HPO call, then share them across all repetitions.
    foreach (var entry in selectedInstances) { entry.Value.TargetNoise = targetNoise; entry.Value.SampleRows = sampleRows; }
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
                    Console.WriteLine($"[{DateTime.UtcNow:HH:mm:ss}Z] Executing {algorithm.GetType().Name} on {benchmarkKey} " +
                        $"(Repetition {repetition}, Configurations: {numConfigurations}, " +
                        (algorithm is AgentSearch or RuleControlledSearch or RandomRunLengthSearch or AshaSearch
                            ? $"Shared evaluations: {checked(numConfigurations * evaluationsPerConfiguration)})"
                            : $"Evaluations per Configuration: {evaluationsPerConfiguration})"));

                    JsonlProtocolLog? protocol = null;
                    try
                    {
                        string? modelName = algorithm is AgentSearch ? agentParameters.ModelName : null;
                        string jobName = ExperimentResultStore.JobName(algorithm.GetType().Name, benchmarkKey,
                            repetition, numConfigurations, evaluationsPerConfiguration, assessmentRepetitions, modelName, algorithm.Variant, targetNoise, sampleRows);
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
                            // Constant fitting changes GP results but is not part of the job name.
                            if (!provenance.SameFitting(cached.Provenance))
                                throw new InvalidDataException("Cached result used a different constant-fitting implementation or settings. Use a fresh run directory.");
                            // Results of different code versions must not be mixed in one campaign; --allow-mixed-code is for development only.
                            if (provenance.CacheConflict(cached.Provenance) is { } conflict && !parsedArgs.ContainsKey("allow-mixed-code"))
                                throw new InvalidDataException($"Cached result not reused: {conflict}. Use a fresh run directory or commit first (development: --allow-mixed-code).");
                            // Model tags are mutable: the same name can refer to other weights after a pull, and the server
                            // version changes decoding. Both must match, also in development.
                            if (algorithm is AgentSearch && modelInfo is not null &&
                                (cached.ModelInfo?.Digest != modelInfo.Digest || cached.ModelInfo?.ServerVersion != modelInfo.ServerVersion))
                                throw new InvalidDataException($"Cached result not reused: it was produced with model digest {cached.ModelInfo?.Digest?[..12] ?? "unknown"} " +
                                    $"on Ollama {cached.ModelInfo?.ServerVersion ?? "unknown"}, this run uses {modelInfo.Digest[..12]} on Ollama {modelInfo.ServerVersion}. Use a fresh run directory.");
                            continue;
                        }

                        // Each attempt has its own protocol; an interrupted attempt's protocol stays beside the new one.
                        string protocolFile = $"{jobName}__attempt={Guid.NewGuid().ToString("N")[..8]}.protocol.jsonl";
                        protocol = new JsonlProtocolLog(Path.Combine(outputDirectory, protocolFile));
                        algorithm.Log = protocol;
                        Volatile.Write(ref activeProtocol, protocol);
                        await algorithm.PrepareAsync();
                        var jobClock = Stopwatch.StartNew();
                        var timing = new TimingBreakdown();
                        if (algorithm is AgentSearch && modelInfo is not null)
                        {
                            timing.ModelWarmupMs = await OllamaModels.WarmUpAsync(agentParameters.OllamaBaseUrl, modelInfo.Name);
                            timing.ContextLength = await OllamaModels.RuntimeContextLengthAsync(agentParameters.OllamaBaseUrl, modelInfo.Name);
                            (timing.ModelMemoryBytes, timing.ModelGpuMemoryBytes) = await OllamaModels.RuntimeMemoryAsync(agentParameters.OllamaBaseUrl, modelInfo.Name);
                            if (timing.ModelMemoryBytes is long total && timing.ModelGpuMemoryBytes is long gpu && gpu < total)
                                Console.WriteLine($"Warning: only {gpu / 1e9:F1} of {total / 1e9:F1} GB of the model are in GPU memory; the rest runs on the CPU.");
                        }
                        protocol.Write("job_started", new {
                            Algorithm = algorithm.GetType().Name, Instance = benchmarkKey, Repetition = repetition, Seed = hpoSeed,
                            T = numConfigurations, E = evaluationsPerConfiguration, K = assessmentRepetitions,
                            Model = modelName, Provider = algorithm is AgentSearch ? agentParameters.Provider.ToString() : null,
                            ModelInfo = algorithm is AgentSearch ? modelInfo : null,
                            Agent = algorithm is AgentSearch ? new { agentParameters.Temperature, Configurator = "stateless", AgentSearch.MaxInvalidReplies, DecisionTimeoutS = agentParameters.DecisionTimeout.TotalSeconds,
                                Feedback = FeedbackChannelNames.Format(agentParameters.Feedback), agentParameters.Checkpoints,
                                agentParameters.ConfiguratorSeesCheckpoints, agentParameters.ControllerPrompt,
                                agentParameters.ControllerComparison, agentParameters.FirstControllerCheckpoint,
                                agentParameters.Controller, agentParameters.StopHazard } : null,
                            algorithm.Variant, TargetNoise = targetNoise, SampleRows = sampleRows, Provenance = provenance,
                            timing.ModelWarmupMs, Host = new { Environment.ProcessorCount, OS = Environment.OSVersion.ToString(),
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
                        // Primary selection over all tuning snapshots; its test NMSE was recorded during the run and never shown to the method.
                        var expressionSelection = HyperParameterOptimizationAlgorithm.SelectExpression(algRes);
                        // Freeze both selections before any test data are scored.
                        var recommendation = algorithm.NominateConfiguration(algRes);
                        var nominatedConfiguration = recommendation.Configuration with { };
                        // Every method's nominee is assessed at the same lengths, E and the longest allowed run, whatever its tuning run used.
                        int[] assessmentLengths = HyperParameterOptimizationAlgorithm.AssessmentRunLengths(evaluationsPerConfiguration);
                        bestConfig.EvaluateTest(preparedData);
                        timing.SelectionAndTestScoringMs = phaseClock.Elapsed.TotalMilliseconds;
                        protocol.Write("selection", new { ExpressionSelection = expressionSelection, SelectedTrialIndex = algRes.IndexOf(bestConfig), bestConfig.TestFitness,
                            RecommendedConfiguration = nominatedConfiguration, RecommendedConfigurationIndex = recommendation.ConfigurationIndex,
                            algorithm.RecommendationRule, AssessmentRunLengths = assessmentLengths });

                        phaseClock.Restart();
                        // --assessment-runs 0 skips the fresh-seed reruns (development pilots only).
                        var assessmentResults = assessmentRepetitions == 0 ? [] : await algorithm.AssessConfiguration(regressionDescriptor,
                            nominatedConfiguration, algRes, hpoSeed, assessmentRepetitions, assessmentLengths);
                        timing.AssessmentMs = phaseClock.Elapsed.TotalMilliseconds;
                        FillTiming(timing, algRes, assessmentResults, algorithm.LlmCalls);
                        timing.PeakWorkingSetMb = Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0;
                        timing.TotalJobMs = jobClock.Elapsed.TotalMilliseconds;
                        var allConfigsResult = new AggregatedResultDTO
                        {
                            AlgorithmName = algorithm.GetType().Name,
                            ModelName = modelName,
                            ModelProvider = algorithm is AgentSearch ? agentParameters.Provider.ToString() : null,
                            Variant = algorithm.Variant,
                            ModelInfo = algorithm is AgentSearch ? modelInfo : null,
                            ProtocolFile = protocolFile,
                            Search = (algorithm as AgentSearch)?.Summary,
                            Timing = timing,
                            AssessmentRunLengths = assessmentLengths,
                            ExpressionSelection = expressionSelection,
                            TargetNoise = targetNoise,
                            SampleRows = sampleRows,
                            Provenance = provenance,
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

                        Console.WriteLine($"Evaluated {algRes.Count} configurations. Selected expression (trial {expressionSelection?.TrialIndex}, " +
                            $"{expressionSelection?.Evaluations} evaluations): validation NMSE {expressionSelection?.ValidationNmse:G6}, test NMSE {expressionSelection?.TestNmse:G6}; " +
                            $"{assessmentResults.Count} fresh-seed assessment runs saved separately.");
                        Console.WriteLine($"Time: job {timing.TotalJobMs / 1000:F1} s; tuning {timing.TuningMs / 1000:F1} s = GP {timing.TuningGpMs / 1000:F1} s " +
                            $"(constant fitting {timing.TuningConstantFittingMs / 1000:F1} s) " +
                            $"+ trial overhead {timing.TuningTrialOverheadMs / 1000:F1} s + controller {timing.ControllerInferenceMs / 1000:F1} s " +
                            $"+ optimizer {timing.OptimizerMs / 1000:F1} s (configurator requests {timing.ConfiguratorInferenceMs / 1000:F1} s); " +
                            $"LLM share of tuning {timing.LlmShareOfTuning:P1}; assessment {timing.AssessmentMs / 1000:F1} s.");
                        if (algorithm.LlmCalls.Count > 0)
                            Console.WriteLine($"Context: largest prompt {timing.MaxConfiguratorInputTokens} tokens (configurator), " +
                                $"{timing.MaxControllerInputTokens} (controller); largest request with reply {timing.MaxRequestTokens} " +
                                $"of {timing.ContextLength?.ToString() ?? "unknown"}.");
                        if (timing.RequestsNearContextLimit > 0)
                            Console.WriteLine($"Warning: {timing.RequestsNearContextLimit} model requests filled at least 90% of the {timing.ContextLength}-token context; " +
                                "the server truncates longer prompts silently.");
                    }
                    catch (Exception ex)
                    {
                        Environment.ExitCode = 1;
                        Console.WriteLine($"Error: {ex.Message}");
                        // An agent search that evaluated no configuration fails the job; its summary lets the analysis count it.
                        protocol?.Write("job_failed", new { Error = ex.ToString(), Search = (algorithm as AgentSearch)?.Summary });
                    }
                    finally
                    {
                        algorithm.Log = NullProtocolLog.Instance;
                        Volatile.Write(ref activeProtocol, null);
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
    timing.TuningConstantFittingMs = tuning.Sum(trial => trial.ConstantFittingMs ?? 0);
    timing.ControllerInferenceMs = observations.Sum(observation => observation.InferenceTimeMs);
    timing.RecordControllerDecisions(observations, llmCalls);
    timing.TuningTrialOverheadMs = timing.TuningTrialWallMs - timing.TuningGpMs - timing.ControllerInferenceMs;
    timing.OptimizerMs = timing.TuningMs - timing.TuningTrialWallMs;
    var configurator = llmCalls.Where(call => call.Role == "configurator").ToArray();
    timing.ConfiguratorRequests = configurator.Length;
    timing.ConfiguratorInferenceMs = configurator.Sum(call => call.DurationMs);
    timing.InputTokens = llmCalls.Sum(call => call.InputTokens ?? 0);
    timing.OutputTokens = llmCalls.Sum(call => call.OutputTokens ?? 0);
    timing.ReasoningTokens = llmCalls.Sum(call => call.ReasoningTokens ?? 0);
    timing.RecordContextUsage(llmCalls);
    timing.TuningTrialCpuMs = tuning.Sum(trial => trial.TrialCpuTimeMs ?? 0);
    timing.AssessmentGpMs = assessment.Sum(run => run.ExecutionTimeMs);
}

Dictionary<string, string> ParseArgs(string[] cmdArgs)
{
    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    string[] known = ["help", "h", "algorithm", "instance", "output", "run-id", "repetitions", "repetition-start",
        "configurations", "evaluations", "budget-sweep", "model", "provider", "ollama-url", "decision-timeout", "feedback",
        "checkpoints", "configurator-checkpoints", "aggregate", "fitting-iterations", "fitting-rows", "fitting-probability", "assessment-runs", "stall-minutes", "noise", "rows", "allow-mixed-code", "placebo", "controller-comparison", "first-decision", "controller-prompt", "controller", "stop-hazard", "code-fingerprint"];

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
        nameof(AshaSearch), nameof(IRaceSearch), nameof(RandomRunLengthSearch), nameof(RandomSearch), nameof(RuleControlledSearch) };
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
    Console.WriteLine($"                Development equations (pilots only, not in the paper's benchmark):");
    Console.WriteLine($"                  {string.Join(", ", developmentSets.Keys)}");
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
    Console.WriteLine("  --checkpoints <k>  Controller checkpoints per per-configuration budget E, one every E/k evaluations,");
    Console.WriteLine("                  for AgentSearch and RuleControlledSearch (default: 10)");
    Console.WriteLine($"  --fitting-iterations <n>  Levenberg-Marquardt iterations of constant fitting per new candidate, 0 disables it (default: {ConstantFittingSettings.DefaultIterations})");
    Console.WriteLine($"  --fitting-rows <n>  Evenly spaced training rows used for constant fitting; 0 uses all rows (default: {ConstantFittingSettings.DefaultRows})");
    Console.WriteLine("  --fitting-probability <p>  Share of new candidates whose constants are fitted, in (0, 1] (default: 1)");
    Console.WriteLine("  --assessment-runs <n>  Fresh-seed reruns of the nominated configuration; 0 skips them (default: 5)");
    Console.WriteLine("  --allow-mixed-code  Reuse cached results of other code (another code fingerprint; development only)");
    Console.WriteLine("  --code-fingerprint  Print the content fingerprint of this build's code and exit");
    Console.WriteLine("  --rows <n>    Use n evenly spaced training rows and n validation rows (small-data variant; default: all)");
    Console.WriteLine("  --noise <s>   Gaussian noise on training and validation targets, s x the target's standard deviation (default: 0)");
    Console.WriteLine("  --stall-minutes <n>  Exit with code 3 when a job writes no protocol record for n minutes; 0 disables it (default: 60)");
    Console.WriteLine("  --placebo     Channel values shown to the agent come from another run of the job (length-matched placebo)");
    Console.WriteLine("  --controller-comparison on|off  Pilot: the controller's comparison with earlier runs at the same evaluations");
    Console.WriteLine("  --first-decision N  Pilot: first controller decision at checkpoint N (default 1)");
    Console.WriteLine("  --controller-prompt NAME  Controller instructions other than the study prompt p8 (p9 without the late-improvement prior; pilot: standard, p1, p2, p4, p5, p6, p7)");
    Console.WriteLine("  --controller llm|rule|random  RQ1 controls: who decides at the checkpoints (default llm); rule = stop below 1% improvement");
    Console.WriteLine("  --stop-hazard P1,P2,...  Random controller: stop probability at the k-th decision of a run (last value repeats)");
    Console.WriteLine("  --configurator-checkpoints shown|hidden  hidden: the controller acts, the configurator sees final results only");
    Console.WriteLine("  --aggregate   Combine completed *.result.json files into all_results.json, without running GP");
    Console.WriteLine("  --configurations <n> --evaluations <n>  Override the default budget (default: 30 configs / 30000 evals)");
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
