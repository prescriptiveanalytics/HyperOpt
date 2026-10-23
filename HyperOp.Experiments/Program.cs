using HyperOp.Algorithms;
using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.Feynman.Instances;
using HyperOp.Algorithms.HyperParameterOptimization;
using HyperOp.Experiments;

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
        { "III.10.19", new Feynman91(instanceSeed) },
        { "SteelIndustry", new SteelIndustryDescriptor() }
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
    var agentParameters = AgentSearch.CreateOllamaParameters();
    if (parsedArgs.TryGetValue("model", out var model)) agentParameters.ModelName = model;
    if (parsedArgs.TryGetValue("ollama-url", out var ollamaUrl)) agentParameters.OllamaBaseUrl = ollamaUrl;
    if (string.IsNullOrWhiteSpace(agentParameters.ModelName)) throw new ArgumentException("--model cannot be empty.");
    List<HyperParameterOptimizationAlgorithm> hyperOpAlgs = algorithmNames.Select(name => name switch {
        nameof(IRaceSearch) => (HyperParameterOptimizationAlgorithm)new IRaceSearch(),
        nameof(AgentSearch) => new AgentSearch(agentParameters),
        nameof(GridSearch) => new GridSearch(),
        nameof(HillClimbingSearch) => new HillClimbingSearch(),
        _ => new RandomSearch()
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
                        $"(Repetition {repetition}, Configurations: {numConfigurations}, Evaluations per Configuration: {evaluationsPerConfiguration})");

                    try
                    {
                        string? modelName = algorithm is AgentSearch ? agentParameters.ModelName : null;
                        string fileName = ExperimentResultStore.JobName(algorithm.GetType().Name, benchmarkKey,
                            repetition, numConfigurations, evaluationsPerConfiguration, assessmentRepetitions, modelName) + ".result.json";
                        string resultFilePath = Path.Combine(outputDirectory, fileName);
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

                        List<ResultDTO> algRes = await algorithm.Execute(
                            regressionDescriptor,
                            hpoSeed,
                            numConfigurations,
                            evaluationsPerConfiguration
                            );

                        var bestConfig = HyperParameterOptimizationAlgorithm.BestValidationTrial(algRes);
                        // Freeze both selections before any test data are scored.
                        var recommendation = algorithm.NominateConfiguration(algRes);
                        var nominatedConfiguration = recommendation.Configuration with { };
                        bestConfig.EvaluateTest(preparedData);
                        var assessmentResults = await algorithm.AssessConfiguration(regressionDescriptor,
                            nominatedConfiguration, algRes, hpoSeed, assessmentRepetitions, evaluationsPerConfiguration);
                        var allConfigsResult = new AggregatedResultDTO
                        {
                            AlgorithmName = algorithm.GetType().Name,
                            ModelName = modelName,
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

                        Console.WriteLine($"Evaluated {algRes.Count} configurations. Best validation NMSE: {bestConfig.BestFitness:G6}; " +
                            $"selected expression test NMSE: {bestConfig.TestFitness:G6}; " +
                            $"{assessmentResults.Count} fresh-seed assessment runs saved separately.");
                    }
                    catch (Exception ex)
                    {
                        Environment.ExitCode = 1;
                        Console.WriteLine($"Error: {ex.Message}");
                    }
                }
            }
        }
    }

    Console.WriteLine($"Experiment complete. Results saved to {outputDirectory}");
}

Dictionary<string, string> ParseArgs(string[] cmdArgs)
{
    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    string[] known = ["help", "h", "algorithm", "instance", "output", "run-id", "repetitions", "repetition-start",
        "configurations", "evaluations", "budget-sweep", "model", "ollama-url", "aggregate"];

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
    return new[] { nameof(AgentSearch), nameof(GridSearch), nameof(HillClimbingSearch), nameof(IRaceSearch), nameof(RandomSearch) };
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
    Console.WriteLine("  --model <name> --ollama-url <url>  Agent model / server (default: qwen3:8b / http://localhost:11434)");
    Console.WriteLine("  --aggregate   Combine completed *.result.json files into all_results.json, without running GP");
    Console.WriteLine("  --configurations <n> --evaluations <n>  Override the default budget (default: 100 configs / 1000 evals)");
    Console.WriteLine("  --budget-sweep Run the full five budget variations instead of the default single budget");
    Console.WriteLine("  --help        Show this message");
    Console.WriteLine();
    Console.WriteLine("Examples:");
    Console.WriteLine("  HyperOp.Experiments --algorithm RandomSearch --instance I.6.20");
    Console.WriteLine("  HyperOp.Experiments --algorithm RandomSearch --instance I.6.20,I.9.18,II.11.27");
    Console.WriteLine("  HyperOp.Experiments --instance I.6.20 --run-id campaign01 --output /data/results");
    Console.WriteLine("  HyperOp.Experiments --aggregate --output /data/results/campaign01");
}
