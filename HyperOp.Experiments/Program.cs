using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms;
using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.Feynman.Instances;
using HyperOp.Algorithms.HyperParameterOptimization;
using System.Text.Json;
using System.Text.Json.Serialization;

var parsedArgs = ParseArgs(args);
const int instanceSeed = 42;
Dictionary<string, RegressionDescriptor> relevantBenchmarksets = new Dictionary<string, RegressionDescriptor>(StringComparer.OrdinalIgnoreCase)
{
        { "I.6.20",    new Feynman1(instanceSeed) },
        { "I.9.18",    new Feynman5(instanceSeed) },
        { "I.15.3.x",  new Feynman17(instanceSeed) },
        { "I.30.5",    new Feynman31(instanceSeed) },
        { "I.32.17",   new Feynman33(instanceSeed) },
        { "I.41.16",   new Feynman44(instanceSeed) },
    //TODO Habs & HFo: Double-Check, we have I.48.2 instead of I.48.20, but the formula is the same
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
var runId = parsedArgs.TryGetValue("run-id", out var runIdValue)
    ? SanitizePathSegment(runIdValue)
    : $"run_{DateTime.UtcNow:yyyyMMdd_HHmmss_fff}_{Environment.ProcessId}";

var runOutputDirectory = Path.Combine(outputRoot, runId);

if (!Directory.Exists(runOutputDirectory)) { Directory.CreateDirectory(runOutputDirectory); }

await ExecuteExperiments(runOutputDirectory, algorithmFilter, instanceFilters);

async Task ExecuteExperiments(string outputDirectory, string? algorithmNameFilter, List<string> instanceNameFilters)
{
    List<HyperParameterOptimizationAlgorithm> hyperOpAlgs = new() {
        new IRaceSearch(),
        new AgentSearch(useOLlama: true),
        new GridSearch(),
        new HillClimbingSearch(),
        new RandomSearch()
    };

    if (!string.IsNullOrWhiteSpace(algorithmNameFilter) &&
        !string.Equals(algorithmNameFilter, "all", StringComparison.OrdinalIgnoreCase))
    {
        hyperOpAlgs = hyperOpAlgs
            .Where(algorithm => NormalizeKey(algorithm.GetType().Name) == NormalizeKey(algorithmNameFilter))
            .ToList();
    }

    if (hyperOpAlgs.Count == 0)
    {
        Console.WriteLine($"No algorithm matched '{algorithmNameFilter}'.");
        Console.WriteLine($"Available: {string.Join(", ", GetAvailableAlgorithmNames())}");
        return;
    }

    // Resolve which benchmark instances to run
    IEnumerable<KeyValuePair<string, RegressionDescriptor>> selectedInstances;
    if (instanceFilters.Count == 0)
    {
        selectedInstances = relevantBenchmarksets;
    }
    else
    {
        selectedInstances = instanceFilters
            .Select(filter =>
            {
                if (relevantBenchmarksets.TryGetValue(filter, out var descriptor))
                    return (found: true, key: filter, descriptor);
                Console.WriteLine($"Unknown benchmark instance '{filter}'. Available: {string.Join(", ", relevantBenchmarksets.Keys)}");
                return (found: false, key: filter, descriptor: (RegressionDescriptor?)null);
            })
            .Where(x => x.found)
            .Select(x => new KeyValuePair<string, RegressionDescriptor>(x.key, x.descriptor!));
    }

    var allResults = new List<AggregatedResultDTO>();
    // Freeze selected instances before any HPO call, then share them across all repetitions.
    var frozenInstances = selectedInstances.Select(entry =>
        (entry.Key, Descriptor: entry.Value, Data: entry.Value.PreparedData)).ToList();

    foreach (var algorithm in hyperOpAlgs)
    {
        foreach (var (benchmarkKey, regressionDescriptor, preparedData) in frozenInstances)
        {
            foreach (int repetition in Enumerable.Range(1, repetitions))
            {
                foreach (var entry in budgetVariations)
                {
                    int numConfigurations = entry.Item1;
                    int evaluationsPerConfiguration = entry.Item2;
                    Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Executing {algorithm.GetType().Name} on {benchmarkKey} " +
                        $"(Repetition {repetition}/{repetitions}, Configurations: {numConfigurations}, Evaluations per Configuration: {evaluationsPerConfiguration})");

                    try
                    {
                        // Use the benchmark key (e.g. "I.6.20") for the filename, sanitized to replace dots/slashes
                        string instanceSegment = SanitizePathSegment(benchmarkKey);
                        string fileName = $"{algorithm.GetType().Name}_{instanceSegment}_rep{repetition}_{numConfigurations}cfg_{evaluationsPerConfiguration}eval.json";
                        string resultFilePath = Path.Combine(outputDirectory, fileName);

                        if (File.Exists(resultFilePath))
                        {
                            Console.WriteLine("  → Result already exists. Loading from cache.");
                            var cached = LoadResult(resultFilePath);
                            if (cached.TrainingRows != preparedData.TrainingRows.Count ||
                                cached.ValidationRows != preparedData.ValidationRows.Count ||
                                cached.TestRows != preparedData.TestRows.Count ||
                                cached.ConfigurationResults.Any(result => result.FitnessMetric != "validation_nmse"))
                                throw new InvalidDataException("Cached result does not use the current partitions and validation scoring.");
                            allResults.Add(cached);
                            continue;
                        }

                        int hpoSeed = 42 + (100 * repetition);

                        List<ResultDTO> algRes = await algorithm.Execute(
                            regressionDescriptor,
                            hpoSeed,
                            numConfigurations,
                            evaluationsPerConfiguration
                            );

                        var bestConfig = algRes.Where(r => double.IsFinite(r.BestFitness))
                            .OrderBy(r => r.BestFitness).FirstOrDefault()
                            ?? throw new InvalidOperationException("No trial produced finite validation NMSE.");
                        bestConfig.EvaluateTest(preparedData);
                        var allConfigsResult = new AggregatedResultDTO
                        {
                            AlgorithmName = algorithm.GetType().Name,
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
                            SelectedTrialIndex = algRes.IndexOf(bestConfig)
                        };

                        allResults.Add(allConfigsResult);
                        SaveResult(resultFilePath, allConfigsResult);

                        Console.WriteLine($"Evaluated {algRes.Count} configurations. Best validation NMSE: {bestConfig.BestFitness:G6}; " +
                            $"selected expression test NMSE: {bestConfig.TestFitness:G6}");
                    }
                    catch (NotImplementedException)
                    {
                        Console.WriteLine("Not implemented yet.");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error: {ex.Message}");
                    }
                }
            }
        }
    }

    string aggregatePath = Path.Combine(outputDirectory, "all_results.json");
    SaveAllResults(aggregatePath, allResults);
    Console.WriteLine($"Experiment complete. Results saved to {outputDirectory}");
}

Dictionary<string, string> ParseArgs(string[] cmdArgs)
{
    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    for (int i = 0; i < cmdArgs.Length; i++)
    {
        var arg = cmdArgs[i];
        if (!arg.StartsWith("--")) continue;

        var key = arg[2..];
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

bool MatchesInstanceName(RegressionDescriptor descriptor, string filter)
{
    return NormalizeKey(descriptor.Name) == NormalizeKey(filter) ||
           NormalizeKey(descriptor.GetType().Name) == NormalizeKey(filter);
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

string SanitizePathSegment(string raw)
{
    var invalid = Path.GetInvalidFileNameChars();
    var cleaned = new string(raw.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray());
    return string.IsNullOrWhiteSpace(cleaned) ? "unknown" : cleaned;
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
    Console.WriteLine("  --output      Output root directory (default: ./results)");
    Console.WriteLine("  --run-id      Unique subdirectory name for this run (default: timestamp + process id)");
    Console.WriteLine("  --repetitions Number of repetitions (default: 1)");
    Console.WriteLine("  --configurations <n> --evaluations <n>  Override the default budget (default: 100 configs / 1000 evals)");
    Console.WriteLine("  --budget-sweep Run the full five budget variations instead of the default single budget");
    Console.WriteLine("  --help        Show this message");
    Console.WriteLine();
    Console.WriteLine("Examples:");
    Console.WriteLine("  HyperOp.Experiments --algorithm RandomSearch --instance I.6.20");
    Console.WriteLine("  HyperOp.Experiments --algorithm RandomSearch --instance I.6.20,I.9.18,II.11.27");
    Console.WriteLine("  HyperOp.Experiments --instance I.6.20 --run-id job42 --output /data/results");
}

void SaveResult(string filePath, AggregatedResultDTO dto)
{
    var json = JsonSerializer.Serialize(dto, new JsonSerializerOptions { WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals });
    File.WriteAllText(filePath, json);
}

void SaveAllResults(string filePath, List<AggregatedResultDTO> allResults)
{
    var json = JsonSerializer.Serialize(allResults, new JsonSerializerOptions { WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals });
    File.WriteAllText(filePath, json);
}

AggregatedResultDTO LoadResult(string filePath)
{
    var json = File.ReadAllText(filePath);
    return JsonSerializer.Deserialize<AggregatedResultDTO>(json, new JsonSerializerOptions { NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals })!;
}
