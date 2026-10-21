using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.Feynman.Instances;
using HyperOp.Algorithms.HyperParameterOptimization;
using System.Text.Json;
using System.Text.Json.Serialization;

Dictionary<string, FeynmanDescriptor> relevantBenchmarksets = new Dictionary<string, FeynmanDescriptor>()
{
        { "I.6.20", new Feynman1() },
        { "I.9.18", new Feynman5() },
        { "I.15.3.x", new Feynman17() },
        { "I.30.5", new Feynman31() },
        { "I.32.17", new Feynman33() },
        { "I.41.16", new Feynman44() },
    //TODO Habs & HFo: Double-Check, we have I.48.2 instead of I.48.20, but the formula is the same    
        { "I.48.20",   new Feynman50() },
        { "II.6.15a", new Feynman56() },
        { "II.11.27", new Feynman64() },
        { "II.11.28", new Feynman65() },
        { "II.35.21", new Feynman81() },
        { "III.9.52", new Feynman90() },
        { "III.10.19", new Feynman91() }
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

var parsedArgs = ParseArgs(args);
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
var instanceFilter = parsedArgs.TryGetValue("instance", out var instValue) ? instValue : defaultInstance;
var outputRoot = parsedArgs.TryGetValue("output", out var outputValue) ? outputValue : "./results";
var runId = parsedArgs.TryGetValue("run-id", out var runIdValue)
    ? SanitizePathSegment(runIdValue)
    : $"run_{DateTime.UtcNow:yyyyMMdd_HHmmss_fff}_{Environment.ProcessId}";

var runOutputDirectory = Path.Combine(outputRoot, runId);

if (!Directory.Exists(runOutputDirectory)) { Directory.CreateDirectory(runOutputDirectory); }

await ExecuteExperiments(runOutputDirectory, algorithmFilter, instanceFilter);

async Task ExecuteExperiments(string outputDirectory, string? algorithmNameFilter, string? instanceNameFilter)
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

    int instanceSeed = 42;
    if (string.IsNullOrWhiteSpace(instanceNameFilter))
        throw new ArgumentException("Select a benchmark using --instance (for example I.6.20 or Feynman1).");
    var benchmarkName = relevantBenchmarksets.TryGetValue(instanceNameFilter, out var benchmark)
        ? benchmark.Name : instanceNameFilter;
    var feynmanDescriptor = FeynmanInstanceProvider.LoadAvailableInstances(instanceSeed)
        .FirstOrDefault(v => MatchesInstanceName(v, benchmarkName))
        ?? throw new ArgumentException($"No instance matched '{instanceNameFilter}'.");

    var allResults = new List<AggregatedResultDTO>();

    foreach (var algorithm in hyperOpAlgs)
    {
        foreach (int repetition in Enumerable.Range(1, repetitions))
        {
            foreach (var entry in budgetVariations)
            {
                int numConfigurations = entry.Item1;
                int evaluationsPerConfiguration = entry.Item2;
                Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Executing {algorithm.GetType().Name} on {feynmanDescriptor.Name} " +
                    $"(Repetition {repetition}/{repetitions}, Configurations: {numConfigurations}, Evaluations per Configuration: {evaluationsPerConfiguration})");

                try
                {
                    string instanceSegment = SanitizePathSegment(feynmanDescriptor.Name);
                    //TODO: Add instance name
                    string fileName = $"{algorithm.GetType().Name}_{instanceSegment}_rep{repetition}_{numConfigurations}cfg_{evaluationsPerConfiguration}eval.json";
                    string resultFilePath = Path.Combine(outputDirectory, fileName);

                    if (File.Exists(resultFilePath))
                    {
                        Console.WriteLine("  → Result already exists. Loading from cache.");
                        allResults.Add(LoadResult(resultFilePath));
                        continue;
                    }

                    int hpoSeed = 42 + (100 * repetition);

                    List<ResultDTO> algRes = await algorithm.Execute(
                        feynmanDescriptor,
                        hpoSeed,
                        numConfigurations,
                        evaluationsPerConfiguration
                        );

                    var allConfigsResult = new AggregatedResultDTO {
                        AlgorithmName = algorithm.GetType().Name,
                        InstanceName = feynmanDescriptor.Name,
                        Repetition = repetition,
                        Seed = hpoSeed,
                        NumConfigurations = numConfigurations,
                        EvaluationsPerConfiguration = evaluationsPerConfiguration,
                        TotalConfigurationsEvaluated = algRes.Count,
                        ConfigurationResults = algRes
                    };

                    allResults.Add(allConfigsResult);
                    SaveResult(resultFilePath, allConfigsResult);

                    var bestConfig = algRes.OrderByDescending(r => r.BestFitness).First();
                    Console.WriteLine($"Evaluated {algRes.Count} configurations. Best fitness: {bestConfig.BestFitness:F4}");
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

bool MatchesInstanceName(FeynmanDescriptor descriptor, string filter)
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
    Console.WriteLine("  HyperOp.Experiments --algorithm <name> --instance <name> [--output <dir>] [--run-id <id>]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine($"  --algorithm   Algorithm name or all (default: {defaultAlgorithm})");
    Console.WriteLine($"  --instance    Benchmark alias, descriptor Name or class name (default: {defaultInstance})");
    Console.WriteLine("  --output      Output root directory (default: ./results)");
    Console.WriteLine("  --run-id      Unique run id used as subdirectory name (default: timestamp + process id)");
    Console.WriteLine("  --configurations <n> --evaluations <n>  Override the default budget of 100 trials / 1000 evaluations");
    Console.WriteLine("  --budget-sweep Run the original five budget variations");
    Console.WriteLine("  --repetitions Number of repetitions (default: 1)");
    Console.WriteLine("  --help        Show this message");
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
