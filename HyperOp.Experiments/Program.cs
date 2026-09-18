using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.HyperParameterOptimization;
using System.Text.Json;
using System.Text.Json.Serialization;

const int numConfigurations = 100;
const int evaluationsPerConfiguration = 10000;
const int repetitions = 3;
const int totalEvaluationBudget = numConfigurations * evaluationsPerConfiguration;
//TODO: Think about whether we also want to include more configurations with fewer evaluations per configuration
//or fewer configurations with more evaluations per configuration.
//This is a trade-off between exploring more configurations and evaluating each configuration more thoroughly.
//This variation could be utilized to analyze the impact of the number of configurations vs. evaluations
//per configuration on the performance of the HPO algorithms and reduce our bias, basically making the experiments
//more robust and fair in a sense.

//Clean-Up for Debugging
#if DEBUG
if (Directory.Exists("./results")) { Directory.Delete("./results", true); }
#endif

if (!Directory.Exists("./results")) { Directory.CreateDirectory("./results"); }
if (!File.Exists("./results/all_results.json")) { await ExecuteExperiments(); }


async Task ExecuteExperiments()
{
    List<HyperParameterOptimizationAlgorithm> hyperOpAlgs = new List<HyperParameterOptimizationAlgorithm>() {
        new AgentSearch(),
        new GridSearch(),
        new HillClimbingSearch(),
        new IRaceSearch(),
        new RandomSearch()
    };

    // Fixed Feynman data per instance (same across all repetitions)
    // This ensures variation comes from HPO randomness, not data variation
    int instanceSeed = 42;
    List<FeynmanDescriptor> feynmanDescriptors = FeynmanInstanceProvider.LoadAvailableInstances(instanceSeed).Take(10).ToList();
    var allResults = new List<AggregatedResultDTO>();

    foreach (var algorithm in hyperOpAlgs)
    {
        foreach (var feynmanInstance in feynmanDescriptors)
        {
            foreach (int repetition in Enumerable.Range(1, repetitions))
            {
                Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Executing {algorithm.GetType().Name} on {feynmanInstance.GetType().Name} " +
                    $"(Repetition {repetition}/{repetitions}, Budget: {totalEvaluationBudget:N0} evals)");

                try
                {
                    string resultFilePath = $"./results/{algorithm.GetType().Name}_{feynmanInstance.GetType().Name}_{repetition}.json";
                    if (File.Exists(resultFilePath))
                    {
                        Console.WriteLine($"  → Result already exists. Loading from cache.");
                        allResults.Add(LoadResult(resultFilePath));
                        continue;
                    }

                    // HPO algorithm seed varies per repetition (arbitrary base seed diversification)
                    int hpoSeed = 42 + (100 * repetition);

                    // Execute with a somewhat fair budget: numConfigurations × evaluationsPerConfiguration
                    List<ResultDTO> algRes = await algorithm.Execute(
                        feynmanInstance,
                        hpoSeed,
                        numConfigurations,
                        evaluationsPerConfiguration
                        );

                    // Save results directly
                    var allConfigsResult = new AggregatedResultDTO {
                        AlgorithmName = algorithm.GetType().Name,
                        InstanceName = feynmanInstance.GetType().Name,
                        Repetition = repetition,
                        Seed = hpoSeed,
                        TotalConfigurationsEvaluated = algRes.Count,
                        ConfigurationResults = algRes
                    };

                    allResults.Add(allConfigsResult);
                    SaveResult(resultFilePath, allConfigsResult);

                    // Summary stats
                    var bestConfig = algRes.OrderByDescending(r => r.BestFitness).First();
                    Console.WriteLine($"Evaluated {algRes.Count} configurations. Best fitness: {bestConfig.BestFitness:F4}");
                }
                catch (NotImplementedException)
                {
                    Console.WriteLine($"Not implemented yet.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error: {ex.Message}");
                }
            }
        }
    }

    // Save aggregate results
    SaveAllResults("./results/all_results.json", allResults);
    Console.WriteLine($"Experiment complete. Results saved to ./results/");
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
