using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.HyperParameterOptimization;
using System.Text.Json;
using System.Text.Json.Serialization;


int evaluations = 10000;
int repetitions = 3;

//Clean-Up for Debugging
#if DEBUG
if (Directory.Exists("./results")) { Directory.Delete("./results", true); }
#endif

if (!Directory.Exists("./results")) { Directory.CreateDirectory("./results"); }
if (!File.Exists("./results/all_results.json")) { await ExecuteExperiments(); }


//AnalyzeResults();
//void AnalyzeResults()
//{
//    if (File.Exists("./results/all_results.json"))
//    {
//        var content = File.ReadAllText("./results/all_results.json");
//        var allResults = JsonSerializer.Deserialize<List<ResultDTO>>(content, new JsonSerializerOptions() { NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals });
//    }
//}

async Task ExecuteExperiments()
{
    List<HyperParameterOptimizationAlgorithm> hyperOpAlgs = new List<HyperParameterOptimizationAlgorithm>() {
        new AgentSearch(),
        new GridSearch(),
        new HillClimbingSearch(),
        new IRaceSearch(),
        new RandomSearch()
    };

    //Fixed Feynman instances seed per instance.
    List<FeynmanDescriptor> feynmanDescriptors = FeynmanInstanceProvider.LoadAvailableInstances(42).Take(10).ToList();
    var allResults = new List<ResultDTO>();

    foreach (var algorithm in hyperOpAlgs)
    {
        foreach (var feynmanInstance in feynmanDescriptors)
        {
            foreach (int repetition in Enumerable.Range(1, repetitions))
            {
                Console.WriteLine($"Executing repetition {repetition} of {algorithm.GetType().Name} on instance {feynmanInstance.GetType().Name} with budget of {evaluations} evaluations.");
                try
                {
                    string resultFilePath = $"./results/results_{algorithm.GetType().Name}_{feynmanInstance.GetType().Name}_{repetition}.json";
                    if (File.Exists(resultFilePath))
                    {
                        Console.WriteLine($"Result file {resultFilePath} already exists. Skipping execution for this repetition.");

                        //Load already existing results so we won't lose them. This is important if we want to resume the experiment after an interruption.
                        allResults.Add(ResultHandler.ReadResultFromFile(resultFilePath));
                        continue;
                    }

                    int seed = 42 + 100 * repetition;
                    List<(AlgorithmParameter, Population<ExpressionTree>)> algRes = await algorithm.Execute(feynmanInstance, seed, evaluations);
                    allResults.Add(ResultHandler.ConvertToResultDTO(algRes, algorithm.GetType().Name, feynmanInstance.GetType().Name, repetition, seed));

                    //Immediately save all available results to a file, so that we can analyze them later or resume the experiment if it was interrupted.
                    ResultHandler.WriteResultToFile(resultFilePath, allResults.Last());
                }
                catch (NotImplementedException) { /* Simply ignore the NotImplemented for now */}
                catch (Exception ex)
                {
                    // Log the error and continue with the next algorithm-instance pair, we don't want one potential failure to stop the entire experiment.
                    Console.WriteLine($"Error executing algorithm {algorithm.GetType().Name} on instance {feynmanInstance.GetType().Name}: {ex.Message}");
                }
            }
        }
        ResultHandler.WriteResultsToFile($"./results/all_results.json", allResults);
    }
}

