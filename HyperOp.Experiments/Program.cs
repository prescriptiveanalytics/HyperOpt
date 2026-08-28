using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.HyperParameterOptimization;
using System.Text.Json;
using System.Text.Json.Serialization;

//Clean-Up
Directory.Delete("./results", true);

if (!Directory.Exists("./results")) { Directory.CreateDirectory("./results"); }
if (!File.Exists("./results/all_results.json")) { await ExecuteExperiments(); }
AnalyzeResults();

async Task ExecuteExperiments()
{
    List<HyperParameterOptimizationAlgorithm> hyperOpAlgs = new List<HyperParameterOptimizationAlgorithm>() {
        new AgentSearch(),
        new GridSearch(),
        new HillClimbingSearch(),
        new IRaceSearch(),
        new RandomSearch()
    };
    var serializerOptions = new JsonSerializerOptions() {
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        WriteIndented = true
    };
    List<FeynmanDescriptor> feynmanDescriptors = FeynmanInstanceProvider.LoadAvailableInstances().Take(10).ToList();
    var budget = TimeSpan.FromSeconds(30);
    var allResults = new List<ResultDTO>();

    foreach (var algorithm in hyperOpAlgs)
    {
        foreach (var feynmanInstance in feynmanDescriptors)
        {
            Console.WriteLine($"Executing {algorithm.GetType().Name} on instance {feynmanInstance.GetType().Name} with budget {budget.TotalMinutes} minutes.");
            try
            {
                var algRes = await algorithm.Execute(feynmanInstance, new CancellationTokenSource(budget).Token);
                allResults.Add(new ResultDTO {
                    AlgorithmName = algorithm.GetType().Name,
                    FeynmanInstanceName = feynmanInstance.GetType().Name,
                    ResultDetails = algRes.Select(r => new ResultDTO.ResultDetail {
                        Parameter = r.Item1,
                        Population = r.Item2.Select(ind => new ResultDTO.ResultDetail.SimplifiedIndividual {
                            Objective = ind.ObjectiveVector[0],
                            Depth = ind.Candidate.Depth,
                            Complexity = ind.Candidate.Complexity,
                            Length = ind.Candidate.Length,
                            InfixRepresentation = ind.Candidate.ToInfixString()
                        }).ToList()
                    }).ToList()
                });
                //Immediately save all available results to a file, so that we can analyze them later or resume the experiment if it was interrupted.
                File.WriteAllText($"./results/results_{algorithm.GetType().Name}_{feynmanInstance.GetType().Name}.json", JsonSerializer.Serialize(allResults.Last(), serializerOptions));
            }
            catch (NotImplementedException) { /* Simply ignore the NotImplemented for now */}
            catch (Exception ex)
            {
                // Log the error and continue with the next algorithm-instance pair, we don't want one potential failure to stop the entire experiment.
                Console.WriteLine($"Error executing algorithm {algorithm.GetType().Name} on instance {feynmanInstance.GetType().Name}: {ex.Message}");
            }
        }

    }
    File.WriteAllText($"./results/all_results.json", JsonSerializer.Serialize(allResults, serializerOptions));
}

void AnalyzeResults()
{
    if (File.Exists("./results/all_results.json"))
    {
        var content = File.ReadAllText("./results/all_results.json");
        var allResults = JsonSerializer.Deserialize<List<ResultDTO>>(content, new JsonSerializerOptions() { NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals });
    }
}

public class ResultDTO
{
    public string AlgorithmName { get; set; }
    public string FeynmanInstanceName { get; set; }
    public List<ResultDetail> ResultDetails { get; set; }

    public class ResultDetail
    {
        public AlgorithmParameter Parameter { get; set; }
        public List<SimplifiedIndividual> Population { get; set; }


        public class SimplifiedIndividual
        {
            public double Objective { get; set; }
            public double Depth { get; set; }
            public double Complexity { get; set; }
            public double Length { get; set; }
            public string InfixRepresentation { get; set; }
        }
    }

}