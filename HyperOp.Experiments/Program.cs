using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.HyperParameterOptimization;
using System.Text.Json;

//Clean-Up
//Directory.Delete("./results", true);

if (!Directory.Exists("./results")) { Directory.CreateDirectory("./results"); }
if (!File.Exists("./results/all_results.json")) { ExecuteExperiments(); }
AnalyzeResults();

void ExecuteExperiments()
{
    List<IHyperParameterOptimizationAlgorithm> hyperOpAlgs = new List<IHyperParameterOptimizationAlgorithm>();
    List<FeynmanDescriptor> feynmanDescriptors = FeynmanInstanceProvider.LoadAvailableInstances();
    var budget = TimeSpan.FromMinutes(1);
    var allResults = new List<(string AlgorithmName, string FeynmanInstanceName, List<(AlgorithmParameter, Population<ExpressionTree>)> Results)>();

    foreach (var algorithm in hyperOpAlgs)
    {
        foreach (var feynmanInstance in feynmanDescriptors)
        {
            var algRes = algorithm.Execute(feynmanInstance, new CancellationTokenSource(budget).Token);
            allResults.Add((algorithm.GetType().Name, feynmanInstance.Name, algRes));
            //Immediately save all available results to a file, so that we can analyze them later or resume the experiment if it was interrupted.
            File.WriteAllText($"./results/results_{algorithm.GetType().Name}_{feynmanInstance.Name}.json", JsonSerializer.Serialize(algRes));
        }
    }

    File.WriteAllText($"./results/all_results.json", JsonSerializer.Serialize(allResults));
}

void AnalyzeResults()
{

}