using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms;

var res = await new GPSR().Execute(new AlgorithmParameter());

var bestCandidates = res.Population.OrderByDescending(v => v.ObjectiveVector.First()).Take(10);
foreach (var candidate in bestCandidates)
{
    var bestFormula = new InfixExpressionFormatter().Format(candidate.Candidate);
    Console.WriteLine($"bestFormula: {bestFormula}, R²: {candidate.ObjectiveVector.First()}");
}


