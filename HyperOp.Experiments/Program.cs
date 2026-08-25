using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HyperOp.Algorithms;

var res = await new GSPR().Execute(new AlgorithmParameter());

var bestCandidates = res.Population.OrderByDescending(v => v.ObjectiveVector[0]).Take(10);
foreach (var candidate in bestCandidates)
{
    var bestFormula = new InfixExpressionFormatter().Format(candidate.Candidate);
    Console.WriteLine($"bestFormula: {bestFormula}, R²: {candidate.ObjectiveVector[0]}");
}
