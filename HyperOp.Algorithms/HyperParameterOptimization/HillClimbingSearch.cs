using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    // Steepest-descent hill climbing on the grid with random restarts: each climb starts at a seeded random
    // unevaluated grid point, evaluates its unevaluated coordinate neighbors in seeded random order, moves to the
    // best neighbor while it improves validation NMSE, and restarts at a local optimum until the trial cap is reached.
    public class HillClimbingSearch : HyperParameterOptimizationAlgorithm
    {
        private static IEnumerable<AlgorithmParameter> GenerateNeighbors(AlgorithmParameter current)
        {
            foreach (int value in ParameterSpace.PopulationSizes)
                if (value != current.PopulationSize) yield return current with { PopulationSize = value };
            foreach (double value in ParameterSpace.MutationRates)
                if (value != current.MutationRate) yield return current with { MutationRate = value };
            foreach (int value in ParameterSpace.MaxTreeDepths)
                if (value != current.MaxTreeDepth) yield return current with { MaxTreeDepth = value };
            foreach (int value in ParameterSpace.MaxTreeLengths)
                if (value != current.MaxTreeLength) yield return current with { MaxTreeLength = value };
            foreach (var value in ParameterSpace.MutatorTypes)
                if (value != current.MutatorType) yield return current with { MutatorType = value };
        }

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            var random = new Random(seed);
            var startOrder = ParameterSpace.Grid();
            random.Shuffle(startOrder);
            int nextStart = 0;
            var evaluated = new HashSet<AlgorithmParameter>();

            async Task<ResultDTO> Evaluate(AlgorithmParameter configuration)
            {
                int trialIndex = results.Count;
                var result = await EvaluateConfiguration(feynmanInstance, configuration,
                    trialIndex, TrialSeed(seed, trialIndex), evaluationsPerConfiguration);
                // Every executed configuration consumes a trial, including unsuccessful neighbors.
                results.Add(result);
                evaluated.Add(configuration);
                return result;
            }

            for (int climb = 0; results.Count < numConfigurations; climb++)
            {
                while (nextStart < startOrder.Length && evaluated.Contains(startOrder[nextStart])) nextStart++;
                // The whole grid has been evaluated.
                if (nextStart == startOrder.Length) break;
                var current = await Evaluate(startOrder[nextStart]);
                Log.Write("hill_climb_start", new { Climb = climb, Trial = results.Count - 1, current.Configuration });

                while (results.Count < numConfigurations)
                {
                    var neighbors = GenerateNeighbors(current.Configuration).Where(n => !evaluated.Contains(n)).ToArray();
                    // Random order matters when the remaining trials truncate the neighborhood.
                    random.Shuffle(neighbors);
                    var neighborResults = new List<ResultDTO>();
                    foreach (var neighbor in neighbors.Take(numConfigurations - results.Count))
                        neighborResults.Add(await Evaluate(neighbor));
                    var bestNeighbor = neighborResults.Where(r => double.IsFinite(r.BestFitness))
                        .OrderBy(r => r.BestFitness).FirstOrDefault();
                    if (bestNeighbor is null || !(bestNeighbor.BestFitness < current.BestFitness))
                        break;
                    current = bestNeighbor;
                }
            }
            return results;
        }
    }
}
