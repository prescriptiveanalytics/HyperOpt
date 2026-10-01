using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Analysis;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization.Util
{
    public static class HyperParameterOptimizationAlgorithmExtension
    {
        extension(HyperParameterOptimizationAlgorithm algo)
        {
            public (int generationCount,double bestFitness,double meanFitness,double worstFitness,List<ResultDTO.IndividualSnapshot> snapshots,List<ResultDTO.GenerationQualitySnapshot> qualitySnapshots) EvaluateFinalState(PopulationState<ExpressionTree> finalState, List<BestMedianWorstEntry<ExpressionTree>> qualityCurve)
            {
                int generationCount = qualityCurve.Count;
                var worstOfFirstGeneration = qualityCurve.First().Worst.ObjectiveVector[0];

                // Extract metrics from population using API's ObjectiveVector
                var objectives = finalState.Population.Select(ind => ind.ObjectiveVector[0]).ToList();
                var bestFitness = objectives.Max();
                var meanFitness = objectives.Average();
                var worstFitness = objectives.Min();

                // Create individual snapshots for the best individuals
                var snapshots = finalState.Population
                    .Select(ind => new ResultDTO.IndividualSnapshot
                    {
                        ObjectiveValue = ind.ObjectiveVector[0],
                        Depth = ind.Candidate.Depth,
                        Length = ind.Candidate.Length,
                        Complexity = ind.Candidate.Complexity,
                        InfixRepresentation = ind.Candidate.ToInfixString()
                    })
                    .ToList();

                // Create per-generation quality curve snapshots
                List<ResultDTO.GenerationQualitySnapshot> qualitySnapshots = qualityCurve
                    .Select((gen, genIndex) => new ResultDTO.GenerationQualitySnapshot
                    {
                        GenerationNumber = genIndex,
                        BestQuality = gen.Best.ObjectiveVector[0],
                        MedianQuality = gen.Median.ObjectiveVector[0],
                        WorstQuality = gen.Worst.ObjectiveVector[0]
                    })
                    .ToList();

                return (generationCount, bestFitness, meanFitness, worstFitness, snapshots, qualitySnapshots);
            }

            public IEnumerable<double> Discretize(RangeAttribute range, int numberOfPoints)
            {
                if (numberOfPoints <= 0)
                    throw new ArgumentOutOfRangeException(nameof(numberOfPoints));

                double min = Convert.ToDouble(range.Minimum);
                double max = Convert.ToDouble(range.Maximum);

                if (numberOfPoints == 1)
                {
                    yield return min;
                    yield break;
                }

                for (int i = 0; i < numberOfPoints; i++)
                {
                    yield return min + i * (max - min) / (numberOfPoints - 1);
                }
            }

            public T GetMiddleValue<T>(IReadOnlyList<T> values)
            {
                if (values.Count == 0)
                    throw new ArgumentException(
                        "The search space must contain at least one value.");

                return values[values.Count / 2];
            }
        }
    }
}
