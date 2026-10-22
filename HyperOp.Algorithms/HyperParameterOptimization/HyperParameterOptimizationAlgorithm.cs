using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Operators;
using HEAL.HeuristicLib.Problems;
using HEAL.HeuristicLib.Problems.MachineLearning;
using HyperOp.Algorithms.Benchmarks;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public abstract class HyperParameterOptimizationAlgorithm
    {
        public abstract Task<List<ResultDTO>> Execute(
            RegressionDescriptor feynmanInstance,
            int seed,
            int numConfigurations,
            int evaluationsPerConfiguration
            );

        protected (GeneticAlgorithm<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>>, SymbolicRegressionProblem) PrepareAlgorithm(
            RegressionDescriptor feynmanInstance,
            AlgorithmParameter parameter,
            int evaluationsLimit)
        {
            var regressionData = feynmanInstance.PreparedData.Training;

            var searchSpace = new ExpressionTreeSearchSpace(
                maximumLength: parameter.MaxTreeLength,
                maximumDepth: parameter.MaxTreeDepth,
                operations: Symbols.DefaultOperations,
                variables: feynmanInstance.AllowedInputVariables
            );
            
            var problem = new SymbolicRegressionProblem(
                regressionData,
                Metrics.PearsonR2,
                searchSpace
                );

            var mutator = CreateMutator(parameter.MutatorType);
            var algorithm = GeneticAlgorithm.Create(
                new RampedHalfAndHalfTreeCreator(),
                new SubtreeCrossover(),
                mutator,
                selector: TournamentSelector.For(problem, tournamentSize: 5),
                populationSize: parameter.PopulationSize,
                maximumGenerations: int.MaxValue,  // Let evaluationLimit be the stopping criterion
                mutationRate: parameter.MutationRate
                );
            //algorithm.Evaluator.LimitEvaluations(evaluationsLimit);

            return (algorithm, problem);
        }

        private IMutator<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>> CreateMutator(MutatorType mutatorType)
        {
            return mutatorType switch {
                MutatorType.NodeReplacement => new NodeReplacementMutator(),
                MutatorType.Subtree => new SubtreeMutator(),
                MutatorType.LocalPerturbation => new LocalPerturbationMutator(),
                MutatorType.Combined => ChooseOneMutator.Create(
                    new NodeReplacementMutator(),
                    new SubtreeMutator(),
                    new LocalPerturbationMutator()),
                _ => throw new ArgumentException($"Unknown mutator type: {mutatorType}")
            };
        }

        protected static void CompleteResult(RegressionDescriptor descriptor, ResultDTO result,
            PopulationState<ExpressionTree> finalState)
        {
            var best = finalState.Population.Where(individual => double.IsFinite(individual.ObjectiveVector[0]))
                .OrderByDescending(individual => individual.ObjectiveVector[0]).FirstOrDefault()
                ?? throw new InvalidOperationException("GP produced no finite training objective.");
            result.TrainingBestFitness = best.ObjectiveVector[0];
            result.SelectedModel = best.Candidate;
            result.SelectedExpression = best.Candidate.ToInfixString();
            result.BestFitness = descriptor.PreparedData.ValidationFitness(best.Candidate);
            result.FitnessMetric = "validation_pearson_r2";
        }
    }
}
