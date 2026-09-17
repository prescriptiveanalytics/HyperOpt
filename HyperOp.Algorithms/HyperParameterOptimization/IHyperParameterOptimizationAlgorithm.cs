using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Operators;
using HEAL.HeuristicLib.Problems;
using HEAL.HeuristicLib.Problems.MachineLearning;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public abstract class HyperParameterOptimizationAlgorithm
    {
        //We simply store the final population of each hyperparameter configuration, so we can analyze them later. Better to have the raw results than to later realize we missed something
        //TODO: Rethink how to track the best individuals across all hyperparameter configurations. We could store the best individual of each configuration, but then we would lose the context of the population. Maybe we can store the best individual and its fitness along with the hyperparameter configuration, so we can analyze them later.
        public abstract Task<List<(AlgorithmParameter, Population<ExpressionTree>)>> Execute(Feynman.FeynmanDescriptor feynmanInstance, int seed, int evaluations);

        protected (IAlgorithm<ExpressionTree, ExpressionTreeSearchSpace, SymbolicRegressionProblem, PopulationState<ExpressionTree>> Alg, SymbolicRegressionProblem Problem) PrepareAlgorithm(Feynman.FeynmanDescriptor feynmanInstance, AlgorithmParameter parameter)
        {
            var data = feynmanInstance.GenerateValues();
            var inputVariables = data.Take(data.Count - 1).ToList();
            var targetData = data.Last();

            var regressionData = new RegressionData(
                DataFrame.FromMatrix(
                    feynmanInstance.AllowedInputVariables,
                    ConvertInputData(inputVariables)),
                new Series<double>(feynmanInstance.TargetVariable, targetData));

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
            var innerAlg = GeneticAlgorithm.Create(
                new RampedHalfAndHalfTreeCreator(),
                new SubtreeCrossover(),
                mutator,
                maximumGenerations: int.MaxValue,
                selector: TournamentSelector.For(problem, tournamentSize: 5),
                evaluator: new ProblemEvaluator<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>>().LimitEvaluations(parameter.Evaluations),
                populationSize: parameter.PopulationSize,
                mutationRate: parameter.MutationRate
                );
            var alg = innerAlg.WithMaxEvaluatedCandidates(innerAlg.Evaluator, 1000);

            return (alg, problem);
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

        private double[,] ConvertInputData(List<List<double>> inputDataLists)
        {
            int rowCount = inputDataLists[0].Count;
            int colCount = inputDataLists.Count;
            var inputMatrix = new double[rowCount, colCount];
            for (int col = 0; col < colCount; col++)
            {
                for (int row = 0; row < rowCount; row++)
                {
                    inputMatrix[row, col] = inputDataLists[col][row];
                }
            }
            return inputMatrix;
        }
    }
}
