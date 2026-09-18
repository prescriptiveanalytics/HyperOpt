using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Operators;
using HEAL.HeuristicLib.Problems;
using HEAL.HeuristicLib.Problems.MachineLearning;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public abstract class HyperParameterOptimizationAlgorithm
    {
        public abstract Task<List<ResultDTO>> Execute(
            Feynman.FeynmanDescriptor feynmanInstance,
            int seed,
            int numConfigurations,
            int evaluationsPerConfiguration
            );

        protected (GeneticAlgorithm<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>>, SymbolicRegressionProblem) PrepareAlgorithm(
            Feynman.FeynmanDescriptor feynmanInstance,
            AlgorithmParameter parameter,
            int evaluationsLimit)
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
            var algorithm = GeneticAlgorithm.Create(
                new RampedHalfAndHalfTreeCreator(),
                new SubtreeCrossover(),
                mutator,
                selector: TournamentSelector.For(problem, tournamentSize: 5),
                populationSize: parameter.PopulationSize,
                maximumGenerations: int.MaxValue,  // Let evaluationLimit be the stopping criterion
                mutationRate: parameter.MutationRate
                );
            algorithm.Evaluator.LimitEvaluations(evaluationsLimit);

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
