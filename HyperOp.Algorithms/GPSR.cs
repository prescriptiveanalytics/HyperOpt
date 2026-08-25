using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.Experiments;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Operators;
using HEAL.HeuristicLib.Problems.MachineLearning;
using HEAL.HeuristicLib.Random;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace HyperOp.Algorithms
{
    public class GPSR
    {
        private double TestFunctiongenerator(double x0, double x1, double x2, double x3, double x4)
        {
            return (x0 - x1) + Math.Pow((3 * x2), x3) / 2 + Math.Exp(Math.Log(2) * x4) - 3.14;
        }

        private (double[,], double[]) GenerateTestData()
        {
            const int sampleCount = 100;
            const int featureCount = 5;

            double[,] inputArray = new double[sampleCount, featureCount];
            double[] targetArray = new double[sampleCount];
            Random r = new Random(42);

            for (int i = 0; i < sampleCount; i++)
            {
                double x0 = r.NextDouble() * 10;
                double x1 = r.NextDouble() * 10;
                double x2 = r.NextDouble() * 10;
                double x3 = r.NextDouble() * 10;
                double x4 = r.NextDouble() * 10;

                inputArray[i, 0] = x0;
                inputArray[i, 1] = x1;
                inputArray[i, 2] = x2;
                inputArray[i, 3] = x3;
                inputArray[i, 4] = x4;

                targetArray[i] = TestFunctiongenerator(x0, x1, x2, x3, x4);
            }

            return (inputArray, targetArray);
        }

        public async Task<PopulationState<ExpressionTree>> Execute(AlgorithmParameter hyperParameter)
        {
            IReadOnlyList<string> variableNames = ["x0", "x1", "x2", "x3", "x4"];
            string targetVariableName = "y";

            var (inputData, targetData) = GenerateTestData();
            var regressionData = new RegressionData(
                DataFrame.FromMatrix(
                    variableNames,
                    inputData),
                new Series<double>(targetVariableName, targetData));

            var searchSpace = new ExpressionTreeSearchSpace(
                maximumLength: hyperParameter.MaxTreeLength,
                maximumDepth: hyperParameter.MaxTreeDepth,
                operations: Symbols.DefaultOperations,
                variables: variableNames
            );

            var problem = new SymbolicRegressionProblem(
                regressionData,
                Metrics.PearsonR2,
                searchSpace
                );


            var algorithm = GeneticAlgorithm.Create(
                new RampedHalfAndHalfTreeCreator(),
                new SubtreeCrossover(),
                ChooseOneMutator.Create(
                    new NodeReplacementMutator(),
                    new SubtreeMutator(),
                    new LocalPerturbationMutator()),
                selector: TournamentSelector.For(problem, tournamentSize: 5),
                populationSize: hyperParameter.PopulationSize,
                maximumGenerations: hyperParameter.Generations,
                mutationRate: hyperParameter.MutationRate);

            var experiment = algorithm.AsGrid();


            IReadOnlyList<AlgorithmParameter> parameterGrid1 = new List<AlgorithmParameter> {
                new AlgorithmParameter { MutationRate = 1, Elites = 2 },
                new AlgorithmParameter { MutationRate = 3, Elites = 4 }
            };

            var aux = experiment.VaryBy(parameterGrid1, static (algorithm, parameters) => algorithm with { MutationRate = parameters.MutationRate, Elites = parameters.Elites })
                .CreateRun(problem, RandomNumberGenerator.Create(42));

            //Should.Throw<InvalidOperationException>(() => experiment.CreateRun(
            //    MetaAlgorithmTestHelpers.CreateIntegerProblem(),
            //    RandomNumberGenerator.Create(42)));

            return await algorithm.CompleteAsync(problem, RandomNumberGenerator.Create(123));
        }
    }
}