using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;

namespace HyperOp.Algorithms.Benchmarks;

/// <summary>A frozen data snapshot with disjoint fitting, selection and assessment rows.</summary>
public sealed class PreparedRegressionData
{
    private readonly RegressionData validation;
    private readonly RegressionData test;
    private readonly double[] validationTargets;
    private readonly double[] testTargets;

    public RegressionData Training { get; }
    public IReadOnlyList<int> TrainingRows { get; }
    public IReadOnlyList<int> ValidationRows { get; }
    public IReadOnlyList<int> TestRows { get; }

    private PreparedRegressionData(RegressionDescriptor descriptor, double[][] columns, int[][] partitions)
    {
        TrainingRows = Array.AsReadOnly(partitions[0]);
        ValidationRows = Array.AsReadOnly(partitions[1]);
        TestRows = Array.AsReadOnly(partitions[2]);
        RegressionData Partition(int[] rows)
        {
            var inputs = new double[rows.Length, columns.Length - 1];
            for (int row = 0; row < rows.Length; row++)
                for (int column = 0; column < columns.Length - 1; column++)
                    inputs[row, column] = columns[column][rows[row]];
            return new RegressionData(DataFrame.FromMatrix(descriptor.AllowedInputVariables, inputs),
                new Series<double>(descriptor.TargetVariable, rows.Select(row => columns[^1][row]).ToArray()));
        }
        Training = Partition(partitions[0]);
        validation = Partition(partitions[1]);
        test = Partition(partitions[2]);
        validationTargets = partitions[1].Select(row => columns[^1][row]).ToArray();
        testTargets = partitions[2].Select(row => columns[^1][row]).ToArray();

    }

    public static PreparedRegressionData Create(RegressionDescriptor descriptor, List<List<double>> source,
        IEnumerable<int> trainingRows, IEnumerable<int> validationRows, IEnumerable<int> testRows)
    {
        string[] names = descriptor.VariableNames;
        if (source.Count != names.Length || source.Count == 0 || names.Distinct().Count() != names.Length)
            throw new ArgumentException("Data columns must match distinct declared variable names.");
        int count = source[0].Count;
        if (source.Any(column => column.Count != count))
            throw new ArgumentException("Data columns have unequal lengths.");
        if (descriptor.AllowedInputVariables.Length == 0 || descriptor.AllowedInputVariables.Contains(descriptor.TargetVariable))
            throw new ArgumentException("Declare inputs separately from the target.");
        // Select by name: a clean target column must not become an input in noisy Feynman variants.
        var selected = descriptor.AllowedInputVariables.Append(descriptor.TargetVariable).Select(name =>
        {
            int index = Array.IndexOf(names, name);
            if (index < 0) throw new ArgumentException($"Unknown variable: {name}");
            return source[index].ToArray();
        }).ToArray();
        if (selected.Any(column => column.Any(value => !double.IsFinite(value))))
            throw new ArgumentException("Prepared observations must be finite.");
        int[][] partitions = [trainingRows.ToArray(), validationRows.ToArray(), testRows.ToArray()];
        var allRows = partitions.SelectMany(rows => rows).ToArray();
        if (partitions.Any(rows => rows.Length < 2) || allRows.Any(row => row < 0 || row >= count) ||
            allRows.Distinct().Count() != allRows.Length)
            throw new ArgumentException("Partitions must have at least two rows each, valid indices and no overlap.");
        return new PreparedRegressionData(descriptor, selected, partitions);
    }

    public double ValidationFitness(ExpressionTree model) => PearsonR2(validationTargets,
        model.Compile(optimize: true).Evaluate(validation.Inputs).ToArray());

    public double TestFitness(ExpressionTree model) => PearsonR2(testTargets,
        model.Compile(optimize: true).Evaluate(test.Inputs).ToArray());

    // Squared correlation, matching the GP objective; this is not 1 - SSE/SST.
    public static double PearsonR2(double[] actual, double[] predicted)
    {
        if (actual.Length < 2 || actual.Length != predicted.Length)
            throw new ArgumentException("Correlation requires equal arrays of at least two observations.");
        if (actual.Any(x => !double.IsFinite(x)) || predicted.Any(x => !double.IsFinite(x))) return 0;
        double actualScale = actual.Max(x => Math.Abs(x)), predictedScale = predicted.Max(x => Math.Abs(x));
        if (actualScale == 0 || predictedScale == 0) return 0;
        // Scaling before centering avoids overflow for finite but very large GP predictions.
        double actualMean = actual.Average(x => x / actualScale), predictedMean = predicted.Average(x => x / predictedScale);
        double covariance = 0, actualVariance = 0, predictedVariance = 0;
        for (int row = 0; row < actual.Length; row++)
        {
            double a = actual[row] / actualScale - actualMean, p = predicted[row] / predictedScale - predictedMean;
            covariance += a * p;
            actualVariance += a * a;
            predictedVariance += p * p;
        }
        if (actualVariance == 0 || predictedVariance == 0) return 0;
        double correlation = covariance / Math.Sqrt(actualVariance) / Math.Sqrt(predictedVariance);
        return Math.Clamp(correlation * correlation, 0, 1);
    }
}
