using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.MachineLearning;

namespace HyperOp.Algorithms.Benchmarks;

/// <summary>A frozen data snapshot with disjoint fitting, selection and assessment rows.</summary>
public sealed class PreparedRegressionData
{
    private readonly RegressionData validation;
    private readonly RegressionData test;

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
            var targets = rows.Select(row => columns[^1][row]).ToArray();
            if (targets.All(value => value == targets[0]))
                throw new ArgumentException("NMSE requires nonconstant targets in every partition.");
            return new RegressionData(DataFrame.FromMatrix(descriptor.AllowedInputVariables, inputs),
                new Series<double>(descriptor.TargetVariable, targets));
        }
        Training = Partition(partitions[0]);
        validation = Partition(partitions[1]);
        test = Partition(partitions[2]);
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

    public double ValidationFitness(IRegressor model) => Evaluate(model, validation);

    public double TestFitness(IRegressor model) => Evaluate(model, test);

    // The model already contains training-fitted scaling; held-out targets are used only for scoring.
    private static double Evaluate(IRegressor model, RegressionData data)
    {
        var predictions = model.Predict(data.Inputs);
        foreach (double prediction in predictions.Values.Span)
            if (!double.IsFinite(prediction)) return double.PositiveInfinity;
        double error = Metrics.NMSE.Evaluate(predictions, data.Target);
        return double.IsFinite(error) ? error : double.PositiveInfinity;
    }
}
