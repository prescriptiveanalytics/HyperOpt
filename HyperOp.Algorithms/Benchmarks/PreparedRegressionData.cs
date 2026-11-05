using HEAL.HeuristicLib.Data;
using HEAL.HeuristicLib.MachineLearning;

namespace HyperOp.Algorithms.Benchmarks;

/// <summary>A frozen data snapshot with disjoint fitting, selection and assessment rows.</summary>
public sealed class PreparedRegressionData
{
    private readonly RegressionData validation;
    private readonly RegressionData test;
    // Test inputs with the noisy targets; null for noiseless data.
    private readonly RegressionData? noisyTest;
    private readonly RegressionDescriptor descriptor;
    private readonly double[][] columns;
    private readonly int[][] partitions;

    // Standard deviation of the target noise as a fraction of the clean target's standard deviation; 0 is noiseless.
    public double TargetNoise { get; }
    // Fixed noise stream: every method, budget and repetition sees the same noisy data.
    public const int NoiseSeed = 4242;

    public RegressionData Training { get; }
    // Inputs of 64 evenly spaced training rows (all rows if fewer), used to compare the behavior of expressions.
    public DataFrame DiversityInputs { get; }
    public IReadOnlyList<int> TrainingRows { get; }
    public IReadOnlyList<int> ValidationRows { get; }
    public IReadOnlyList<int> TestRows { get; }
    // Rows the generalization feedback reads; disjoint from ValidationRows in the small-data variant, otherwise the same.
    public IReadOnlyList<int> MonitoringRows { get; }
    private readonly RegressionData monitoring;

    private PreparedRegressionData(RegressionDescriptor descriptor, double[][] columns, int[][] partitions, double targetNoise = 0)
    {
        this.descriptor = descriptor; this.columns = columns; this.partitions = partitions; TargetNoise = targetNoise;
        TrainingRows = Array.AsReadOnly(partitions[0]);
        ValidationRows = Array.AsReadOnly(partitions[1]);
        TestRows = Array.AsReadOnly(partitions[2]);
        // Noise is drawn for every row in row order, so a row's noise does not depend on the partitioning.
        double[]? noisyTarget = null;
        if (targetNoise > 0)
        {
            var clean = columns[^1];
            double mean = clean.Average();
            double sigma = targetNoise * Math.Sqrt(clean.Sum(value => (value - mean) * (value - mean)) / clean.Length);
            var random = new Random(NoiseSeed);
            noisyTarget = clean.Select(value => value + sigma * StandardNormal(random)).ToArray();
        }
        RegressionData Partition(int[] rows, bool noisy = false)
        {
            var inputs = new double[rows.Length, columns.Length - 1];
            for (int row = 0; row < rows.Length; row++)
                for (int column = 0; column < columns.Length - 1; column++)
                    inputs[row, column] = columns[column][rows[row]];
            var targets = rows.Select(row => noisy && noisyTarget is not null ? noisyTarget[row] : columns[^1][row]).ToArray();
            if (targets.All(value => value == targets[0]))
                throw new ArgumentException("NMSE requires nonconstant targets in every partition.");
            return new RegressionData(DataFrame.FromMatrix(descriptor.AllowedInputVariables, inputs),
                new Series<double>(descriptor.TargetVariable, targets));
        }
        // Methods see only training and validation, which carry the noise; test is scored against the clean target
        // (recovery of the true equation) and, for noisy data, also against the noisy one.
        Training = Partition(partitions[0], noisy: true);
        validation = Partition(partitions[1], noisy: true);
        test = Partition(partitions[2]);
        noisyTest = noisyTarget is null ? null : Partition(partitions[2], noisy: true);
        // A fourth partition separates the rows of the generalization feedback from the selection rows.
        MonitoringRows = partitions.Length > 3 ? Array.AsReadOnly(partitions[3]) : ValidationRows;
        monitoring = partitions.Length > 3 ? Partition(partitions[3], noisy: true) : validation;
        var sample = Enumerable.Range(0, Math.Min(64, partitions[0].Length))
            .Select(i => partitions[0][(int)((long)i * partitions[0].Length / Math.Min(64, partitions[0].Length))]).ToArray();
        var sampleInputs = new double[sample.Length, columns.Length - 1];
        for (int row = 0; row < sample.Length; row++)
            for (int column = 0; column < columns.Length - 1; column++)
                sampleInputs[row, column] = columns[column][sample[row]];
        DiversityInputs = DataFrame.FromMatrix(descriptor.AllowedInputVariables, sampleInputs);
    }

    // Box-Muller from System.Random(seed), whose sequence is fixed for a given seed.
    private static double StandardNormal(Random random)
    {
        double u1 = 1 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    // The same data with noisy training and validation targets and, for sampleRows > 0, evenly spaced subsets of that many
    // training and validation rows; noise is drawn for every row of the column, so a kept row has the same noise as in the
    // full noisy data. Test rows are never reduced. Noise 0 and sampleRows 0 return the data unchanged.
    public static PreparedRegressionData Derive(PreparedRegressionData data, double targetNoise, int sampleRows)
    {
        if (targetNoise == 0 && sampleRows == 0) return data;
        int[] Subset(int[] rows) => sampleRows == 0 || sampleRows >= rows.Length ? rows
            : Enumerable.Range(0, sampleRows).Select(i => rows[(int)((long)i * rows.Length / sampleRows)]).ToArray();
        if (sampleRows == 0)
            return new PreparedRegressionData(data.descriptor, data.columns, [data.partitions[0], data.partitions[1], data.partitions[2]], targetNoise);
        // Small data: the validation subset is split in alternating order into selection rows and monitoring rows, so the
        // generalization feedback, read at every checkpoint, never sees the rows that select the expression.
        int[] validationSubset = Subset(data.partitions[1]);
        return new PreparedRegressionData(data.descriptor, data.columns,
            [Subset(data.partitions[0]), validationSubset.Where((_, i) => i % 2 == 0).ToArray(), data.partitions[2],
             validationSubset.Where((_, i) => i % 2 == 1).ToArray()], targetNoise);
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

    private readonly System.Collections.Concurrent.ConcurrentDictionary<int, RegressionData> fittingSamples = new();

    // Evenly spaced training rows for constant fitting (all rows if fewer); fixed per row count, so every candidate,
    // method and repetition is fitted to the same rows. Selection and scoring still use the full partitions.
    public RegressionData FittingSample(int rows) => rows >= TrainingRows.Count ? Training : fittingSamples.GetOrAdd(rows, count =>
    {
        var names = Training.Inputs.Columns.Select(column => column.Name).ToArray();
        var columns = names.Select(name => Training.Inputs.Get<double>(name).Values.ToArray()).ToArray();
        var source = Training.Target.Values.ToArray();
        var inputs = new double[count, names.Length];
        var targets = new double[count];
        for (int i = 0; i < count; i++)
        {
            int row = (int)((long)i * TrainingRows.Count / count);
            for (int c = 0; c < columns.Length; c++) inputs[i, c] = columns[c][row];
            targets[i] = source[row];
        }
        return new RegressionData(DataFrame.FromMatrix(names, inputs), new Series<double>(Training.Target.Name, targets));
    });

    public double ValidationFitness(IRegressor model) => Evaluate(model, validation);

    // NMSE on the monitoring rows (the generalization feedback); the validation rows unless the small-data variant splits them.
    public double MonitoringFitness(IRegressor model) => Evaluate(model, monitoring);

    public double TestFitness(IRegressor model) => Evaluate(model, test);

    // Test NMSE against the noisy test targets; null for noiseless data.
    public double? NoisyTestFitness(IRegressor model) => noisyTest is null ? null : Evaluate(model, noisyTest);

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
