namespace HyperOp.Algorithms.Benchmarks;

public abstract class RegressionDescriptor
{
    private readonly Lazy<PreparedRegressionData> preparedData;
    private double targetNoise;

    protected RegressionDescriptor()
    {
        preparedData = new Lazy<PreparedRegressionData>(() => PreparedRegressionData.Derive(PrepareData(), TargetNoise, SampleRows));
    }

    // Small-data variant: evenly spaced subsets of this many training rows and this many validation rows (0: all rows);
    // the test partition is unchanged. Set before PreparedData is used.
    public int SampleRows
    {
        get => sampleRows;
        set
        {
            if (preparedData.IsValueCreated) throw new InvalidOperationException("Set SampleRows before the data are prepared.");
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            sampleRows = value;
        }
    }
    private int sampleRows;

    // Gaussian noise on the training and validation targets, with a standard deviation of this fraction of the clean
    // target's standard deviation; 0 is noiseless. Test is scored against clean and noisy targets. Set before PreparedData is used.
    public double TargetNoise
    {
        get => targetNoise;
        set
        {
            if (preparedData.IsValueCreated) throw new InvalidOperationException("Set TargetNoise before the data are prepared.");
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            targetNoise = value;
        }
    }

    public abstract string Name { get; }
    public abstract string Description { get; }
    public abstract string TargetVariable { get; }
    public abstract string[] VariableNames { get; }
    public abstract string[] AllowedInputVariables { get; }

    // One list per variable, with the target in the final list.
    public abstract List<List<double>> GenerateValues();

    // Prepare once per descriptor; HPO trials and repetitions reuse the same snapshot.
    public PreparedRegressionData PreparedData => preparedData.Value;

    protected virtual PreparedRegressionData PrepareData()
    {
        var columns = GenerateValues();
        int count = columns[0].Count;
        int trainingEnd = (int)(count * 0.6);
        int validationEnd = (int)(count * 0.8);
        return PreparedRegressionData.Create(this, columns,
            Enumerable.Range(0, trainingEnd),
            Enumerable.Range(trainingEnd, validationEnd - trainingEnd),
            Enumerable.Range(validationEnd, count - validationEnd));
    }
}
