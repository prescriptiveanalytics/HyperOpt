namespace HyperOp.Algorithms.Benchmarks;

public abstract class RegressionDescriptor
{
    private readonly Lazy<PreparedRegressionData> preparedData;

    protected RegressionDescriptor()
    {
        preparedData = new Lazy<PreparedRegressionData>(PrepareData);
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
