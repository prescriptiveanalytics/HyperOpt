namespace HyperOp.Algorithms.Benchmarks;

public abstract class RegressionDescriptor
{
    public abstract string Name { get; }
    public abstract string Description { get; }
    public abstract string TargetVariable { get; }
    public abstract string[] VariableNames { get; }
    public abstract string[] AllowedInputVariables { get; }

    // One list per variable, with the target in the final list.
    public abstract List<List<double>> GenerateValues();
}
