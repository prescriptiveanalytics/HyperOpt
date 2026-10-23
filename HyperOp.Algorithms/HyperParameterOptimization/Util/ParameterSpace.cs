using System.ComponentModel.DataAnnotations;
using System.Reflection;

namespace HyperOp.Algorithms.HyperParameterOptimization.Util;

// Grid and hill climbing share the same discrete subset of the declared ranges.
internal static class ParameterSpace
{
    public static readonly int[] PopulationSizes = IntegerValues(nameof(AlgorithmParameter.PopulationSize), 10);
    public static readonly double[] MutationRates = Discretize(nameof(AlgorithmParameter.MutationRate), 5).ToArray();
    public static readonly int[] MaxTreeDepths = IntegerValues(nameof(AlgorithmParameter.MaxTreeDepth), 5);
    public static readonly int[] MaxTreeLengths = IntegerValues(nameof(AlgorithmParameter.MaxTreeLength), 5);
    public static readonly MutatorType[] MutatorTypes = Enum.GetValues<MutatorType>();

    public static RangeAttribute Range(string propertyName) =>
        typeof(AlgorithmParameter).GetProperty(propertyName)!.GetCustomAttribute<RangeAttribute>()!;

    private static int[] IntegerValues(string propertyName, int count) =>
        Discretize(propertyName, count).Select(value => (int)Math.Round(value)).Distinct().ToArray();

    // All callers use the fixed counts above (at least two).
    private static IEnumerable<double> Discretize(string propertyName, int count)
    {
        var range = Range(propertyName);
        double min = Convert.ToDouble(range.Minimum);
        double max = Convert.ToDouble(range.Maximum);
        for (int i = 0; i < count; i++)
            yield return min + i * (max - min) / (count - 1);
    }
}
