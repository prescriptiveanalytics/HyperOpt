using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;
using HyperOp.Algorithms.Benchmarks;

namespace HyperOp.Algorithms.HyperParameterOptimization;

// Builds checkpoint feedback for one GP run. It observes every generation (for the quality curve) and computes
// the requested channels from the current population at a checkpoint. Only the generalization channel reads
// validation data; nothing reads test data.
public sealed class RunFeedbackBuilder(PreparedRegressionData data, AlgorithmParameter parameter, FeedbackChannels channels)
{
    private const int MaxCurvePoints = 10;
    private readonly List<(int Generation, double Best, double Median)> _sinceCheckpoint = [];
    private double? _previousBest;
    private double? _previousMedianLength;

    public void Observe(int generation, PopulationState<ExpressionTree> state)
    {
        if (!channels.HasFlag(FeedbackChannels.Curve)) return;
        var finite = FiniteObjectives(state);
        if (finite.Length > 0) _sinceCheckpoint.Add((generation, finite[0], Median(finite)));
    }

    public RunProgress Checkpoint(int generation, int evaluationsUsed, int evaluationsRemaining, PopulationState<ExpressionTree> state)
    {
        var finite = FiniteObjectives(state);
        double? best = finite.Length == 0 ? null : finite[0];
        var progress = new RunProgress(generation, evaluationsUsed, evaluationsRemaining, best,
            _previousBest is double before && best is double now ? before - now : null);
        _previousBest = best;
        var population = state.Population.ToArray();

        if (channels.HasFlag(FeedbackChannels.Curve))
        {
            var points = SampleEvenly(_sinceCheckpoint, MaxCurvePoints);
            progress = progress with { Curve = new CurveFeedback(points.Select(p => p.Generation).ToArray(),
                points.Select(p => p.Best).ToArray(), points.Select(p => p.Median).ToArray()) };
            _sinceCheckpoint.Clear();
        }
        if (channels.HasFlag(FeedbackChannels.Diversity))
        {
            double structures = population.Select(i => i.Candidate.ToInfixString()).Distinct().Count() / (double)population.Length;
            var outputs = population.Select(i => new SymbolicRegressor(i.Candidate, "prediction").Predict(data.DiversityInputs).Values.ToArray()).ToArray();
            var valid = outputs.Where(values => values.All(double.IsFinite)).ToArray();
            progress = progress with { Diversity = new DiversityFeedback(structures, PhenotypicDiversity(valid),
                1 - valid.Length / (double)population.Length, Spread(finite)) };
        }
        if (channels.HasFlag(FeedbackChannels.Size))
        {
            var size = Size(population.Select(i => i.Candidate.Length).ToArray(), population.Select(i => i.Candidate.Depth).ToArray(),
                parameter.MaxTreeLength, parameter.MaxTreeDepth, _previousMedianLength);
            progress = progress with { Size = size };
            _previousMedianLength = size.MedianLength;
        }
        if (channels.HasFlag(FeedbackChannels.Generalization) && best is double training)
        {
            var bestTree = population.Where(i => double.IsFinite(i.ObjectiveVector[0])).MinBy(i => i.ObjectiveVector[0])!.Candidate;
            var model = new SymbolicRegressor(bestTree, "prediction").FitLinearScaling(data.Training);
            double validation = data.MonitoringFitness(model);
            progress = progress with { Generalization = new GeneralizationFeedback(validation, validation - training) };
        }
        return progress;
    }

    // At most count elements, evenly spaced, always including the first and the most recent one.
    public static T[] SampleEvenly<T>(IReadOnlyList<T> items, int count) =>
        items.Count <= count ? items.ToArray()
            : Enumerable.Range(0, count).Select(i => items[(int)Math.Round(i * (items.Count - 1) / (double)(count - 1))]).ToArray();

    // Interquartile range of finite training NMSE over its median; undefined for fewer than four values or a zero median.
    public static double? Spread(double[] sortedFinite) => sortedFinite.Length < 4 || Median(sortedFinite) <= 0 ? null
        : (Quantile(sortedFinite, 0.75) - Quantile(sortedFinite, 0.25)) / Median(sortedFinite);

    public static SizeFeedback Size(int[] lengths, int[] depths, int maxLength, int maxDepth, double? previousMedianLength)
    {
        var sortedLengths = lengths.Select(l => (double)l).Order().ToArray();
        double medianLength = Median(sortedLengths);
        return new SizeFeedback(medianLength, lengths.Max(), Median(depths.Select(d => (double)d).Order().ToArray()),
            lengths.Count(l => l >= maxLength) / (double)lengths.Length, depths.Count(d => d >= maxDepth) / (double)depths.Length,
            previousMedianLength is double previous ? medianLength - previous : null);
    }

    // One minus the mean pairwise squared Pearson correlation of the outputs. Two constant outputs are identical
    // (similarity 1); a constant and a varying output share no variation (similarity 0). Zero for fewer than two outputs.
    public static double PhenotypicDiversity(IReadOnlyList<double[]> outputs)
    {
        if (outputs.Count < 2) return 0;
        var standardized = outputs.Select(Standardize).ToArray();
        double similarity = 0;
        int pairs = 0;
        for (int a = 0; a < standardized.Length; a++)
            for (int b = a + 1; b < standardized.Length; b++, pairs++)
            {
                var (x, y) = (standardized[a], standardized[b]);
                if (x is null || y is null) { similarity += x is null && y is null ? 1 : 0; continue; }
                double r = 0;
                for (int k = 0; k < x.Length; k++) r += x[k] * y[k];
                r /= x.Length;
                similarity += Math.Min(1, r * r);
            }
        return 1 - similarity / pairs;
    }

    // Zero mean and unit variance; null for constant outputs, whose correlation is undefined. The values are first divided by
    // their largest magnitude, so the test is invariant to scale (outputs of 1e-14 and of 1 are treated alike) and squares
    // cannot overflow; an output counts as constant when its spread is zero or below 1e-12 of its absolute mean, a chosen
    // numerical tolerance (so a large offset with tiny variation counts as constant, although double precision could still
    // resolve it; the paper and Supplement S5 state this rule).
    private static double[]? Standardize(double[] values)
    {
        double scale = values.Max(v => Math.Abs(v));
        if (!(scale > 0) || !double.IsFinite(scale)) return null;
        var scaled = values.Select(v => v / scale).ToArray();
        double mean = scaled.Average();
        double sd = Math.Sqrt(scaled.Sum(v => (v - mean) * (v - mean)) / scaled.Length);
        return sd == 0 || sd <= 1e-12 * Math.Abs(mean) ? null : scaled.Select(v => (v - mean) / sd).ToArray();
    }

    private static double[] FiniteObjectives(PopulationState<ExpressionTree> state) =>
        state.Population.Select(i => i.ObjectiveVector[0]).Where(double.IsFinite).Order().ToArray();

    private static double Median(double[] sorted) => (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2;

    private static double Quantile(double[] sorted, double q)
    {
        double position = q * (sorted.Length - 1);
        int lower = (int)Math.Floor(position);
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (position - lower) * (sorted[upper] - sorted[lower]);
    }
}
