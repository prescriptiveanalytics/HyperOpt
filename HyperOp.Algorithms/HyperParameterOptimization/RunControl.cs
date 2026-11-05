namespace HyperOp.Algorithms.HyperParameterOptimization;

public enum RunAction { Continue, Stop }

public sealed record RunDecision(RunAction Action, string Reason, int? NextObservationInEvaluations = null);

public static class CheckpointSchedule
{
    public const int DefaultCheckpoints = 10;

    // Candidate evaluations between checkpoints: E/K, so a run that uses E evaluations is observed about K times,
    // whatever its population size, and observation cost grows with the evaluations a run actually uses.
    public static int Interval(int evaluationsPerConfiguration, int checkpoints)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(evaluationsPerConfiguration);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(checkpoints);
        return (evaluationsPerConfiguration + checkpoints - 1) / checkpoints;
    }

    // Part of job and file names, so it must stay file-name safe.
    public static string Name(int checkpoints) => $"checkpoints-per-E={checkpoints}";
}

// Information a run controller receives at a checkpoint. None means no checkpoints: only the final result is seen.
// Progress is the base of every checkpoint; the other channels add sections to it.
[Flags]
public enum FeedbackChannels
{
    None = 0,
    Progress = 1,
    Curve = 2,
    Diversity = 4,
    Size = 8,
    Generalization = 16,
    All = Progress | Curve | Diversity | Size | Generalization
}

public static class FeedbackChannelNames
{
    private static readonly (string Name, FeedbackChannels Channel)[] Names = [
        ("progress", FeedbackChannels.Progress), ("curve", FeedbackChannels.Curve), ("diversity", FeedbackChannels.Diversity),
        ("size", FeedbackChannels.Size), ("generalization", FeedbackChannels.Generalization)];

    // "final", "progress", "all", or a comma- or plus-separated list; any channel implies progress.
    public static FeedbackChannels Parse(string text)
    {
        var parts = text.ToLowerInvariant().Split([',', '+'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts is ["final"]) return FeedbackChannels.None;
        if (parts is ["all"]) return FeedbackChannels.All;
        var channels = FeedbackChannels.None;
        foreach (var part in parts)
        {
            var match = Names.FirstOrDefault(entry => entry.Name == part);
            if (match.Name is null)
                throw new ArgumentException($"Unknown feedback '{part}'. Use final, all, or a combination of {string.Join(", ", Names.Select(entry => entry.Name))}.");
            channels |= match.Channel;
        }
        if (channels == FeedbackChannels.None) throw new ArgumentException("Specify at least one feedback channel.");
        return channels | FeedbackChannels.Progress;
    }

    // Canonical name used in job names and results, e.g. "final", "progress", "progress+curve+size".
    public static string Format(FeedbackChannels channels) => channels == FeedbackChannels.None ? "final"
        : string.Join("+", Names.Where(entry => channels.HasFlag(entry.Channel)).Select(entry => entry.Name));
}

// Training-only state of a run at a checkpoint (Progress), plus the sections of the requested channels.
public sealed record RunProgress(int Generation, int EvaluationsUsed, int EvaluationsRemaining,
    double? BestTrainingNmse, double? RecentImprovement)
{
    public CurveFeedback? Curve { get; init; }
    public DiversityFeedback? Diversity { get; init; }
    public SizeFeedback? Size { get; init; }
    public GeneralizationFeedback? Generalization { get; init; }
}

// Best and median training NMSE of the population at generations since the previous checkpoint (at most ten, evenly spaced).
public sealed record CurveFeedback(int[] Generations, double[] BestTrainingNmse, double[] MedianTrainingNmse);

// UniqueStructures: fraction of distinct expression strings (genotypic). PhenotypicDiversity: one minus the mean
// pairwise phenotypic similarity, the squared Pearson correlation of two expressions' outputs on a fixed sample of
// training inputs (Burlacu et al. 2023); it is invariant to offset and scale, as is fitness under linear scaling.
// Individuals with non-finite outputs are excluded and counted in InvalidOutputs.
// TrainingNmseSpread: interquartile range of finite training NMSE divided by its median.
public sealed record DiversityFeedback(double UniqueStructures, double PhenotypicDiversity, double InvalidOutputs,
    double? TrainingNmseSpread);

// Expression size of the population; AtLengthLimit and AtDepthLimit are fractions of trees at the configured limits.
public sealed record SizeFeedback(double MedianLength, int MaxLength, double MedianDepth, double AtLengthLimit,
    double AtDepthLimit, double? MedianLengthChange);

// Validation NMSE of the current best expression (training-fitted scaling) and its gap to the best training NMSE.
public sealed record GeneralizationFeedback(double ValidationNmseOfBest, double Gap);

public sealed record RunControlObservation(RunProgress Progress, RunDecision Decision,
    double InferenceTimeMs, bool UsedFallback);

// What the LLM sees of a checkpoint: the feedback and the decision with its reason. Inference time, the fallback
// flag and the fixed checkpoint delay are bookkeeping; they stay in the protocol and results.
// The primary selection of a job: the snapshot chosen over all tuning runs and its scores.
public sealed record ExpressionSelection(int TrialIndex, int ConfigurationIndex, int Evaluations, double ValidationNmse, double TestNmse,
    int Length, string Expression)
{
    // Against the noisy test targets; null for noiseless data.
    public double? NoisyTestNmse { get; init; }
}

public sealed record CheckpointView(RunProgress Progress, RunAction Action, string Reason)
{
    public static CheckpointView From(RunControlObservation observation) =>
        new(observation.Progress, observation.Decision.Action, observation.Decision.Reason);
}
