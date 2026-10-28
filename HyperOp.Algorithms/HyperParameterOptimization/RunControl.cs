namespace HyperOp.Algorithms.HyperParameterOptimization;

public enum RunAction { Continue, Stop }

public sealed record RunDecision(RunAction Action, string Reason);

// Training-only observations; structural diversity is the fraction of unique infix expressions.
public sealed record RunProgress(int Generation, int EvaluationsUsed, int EvaluationsRemaining,
    double? BestTrainingNmse, double? RecentImprovement, double Diversity, double MedianTreeLength);

public sealed record RunControlObservation(RunProgress Progress, RunDecision Decision,
    double InferenceTimeMs, bool UsedFallback);
