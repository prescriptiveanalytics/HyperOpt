using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    // The agent's setting with both LLM roles replaced by fixed rules: random configurations share a pool of
    // T x E candidate evaluations across at most T trials, and each run is checked every CheckpointInterval
    // generations using the Progress feedback the LLM controller receives. A run stops when
    //  - its best training NMSE improved by less than MinRelativeImprovement since the previous checkpoint, or
    //  - at least MedianMinimumTrials runs have completed and its best training NMSE is worse than the median of
    //    their best training NMSE at the same evaluation count (median stopping rule).
    // Otherwise it continues until the pool is exhausted.
    public class RuleControlledSearch(int checkpointInterval = 10) : HyperParameterOptimizationAlgorithm
    {
        // Generations between checkpoints; the same setting as the agent's controller.
        public int CheckpointInterval { get; } = checkpointInterval > 0 ? checkpointInterval
            : throw new ArgumentOutOfRangeException(nameof(checkpointInterval));
        public override string Variant => $"interval={CheckpointInterval}";
        public const double MinRelativeImprovement = 0.01;
        public const int MedianMinimumTrials = 3;

        public override string RecommendationRule => "minimum_validation_nmse";

        // As for the agent, run length is part of the configuration choice.
        public override int AssessmentEvaluations(ResultDTO recommendation, int evaluationsPerConfiguration) => recommendation.EvaluationsUsed;

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            int pool = checked(numConfigurations * evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            var random = new Random(seed);
            while (results.Count < numConfigurations)
            {
                int remaining = pool - results.Sum(result => result.EvaluationsUsed);
                // The same end condition as the agent: the smallest legal population no longer fits.
                if (remaining < (int)ParameterSpace.Range(nameof(AlgorithmParameter.PopulationSize)).Minimum) break;
                // Rejection sampling keeps the remaining pool usable: the initial population must fit.
                AlgorithmParameter configuration;
                do configuration = ParameterSpace.Sample(random); while (configuration.PopulationSize > remaining);
                int trialIndex = results.Count;
                var observations = new List<RunControlObservation>();
                var completed = results.ToArray();
                Task<RunDecision> Decide(RunProgress progress, CancellationToken token)
                {
                    var decision = DecideRun(progress, observations, completed, CheckpointInterval);
                    observations.Add(new RunControlObservation(progress, decision, 0, false));
                    Log.Write("checkpoint", new { Trial = trialIndex, Observation = observations[^1] });
                    return Task.FromResult(decision);
                }
                var result = await EvaluateConfiguration(feynmanInstance, configuration, trialIndex,
                    TrialSeed(seed, trialIndex), remaining, Decide, CheckpointInterval);
                result.ControlObservations = observations;
                results.Add(result);
            }
            return results;
        }

        public static RunDecision DecideRun(RunProgress progress, IReadOnlyList<RunControlObservation> previous, IReadOnlyList<ResultDTO> completed,
            int checkpointInterval)
        {
            double? best = progress.BestTrainingNmse;
            if (best is double now && previous.Count > 0 && previous[^1].Progress.BestTrainingNmse is double before && before > 0 &&
                (before - now) / before < MinRelativeImprovement)
                return new RunDecision(RunAction.Stop, $"stagnation: relative improvement {(before - now) / before:G3} < {MinRelativeImprovement}");
            if (best is double current && completed.Count >= MedianMinimumTrials)
            {
                var references = completed.Select(trial => BestTrainingAt(trial, progress.EvaluationsUsed))
                    .OfType<double>().Order().ToArray();
                if (references.Length >= MedianMinimumTrials)
                {
                    double median = (references[(references.Length - 1) / 2] + references[references.Length / 2]) / 2;
                    if (current > median)
                        return new RunDecision(RunAction.Stop, $"median rule: {current:G4} > median {median:G4} of {references.Length} runs at {progress.EvaluationsUsed} evaluations");
                }
            }
            return new RunDecision(RunAction.Continue, "rules not triggered", checkpointInterval);
        }

        // Best training NMSE a completed run had reached after at most the given number of evaluations;
        // generation g has used population x (g + 1) evaluations. Null when the run never got that far.
        public static double? BestTrainingAt(ResultDTO trial, int evaluations)
        {
            if (trial.QualityCurve is not { Count: > 0 } curve || trial.EvaluationsUsed < evaluations) return null;
            var reached = curve.Where(point => (long)trial.PopulationSize * (point.GenerationNumber + 1) <= evaluations
                    && double.IsFinite(point.BestQuality))
                .Select(point => point.BestQuality).ToArray();
            return reached.Length == 0 ? null : reached.Min();
        }
    }
}
