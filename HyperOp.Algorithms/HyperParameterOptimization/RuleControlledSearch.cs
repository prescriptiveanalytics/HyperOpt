using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    // The agent's setting with both LLM roles replaced by fixed rules: random configurations share a pool of
    // T x E candidate evaluations (the pool bounds the number of trials, see MaximumSharedPoolTrials), and each run is checked every E/Checkpoints candidate
    // evaluations using the Progress feedback the LLM controller receives. A run stops when
    //  - its best training NMSE improved by less than MinRelativeImprovement since the previous checkpoint, or
    //  - at least MedianMinimumTrials runs have completed and its best training NMSE is worse than the median of
    //    their best training NMSE at the same evaluation count (median stopping rule).
    // Otherwise it continues until the pool is exhausted.
    public class RuleControlledSearch(int checkpoints = CheckpointSchedule.DefaultCheckpoints) : HyperParameterOptimizationAlgorithm
    {
        // Checkpoints per per-configuration budget E; the same setting as the agent's controller.
        public int Checkpoints { get; } = checkpoints > 0 ? checkpoints
            : throw new ArgumentOutOfRangeException(nameof(checkpoints));
        public override string Variant => CheckpointSchedule.Name(Checkpoints);
        public const double MinRelativeImprovement = 0.01;
        public const int MedianMinimumTrials = 3;

        public override string RecommendationRule => "minimum_validation_nmse";

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            int pool = checked(numConfigurations * evaluationsPerConfiguration);
            int interval = CheckpointSchedule.Interval(evaluationsPerConfiguration, Checkpoints);
            var results = new List<ResultDTO>();
            var random = new Random(seed);
            // As for the agent, the pool is the budget: freed evaluations go to further trials, up to the loose cap.
            int maximumTrials = MaximumSharedPoolTrials(numConfigurations, evaluationsPerConfiguration,
                Math.Min(interval, MinimumRunEvaluations(evaluationsPerConfiguration)));
            while (results.Count < maximumTrials)
            {
                int remaining = pool - results.Sum(result => result.EvaluationsUsed);
                // The same end condition as the agent: the smallest legal population no longer fits.
                if (remaining < (int)ParameterSpace.Range(nameof(AlgorithmParameter.PopulationSize)).Minimum) break;
                // Each run may use up to the longest allowed run; checkpoints and the median rule decide how much it uses.
                int allowance = BoundRunAllowance(MaximumRunEvaluations(evaluationsPerConfiguration), evaluationsPerConfiguration, remaining);
                // Rejection sampling keeps the allowance usable: the initial population must fit.
                AlgorithmParameter configuration;
                do configuration = ParameterSpace.Sample(random); while (configuration.PopulationSize > allowance);
                int trialIndex = results.Count;
                var observations = new List<RunControlObservation>();
                var completed = results.ToArray();
                Task<RunDecision> Decide(RunProgress progress, CancellationToken token)
                {
                    var decision = DecideRun(progress, observations, completed, interval);
                    observations.Add(new RunControlObservation(progress, decision, 0, false));
                    Log.Write("checkpoint", new { Trial = trialIndex, Observation = observations[^1] });
                    return Task.FromResult(decision);
                }
                var result = await EvaluateConfiguration(feynmanInstance, configuration, trialIndex,
                    TrialSeed(seed, trialIndex), allowance, Decide, interval);
                result.ControlObservations = observations;
                results.Add(result);
            }
            return results;
        }

        // The stagnation part of the rule, also the agent's rule controller (AgentParameter.Controller "rule"): Stop when the best
        // training NMSE improved by less than MinRelativeImprovement since the previous checkpoint, otherwise null.
        public static RunDecision? StagnationDecision(RunProgress progress, IReadOnlyList<RunControlObservation> previous) =>
            progress.BestTrainingNmse is double now && previous.Count > 0 && previous[^1].Progress.BestTrainingNmse is double before && before > 0 &&
                (before - now) / before < MinRelativeImprovement
                ? new RunDecision(RunAction.Stop, FormattableString.Invariant($"stagnation: relative improvement {(before - now) / before:G3} < {MinRelativeImprovement}"))
                : null;

        public static RunDecision DecideRun(RunProgress progress, IReadOnlyList<RunControlObservation> previous, IReadOnlyList<ResultDTO> completed,
            int checkpointEvaluations)
        {
            double? best = progress.BestTrainingNmse;
            if (StagnationDecision(progress, previous) is RunDecision stagnation)
                return stagnation;
            if (best is double current && completed.Count >= MedianMinimumTrials)
            {
                var references = TrainingReferences(completed, progress.EvaluationsUsed);
                if (references.Length >= MedianMinimumTrials)
                {
                    double median = Median(references);
                    if (current > median)
                        return new RunDecision(RunAction.Stop, FormattableString.Invariant($"median rule: {current:G4} > median {median:G4} of {references.Length} runs at {progress.EvaluationsUsed} evaluations"));
                }
            }
            return new RunDecision(RunAction.Continue, "rules not triggered", checkpointEvaluations);
        }

        // Best training NMSE of every completed run that got at least this far, at this evaluation count, in ascending order.
        // The agent's controller receives a summary of the same values, so both controllers compare runs alike.
        public static double[] TrainingReferences(IEnumerable<ResultDTO> completed, int evaluations) =>
            completed.Select(trial => BestTrainingAt(trial, evaluations)).OfType<double>().Order().ToArray();

        public static double Median(double[] ascending) => (ascending[(ascending.Length - 1) / 2] + ascending[ascending.Length / 2]) / 2;

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
