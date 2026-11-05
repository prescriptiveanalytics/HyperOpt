using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    // Stop-only asynchronous successive halving (after ASHA; Li et al., MLSys 2020) with random configurations, the
    // standard non-LLM way to move budget between configurations by stopping runs. Trials draw from the shared pool of
    // T x E evaluations like the agent's: each run may use up to 4E. Rungs lie at E/10 x Eta^k evaluations (3k, 9k, 27k,
    // 81k at E = 30k). At a rung, the run's best training NMSE after at most the rung's evaluations is recorded, and the
    // run continues only if it ranks within the best floor(n/Eta) of the n runs recorded there, itself included, so the
    // first Eta - 1 runs at a rung stop there (as in ASHA, which promotes nothing without comparison; a grace rule would
    // instead send the first runs to 4E unconditionally, 2 x 4E = 27% of the pool at T = 30); past the last rung a run
    // continues to its allowance. GP runs cannot be resumed, so a run that is not promoted stops for good: unlike ASHA,
    // there are no later promotions. Rungs compare training NMSE, as the rule baseline and the LLM controller do.
    public class AshaSearch(int eta = AshaSearch.DefaultEta) : HyperParameterOptimizationAlgorithm
    {
        public const int DefaultEta = 3;
        public int Eta { get; } = eta >= 2 ? eta : throw new ArgumentOutOfRangeException(nameof(eta));
        public override string Variant => $"eta={Eta}";

        // Rung positions in evaluations: E/10 x Eta^k below the longest allowed run.
        public static int[] Rungs(int evaluationsPerConfiguration, int eta)
        {
            var rungs = new List<int>();
            for (long rung = MinimumRunEvaluations(evaluationsPerConfiguration); rung < MaximumRunEvaluations(evaluationsPerConfiguration); rung *= eta)
                rungs.Add((int)rung);
            return [.. rungs];
        }

        // Promotion rule at one rung: continue if within the best floor(n / Eta) of the n values recorded there (itself included).
        public static bool Promote(double value, IReadOnlyCollection<double> recordedIncludingThis, int eta)
        {
            int rank = 1 + recordedIncludingThis.Count(other => other < value);
            return rank <= recordedIncludingThis.Count / eta;
        }

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            int pool = checked(numConfigurations * evaluationsPerConfiguration);
            int[] rungs = Rungs(evaluationsPerConfiguration, Eta);
            var recorded = rungs.Select(_ => new List<double>()).ToArray();
            var results = new List<ResultDTO>();
            var random = new Random(seed);
            int maximumTrials = MaximumSharedPoolTrials(numConfigurations, evaluationsPerConfiguration, rungs[0]);
            while (results.Count < maximumTrials)
            {
                int remaining = pool - results.Sum(result => result.EvaluationsUsed);
                if (remaining < (int)ParameterSpace.Range(nameof(AlgorithmParameter.PopulationSize)).Minimum) break;
                int allowance = BoundRunAllowance(MaximumRunEvaluations(evaluationsPerConfiguration), evaluationsPerConfiguration, remaining);
                AlgorithmParameter configuration;
                do configuration = ParameterSpace.Sample(random); while (configuration.PopulationSize > allowance);
                int trialIndex = results.Count;
                var observations = new List<RunControlObservation>();
                int rung = 0;
                // The run is observed at every generation boundary; the best training NMSE after at most the rung's
                // evaluations is the rung value, so populations that overshoot a rung are not compared later than others.
                double? bestWithinRung = null;
                Task<RunDecision> Decide(RunProgress progress, CancellationToken token)
                {
                    if (rung >= rungs.Length)
                        return Task.FromResult(new RunDecision(RunAction.Continue, "past the last rung", Math.Max(1, allowance)));
                    if (progress.EvaluationsUsed < rungs[rung])
                    {
                        if (progress.BestTrainingNmse is double below && double.IsFinite(below)) bestWithinRung = below;
                        return Task.FromResult(new RunDecision(RunAction.Continue, "between rungs", 1));
                    }
                    if (progress.EvaluationsUsed == rungs[rung] && progress.BestTrainingNmse is double exact && double.IsFinite(exact)) bestWithinRung = exact;
                    // A first generation already past the rung has no earlier value; its current one is used.
                    double? rungValue = bestWithinRung ?? progress.BestTrainingNmse;
                    RunDecision decision;
                    if (rungValue is not double value || !double.IsFinite(value))
                        decision = new RunDecision(RunAction.Stop, $"rung {rung}: no finite training NMSE");
                    else
                    {
                        recorded[rung].Add(value);
                        bool promoted = Promote(value, recorded[rung], Eta);
                        string reason = $"rung {rung} ({rungs[rung]} evaluations): {(promoted ? "promoted" : "not promoted")}, " +
                            $"rank {1 + recorded[rung].Count(other => other < value)} of {recorded[rung].Count}";
                        rung++;
                        bestWithinRung = progress.BestTrainingNmse is double now && double.IsFinite(now) ? now : null;
                        decision = !promoted ? new RunDecision(RunAction.Stop, reason)
                            // Past the last rung the run continues to its allowance without further checks.
                            : new RunDecision(RunAction.Continue, reason, rung < rungs.Length ? 1 : Math.Max(1, allowance));
                    }
                    observations.Add(new RunControlObservation(progress, decision, 0, false));
                    Log.Write("checkpoint", new { Trial = trialIndex, Observation = observations[^1] });
                    return Task.FromResult(decision);
                }
                var result = await EvaluateConfiguration(feynmanInstance, configuration, trialIndex,
                    TrialSeed(seed, trialIndex), allowance, Decide, firstObservationEvaluations: 1);
                result.ControlObservations = observations;
                results.Add(result);
            }
            return results;
        }
    }
}
