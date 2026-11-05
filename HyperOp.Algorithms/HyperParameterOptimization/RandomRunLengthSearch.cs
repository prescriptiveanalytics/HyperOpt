using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.HyperParameterOptimization.Util;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    // Control with the agent's freedom over run length but without any model or feedback: random configurations, each run
    // with a log-uniform length between the per-run bounds (E/10 to 4E), drawn from the shared pool of T x E evaluations
    // until it is used up (the pool bounds the number of runs). It separates the value of choosing run lengths
    // from the value of the LLM's choices; the expected run length is about 1.06E, so about T runs fit the pool.
    public class RandomRunLengthSearch : HyperParameterOptimizationAlgorithm
    {
        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            int pool = checked(numConfigurations * evaluationsPerConfiguration);
            double logMinimum = Math.Log(MinimumRunEvaluations(evaluationsPerConfiguration));
            double logMaximum = Math.Log(MaximumRunEvaluations(evaluationsPerConfiguration));
            var results = new List<ResultDTO>();
            var random = new Random(seed);
            int maximumTrials = MaximumSharedPoolTrials(numConfigurations, evaluationsPerConfiguration, MinimumRunEvaluations(evaluationsPerConfiguration));
            while (results.Count < maximumTrials)
            {
                int remaining = pool - results.Sum(result => result.EvaluationsUsed);
                // The same end condition as the shared-pool methods: the smallest legal population no longer fits.
                if (remaining < (int)ParameterSpace.Range(nameof(AlgorithmParameter.PopulationSize)).Minimum) break;
                int length = (int)Math.Round(Math.Exp(logMinimum + random.NextDouble() * (logMaximum - logMinimum)));
                int allowance = BoundRunAllowance(length, evaluationsPerConfiguration, remaining);
                // Rejection sampling keeps the allowance usable: the initial population must fit.
                AlgorithmParameter configuration;
                do configuration = ParameterSpace.Sample(random); while (configuration.PopulationSize > allowance);
                results.Add(await EvaluateConfiguration(feynmanInstance, configuration, results.Count,
                    TrialSeed(seed, results.Count), allowance));
            }
            return results;
        }
    }
}
