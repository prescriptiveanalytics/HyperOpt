using HyperOp.Algorithms.Benchmarks;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    // Untuned reference: the default configuration repeated T times with new GP seeds and E evaluations each,
    // so that validation selection over T runs has the same number of chances as the configurators.
    public class DefaultConfigurationSearch : HyperParameterOptimizationAlgorithm
    {
        public override string RecommendationRule => "default_configuration";

        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            var results = new List<ResultDTO>();
            for (int trialIndex = 0; trialIndex < numConfigurations; trialIndex++)
                results.Add(await EvaluateConfiguration(feynmanInstance, new AlgorithmParameter(),
                    trialIndex, TrialSeed(seed, trialIndex), evaluationsPerConfiguration));
            return results;
        }

        // The recommendation does not depend on tuning: every trial ran the default.
        public override ResultDTO NominateConfiguration(IReadOnlyList<ResultDTO> tuningResults) => tuningResults[0];
    }
}
