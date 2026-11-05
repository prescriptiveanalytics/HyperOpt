using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms
{
    public class ResultDTO
    {
        public required AlgorithmParameter Configuration { get; set; }

        public required int ConfigurationIndex { get; set; }
        public int? GPSeed { get; set; }

        public required double BestFitness { get; set; }

        // BestFitness is validation NMSE (lower is better); population statistics are training NMSE.
        public string FitnessMetric { get; set; } = "training_nmse";
        public double? TrainingBestFitness { get; set; }
        // Test NMSE against the clean target; NoisyTestFitness against the noisy test target (noisy data only).
        public double? TestFitness { get; set; }
        public double? NoisyTestFitness { get; set; }
        public string? SelectedExpression { get; set; }
        // Prediction = intercept + slope * SelectedExpression, fitted on training data only.
        public double? LinearScalingSlope { get; set; }
        public double? LinearScalingIntercept { get; set; }

        [System.Text.Json.Serialization.JsonIgnore]
        internal HEAL.HeuristicLib.MachineLearning.LinearlyScaledRegressor? SelectedModel { get; set; }

        public void EvaluateTest(Benchmarks.PreparedRegressionData data)
        {
            if (SelectedModel is null) throw new InvalidOperationException("No selected GP expression is available.");
            TestFitness = data.TestFitness(SelectedModel);
            NoisyTestFitness = data.NoisyTestFitness(SelectedModel);
        }

        public required double MeanPopulationFitness { get; set; }

        public required double WorstPopulationFitness { get; set; }

        // GP execution only; controller inference while GP waits is excluded.
        public required double ExecutionTimeMs { get; set; }
        // Constant fitting (Levenberg-Marquardt) within ExecutionTimeMs, and the candidates it was applied to;
        // its work is outside the evaluation budget.
        public double? ConstantFittingMs { get; set; }
        public int? ConstantFittingCandidates { get; set; }
        // Trial points of constant fitting in this run, each one evaluation of a candidate on the fitting rows.
        public long? ConstantFittingTrialPoints { get; set; }
        public HyperParameterOptimization.ConstantFittingSettings? ConstantFitting { get; set; }
        // Entire trial: setup, GP, controller decisions, calibration and validation scoring.
        public double? TrialWallTimeMs { get; set; }
        // Process CPU time during the trial (all threads); unaffected by other processes competing for cores.
        public double? TrialCpuTimeMs { get; set; }

        public required int EvaluationsUsed { get; set; }

        public required int EvaluationsLimit { get; set; }

        public required int GenerationsCompleted { get; set; }

        public required int PopulationSize { get; set; }

        public required List<IndividualSnapshot> BestIndividualsSnapshot { get; set; }

        public List<GenerationQualitySnapshot>? QualityCurve { get; set; }

        public List<HyperParameterOptimization.RunControlObservation>? ControlObservations { get; set; }
        // Length of the run's final best expression (nodes); breaks ties between trials below the solved threshold.
        public int? SelectedExpressionLength { get; set; }
        // Tuning runs only: the current best expression every E/10 evaluations and at the end, for selection and the
        // anytime analysis. Test NMSE is recorded for analysis only; no method sees it and selection never uses it.
        public List<RunSnapshot>? Snapshots { get; set; }
        public string TerminationReason { get; set; } = "evaluation_limit";

        public class IndividualSnapshot
        {
            public required double ObjectiveValue { get; set; }
            public required int Depth { get; set; }
            public required int Length { get; set; }
            public required double Complexity { get; set; }
            public required string InfixRepresentation { get; set; }
        }

        public class RunSnapshot
        {
            public required int Evaluations { get; set; }
            public required int Generation { get; set; }
            [System.Text.Json.Serialization.JsonNumberHandling(System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals)]
            public required double ValidationNmse { get; set; }
            [System.Text.Json.Serialization.JsonNumberHandling(System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals)]
            public required double TestNmse { get; set; }
            [System.Text.Json.Serialization.JsonNumberHandling(System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals)]
            public double? NoisyTestNmse { get; set; }
            public required int Length { get; set; }
            public required string Expression { get; set; }
            // The run's final state (also when it ended between two snapshot points).
            public bool Final { get; set; }
        }

        public class GenerationQualitySnapshot
        {
            public required int GenerationNumber { get; set; }
            public required double BestQuality { get; set; }
            public required double MedianQuality { get; set; }
            public required double WorstQuality { get; set; }
        }
    }
}
