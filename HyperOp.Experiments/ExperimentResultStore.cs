using HyperOp.Algorithms;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HyperOp.Experiments;

public static class ExperimentResultStore
{
    private static readonly JsonSerializerOptions JsonOptions = new() {
        WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    // A file name may have at most 255 characters on NTFS; the longest suffix is the protocol's
    // "__attempt=xxxxxxxx.protocol.jsonl" (33) or ".result.json" plus the temporary ".xxxxxxxx.tmp" (25). Longer job names
    // (all channels with several pilot switches reached 240) replace the variant by a hash of it; the result keeps the full
    // variant, and Run-Experiments.ps1 applies the same rule.
    public const int MaxJobNameLength = 200;

    // Escape values reversibly: replacing '/' and ':' with '_' would merge distinct model names.
    public static string JobName(string algorithm, string instance, int repetition, int trials,
        int evaluations, int assessments, string? model = null, string? variant = null, double targetNoise = 0, int sampleRows = 0)
    {
        string Name(string? shownVariant) =>
            $"{Uri.EscapeDataString(algorithm)}" +
            (model is null ? "" : $"__model={Uri.EscapeDataString(model)}") +
            (shownVariant is null ? "" : $"__{shownVariant}") +
            $"__instance={Uri.EscapeDataString(instance)}" +
            // Noisy data are a different task variant; noiseless names stay unchanged.
            (targetNoise == 0 ? "" : $"__noise={targetNoise.ToString("R", System.Globalization.CultureInfo.InvariantCulture)}") +
            (sampleRows == 0 ? "" : $"__rows={sampleRows}") +
            $"__rep={repetition}__T={trials}__E={evaluations}__K={assessments}";
        string name = Name(variant);
        return name.Length <= MaxJobNameLength || variant is null ? name : Name($"variant={VariantHash(variant)}");
    }

    // First 12 hex digits of the SHA-256 of the variant (UTF-8).
    public static string VariantHash(string variant) =>
        Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(variant)))[..12].ToLowerInvariant();

    public static string FileName(AggregatedResultDTO result) => JobName(result.AlgorithmName,
        result.InstanceName, result.Repetition, result.NumConfigurations, result.EvaluationsPerConfiguration,
        // K counts assessment seeds; every seed runs at each assessment length.
        (result.ConfigurationAssessmentResults?.Count ?? 0) / Math.Max(1, result.AssessmentRunLengths?.Length ?? 1), result.ModelName, result.Variant,
        result.TargetNoise, result.SampleRows) + ".result.json";

    public static AggregatedResultDTO Load(string path)
    {
        var result = JsonSerializer.Deserialize<AggregatedResultDTO>(File.ReadAllText(path), JsonOptions)
            ?? throw new InvalidDataException($"Empty result: {path}");
        Validate(result);
        // This also prevents renamed copies of the same job from entering an aggregate.
        if (Path.GetFileName(path) != FileName(result))
            throw new InvalidDataException($"Result filename does not match its experiment: {path}");
        return result;
    }

    public static void Save(string directory, AggregatedResultDTO result)
    {
        WriteAtomically(Path.Combine(directory, FileName(result)), result, overwrite: false);
    }

    // Only this explicit post-processing step writes all_results.json; workers never do.
    public static int Aggregate(string directory)
    {
        var results = Directory.EnumerateFiles(directory, "*.result.json")
            .Order(StringComparer.Ordinal).Select(Load).ToList();
        if (results.Count == 0) throw new InvalidDataException("No completed *.result.json experiments found.");
        WriteAtomically(Path.Combine(directory, "all_results.json"), results, overwrite: true);
        return results.Count;
    }

    private static void Validate(AggregatedResultDTO result)
    {
        var tuning = result.ConfigurationResults;
        var assessment = result.ConfigurationAssessmentResults;
        int[] lengths = result.AssessmentRunLengths is { Length: > 0 } recorded ? recorded : [result.EvaluationsPerConfiguration];
        // Methods that share the pool start as many runs as the pool allows (each run initializes at least the smallest
        // population of 100); the others start exactly T.
        bool sharedPool = result.AlgorithmName is "AgentSearch" or "RuleControlledSearch" or "RandomRunLengthSearch" or "AshaSearch";
        int maximumTrials = sharedPool ? HyperOp.Algorithms.HyperParameterOptimization.HyperParameterOptimizationAlgorithm.MaximumSharedPoolTrials(
                result.NumConfigurations, result.EvaluationsPerConfiguration, 100) : result.NumConfigurations;
        if ((result.AlgorithmName == "AgentSearch" && string.IsNullOrWhiteSpace(result.ModelName)) ||
            result.TrainingRows <= 0 || result.ValidationRows <= 0 || result.TestRows <= 0 ||
            tuning is null || tuning.Count == 0 || tuning.Count > maximumTrials ||
            result.TotalConfigurationsEvaluated != tuning.Count ||
            result.SelectedTrialIndex is not int selected || selected < 0 || selected >= tuning.Count ||
            tuning[selected].TestFitness is null || !double.IsFinite(tuning[selected].BestFitness) ||
            tuning.Count(trial => trial.TestFitness is not null) != 1 ||
            result.RecommendedConfiguration is null || string.IsNullOrWhiteSpace(result.RecommendationRule) ||
            !tuning.Any(trial => trial.ConfigurationIndex == result.RecommendedConfigurationIndex &&
                trial.Configuration == result.RecommendedConfiguration) ||
            // An empty assessment is valid: pilots run with --assessment-runs 0 (K = 0 in the job name).
            assessment is null ||
            assessment.Any(trial => trial.TestFitness is null || trial.Configuration != result.RecommendedConfiguration ||
                tuning.Any(observed => observed.GPSeed == trial.GPSeed)) ||
            assessment.Count % lengths.Length != 0 ||
            assessment.Any(trial => trial.TerminationReason != "evaluation_limit" || !lengths.Contains(trial.EvaluationsUsed)) ||
            lengths.Any(length => assessment.Count(trial => trial.EvaluationsLimit == length) != assessment.Count / lengths.Length) ||
            // The same seeds at every length, each seed once per length.
            assessment.Select(trial => (trial.GPSeed, trial.EvaluationsLimit)).Distinct().Count() != assessment.Count ||
            assessment.Select(trial => trial.GPSeed).Distinct().Count() != assessment.Count / lengths.Length ||
            tuning.Concat(assessment).Any(trial => trial.GPSeed is null || trial.FitnessMetric != "validation_nmse" ||
                trial.EvaluationsUsed <= 0 || trial.EvaluationsUsed > trial.EvaluationsLimit ||
                (trial.TerminationReason == "evaluation_limit"
                    ? trial.EvaluationsUsed != trial.EvaluationsLimit :
                    trial.TerminationReason != "controller_stop" || result.AlgorithmName is not ("AgentSearch" or "RuleControlledSearch" or "AshaSearch") ||
                    trial.ControlObservations is not { Count: > 0 } ||
                    trial.ControlObservations[^1].Decision.Action != HyperOp.Algorithms.HyperParameterOptimization.RunAction.Stop ||
                    trial.ControlObservations[^1].Progress.EvaluationsUsed != trial.EvaluationsUsed) ||
                trial.EvaluationsLimit <= 0) ||
            assessment.Any(trial => !lengths.Contains(trial.EvaluationsLimit)))
            throw new InvalidDataException("Incomplete or incompatible experiment result. Use a fresh campaign directory for a changed protocol.");
    }

    private static void WriteAtomically<T>(string path, T value, bool overwrite)
    {
        var temporary = path + $".{Guid.NewGuid().ToString("N")[..8]}.tmp";
        try
        {
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write))
            {
                JsonSerializer.Serialize(stream, value, JsonOptions);
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporary, path, overwrite);
        }
        finally
        {
            File.Delete(temporary);
        }
    }
}
