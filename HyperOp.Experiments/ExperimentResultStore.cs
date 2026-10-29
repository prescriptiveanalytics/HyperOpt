using HyperOp.Algorithms;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HyperOp.Experiments;

public static class ExperimentResultStore
{
    private static readonly JsonSerializerOptions JsonOptions = new() {
        WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    // Escape values reversibly: replacing '/' and ':' with '_' would merge distinct model names.
    public static string JobName(string algorithm, string instance, int repetition, int trials,
        int evaluations, int assessments, string? model = null) =>
        $"{Uri.EscapeDataString(algorithm)}" +
        (model is null ? "" : $"__model={Uri.EscapeDataString(model)}") +
        $"__instance={Uri.EscapeDataString(instance)}__rep={repetition}__T={trials}__E={evaluations}__K={assessments}";

    public static string FileName(AggregatedResultDTO result) => JobName(result.AlgorithmName,
        result.InstanceName, result.Repetition, result.NumConfigurations, result.EvaluationsPerConfiguration,
        result.ConfigurationAssessmentResults?.Count ?? 0, result.ModelName) + ".result.json";

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
        if ((result.AlgorithmName == "AgentSearch" && string.IsNullOrWhiteSpace(result.ModelName)) ||
            result.TrainingRows <= 0 || result.ValidationRows <= 0 || result.TestRows <= 0 ||
            tuning is null || tuning.Count == 0 || tuning.Count > result.NumConfigurations ||
            result.TotalConfigurationsEvaluated != tuning.Count ||
            result.SelectedTrialIndex is not int selected || selected < 0 || selected >= tuning.Count ||
            tuning[selected].TestFitness is null || !double.IsFinite(tuning[selected].BestFitness) ||
            tuning.Count(trial => trial.TestFitness is not null) != 1 ||
            result.RecommendedConfiguration is null || string.IsNullOrWhiteSpace(result.RecommendationRule) ||
            !tuning.Any(trial => trial.ConfigurationIndex == result.RecommendedConfigurationIndex &&
                trial.Configuration == result.RecommendedConfiguration) ||
            assessment is null || assessment.Count == 0 ||
            assessment.Any(trial => trial.TestFitness is null || trial.Configuration != result.RecommendedConfiguration ||
                tuning.Any(observed => observed.GPSeed == trial.GPSeed)) ||
            assessment.Any(trial => trial.TerminationReason != "evaluation_limit" || trial.EvaluationsUsed != result.EvaluationsPerConfiguration) ||
            assessment.Select(trial => trial.GPSeed).Distinct().Count() != assessment.Count ||
            tuning.Concat(assessment).Any(trial => trial.GPSeed is null || trial.FitnessMetric != "validation_nmse" ||
                trial.EvaluationsUsed <= 0 || trial.EvaluationsUsed > trial.EvaluationsLimit ||
                (trial.TerminationReason == "evaluation_limit"
                    ? trial.EvaluationsUsed != trial.EvaluationsLimit :
                    trial.TerminationReason != "controller_stop" || result.AlgorithmName != "AgentSearch" ||
                    trial.ControlObservations is not { Count: > 0 } ||
                    trial.ControlObservations[^1].Decision.Action != HyperOp.Algorithms.HyperParameterOptimization.RunAction.Stop ||
                    trial.ControlObservations[^1].Progress.EvaluationsUsed != trial.EvaluationsUsed) ||
                trial.EvaluationsLimit <= 0) ||
            assessment.Any(trial => trial.EvaluationsLimit != result.EvaluationsPerConfiguration))
            throw new InvalidDataException("Incomplete or incompatible experiment result. Use a fresh campaign directory for a changed protocol.");
    }

    private static void WriteAtomically<T>(string path, T value, bool overwrite)
    {
        var temporary = path + $".{Guid.NewGuid():N}.tmp";
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
