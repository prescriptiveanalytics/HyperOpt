using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;

namespace HyperOp.Algorithms.HyperParameterOptimization.AgentComponents
{
    // Identity of a model as served by Ollama; the digest pins the exact weights used in a job.
    public sealed record OllamaModelInfo(string Name, string Digest, long SizeBytes, string? Family,
        string? ParameterSize, string? QuantizationLevel, string ServerVersion);

    public static class OllamaModels
    {
        // One family and quantization at three sizes, so that size is the only factor that differs.
        public static readonly string[] StudyModels = ["qwen3.5:2b-q4_K_M", "qwen3.5:4b-q4_K_M", "qwen3.5:9b-q4_K_M"];

        private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromMinutes(10) };

        public static async Task<OllamaModelInfo> ResolveAsync(string baseUrl, string model)
        {
            var root = new Uri(baseUrl.TrimEnd('/') + "/");
            using var version = JsonDocument.Parse(await Http.GetStringAsync(new Uri(root, "api/version")));
            using var tags = JsonDocument.Parse(await Http.GetStringAsync(new Uri(root, "api/tags")));
            string wanted = model.Contains(':') ? model : model + ":latest";
            foreach (var entry in tags.RootElement.GetProperty("models").EnumerateArray())
            {
                if (entry.GetProperty("name").GetString() != wanted) continue;
                var details = entry.TryGetProperty("details", out var d) ? d : default;
                string? Detail(string name) => details.ValueKind == JsonValueKind.Object && details.TryGetProperty(name, out var value)
                    ? value.GetString() : null;
                return new OllamaModelInfo(wanted, entry.GetProperty("digest").GetString()!, entry.GetProperty("size").GetInt64(),
                    Detail("family"), Detail("parameter_size"), Detail("quantization_level"),
                    version.RootElement.GetProperty("version").GetString()!);
            }
            throw new InvalidOperationException($"Model '{model}' is not available on the Ollama server at {baseUrl}. " +
                $"Pull it first, e.g. docker compose -f docker/agent.compose.yml exec ollama ollama pull {model}");
        }

        // Context window of the loaded model as the server runs it (not the model's maximum); requests beyond it are truncated.
        public static async Task<int?> RuntimeContextLengthAsync(string baseUrl, string model)
        {
            using var running = JsonDocument.Parse(await Http.GetStringAsync(new Uri(new Uri(baseUrl.TrimEnd('/') + "/"), "api/ps")));
            foreach (var entry in running.RootElement.GetProperty("models").EnumerateArray())
                if (entry.GetProperty("name").GetString() == model && entry.TryGetProperty("context_length", out var length))
                    return length.GetInt32();
            return null;
        }

        // Memory of the loaded model as the server reports it: total size and the part in GPU memory. Less in GPU memory than in
        // total means layers run on the CPU, which slows every request.
        public static async Task<(long? Size, long? Vram)> RuntimeMemoryAsync(string baseUrl, string model)
        {
            using var running = JsonDocument.Parse(await Http.GetStringAsync(new Uri(new Uri(baseUrl.TrimEnd('/') + "/"), "api/ps")));
            foreach (var entry in running.RootElement.GetProperty("models").EnumerateArray())
                if (entry.GetProperty("name").GetString() == model)
                    return (entry.TryGetProperty("size", out var size) ? size.GetInt64() : null,
                        entry.TryGetProperty("size_vram", out var vram) ? vram.GetInt64() : null);
            return (null, null);
        }

        // Loads the model into memory and keeps it resident, so loading is not charged to the first decision.
        public static async Task<double> WarmUpAsync(string baseUrl, string model)
        {
            var clock = Stopwatch.StartNew();
            using var response = await Http.PostAsJsonAsync(new Uri(new Uri(baseUrl.TrimEnd('/') + "/"), "api/generate"),
                new { model, prompt = "", keep_alive = -1 });
            response.EnsureSuccessStatusCode();
            return clock.Elapsed.TotalMilliseconds;
        }
    }
}
