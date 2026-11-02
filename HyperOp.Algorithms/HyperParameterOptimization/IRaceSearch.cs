using HyperOp.Algorithms.Benchmarks;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    public class IRaceSearch : HyperParameterOptimizationAlgorithm
    {
        private const string RequestPrefix = "HYPEROP_EVALUATE\t";
        private const string RecommendationPrefix = "HYPEROP_RECOMMEND\t";
        private const string InfoPrefix = "HYPEROP_IRACE\t";
        private static readonly SemaphoreSlim ImageLock = new(1, 1);
        // Tagged with a hash of the Dockerfile, so a changed Dockerfile builds a new image instead of reusing a stale one.
        private static string? dockerImage;
        private int? recommendedConfigurationId;

        public override string RecommendationRule => "irace_final_elite";

        public override Task PrepareAsync() => EnsureDockerImage();

        public override ResultDTO NominateConfiguration(IReadOnlyList<ResultDTO> tuningResults) =>
            tuningResults.FirstOrDefault(result => result.ConfigurationIndex == recommendedConfigurationId)
            ?? throw new InvalidOperationException("irace did not nominate an evaluated configuration.");

        // numConfigurations caps target-runner calls, including repeated configurations.
        // ConfigurationIndex is irace's configuration ID, shared by repeated trials.
        public override async Task<List<ResultDTO>> Execute(RegressionDescriptor feynmanInstance, int seed, int numConfigurations, int evaluationsPerConfiguration)
        {
            // irace 4.4.0 rejects smaller budgets with these racing settings (firstTest = 2, one iteration).
            if (numConfigurations < 6)
                throw new ArgumentOutOfRangeException(nameof(numConfigurations), "irace requires at least six target evaluations.");
            ValidateBudget(numConfigurations, evaluationsPerConfiguration);
            recommendedConfigurationId = null;
            await EnsureDockerImage();
            var directory = Directory.CreateTempSubdirectory("hyperop-irace-");
            try
            {
                using (var resource = typeof(IRaceSearch).Assembly.GetManifestResourceStream("HyperOp.Algorithms.Resources.irace.R")
                    ?? throw new InvalidOperationException("The embedded irace runner is missing."))
                using (var script = File.Create(Path.Combine(directory.FullName, "irace.R")))
                    await resource.CopyToAsync(script);
                await File.WriteAllTextAsync(Path.Combine(directory.FullName, "parameters.txt"), CreateParameterDefinitions());

                var start = new ProcessStartInfo {
                    FileName = "docker",
                    WorkingDirectory = directory.FullName,
                    RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
                    UseShellExecute = false, CreateNoWindow = true
                };
                foreach (var arg in new[] { "run", "--rm", "-i", "--mount",
                    $"type=bind,source={directory.FullName},target=/work,readonly", dockerImage! })
                    start.ArgumentList.Add(arg);
                start.ArgumentList.Add("/work/irace.R");
                start.ArgumentList.Add("/work/parameters.txt");
                start.ArgumentList.Add(((long)seed & int.MaxValue).ToString(CultureInfo.InvariantCulture));
                start.ArgumentList.Add(numConfigurations.ToString(CultureInfo.InvariantCulture));

                using var process = new Process { StartInfo = start };
                try { process.Start(); }
                catch (Win32Exception ex)
                {
                    throw new InvalidOperationException("Cannot start irace. Install Docker Desktop and start its Linux container engine.", ex);
                }
                var errors = process.StandardError.ReadToEndAsync();
                var diagnostics = new StringBuilder();
                var results = new List<ResultDTO>();
                try
                {
                    while (await process.StandardOutput.ReadLineAsync() is { } line)
                    {
                        if (line.StartsWith(RecommendationPrefix, StringComparison.Ordinal))
                        {
                            if (recommendedConfigurationId is not null)
                                throw new InvalidOperationException("irace returned more than one final recommendation.");
                            recommendedConfigurationId = int.Parse(line[RecommendationPrefix.Length..], CultureInfo.InvariantCulture);
                            continue;
                        }
                        if (line.StartsWith(InfoPrefix, StringComparison.Ordinal))
                        {
                            var info = line[InfoPrefix.Length..].Split('\t');
                            Log.Write("irace_started", new { IraceVersion = info[0], Iterations = int.Parse(info[1], CultureInfo.InvariantCulture),
                                Image = dockerImage, MaxExperiments = numConfigurations });
                            continue;
                        }
                        if (!line.StartsWith(RequestPrefix, StringComparison.Ordinal))
                        {
                            if (diagnostics.Length < 16_384) diagnostics.AppendLine(line);
                            continue;
                        }
                        if (results.Count >= numConfigurations)
                            throw new InvalidOperationException("irace exceeded its target evaluation budget.");
                        var fields = line[RequestPrefix.Length..].Split('\t');
                        if (fields.Length != 7) throw new InvalidOperationException($"Invalid irace request: {line}");
                        var parameter = new AlgorithmParameter {
                            PopulationSize = int.Parse(fields[2], CultureInfo.InvariantCulture),
                            MutationRate = double.Parse(fields[3], CultureInfo.InvariantCulture),
                            MaxTreeDepth = int.Parse(fields[4], CultureInfo.InvariantCulture),
                            MaxTreeLength = int.Parse(fields[5], CultureInfo.InvariantCulture),
                            MutatorType = Enum.Parse<MutatorType>(fields[6])
                        };
                        var result = await EvaluateConfiguration(feynmanInstance, parameter,
                            int.Parse(fields[0], CultureInfo.InvariantCulture),
                            int.Parse(fields[1], CultureInfo.InvariantCulture), evaluationsPerConfiguration);
                        results.Add(result);
                        // irace minimizes cost, matching validation NMSE.
                        var cost = double.IsFinite(result.BestFitness) ? result.BestFitness : 1e100;
                        await process.StandardInput.WriteLineAsync(cost.ToString("R", CultureInfo.InvariantCulture));
                        await process.StandardInput.FlushAsync();
                    }
                    await process.WaitForExitAsync();
                    var errorText = await errors;
                    if (process.ExitCode != 0)
                        throw new InvalidOperationException($"irace exited with code {process.ExitCode}.\n{errorText}\n{diagnostics}");
                    return results;
                }
                finally
                {
                    if (!process.HasExited)
                    {
                        process.Kill(entireProcessTree: true);
                        await process.WaitForExitAsync();
                    }
                }
            }
            finally { directory.Delete(recursive: true); }
        }

        private static async Task EnsureDockerImage()
        {
            await ImageLock.WaitAsync();
            try
            {
                if (dockerImage is not null) return;
                var daemon = await RunDocker("info", "--format", "{{.ServerVersion}}");
                if (daemon.ExitCode != 0)
                    throw new InvalidOperationException($"Docker is unavailable. Start Docker Desktop with Linux containers enabled.\n{daemon.Output}");
                // MSBuild copies this file beside the executable, independent of the F5 working directory.
                var dockerfile = Path.Combine(AppContext.BaseDirectory, "docker", "irace.Dockerfile");
                if (!File.Exists(dockerfile))
                    throw new FileNotFoundException("The irace Dockerfile is missing from the application output. Rebuild the solution.", dockerfile);
                string tag = "hyperop-irace:" + Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(dockerfile)))[..12];
                var image = await RunDocker("image", "inspect", tag);
                if (image.ExitCode != 0)
                {
                    Console.WriteLine($"Building {tag} from {dockerfile}. The first build may take a few minutes.");
                    var build = await RunDocker("build", "--file", dockerfile, "--tag", tag, Path.GetDirectoryName(dockerfile)!);
                    if (build.ExitCode != 0)
                        throw new InvalidOperationException($"Failed to build the irace Docker image.\n{build.Output}");
                    Console.WriteLine($"Docker image {tag} is ready.");
                }
                dockerImage = tag;
            }
            finally { ImageLock.Release(); }
        }

        private static async Task<(int ExitCode, string Output)> RunDocker(params string[] arguments)
        {
            var start = new ProcessStartInfo("docker") {
                RedirectStandardOutput = true, RedirectStandardError = true,
                UseShellExecute = false, CreateNoWindow = true
            };
            foreach (var argument in arguments) start.ArgumentList.Add(argument);
            using var process = new Process { StartInfo = start };
            try { process.Start(); }
            catch (Win32Exception ex)
            {
                throw new InvalidOperationException("Docker was not found. Install Docker Desktop and start its Linux container engine.", ex);
            }
            var output = process.StandardOutput.ReadToEndAsync();
            var errors = process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();
            return (process.ExitCode, await output + Environment.NewLine + await errors);
        }

        private static string CreateParameterDefinitions()
        {
            var definitions = new StringBuilder();
            foreach (var property in typeof(AlgorithmParameter).GetProperties())
            {
                if (property.GetCustomAttribute<RangeAttribute>() is { } range)
                {
                    var type = property.PropertyType == typeof(int) ? "i" : "r";
                    definitions.AppendLine(FormattableString.Invariant($"{property.Name} \"\" {type} ({range.Minimum}, {range.Maximum})"));
                }
            }
            definitions.AppendLine($"MutatorType \"\" c ({string.Join(", ", Enum.GetNames<MutatorType>())})");
            return definitions.ToString();
        }
    }
}
