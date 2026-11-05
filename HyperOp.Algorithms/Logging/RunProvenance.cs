using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;
using HyperOp.Algorithms.HyperParameterOptimization;

namespace HyperOp.Algorithms.Logging
{
    // What produced a result beyond its job name: code version, constant-fitting implementation and settings, and the
    // CPU's vector support, which the vectorized fitting depends on. Results are only comparable when these agree.
    // CodeFingerprint (CodeFingerprint.cs) identifies the code by content; it is the same in a public copy of the code with
    // other commits. GitCommit and GitDirty are no longer recorded (5 November 2026): the commit of the private development
    // repository identifies nothing outside it and must not reach an anonymous artifact. They remain for reading older
    // results, whose cache rule still uses them.
    public sealed record RunProvenance(string? GitCommit, bool? GitDirty, string FitterVersion, ConstantFittingSettings ConstantFitting,
        bool Vector256Accelerated, bool Vector512Accelerated, string Runtime, string OperatingSystem)
    {
        public string? CodeFingerprint { get; init; }

        // Settings and implementation that change GP results; a cached result with others must not be reused.
        public bool SameFitting(RunProvenance? other) =>
            other is not null && other.FitterVersion == FitterVersion && other.ConstantFitting == ConstantFitting;

        // Whether a cached result may be reused by this process: the same code fingerprint. Results without one (before
        // 5 November 2026) fall back to the old rule: same commit, and neither built from uncommitted changes (whose content
        // the commit does not identify). Null when reuse is allowed, otherwise the reason.
        public string? CacheConflict(RunProvenance? cached)
        {
            if (cached?.CodeFingerprint is { } before && CodeFingerprint is { } now)
                return before == now ? null : $"the cached result comes from code {before[..12]}, this build is code {now[..12]}";
            if (cached?.GitCommit is null || GitCommit is null) return "the code version of the cached result or of this build is unknown";
            if (cached.GitCommit != GitCommit) return $"the cached result comes from commit {cached.GitCommit}, this build from {GitCommit}";
            if (cached.GitDirty != false || GitDirty != false) return "the cached result or this build includes uncommitted changes";
            return null;
        }

        public static RunProvenance Capture(ConstantFittingSettings fitting) =>
            new(null, null, ScaledConstantFittingRefiner.Version, fitting,
                Vector256.IsHardwareAccelerated, Vector512.IsHardwareAccelerated,
                RuntimeInformation.FrameworkDescription, RuntimeInformation.OSDescription) { CodeFingerprint = Logging.CodeFingerprint.OfRunningCode() };
    }
}
