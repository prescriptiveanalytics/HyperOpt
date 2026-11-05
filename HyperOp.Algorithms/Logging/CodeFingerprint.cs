using System.Security.Cryptography;
using System.Text;

namespace HyperOp.Algorithms.Logging
{
    // Content fingerprint of the code and settings that determine results, independent of git history: the same files give
    // the same value in the private development repository and in a public copy with other commits. scripts/code_fingerprint.py
    // computes the identical value for any checkout; keep both in step.
    //
    // Definition: every file below the roots, except in directories named bin, obj, .artifacts or Properties; its path
    // relative to the repository with '/' separators; its bytes with every CR LF replaced by LF (so Windows and Linux
    // checkouts agree); the files in ordinal order of their paths; SHA-256 over the lines "<path>\0<SHA-256 of content, hex>\n".
    public static class CodeFingerprint
    {
        public static readonly string[] Roots = ["HyperOp.Algorithms", "HyperOp.Experiments", "data", "docker"];
        private static readonly HashSet<string> Excluded = new(StringComparer.Ordinal) { "bin", "obj", ".artifacts", "Properties" };

        // The repository root above the running binaries: the first directory that holds all roots' first two entries.
        public static string? FindRoot(string? start = null)
        {
            for (var directory = new DirectoryInfo(start ?? AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
                if (Directory.Exists(Path.Combine(directory.FullName, Roots[0])) && Directory.Exists(Path.Combine(directory.FullName, Roots[1])))
                    return directory.FullName;
            return null;
        }

        public static IReadOnlyList<string> Files(string root) => Roots
            .Where(name => Directory.Exists(Path.Combine(root, name)))
            .SelectMany(name => Directory.EnumerateFiles(Path.Combine(root, name), "*", SearchOption.AllDirectories))
            .Select(path => Path.GetRelativePath(root, path).Replace('\\', '/'))
            .Where(relative => !relative.Split('/').SkipLast(1).Any(Excluded.Contains))
            .Order(StringComparer.Ordinal)
            .ToList();

        public static string Compute(string root)
        {
            using var outer = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            foreach (var relative in Files(root))
            {
                byte[] content = NormalizeLineEndings(File.ReadAllBytes(Path.Combine(root, relative)));
                outer.AppendData(Encoding.UTF8.GetBytes($"{relative}\0{Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant()}\n"));
            }
            return Convert.ToHexString(outer.GetHashAndReset()).ToLowerInvariant();
        }

        // The fingerprint of the repository that holds the running binaries; null when the sources are not found.
        public static string? OfRunningCode() => FindRoot() is { } root ? Compute(root) : null;

        public static byte[] NormalizeLineEndings(byte[] content)
        {
            var output = new List<byte>(content.Length);
            for (int i = 0; i < content.Length; i++)
            {
                if (content[i] == (byte)'\r' && i + 1 < content.Length && content[i + 1] == (byte)'\n') continue;
                output.Add(content[i]);
            }
            return output.ToArray();
        }
    }
}
