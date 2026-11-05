using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HyperOp.Algorithms.Logging
{
    // An append-only event protocol of one experiment attempt; every record is flushed when written,
    // so an interrupted attempt keeps everything that happened up to the interruption.
    public interface IProtocolLog
    {
        void Write(string type, object? data = null);
    }

    public sealed class NullProtocolLog : IProtocolLog
    {
        public static readonly NullProtocolLog Instance = new();
        public void Write(string type, object? data = null) { }
    }

    public sealed class JsonlProtocolLog : IProtocolLog, IDisposable
    {
        public static readonly JsonSerializerOptions JsonOptions = new() {
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            Converters = { new JsonStringEnumConverter() },
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        private readonly StreamWriter _writer;
        private readonly Stopwatch _clock = Stopwatch.StartNew();
        private readonly Lock _gate = new();
        private long _lastWriteTicks = DateTime.UtcNow.Ticks;

        public string Path { get; }

        // Time of the latest record; the experiment runner treats a long silence as a hung job.
        public DateTime LastWriteUtc => new(Interlocked.Read(ref _lastWriteTicks), DateTimeKind.Utc);

        public JsonlProtocolLog(string path)
        {
            Path = path;
            _writer = new StreamWriter(new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
        }

        public void Write(string type, object? data = null)
        {
            lock (_gate)
            {
                // ElapsedMs is monotonic and comparable across records of one attempt.
                _writer.WriteLine(JsonSerializer.Serialize(new {
                    Utc = DateTime.UtcNow, ElapsedMs = _clock.Elapsed.TotalMilliseconds, Type = type, Data = data
                }, JsonOptions));
                Interlocked.Exchange(ref _lastWriteTicks, DateTime.UtcNow.Ticks);
            }
        }

        public void Dispose()
        {
            lock (_gate) _writer.Dispose();
        }
    }
}
