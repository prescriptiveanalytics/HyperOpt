using System.Diagnostics;
using LlmTornado;

namespace HyperOp.Algorithms.Logging
{
    // Receives every model request made in its asynchronous scope, timed from sending until the
    // complete response body has been read.
    public interface ILlmHttpObserver
    {
        double Now();
        void Completed(double startMs, int? statusCode, string? body, Exception? error);
    }

    // Times model requests at the HTTP layer. LlmTornado's runner events do not mark every request
    // (requests that follow a tool result have no "prepared" event), and its usage events arrive late.
    public static class LlmHttpTiming
    {
        private static readonly AsyncLocal<ILlmHttpObserver?> CurrentObserver = new();
        private static readonly HttpClient Client = new(new TimingHandler(new SocketsHttpHandler()))
            { Timeout = TimeSpan.FromMinutes(30) };

        static LlmHttpTiming() => EndpointBase.OnHttpClientRequested = _ => Client;

        // Applies to requests made later in the calling asynchronous flow, including awaited callees.
        public static ILlmHttpObserver? Observer
        {
            get => CurrentObserver.Value;
            set => CurrentObserver.Value = value;
        }

        private sealed class TimingHandler(HttpMessageHandler inner) : DelegatingHandler(inner)
        {
            protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            {
                var observer = CurrentObserver.Value;
                if (observer is null) return await base.SendAsync(request, cancellationToken);
                double start = observer.Now();
                try
                {
                    var response = await base.SendAsync(request, cancellationToken);
                    await response.Content.LoadIntoBufferAsync(cancellationToken);
                    observer.Completed(start, (int)response.StatusCode, await response.Content.ReadAsStringAsync(cancellationToken), null);
                    return response;
                }
                catch (Exception ex)
                {
                    observer.Completed(start, null, null, ex);
                    throw;
                }
            }
        }
    }
}
