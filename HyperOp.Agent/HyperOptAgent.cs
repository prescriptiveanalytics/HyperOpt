using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;
using HyperOp.Agent.Util;
using HyperOp.Algorithms;
using LlmTornado;
using LlmTornado.Agents;
using LlmTornado.Chat;
using LlmTornado.Chat.Models;
using Microsoft.Extensions.Configuration;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Reflection;

namespace HyperOp.Agent
{
    // Agent can easier handle this than PopulationState<ExpressionTree>
    public record SymbolicRegressionResult
    (
        AlgorithmParameter UsedParameters,
        double QualityMetrik,
        TimeSpan Runtime,
        string? Expression = null
    );

    public enum ToolErrorType
    {
        InvalidParameters,
        ConfigurationAlreadyEvaluated
    }

    public record ToolResult<T>(
        bool Success,
        T? Result = default,
        ToolErrorType? ErrorType = null,
        string? ErrorMessage = null
    );

    public record Experiment {
        public required AlgorithmParameter HyperParameters { get; init; }
        public required SymbolicRegressionResult Result { get; init; }
    }

    public class HyperOptAgent
    {
        // Initialize the respective LLM Tornado components
        private TornadoApi _api;
        private ChatModel _model;
        private TornadoAgent _agent;
        private int _maxTurns;
        private string _systemPrompt;
        //private string _agentInput;
        private TimeSpan _timeLimit;

        // TODO: Think about this
        private GPSR symRegInstance;
        private RegressionData? ProblemData;

        public HashSet<AlgorithmParameter> ParameterHistory = new HashSet<AlgorithmParameter>();
        private List<Experiment> _experimentHistory = new();

        public HyperOptAgent(AgentParameter agentParameters, GPSR symReg)
        {
            symRegInstance = symReg;
            string openRouterAPIKey = GetAPIKeyFromConfiguration();
            _api = new TornadoApi(LlmTornado.Code.LLmProviders.OpenRouter, openRouterAPIKey);
            _model = new ChatModel(agentParameters.ModelName, LlmTornado.Code.LLmProviders.OpenRouter);
            _timeLimit = agentParameters.TimeLimit;
            // Taken from aisra Code
            _systemPrompt = Assembly.GetExecutingAssembly()
                    .Let(asm => asm.ReadEmbeddedTextFile($"{asm.GetName().Name}.Resources.{agentParameters.SytstemPrompt}.md"));

            _agent = new TornadoAgent(
                    client: _api,
                    model: _model,
                    instructions: _systemPrompt,
                    tools: [],
                    streaming: agentParameters.Streaming
                    );
            _agent.Options.Temperature = agentParameters.Temperature;
            _agent.Options.ServiceTier = agentParameters.ServiceTiers;
            _maxTurns = agentParameters.Turns;
        }

        private string GetAPIKeyFromConfiguration()
        {
            // Im HyperOptAgent Project
            // dotnet user-secrets init
            // dotnet user-secrets set "OpenRouterAPIKey" "sk-..."
            IConfiguration configuration = new ConfigurationBuilder()
                .AddUserSecrets<HyperOptAgent>() // Loads user secrets for the specified assembly
                .Build();
            var openRouterAPIKey = configuration["OpenRouterAPIKey"];
            if (string.IsNullOrEmpty(openRouterAPIKey)) throw new Exception("No open router api has been configured in user secrets for OptimizerSystem.csproj");

            return openRouterAPIKey;
        }

        private SymbolicRegressionResult ConvertSymRegResults(PopulationState<ExpressionTree> popResult, AlgorithmParameter hyperParameter)
        {
            // TODO: Implement logic that parses results
            return new SymbolicRegressionResult(hyperParameter, -1.0, _timeLimit, null);
        }

        public async Task Run(string agentInput, RegressionData regData)
        {
            try
            {
                ProblemData = regData;
                using var cts = new CancellationTokenSource(_timeLimit);
                var sw = Stopwatch.StartNew();

                Conversation result = await _agent.Run(
                    agentInput, 
                    maxTurns: _maxTurns, 
                    cancellationToken: cts.Token
                ); 

                var responseTime = sw.Elapsed;
                sw.Restart();
                var conversation = result.Messages;
                // TODO: Think about return value
            }
            catch // Think about error handling
            {
            }
        }

        #region Tool Calls

        [Description(
            "Returns the history of symbolic regression experiments that have already " +
            "been performed, including their hyperparameters and resulting quality metrics. " +
            "Use this to compare previous experiments before selecting the next configuration."
        )]
        public IReadOnlyList<Experiment> GetExperimentHistory()
        {
            return _experimentHistory;
        }

        [Description(
            "Runs a symbolic regression algorithm with the given algorithm parameters and the current problem data." +
            "Returns quality metrics of the respecitve model."
            )]
        public async Task<ToolResult<SymbolicRegressionResult>> RunSymbolicRegression(AlgorithmParameter hyperParameter)
        {
            try
            {
                if (ProblemData == null) throw new Exception("Problem data must not be null"); // This cannot be here hence real exception

                if (_experimentHistory.Any(e => e.HyperParameters == hyperParameter))
                {
                    return new ToolResult<SymbolicRegressionResult>(false, null, ToolErrorType.ConfigurationAlreadyEvaluated, $"Configuration {hyperParameter} has already been used! Try again with different configuration.");
                }
            
                Validator.ValidateObject(
                hyperParameter,
                new ValidationContext(hyperParameter),
                validateAllProperties: true);
            } 
            catch(ValidationException valEx)
            {
                return new ToolResult<SymbolicRegressionResult>(false, null, ToolErrorType.InvalidParameters, $"Configuration {hyperParameter} has invalid values {valEx.Message}. Correct parameters and try again.");
            }

            ParameterHistory.Add(hyperParameter);

            var symRegres = await symRegInstance.ExecuteOnProblem(hyperParameter, ProblemData);
            var result = ConvertSymRegResults(symRegres, hyperParameter);
            _experimentHistory.Add(new Experiment { HyperParameters = hyperParameter, Result=result});
            return  new ToolResult<SymbolicRegressionResult>(true, result);
            
        }

        #endregion ToolCalls
    }
}
