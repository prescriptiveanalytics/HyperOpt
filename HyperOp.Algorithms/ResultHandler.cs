using HEAL.HeuristicLib.Algorithms;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HyperOp.Algorithms
{
    public static class ResultHandler
    {
        private static readonly JsonSerializerOptions _serializerOptions = new JsonSerializerOptions() {
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            WriteIndented = true
        };

        public static ResultDTO ConvertToResultDTO(List<(AlgorithmParameter, Population<ExpressionTree>)> result, string algorithmName, string feynmanInstanceName, int repetition, int seed)
        {
            return new ResultDTO {
                AlgorithmName = algorithmName,
                FeynmanInstanceName = feynmanInstanceName,
                Repetition = repetition,
                Seed = seed,
                ResultDetails = result.Select(r => new ResultDTO.ResultDetail {
                    Parameter = r.Item1,
                    Population = r.Item2.Select(ind => new ResultDTO.ResultDetail.SimplifiedIndividual {
                        Objective = ind.ObjectiveVector[0],
                        Depth = ind.Candidate.Depth,
                        Complexity = ind.Candidate.Complexity,
                        Length = ind.Candidate.Length,
                        InfixRepresentation = ind.Candidate.ToInfixString()
                    }).ToList()
                }).ToList()
            };
        }

        public static void WriteResultToFile(string filePath, ResultDTO result)
        {
            string jsonString = JsonSerializer.Serialize(result, _serializerOptions);
            System.IO.File.WriteAllText(filePath, jsonString);
        }

        public static void WriteResultsToFile(string filePath, List<ResultDTO> results)
        {
            string jsonString = JsonSerializer.Serialize(results, _serializerOptions);
            System.IO.File.WriteAllText(filePath, jsonString);
        }

        public static ResultDTO ReadResultFromFile(string filePath)
        {
            string jsonString = System.IO.File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<ResultDTO>(jsonString, _serializerOptions);
        }
    }
}
