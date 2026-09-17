using System;
using System.Collections.Generic;
using System.Text;

namespace HyperOp.Algorithms
{
    public class ResultDTO
    {
        public string AlgorithmName { get; set; }
        public string FeynmanInstanceName { get; set; }
        public int Repetition { get; set; }
        public int Seed { get; set; }
        public List<ResultDetail> ResultDetails { get; set; }

        public class ResultDetail
        {
            public AlgorithmParameter Parameter { get; set; }
            public List<SimplifiedIndividual> Population { get; set; }


            public class SimplifiedIndividual
            {
                public double Objective { get; set; }
                public double Depth { get; set; }
                public double Complexity { get; set; }
                public double Length { get; set; }
                public string InfixRepresentation { get; set; }
            }
        }

    }
}
