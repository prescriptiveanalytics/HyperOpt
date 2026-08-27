using HeuristicLab.Common;

namespace HyperOp.Algorithms.Feynman
{
    public abstract class FeynmanDescriptor
    {
        protected double? noiseRatio;
        public abstract string Name { get; }
        public string Description
        {
            get
            {
                return "Feynman instances ... descriptions follows: " + Environment.NewLine;
            }
        }
        protected abstract string TargetVariable { get; }
        protected abstract string[] VariableNames { get; }
        protected abstract string[] AllowedInputVariables { get; }
        protected abstract int TrainingPartitionStart { get; }
        protected abstract int TrainingPartitionEnd { get; }
        protected abstract int TestPartitionStart { get; }
        protected abstract int TestPartitionEnd { get; }


        public List<double> GetNoisyTarget(List<double> target, IRandom rand)
        {
            if (noiseRatio == null) return null;

            var targetNoise = new List<double>();
            var sigmaNoise = Math.Sqrt(noiseRatio.Value) * target.StandardDeviationPop();
            targetNoise.AddRange(target.Select(md => md + NormalDistributedRandomPolar.NextDouble(rand, 0, sigmaNoise)));
            return targetNoise;
        }
        //protected int TrainingPartitionStart { get { return 0; } }
        //protected int TrainingPartitionEnd { get { return 100; } }
        //protected int TestPartitionStart { get { return 100; } }
        //protected int TestPartitionEnd { get { return 200; } }

        protected abstract List<List<double>> GenerateValues();

    }
}
