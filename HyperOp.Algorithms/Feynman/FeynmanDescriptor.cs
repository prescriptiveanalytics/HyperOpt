namespace HyperOp.Algorithms.Feynman
{
    public abstract class FeynmanDescriptor
    {
        public double? noiseRatio;
        public abstract string Name { get; }
        public string Description
        {
            get
            {
                return "Feynman instances ... descriptions follows: " + Environment.NewLine;
            }
        }
        public abstract string TargetVariable { get; }
        public abstract string[] VariableNames { get; }
        public abstract string[] AllowedInputVariables { get; }
        public abstract int TrainingPartitionStart { get; }
        public abstract int TrainingPartitionEnd { get; }
        public abstract int TestPartitionStart { get; }
        public abstract int TestPartitionEnd { get; }


        public List<double> GetNoisyTarget(List<double> target, IRandom rand)
        {
            if (noiseRatio == null) return null;

            var targetNoise = new List<double>();
            var sigmaNoise = Math.Sqrt(noiseRatio.Value) * target.StandardDeviationPop();
            targetNoise.AddRange(target.Select(md => md + NormalDistributedRandomPolar.NextDouble(rand, 0, sigmaNoise)));
            return targetNoise;
        }
        //public int TrainingPartitionStart { get { return 0; } }
        //public int TrainingPartitionEnd { get { return 100; } }
        //public int TestPartitionStart { get { return 100; } }
        //public int TestPartitionEnd { get { return 200; } }

        public abstract List<List<double>> GenerateValues();

    }
}
