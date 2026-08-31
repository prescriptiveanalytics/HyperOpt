namespace HyperOp.Algorithms.Feynman.Instances
{
    public class Feynman98 : FeynmanDescriptor
    {
        private readonly int testSamples;
        private readonly int trainingSamples;

        public Feynman98() : this((int)DateTime.Now.Ticks, 10000, 10000, null) { }

        public Feynman98(int seed)
        {
            Seed = seed;
            trainingSamples = 10000;
            testSamples = 10000;
            noiseRatio = null;
        }

        public Feynman98(int seed, int trainingSamples, int testSamples, double? noiseRatio)
        {
            Seed = seed;
            this.trainingSamples = trainingSamples;
            this.testSamples = testSamples;
            this.noiseRatio = noiseRatio;
        }

        public override string Name
        {
            get
            {
                return string.Format("III.17.37 beta*(1+alpha*cos(theta)) | {0}",
                  noiseRatio == null ? "no noise" : string.Format(System.Globalization.CultureInfo.InvariantCulture, "noise={0:g}", noiseRatio));
            }
        }

        public override string TargetVariable { get { return noiseRatio == null ? "f" : "f_noise"; } }

        public override string[] VariableNames
        {
            get { return noiseRatio == null ? new[] { "beta", "alpha", "theta", "f" } : new[] { "beta", "alpha", "theta", "f", "f_noise" }; }
        }

        public override string[] AllowedInputVariables { get { return new[] { "beta", "alpha", "theta" }; } }

        public int Seed { get; private set; }

        public override int TrainingPartitionStart { get { return 0; } }
        public override int TrainingPartitionEnd { get { return trainingSamples; } }
        public override int TestPartitionStart { get { return trainingSamples; } }
        public override int TestPartitionEnd { get { return trainingSamples + testSamples; } }

        public override List<List<double>> GenerateValues()
        {
            var rand = new MersenneTwister((uint)Seed);

            var data = new List<List<double>>();
            var beta = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();
            var alpha = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();
            var theta = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();

            var f = new List<double>();

            data.Add(beta);
            data.Add(alpha);
            data.Add(theta);
            data.Add(f);

            for (var i = 0; i < beta.Count; i++)
            {
                var res = beta[i] * (1 + alpha[i] * Math.Cos(theta[i]));
                f.Add(res);
            }

            var targetNoise = ValueGenerator.GenerateNoise(f, rand, noiseRatio);
            if (targetNoise != null) data.Add(targetNoise);

            return data;
        }
    }
}