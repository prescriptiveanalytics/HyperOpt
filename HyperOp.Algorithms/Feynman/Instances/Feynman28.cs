namespace HyperOp.Algorithms.Feynman.Instances
{
    public class Feynman28 : FeynmanDescriptor
    {
        private readonly int testSamples;
        private readonly int trainingSamples;

        public Feynman28() : this((int)DateTime.Now.Ticks, 10000, 10000, null) { }

        public Feynman28(int seed)
        {
            Seed = seed;
            trainingSamples = 10000;
            testSamples = 10000;
            noiseRatio = null;
        }

        public Feynman28(int seed, int trainingSamples, int testSamples, double? noiseRatio)
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
                return string.Format("I.29.4 omega/c | {0}",
                  noiseRatio == null ? "no noise" : string.Format(System.Globalization.CultureInfo.InvariantCulture, "noise={0:g}", noiseRatio));
            }
        }

        public override string TargetVariable { get { return noiseRatio == null ? "k" : "k_noise"; } }

        public override string[] VariableNames
        {
            get { return noiseRatio == null ? new[] { "omega", "c", "k" } : new[] { "omega", "c", "k", "k_noise" }; }
        }

        public override string[] AllowedInputVariables { get { return new[] { "omega", "c" }; } }

        public int Seed { get; private set; }

        public override int TrainingPartitionStart { get { return 0; } }
        public override int TrainingPartitionEnd { get { return trainingSamples; } }
        public override int TestPartitionStart { get { return trainingSamples; } }
        public override int TestPartitionEnd { get { return trainingSamples + testSamples; } }

        public override List<List<double>> GenerateValues()
        {
            var rand = new MersenneTwister((uint)Seed);

            var data = new List<List<double>>();
            var omega = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 10).ToList();
            var c = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 10).ToList();

            var k = new List<double>();

            data.Add(omega);
            data.Add(c);
            data.Add(k);

            for (var i = 0; i < omega.Count; i++)
            {
                var res = omega[i] / c[i];
                k.Add(res);
            }

            var targetNoise = ValueGenerator.GenerateNoise(k, rand, noiseRatio);
            if (targetNoise != null) data.Add(targetNoise);

            return data;
        }
    }
}