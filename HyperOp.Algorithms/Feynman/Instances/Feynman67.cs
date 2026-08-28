namespace HyperOp.Algorithms.Feynman.Instances
{
    public class Feynman67 : FeynmanDescriptor
    {
        private readonly int testSamples;
        private readonly int trainingSamples;

        public Feynman67() : this((int)DateTime.Now.Ticks, 10000, 10000, null) { }

        public Feynman67(int seed)
        {
            Seed = seed;
            trainingSamples = 10000;
            testSamples = 10000;
            noiseRatio = null;
        }

        public Feynman67(int seed, int trainingSamples, int testSamples, double? noiseRatio)
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
                return string.Format("II.13.23 rho_c_0/sqrt(1-v**2/c**2) | {0}",
                  noiseRatio == null ? "no noise" : string.Format(System.Globalization.CultureInfo.InvariantCulture, "noise={0:g}", noiseRatio));
            }
        }

        public override string TargetVariable { get { return noiseRatio == null ? "rho_c" : "rho_c_noise"; } }

        public override string[] VariableNames
        {
            get { return noiseRatio == null ? new[] { "rho_c_0", "v", "c", "rho_c" } : new[] { "rho_c_0", "v", "c", "rho_c", "rho_c_noise" }; }
        }

        public override string[] AllowedInputVariables { get { return new[] { "rho_c_0", "v", "c" }; } }

        public int Seed { get; private set; }

        public override int TrainingPartitionStart { get { return 0; } }
        public override int TrainingPartitionEnd { get { return trainingSamples; } }
        public override int TestPartitionStart { get { return trainingSamples; } }
        public override int TestPartitionEnd { get { return trainingSamples + testSamples; } }

        public override List<List<double>> GenerateValues()
        {
            var rand = new MersenneTwister((uint)Seed);

            var data = new List<List<double>>();
            var rho_c_0 = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();
            var v = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 2).ToList();
            var c = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 3, 10).ToList();

            var rho_c = new List<double>();

            data.Add(rho_c_0);
            data.Add(v);
            data.Add(c);
            data.Add(rho_c);

            for (var i = 0; i < rho_c_0.Count; i++)
            {
                var res = rho_c_0[i] / Math.Sqrt(1 - Math.Pow(v[i], 2) / Math.Pow(c[i], 2));
                rho_c.Add(res);
            }

            var targetNoise = ValueGenerator.GenerateNoise(rho_c, rand, noiseRatio);
            if (targetNoise != null) data.Add(targetNoise);

            return data;
        }
    }
}