namespace HyperOp.Algorithms.Feynman.Instances
{
    public class Feynman71 : FeynmanDescriptor
    {
        private readonly int testSamples;
        private readonly int trainingSamples;

        public Feynman71() : this((int)DateTime.Now.Ticks, 10000, 10000, null) { }

        public Feynman71(int seed)
        {
            Seed = seed;
            trainingSamples = 10000;
            testSamples = 10000;
            noiseRatio = null;
        }

        public Feynman71(int seed, int trainingSamples, int testSamples, double? noiseRatio)
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
                return string.Format("II.21.32 q/(4*pi*epsilon*r*(1-v/c)) | {0}",
                  noiseRatio == null ? "no noise" : string.Format(System.Globalization.CultureInfo.InvariantCulture, "noise={0:g}", noiseRatio));
            }
        }

        public override string TargetVariable { get { return noiseRatio == null ? "Volt" : "Volt_noise"; } }

        public override string[] VariableNames
        {
            get { return noiseRatio == null ? new[] { "q", "epsilon", "r", "v", "c", "Volt" } : new[] { "q", "epsilon", "r", "v", "c", "Volt", "Volt_noise" }; }
        }

        public override string[] AllowedInputVariables { get { return new[] { "q", "epsilon", "r", "v", "c" }; } }

        public int Seed { get; private set; }

        public override int TrainingPartitionStart { get { return 0; } }
        public override int TrainingPartitionEnd { get { return trainingSamples; } }
        public override int TestPartitionStart { get { return trainingSamples; } }
        public override int TestPartitionEnd { get { return trainingSamples + testSamples; } }

        public override List<List<double>> GenerateValues()
        {
            var rand = new MersenneTwister((uint)Seed);

            var data = new List<List<double>>();
            var q = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();
            var epsilon = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();
            var r = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();
            var v = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 2).ToList();
            var c = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 3, 10).ToList();

            var Volt = new List<double>();

            data.Add(q);
            data.Add(epsilon);
            data.Add(r);
            data.Add(v);
            data.Add(c);
            data.Add(Volt);

            for (var i = 0; i < q.Count; i++)
            {
                var res = q[i] / (4 * Math.PI * epsilon[i] * r[i] * (1 - v[i] / c[i]));
                Volt.Add(res);
            }

            var targetNoise = ValueGenerator.GenerateNoise(Volt, rand, noiseRatio);
            if (targetNoise != null) data.Add(targetNoise);

            return data;
        }
    }
}