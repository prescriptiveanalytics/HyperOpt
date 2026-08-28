namespace HyperOp.Algorithms.Feynman.Instances
{
    public class Feynman82 : FeynmanDescriptor
    {
        private readonly int testSamples;
        private readonly int trainingSamples;

        public Feynman82() : this((int)DateTime.Now.Ticks, 10000, 10000, null) { }

        public Feynman82(int seed)
        {
            Seed = seed;
            trainingSamples = 10000;
            testSamples = 10000;
            noiseRatio = null;
        }

        public Feynman82(int seed, int trainingSamples, int testSamples, double? noiseRatio)
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
                return string.Format(
                  "II.36.38 mom*B/(kb*T)+(mom*alpha*M)/(epsilon*c**2*kb*T) | {0}",
                  noiseRatio == null ? "no noise" : string.Format(System.Globalization.CultureInfo.InvariantCulture, "noise={0:g}", noiseRatio));
            }
        }

        public override string TargetVariable { get { return noiseRatio == null ? "f" : "f_noise"; } }

        public override string[] VariableNames
        {
            get { return noiseRatio == null ? new[] { "mom", "B", "kb", "T", "alpha", "epsilon", "c", "M", "f" } : new[] { "mom", "B", "kb", "T", "alpha", "epsilon", "c", "M", "f", "f_noise" }; }
        }

        public override string[] AllowedInputVariables
        {
            get { return new[] { "mom", "B", "kb", "T", "alpha", "epsilon", "c", "M" }; }
        }

        public int Seed { get; private set; }

        public override int TrainingPartitionStart { get { return 0; } }
        public override int TrainingPartitionEnd { get { return trainingSamples; } }
        public override int TestPartitionStart { get { return trainingSamples; } }
        public override int TestPartitionEnd { get { return trainingSamples + testSamples; } }

        public override List<List<double>> GenerateValues()
        {
            var rand = new MersenneTwister((uint)Seed);

            var data = new List<List<double>>();
            var mom = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 3).ToList();
            var B = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 3).ToList();
            var kb = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 3).ToList();
            var T = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 3).ToList();
            var alpha = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 3).ToList();
            var epsilon = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 3).ToList();
            var c = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 3).ToList();
            var M = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 3).ToList();

            var f = new List<double>();

            data.Add(mom);
            data.Add(B);
            data.Add(kb);
            data.Add(T);
            data.Add(alpha);
            data.Add(epsilon);
            data.Add(c);
            data.Add(M);
            data.Add(f);

            for (var i = 0; i < mom.Count; i++)
            {
                var res = mom[i] * B[i] / (kb[i] * T[i]) +
                          mom[i] * alpha[i] * M[i] / (epsilon[i] * Math.Pow(c[i], 2) * kb[i] * T[i]);
                f.Add(res);
            }

            var targetNoise = ValueGenerator.GenerateNoise(f, rand, noiseRatio);
            if (targetNoise != null) data.Add(targetNoise);

            return data;
        }
    }
}