namespace HyperOp.Algorithms.Feynman.Instances
{
    public class Feynman46 : FeynmanDescriptor
    {
        private readonly int testSamples;
        private readonly int trainingSamples;

        public Feynman46() : this((int)DateTime.Now.Ticks, 10000, 10000, null) { }

        public Feynman46(int seed)
        {
            Seed = seed;
            trainingSamples = 10000;
            testSamples = 10000;
            noiseRatio = null;
        }

        public Feynman46(int seed, int trainingSamples, int testSamples, double? noiseRatio)
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
                return string.Format("I.43.31 mob*kb*T | {0}",
                  noiseRatio == null ? "no noise" : string.Format(System.Globalization.CultureInfo.InvariantCulture, "noise={0:g}", noiseRatio));
            }
        }

        public override string TargetVariable { get { return noiseRatio == null ? "D" : "D_noise"; } }

        public override string[] VariableNames
        {
            get { return noiseRatio == null ? new[] { "mob", "T", "kb", "D" } : new[] { "mob", "T", "kb", "D", "D_noise" }; }
        }

        public override string[] AllowedInputVariables { get { return new[] { "mob", "T", "kb" }; } }

        public int Seed { get; private set; }

        public override int TrainingPartitionStart { get { return 0; } }
        public override int TrainingPartitionEnd { get { return trainingSamples; } }
        public override int TestPartitionStart { get { return trainingSamples; } }
        public override int TestPartitionEnd { get { return trainingSamples + testSamples; } }

        public override List<List<double>> GenerateValues()
        {
            var rand = new MersenneTwister((uint)Seed);

            var data = new List<List<double>>();
            var mob = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();
            var T = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();
            var kb = ValueGenerator.GenerateUniformDistributedValues(rand.Next(), TestPartitionEnd, 1, 5).ToList();

            var D = new List<double>();

            data.Add(mob);
            data.Add(T);
            data.Add(kb);
            data.Add(D);

            for (var i = 0; i < mob.Count; i++)
            {
                var res = mob[i] * kb[i] * T[i];
                D.Add(res);
            }

            var targetNoise = ValueGenerator.GenerateNoise(D, rand, noiseRatio);
            if (targetNoise != null) data.Add(targetNoise);

            return data;
        }
    }
}