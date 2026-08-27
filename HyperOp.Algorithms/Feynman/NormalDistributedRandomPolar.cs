
namespace HyperOp.Algorithms.Feynman
{

    /// <summary>
    /// Normally distributed random variable.
    /// Uses Marsaglia's polar method
    /// </summary>

    public sealed class NormalDistributedRandomPolar : IRandom
    {
        private double mu;
        /// <summary>
        /// Gets or sets the value for µ.
        /// </summary>
        public double Mu
        {
            get { return mu; }
            set { mu = value; }
        }

        private double sigma;
        /// <summary>
        /// Gets or sets the value for sigma.
        /// </summary>
        public double Sigma
        {
            get { return sigma; }
            set { sigma = value; }
        }

        private IRandom uniform;

        /// <summary>
        /// Initializes a new instance of <see cref="NormalDistributedRandomPolar"/> with µ = 0 and sigma = 1
        /// and a new random number generator.
        /// </summary>
        public NormalDistributedRandomPolar()
        {
            this.mu = 0.0;
            this.sigma = 1.0;
            this.uniform = new MersenneTwister();
        }

        /// <summary>
        /// Initializes a new instance of <see cref="NormalDistributedRandomPolar"/> with the given parameters.
        /// <note type="caution"> The random number generator is not copied!</note>
        /// </summary>    
        /// <param name="uniformRandom">The random number generator.</param>
        /// <param name="mu">The value for µ.</param>
        /// <param name="sigma">The value for sigma.</param>
        public NormalDistributedRandomPolar(IRandom uniformRandom, double mu, double sigma)
        {
            this.mu = mu;
            this.sigma = sigma;
            this.uniform = uniformRandom;
        }

        #region IRandom Members

        /// <inheritdoc cref="IRandom.Reset()"/>
        public void Reset()
        {
            uniform.Reset();
        }

        /// <inheritdoc cref="IRandom.Reset(int)"/>
        public void Reset(int seed)
        {
            uniform.Reset(seed);
        }

        /// <summary>
        /// This method is not implemented.
        /// </summary>
        public int Next()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// This method is not implemented.
        /// </summary>
        public int Next(int maxVal)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// This method is not implemented.
        /// </summary>
        public int Next(int minVal, int maxVal)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Generates a new double random number.
        /// </summary>
        /// <returns>A double random number.</returns>
        public double NextDouble()
        {
            return NormalDistributedRandomPolar.NextDouble(uniform, mu, sigma);
        }

        #endregion


        /**
         * Polar method due to Marsaglia.
         *
         * Devroye, L. Non-Uniform Random Variates Generation. Springer-Verlag,
         * New York, 1986, Ch. V, Sect. 4.4.
         */
        public static double NextDouble(IRandom uniformRandom, double mu, double sigma)
        {
            // we don't use spare numbers (efficency loss but easier for multi-threaded code)
            double u, v, s;
            do
            {
                u = uniformRandom.NextDouble() * 2 - 1;
                v = uniformRandom.NextDouble() * 2 - 1;
                s = u * u + v * v;
            } while (s > 1 || s == 0);
            s = Math.Sqrt(-2.0 * Math.Log(s) / s);
            return mu + sigma * u * s;
        }
    }
}
