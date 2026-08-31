namespace HyperOp.Algorithms.Feynman
{
    /// <summary>
    /// Represents an interface for random number generators.
    /// </summary>
    public interface IRandom
    {
        /// <summary>
        /// Resets the random number generator.
        /// </summary>
        void Reset();
        /// <summary>
        /// Resets the random number generator with the given <paramref name="seed"/>.
        /// </summary>
        /// <param name="seed">The new seed.</param>
        void Reset(int seed);

        /// <summary>
        /// Gets a new random number.
        /// </summary>
        /// <returns>A random integer number.</returns>
        int Next();
        /// <summary>
        /// Gets a new random number between 0 and <paramref name="maxVal"/>.
        /// </summary>
        /// <param name="maxVal">The maximal value of the random number (exclusive).</param>
        /// <returns>A random integer number smaller than <paramref name="maxVal"/>.</returns>
        int Next(int maxVal);
        /// <summary>
        /// Gets a new random number between <paramref name="minVal"/> and <paramref name="maxVal"/>.
        /// </summary>
        /// <param name="maxVal">The maximal value of the random number (exclusive).</param>
        /// <param name="minVal">The minimal value of the random number (inclusive).</param>
        /// <returns>A random integer number. (<paramref name="minVal"/> &lt;= x &lt; <paramref name="maxVal"/>).</returns>
        int Next(int minVal, int maxVal);
        /// <summary>
        /// Gets a new double random number.
        /// </summary>
        /// <returns>A random double number.</returns>
        double NextDouble();
    }
}
