#region License Information
/* HeuristicLab
 * Copyright (C) Heuristic and Evolutionary Algorithms Laboratory (HEAL)
 *
 * This file is part of HeuristicLab.
 *
 * HeuristicLab is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HeuristicLab is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HeuristicLab. If not, see <http://www.gnu.org/licenses/>.
 */
#endregion

namespace HyperOp.Algorithms.Feynman
{
    public static class EnumerableStatisticExtensions
    {

        /// <summary>
        /// Calculates the sample standard deviation of values.
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double StandardDeviation(this IEnumerable<double> values)
        {
            return Math.Sqrt(Variance(values));
        }

        /// <summary>
        /// Calculates the population standard deviation of values.
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double StandardDeviationPop(this IEnumerable<double> values)
        {
            return Math.Sqrt(VariancePop(values));
        }

        /// <summary>
        /// Calculates the sample variance of values. (sum (x - x_mean)² / (n-1))
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double Variance(this IEnumerable<double> values)
        {
            return Variance(values, true);
        }

        /// <summary>
        /// Calculates the population variance of values. (sum (x - x_mean)² / n)
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double VariancePop(this IEnumerable<double> values)
        {
            return Variance(values, false);
        }

        private static double Variance(IEnumerable<double> values, bool sampleVariance)
        {
            int m_n = 0;
            double m_oldM = 0.0;
            double m_newM = 0.0;
            double m_oldS = 0.0;
            double m_newS = 0.0;
            foreach (double x in values)
            {
                m_n++;
                if (m_n == 1)
                {
                    m_oldM = m_newM = x;
                    m_oldS = 0.0;
                }
                else
                {
                    m_newM = m_oldM + (x - m_oldM) / m_n;
                    m_newS = m_oldS + (x - m_oldM) * (x - m_newM);

                    // set up for next iteration
                    m_oldM = m_newM;
                    m_oldS = m_newS;
                }
            }

            if (m_n == 0) return double.NaN;
            if (m_n == 1) return 0.0;

            if (sampleVariance) return m_newS / (m_n - 1);
            else return m_newS / m_n;
        }
    }
}
