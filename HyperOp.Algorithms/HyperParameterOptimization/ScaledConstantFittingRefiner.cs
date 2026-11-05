using System.Buffers;
using System.Numerics.Tensors;
using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Operators.Refiners;
using HEAL.HeuristicLib.Problems;
using HEAL.HeuristicLib.Problems.MachineLearning;
using HEAL.HeuristicLib.Random;

namespace HyperOp.Algorithms.HyperParameterOptimization;

/// <summary>Trial points of constant fitting in one GP run (each costs one evaluation of the tree on the fitting rows).</summary>
public sealed class ConstantFittingStatistics
{
    private long _trialPoints;
    public long TrialPoints => Interlocked.Read(ref _trialPoints);
    public void AddTrialPoints(int count) => Interlocked.Add(ref _trialPoints, count);
}

/// <summary>Constant-fitting settings shared by every GP run of a job; recorded with each trial.</summary>
public sealed record ConstantFittingSettings
{
    // Development pilot (15 non-benchmark Feynman equations, E=30000): fitting every candidate on 250 rows cost 19% of
    // fitting on 1000 rows at practically equal trial quality; fitting only 10-25% of candidates lost most of the benefit.
    public const int DefaultIterations = 5;
    public const int DefaultRows = 250;
    // 0 disables constant fitting.
    public int Iterations { get; init; } = DefaultIterations;
    // Evenly spaced training rows used for fitting; 0 or at least the training size uses all rows.
    public int Rows { get; init; } = DefaultRows;
    // Share of new candidates that are fitted (each independently); 1 fits every candidate.
    public double Probability { get; init; } = 1;
}

/// <summary>
/// Constant optimization for every GP run: Levenberg-Marquardt on the evolvable constants of each new candidate,
/// fitted before the candidate is evaluated (Lamarckian; the fitted tree enters the population).
/// </summary>
/// <remarks>
/// The problem scores linearly scaled predictions, so the tree is fitted as <c>a + b·k·f</c>: <c>a</c> and <c>b</c>
/// start at the linear-scaling optimum, <c>k = 1/RMS(f)</c> is fixed, and only the inner constants are kept afterwards
/// (evaluation refits <c>a</c> and <c>b</c>). Derivatives are exact (reverse mode over the tree). Steps are accepted only
/// when they lower the squared error, so the scaled fitting error never increases.
///
/// HeuristicLib's NumericParameterFittingRefiner is not used: MathNet 6.0.0-beta2, which it calls, computes a covariance
/// by SVD after every fit, and that SVD never returns on some badly scaled Jacobians (seen on a GP tree with constants
/// from 1e-9 to exp(24.7)). Here every iteration tries one step from a small Cholesky system with Jacobian columns scaled
/// to unit length; as in MathNet, rejected steps count as iterations, so at most <see cref="MaximumIterations"/> trial
/// points are evaluated and the work per candidate is bounded and deterministic. Its work does not count against the
/// evaluation budget.
/// </remarks>
public sealed record ScaledConstantFittingRefiner
    : SingleCandidateRefiner<ExpressionTree, ExpressionTreeSearchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace>>
{
    // Identifies this implementation in result provenance; change it whenever fitting results can change.
    public const string Version = "scaled-lm-cholesky-2";

    // Dot product in a fixed summation order. TensorPrimitives.Dot aligns its vector loop to the memory address, so
    // the last bits of its sums depended on where the arrays happened to lie, and repeated jobs drifted apart.
    public static double Dot(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= x.Length; i += 4)
        {
            s0 += x[i] * y[i]; s1 += x[i + 1] * y[i + 1];
            s2 += x[i + 2] * y[i + 2]; s3 += x[i + 3] * y[i + 3];
        }
        for (; i < x.Length; i++) s0 += x[i] * y[i];
        return (s0 + s1) + (s2 + s3);
    }

    // Least-squares y ≈ intercept + slope·x in a fixed summation order, with HeuristicLib's LinearScaling.Fit conventions
    // (slope 0 and the mean of y when x is constant or not finite); LinearScaling.Fit's sums depend on memory alignment.
    public static (double Slope, double Intercept) FitLine(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        int n = x.Length;
        double sx = 0, sy = 0;
        for (int i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
        double mx = sx / n, my = sy / n, sxx = 0, sxy = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - mx;
            sxx += dx * dx; sxy += dx * (y[i] - my);
        }
        if (!(sxx > 0) || !double.IsFinite(sxx) || !double.IsFinite(sxy)) return (0, my);
        double slope = sxy / sxx;
        return (slope, my - slope * mx);
    }

    public int MaximumIterations { get; init; } = ConstantFittingSettings.DefaultIterations;

    // Counts the trial points evaluated by this refiner (shared by all candidates of one GP run); null counts nothing.
    public ConstantFittingStatistics? Statistics { get; init; }

    // Rows the constants are fitted to; null fits to the problem's training data. Evaluation always uses all training rows.
    public RegressionData? FittingData { get; init; }

    public override ExpressionTree RefineCandidate(ExpressionTree candidate, IRandomNumberGenerator random,
        ExpressionTreeSearchSpace searchSpace, IProblem<ExpressionTree, ExpressionTreeSearchSpace> problem)
    {
        if (problem is not SymbolicRegressionProblem regression)
            throw new ArgumentException("Constant fitting needs a symbolic regression problem.", nameof(problem));
        // Scaling constants alone would only repeat the linear scaling that evaluation applies anyway.
        if (MaximumIterations == 0 || !candidate.TraversePreOrder().Any(node => node.Symbol is EvolvableConstantSymbol))
            return candidate;
        var program = TreeProgram.Compile(candidate);
        if (program is null) return candidate;
        var data = FittingData ?? regression.TrainingData;
        var fitted = program.Fit(data, MaximumIterations, out int trialPoints);
        Statistics?.AddTrialPoints(trialPoints);
        if (fitted is null) return candidate;
        var refined = program.Rebuild(fitted);
        // Constants fitted on a row sample can make a training row undefined (log or sqrt of a negative, division by zero);
        // such a fit is discarded, as the candidate would otherwise lose its finite objective.
        if (!ReferenceEquals(data, regression.TrainingData))
        {
            var predictions = new double[regression.TrainingData.RowCount];
            new SymbolicRegressor(refined).Predict(regression.TrainingData.Inputs, predictions);
            if (!predictions.All(double.IsFinite)) return candidate;
        }
        return refined;
    }

    // A tree flattened to post order, evaluated column-wise over all rows, with reverse-mode derivatives.
    internal sealed class TreeProgram
    {
        private enum Code { Variable, Fixed, Parameter, Add, Subtract, Multiply, Divide, Exp, Log, Sqrt, Square }

        private readonly ExpressionNode _root;
        private readonly Code[] _codes;
        private readonly int[] _left, _right, _payload;       // child indices; parameter index or variable index
        private readonly double[] _fixed;
        private readonly string[] _variables;
        private readonly double[] _initial;
        private readonly int[] _parameterNode;                 // node index of each parameter
        private readonly bool[] _hasParameter;                 // the node's subtree contains a parameter

        private TreeProgram(ExpressionNode root, List<Code> codes, List<int> left, List<int> right, List<int> payload,
            List<double> fixedValues, List<string> variables, List<double> initial)
        {
            _root = root; _codes = [.. codes]; _left = [.. left]; _right = [.. right]; _payload = [.. payload];
            _fixed = [.. fixedValues]; _variables = [.. variables]; _initial = [.. initial];
            _parameterNode = new int[_initial.Length];
            _hasParameter = new bool[_codes.Length];
            for (int i = 0; i < _codes.Length; i++)
            {
                if (_codes[i] == Code.Parameter) _parameterNode[_payload[i]] = i;
                _hasParameter[i] = _codes[i] == Code.Parameter || (_left[i] >= 0 && _hasParameter[_left[i]]) || (_right[i] >= 0 && _hasParameter[_right[i]]);
            }
        }

        public int ParameterCount => _initial.Length;

        // Null when the tree uses an operation this evaluator does not implement; the candidate then stays as it is.
        public static TreeProgram? Compile(ExpressionTree tree)
        {
            var codes = new List<Code>(); var left = new List<int>(); var right = new List<int>(); var payload = new List<int>();
            var fixedValues = new List<double>(); var variables = new List<string>(); var initial = new List<double>();
            int Emit(ExpressionNode node)
            {
                int l = -1, r = -1, value = -1; double constant = 0; Code code;
                switch (node)
                {
                    case NumericConstantExpressionNode { Symbol: EvolvableConstantSymbol } c:
                        code = Code.Parameter; value = initial.Count; initial.Add(c.Value); break;
                    case NumericConstantExpressionNode c:
                        code = Code.Fixed; constant = c.Value; break;
                    case VariableExpressionNode v:
                        code = Code.Variable; value = variables.IndexOf(v.VariableName);
                        if (value < 0) { value = variables.Count; variables.Add(v.VariableName); }
                        break;
                    case UnaryExpressionNode u:
                        code = u.Symbol switch {
                            ExponentialSymbol => Code.Exp, LogarithmSymbol => Code.Log, SquareRootSymbol => Code.Sqrt, SquareSymbol => Code.Square,
                            _ => throw new NotSupportedException()
                        };
                        l = Emit(u.Operand); break;
                    case BinaryExpressionNode b:
                        code = b.Symbol switch {
                            AdditionSymbol => Code.Add, SubtractionSymbol => Code.Subtract, MultiplicationSymbol => Code.Multiply, DivisionSymbol => Code.Divide,
                            _ => throw new NotSupportedException()
                        };
                        l = Emit(b.Left); r = Emit(b.Right); break;
                    default: throw new NotSupportedException();
                }
                codes.Add(code); left.Add(l); right.Add(r); payload.Add(value); fixedValues.Add(constant);
                return codes.Count - 1;
            }
            try { Emit(tree.Root); }
            catch (NotSupportedException) { return null; }
            return new TreeProgram(tree.Root, codes, left, right, payload, fixedValues, variables, initial);
        }

        // Fitted inner constants, or null when the fit does not improve on the start.
        public double[]? Fit(RegressionData data, int maximumIterations, out int trialPoints)
        {
            trialPoints = 0;
            int n = data.RowCount, p = ParameterCount, m = p + 2;  // parameters: a, b, then the tree's constants
            var columns = _variables.Select(name => data.Inputs.Get<double>(name).Values).ToArray();
            var target = data.Target.Values.Span;
            int nodes = _codes.Length;
            var pool = ArrayPool<double>.Shared;
            double[] values = pool.Rent(nodes * n), derivatives = pool.Rent(nodes * n), f = pool.Rent(n), jacobian = pool.Rent(m * n);
            try
            {
                var theta = (double[])_initial.Clone();
                if (!Evaluate(theta, columns, n, values)) return null;
                var output = values.AsSpan((nodes - 1) * n, n);
                double rms = 0;
                foreach (double value in output) rms += value * value;
                rms = Math.Sqrt(rms / n);
                if (!double.IsFinite(rms) || rms == 0) return null;
                double k = 1 / rms;
                for (int i = 0; i < n; i++) f[i] = k * output[i];
                var (b, a) = FitLine(f.AsSpan(0, n), target);
                if (!double.IsFinite(a) || !double.IsFinite(b)) return null;

                double cost = Cost(a, b, f, target, n);
                double startCost = cost;
                double lambda = 1e-2;
                var normal = new double[m * m]; var gradient = new double[m]; var scale = new double[m];
                var system = new double[m * m]; var step = new double[m]; var trial = new double[p];
                // Each iteration tries one step; a rejected step raises the damping, an accepted one lowers it. The Jacobian
                // is recomputed only after an accepted step, when values again hold the current point.
                bool jacobianCurrent = false;
                for (int iteration = 0; iteration < maximumIterations && cost > 0; iteration++, lambda *= 10)
                {
                    if (!jacobianCurrent)
                    {
                        // Jacobian of the residual model a + b·k·f(θ) at the current point; rows are parameters.
                        for (int i = 0; i < n; i++) { jacobian[i] = 1; jacobian[n + i] = f[i]; }
                        Backpropagate(values, n, derivatives);
                        for (int j = 0; j < p; j++)
                        {
                            var row = jacobian.AsSpan((j + 2) * n, n);
                            var d = derivatives.AsSpan(_parameterNode[j] * n, n);
                            TensorPrimitives.Multiply(d, b * k, row);
                        }
                        // Normal equations JᵀJ δ = Jᵀr with unit-length columns, so constants of any magnitude are comparable.
                        var residuals = derivatives.AsSpan(0, n);  // the adjoints are no longer needed
                        for (int i = 0; i < n; i++) residuals[i] = target[i] - a - b * f[i];
                        for (int r = 0; r < m; r++)
                        {
                            var jr = jacobian.AsSpan(r * n, n);
                            gradient[r] = Dot(jr, residuals);
                            for (int c = 0; c <= r; c++)
                                normal[r * m + c] = normal[c * m + r] = Dot(jr, jacobian.AsSpan(c * n, n));
                        }
                        if (!normal.All(double.IsFinite) || !gradient.All(double.IsFinite)) break;
                        for (int r = 0; r < m; r++) scale[r] = normal[r * m + r] > 0 ? 1 / Math.Sqrt(normal[r * m + r]) : 0;
                        jacobianCurrent = true;
                    }

                    for (int r = 0; r < m; r++)
                        for (int c = 0; c < m; c++)
                            system[r * m + c] = scale[r] * normal[r * m + c] * scale[c] + (r == c ? lambda + (scale[r] == 0 ? 1 : 0) : 0);
                    for (int r = 0; r < m; r++) step[r] = scale[r] * gradient[r];
                    if (!CholeskySolve(system, step, m)) continue;
                    for (int r = 0; r < m; r++) step[r] *= scale[r];
                    double newA = a + step[0], newB = b + step[1];
                    for (int j = 0; j < p; j++) trial[j] = theta[j] + step[j + 2];
                    if (!double.IsFinite(newA) || !double.IsFinite(newB) || !trial.All(double.IsFinite)) continue;
                    trialPoints++;
                    if (!Evaluate(trial, columns, n, values)) continue;
                    double newCost = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double residual = target[i] - newA - newB * k * output[i];
                        newCost += residual * residual;
                    }
                    if (!(newCost < cost)) continue;
                    a = newA; b = newB; cost = newCost;
                    Array.Copy(trial, theta, p);
                    for (int i = 0; i < n; i++) f[i] = k * output[i];
                    jacobianCurrent = false;
                    lambda = Math.Max(lambda / 100, 1e-13);  // with the loop's factor 10: damping divided by 10
                }
                return Improved();

                double[]? Improved() => cost < startCost ? theta : null;
            }
            finally
            {
                pool.Return(values); pool.Return(derivatives); pool.Return(f); pool.Return(jacobian);
            }
        }

        // The tree with the given inner constants; the input tree is left unchanged.
        public ExpressionTree Rebuild(double[] theta)
        {
            int next = 0;
            ExpressionNode Copy(ExpressionNode node) => node switch {
                NumericConstantExpressionNode { Symbol: EvolvableConstantSymbol symbol } => new NumericConstantExpressionNode(symbol, theta[next++]),
                UnaryExpressionNode u => new UnaryExpressionNode((OperationSymbol)u.Symbol, Copy(u.Operand)),
                BinaryExpressionNode b => CopyBinary(b),
                _ => node
            };
            ExpressionNode CopyBinary(BinaryExpressionNode b)
            {
                // Post order, as in Compile: the left subtree's constants come first.
                var l = Copy(b.Left);
                var r = Copy(b.Right);
                return new BinaryExpressionNode((OperationSymbol)b.Symbol, l, r);
            }
            return new ExpressionTree(Copy(_root));
        }

        // Node values for all rows in post order; false when the output has a non-finite value.
        internal bool Evaluate(double[] theta, ReadOnlyMemory<double>[] columns, int n, double[] values)
        {
            for (int node = 0; node < _codes.Length; node++)
            {
                var v = values.AsSpan(node * n, n);
                var x = _left[node] >= 0 ? values.AsSpan(_left[node] * n, n) : default;
                var y = _right[node] >= 0 ? values.AsSpan(_right[node] * n, n) : default;
                switch (_codes[node])
                {
                    case Code.Variable: columns[_payload[node]].Span[..n].CopyTo(v); break;
                    case Code.Fixed: v.Fill(_fixed[node]); break;
                    case Code.Parameter: v.Fill(theta[_payload[node]]); break;
                    case Code.Add: TensorPrimitives.Add(x, y, v); break;
                    case Code.Subtract: TensorPrimitives.Subtract(x, y, v); break;
                    case Code.Multiply: TensorPrimitives.Multiply(x, y, v); break;
                    case Code.Divide: TensorPrimitives.Divide(x, y, v); break;
                    case Code.Exp: TensorPrimitives.Exp(x, v); break;
                    case Code.Log: TensorPrimitives.Log(x, v); break;
                    case Code.Sqrt: TensorPrimitives.Sqrt(x, v); break;
                    case Code.Square: TensorPrimitives.Multiply(x, x, v); break;
                }
            }
            foreach (double value in values.AsSpan((_codes.Length - 1) * n, n))
                if (!double.IsFinite(value)) return false;
            return true;
        }

        // Reverse mode: the derivative of the output by every node of a subtree that holds a parameter, from the values
        // of Evaluate. Each node has one parent and parents follow their children in post order, so one backward pass
        // assigns every adjoint; non-finite adjoints surface as a non-finite normal matrix.
        private void Backpropagate(double[] values, int n, double[] adjoints)
        {
            int root = _codes.Length - 1;
            adjoints.AsSpan(root * n, n).Fill(1);
            for (int node = root; node >= 0; node--)
            {
                if (!_hasParameter[node] || _left[node] < 0) continue;
                var a = adjoints.AsSpan(node * n, n);
                var v = values.AsSpan(node * n, n);
                var x = values.AsSpan(_left[node] * n, n);
                var y = _right[node] >= 0 ? values.AsSpan(_right[node] * n, n) : default;
                var ax = adjoints.AsSpan(_left[node] * n, n);
                var ay = _right[node] >= 0 ? adjoints.AsSpan(_right[node] * n, n) : default;
                switch (_codes[node])
                {
                    case Code.Add: a.CopyTo(ax); a.CopyTo(ay); break;
                    case Code.Subtract: a.CopyTo(ax); TensorPrimitives.Negate(a, ay); break;
                    case Code.Multiply: TensorPrimitives.Multiply(a, y, ax); TensorPrimitives.Multiply(a, x, ay); break;
                    case Code.Divide: TensorPrimitives.Divide(a, y, ax); TensorPrimitives.Multiply(ax, v, ay); TensorPrimitives.Negate(ay, ay); break;
                    case Code.Exp: TensorPrimitives.Multiply(a, v, ax); break;
                    case Code.Log: TensorPrimitives.Divide(a, x, ax); break;
                    case Code.Sqrt: TensorPrimitives.Divide(a, v, ax); TensorPrimitives.Multiply(ax, 0.5, ax); break;
                    case Code.Square: TensorPrimitives.Multiply(a, x, ax); TensorPrimitives.Multiply(ax, 2.0, ax); break;
                }
            }
        }

        private static double Cost(double a, double b, double[] f, ReadOnlySpan<double> target, int n)
        {
            double cost = 0;
            for (int i = 0; i < n; i++)
            {
                double residual = target[i] - a - b * f[i];
                cost += residual * residual;
            }
            return cost;
        }

        // Solves the symmetric system in place (right-hand side becomes the solution); false if not positive definite.
        private static bool CholeskySolve(double[] matrix, double[] rhs, int m)
        {
            for (int c = 0; c < m; c++)
            {
                double diagonal = matrix[c * m + c];
                for (int k = 0; k < c; k++) diagonal -= matrix[c * m + k] * matrix[c * m + k];
                if (!(diagonal > 0) || !double.IsFinite(diagonal)) return false;
                double root = Math.Sqrt(diagonal);
                matrix[c * m + c] = root;
                for (int r = c + 1; r < m; r++)
                {
                    double s = matrix[r * m + c];
                    for (int k = 0; k < c; k++) s -= matrix[r * m + k] * matrix[c * m + k];
                    matrix[r * m + c] = s / root;
                }
            }
            for (int r = 0; r < m; r++)
            {
                double s = rhs[r];
                for (int k = 0; k < r; k++) s -= matrix[r * m + k] * rhs[k];
                rhs[r] = s / matrix[r * m + r];
            }
            for (int r = m - 1; r >= 0; r--)
            {
                double s = rhs[r];
                for (int k = r + 1; k < m; k++) s -= matrix[k * m + r] * rhs[k];
                rhs[r] = s / matrix[r * m + r];
            }
            return true;
        }
    }
}
