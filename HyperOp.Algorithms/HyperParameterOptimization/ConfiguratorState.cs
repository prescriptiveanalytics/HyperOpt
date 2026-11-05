using System.Globalization;
using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    // Text rendering of the search state for the stateless LLM roles: one line per trial with fixed columns, numbers to
    // three significant digits. Conditions differ only in the channel columns, so the prompt-length difference between
    // conditions is the cost of their channels. JSON with 17-digit doubles tokenized at about two characters per token
    // and let configurator prompts reach 47k tokens.
    public static class ConfiguratorState
    {
        // All trials up to TableRows; beyond that the TableBest best, the TableRecent most recent and every new best, in
        // trial order, with one aggregate line per mutator for the full history. Trial order is kept because a re-sorted
        // window invites extrapolating the last rows.
        public const int TableRows = 60;
        public const int TableBest = 20;
        public const int TableRecent = 30;
        // Checkpoints of a run kept for its detail block.
        public const int DetailCheckpoints = 10;
        public const int ReasonLength = 120;
        public const int PlanLength = 200;
        public const int ControllerReasonLength = 110;

        // Three significant digits in plain decimals (no exponent, which small models misread); integers from 1000 on.
        public static string Number(double? value, int digits = 3)
        {
            if (value is not double v) return "-";
            if (double.IsNaN(v)) return "nan";
            if (double.IsInfinity(v)) return v > 0 ? "inf" : "-inf";
            if (v == 0) return "0";
            if (Math.Abs(v) >= 1000) return Math.Round(v).ToString("F0", CultureInfo.InvariantCulture);
            int decimals = Math.Clamp(digits - 1 - (int)Math.Floor(Math.Log10(Math.Abs(v))), 0, 15);
            return Math.Round(v, decimals).ToString("0." + new string('#', Math.Max(decimals, 1)), CultureInfo.InvariantCulture);
        }

        public static string Thousands(int evaluations) =>
            evaluations >= 1000 ? Number(evaluations / 1000.0) + "k" : evaluations.ToString(CultureInfo.InvariantCulture);

        public static string Percent(double share) => Math.Round(100 * share).ToString("F0", CultureInfo.InvariantCulture) + "%";

        public static string Clip(string? text, int length)
        {
            string flat = string.Join(' ', (text ?? "").Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
            return flat.Length <= length ? flat : flat[..(length - 1)] + "…";
        }

        // Trials shown in the table, in trial order.
        public static IReadOnlyList<Experiment> ShownTrials(IReadOnlyList<Experiment> trials)
        {
            if (trials.Count <= TableRows) return trials;
            var best = trials.Where(t => double.IsFinite(t.BestFitness)).OrderBy(t => t.BestFitness).ThenBy(t => t.Trial).Take(TableBest);
            return best.Concat(trials.TakeLast(TableRecent)).Concat(trials.Where(t => t.IsNewBest))
                .DistinctBy(t => t.Trial).OrderBy(t => t.Trial).ToList();
        }

        static IEnumerable<(FeedbackChannels Channel, string Header, string Legend)> Cells => [
            (FeedbackChannels.Progress, "trainNMSE first>last imp@cp",
                "trainNMSE first>last: best training NMSE at the run's first and last checkpoint; imp@k/n: the last checkpoint k of n with an improvement"),
            (FeedbackChannels.Curve, "curve",
                "curve: median/best training NMSE ratio at the last checkpoint (near 1: the population has converged on the best)"),
            (FeedbackChannels.Diversity, "unique first>last/pheno",
                "unique first>last/pheno: fraction of distinct expressions at the first and last checkpoint / phenotypic diversity at the last (0: all behave alike)"),
            (FeedbackChannels.Size, "medLen first>last atLen/atDepth",
                "medLen first>last atLen/atDepth: median expression length at the first and last checkpoint, shares of trees at the length and depth limits at the last"),
            (FeedbackChannels.Generalization, "gap last max",
                "gap last max: validation minus training NMSE of the best expression at the last checkpoint, and the largest gap of the run (positive and growing: overfitting)")];

        static string Cell(FeedbackChannels channel, Experiment trial)
        {
            var observations = trial.ControlObservations;
            if (observations.Count == 0) return "-";
            var first = observations[0].Progress; var last = observations[^1].Progress;
            return channel switch {
                FeedbackChannels.Progress => $"{Number(first.BestTrainingNmse)}>{Number(last.BestTrainingNmse)} imp@{trial.LastImprovementCheckpoint}/{trial.CheckpointCount}",
                FeedbackChannels.Curve => last.Curve is { BestTrainingNmse.Length: > 0 } c && c.BestTrainingNmse[^1] > 0
                    ? $"med/best {Number(c.MedianTrainingNmse[^1] / c.BestTrainingNmse[^1], 2)}" : "-",
                FeedbackChannels.Diversity => first.Diversity is { } d0 && last.Diversity is { } d1
                    ? $"{Number(d0.UniqueStructures, 2)}>{Number(d1.UniqueStructures, 2)}/{Number(d1.PhenotypicDiversity, 2)}" : "-",
                FeedbackChannels.Size => first.Size is { } s0 && last.Size is { } s1
                    ? $"{Number(s0.MedianLength)}>{Number(s1.MedianLength)} {Percent(s1.AtLengthLimit)}/{Percent(s1.AtDepthLimit)}" : "-",
                FeedbackChannels.Generalization => last.Generalization is { } g
                    ? $"{Number(g.Gap, 2)} max {Number(observations.Select(o => o.Progress.Generalization?.Gap ?? 0).MaxBy(Math.Abs), 2)}" : "-",
                _ => "-"
            };
        }

        static IEnumerable<FeedbackChannels> Active(FeedbackChannels channels) =>
            Cells.Select(cell => cell.Channel).Where(channel => channels != FeedbackChannels.None && channels.HasFlag(channel));

        // Explanation of the table's columns for the instructions; it names only the columns this view has.
        public static string Legend(FeedbackChannels channels, bool withReason)
        {
            var text = new StringBuilder(
                "Trial table columns: trial; pop, mut, depth, len, mutator: the configuration; granted: evaluations allowed for the run; " +
                "used: evaluations it spent; end: stop if the controller stopped it, full if it used its allowance; " +
                "valNMSE: validation NMSE of its best expression (lower is better); rank: 1 = lowest valNMSE of all trials");
            foreach (var cell in Cells.Where(cell => Active(channels).Contains(cell.Channel))) text.Append("; ").Append(cell.Legend);
            if (withReason) text.Append("; your reason: the analysis you gave when you proposed the trial");
            return text.Append('.').ToString();
        }

        // The trial table of one view; channels None leaves only the final-result columns.
        public static string Table(IReadOnlyList<Experiment> trials, FeedbackChannels channels, bool withReason)
        {
            var columns = new List<string> { "trial", "pop", "mut", "depth", "len", "mutator", "granted", "used", "end", "valNMSE", "rank" };
            var active = Active(channels).ToArray();
            columns.AddRange(Cells.Where(cell => active.Contains(cell.Channel)).Select(cell => cell.Header));
            if (withReason) columns.Add("your reason");
            var rank = trials.OrderBy(t => double.IsNaN(t.BestFitness) ? double.PositiveInfinity : t.BestFitness).ThenBy(t => t.Trial)
                .Select((t, i) => (t.Trial, Rank: i + 1)).ToDictionary(x => x.Trial, x => x.Rank);
            var shown = ShownTrials(trials);
            var text = new StringBuilder(string.Join('|', columns)).Append('\n');
            foreach (var t in shown)
            {
                var h = t.HyperParameters;
                var row = new List<string> { t.Trial.ToString(CultureInfo.InvariantCulture), h.PopulationSize.ToString(CultureInfo.InvariantCulture),
                    Number(h.MutationRate, 2), h.MaxTreeDepth.ToString(CultureInfo.InvariantCulture), h.MaxTreeLength.ToString(CultureInfo.InvariantCulture),
                    h.MutatorType.ToString(), Thousands(t.GrantedEvaluations), Thousands(t.EvaluationsUsed),
                    t.TerminationReason == "controller_stop" ? "stop" : "full", Number(t.BestFitness), rank[t.Trial].ToString(CultureInfo.InvariantCulture) };
                row.AddRange(active.Select(channel => Cell(channel, t)));
                if (withReason) row.Add(Clip(t.Reason, ReasonLength) is { Length: > 0 } reason ? reason : "-");
                text.Append(string.Join('|', row)).Append('\n');
            }
            if (shown.Count < trials.Count)
            {
                text.Append($"{trials.Count - shown.Count} trials not shown (beyond {TableRows} trials the table keeps the {TableBest} best, " +
                    $"the {TableRecent} most recent and every new best). All {trials.Count} trials by mutator (mutator|trials|best valNMSE|median valNMSE|mean used):\n");
                foreach (var group in trials.GroupBy(t => t.HyperParameters.MutatorType).OrderBy(g => g.Key))
                {
                    var finite = group.Select(t => t.BestFitness).Where(double.IsFinite).Order().ToArray();
                    string median = finite.Length == 0 ? "-" : Number(finite.Length % 2 == 1 ? finite[finite.Length / 2] : (finite[finite.Length / 2 - 1] + finite[finite.Length / 2]) / 2);
                    text.Append($"{group.Key}|{group.Count()}|{(finite.Length == 0 ? "-" : Number(finite[0]))}|{median}|{Thousands((int)group.Average(t => t.EvaluationsUsed))}\n");
                }
            }
            return text.ToString().TrimEnd('\n');
        }

        // Every kept checkpoint of one trial with the controller's decision; the configurator sees it for the best and the
        // latest trial in conditions that show checkpoints.
        public static string Detail(Experiment trial, FeedbackChannels channels, string label)
        {
            var active = Active(channels).ToHashSet();
            var columns = new List<string> { "cp", "evals", "trainNMSE" };
            if (active.Contains(FeedbackChannels.Curve)) columns.Add("med/best");
            if (active.Contains(FeedbackChannels.Diversity)) columns.AddRange(["unique", "pheno"]);
            if (active.Contains(FeedbackChannels.Size)) columns.AddRange(["medLen", "atLen", "atDepth"]);
            if (active.Contains(FeedbackChannels.Generalization)) columns.Add("gap");
            columns.Add("controller");
            var text = new StringBuilder($"{label}: trial {trial.Trial}, {trial.ControlObservations.Count} of {trial.CheckpointCount} checkpoints ({string.Join('|', columns)})\n");
            foreach (var (view, index) in trial.ControlObservations.Select((v, i) => (v, i)))
            {
                var p = view.Progress;
                var row = new List<string> { (index + 1).ToString(CultureInfo.InvariantCulture), Thousands(p.EvaluationsUsed), Number(p.BestTrainingNmse) };
                if (active.Contains(FeedbackChannels.Curve))
                    row.Add(p.Curve is { BestTrainingNmse.Length: > 0 } c && c.BestTrainingNmse[^1] > 0 ? Number(c.MedianTrainingNmse[^1] / c.BestTrainingNmse[^1], 2) : "-");
                if (active.Contains(FeedbackChannels.Diversity))
                    row.AddRange(p.Diversity is { } d ? [Number(d.UniqueStructures, 2), Number(d.PhenotypicDiversity, 2)] : ["-", "-"]);
                if (active.Contains(FeedbackChannels.Size))
                    row.AddRange(p.Size is { } s ? [Number(s.MedianLength), Percent(s.AtLengthLimit), Percent(s.AtDepthLimit)] : ["-", "-", "-"]);
                if (active.Contains(FeedbackChannels.Generalization)) row.Add(Number(p.Generalization?.Gap, 2));
                row.Add($"{view.Action}: {Clip(view.Reason, ControllerReasonLength)}");
                text.Append(string.Join('|', row)).Append('\n');
            }
            return text.ToString().TrimEnd('\n');
        }
    }
}
