using System.Text;

namespace HyperOp.Algorithms.HyperParameterOptimization
{
    // Controller instructions. A prompt pilot (4 November 2026, results/cp) compared variants after the 9B controller had
    // stopped almost every run that was not the best so far at its first decision: "standard" is the earlier prompt
    // (AgentSearch.ControllerInstructions), p1 to p5 run from minimal to verbose, p6 frames the decision by the search's
    // objective, and p7 is p4 without the comparison with earlier runs at the same evaluation count (the request omits
    // AtSameEvaluations too). With p7, 9B stopped no run at its first decision.
    // p8, the study prompt since the fourth review (4 November), fixes what the p7 protocols showed the models misreading,
    // not the policy: the budget is stated as neutral facts instead of "if stopped now / if continued" counterfactuals
    // (4B cited "sufficient budget remaining" as a reason to stop), the distinct-expression share is called
    // DistinctExpressionShare with "1 = all distinct" (9B read UniqueStructures 1 as "zero diversity"), and no decision is
    // requested when less than one checkpoint of the allowance remains. p9 is p8 without the sentence that runs with large
    // populations or trees can improve late (ablation of that prior). Earlier variants keep their fields and behaviour.
    public static class ControllerPrompts
    {
        public const string Study = "p8";
        public static readonly string[] Names = [Study, "p9", "standard", "p1", "p2", "p4", "p5", "p6", "p7"];

        public static bool ShowsComparison(string name) => name is not ("p7" or "p8" or "p9");
        // Neutral budget facts, the renamed diversity field and no forced decisions.
        public static bool UsesNeutralFacts(string name) => name is "p8" or "p9";
        public const string DistinctExpressionShare = "DistinctExpressionShare";

        static string Output(bool sentences) => sentences
            ? "Return JSON with Action (Continue or Stop) and Reason, one or two sentences explaining your judgement."
            : "Return JSON with Action (Continue or Stop) and Reason. Reason: keywords, not sentences; report numbers.";

        static void ShortChannels(StringBuilder text, FeedbackChannels channels)
        {
            if (channels.HasFlag(FeedbackChannels.Curve)) text.Append(" Curve: best and median training NMSE at recent generations.");
            if (channels.HasFlag(FeedbackChannels.Diversity))
                text.Append(" Diversity: how varied the population is (UniqueStructures, PhenotypicDiversity); low values mean it has converged.");
            if (channels.HasFlag(FeedbackChannels.Size)) text.Append(" Size: expression length and depth, and the shares of trees at the limits.");
            if (channels.HasFlag(FeedbackChannels.Generalization))
                text.Append(" Generalization: validation NMSE of the current best expression and its gap to training NMSE.");
        }

        static void LongChannels(StringBuilder text, FeedbackChannels channels, bool renamed = false)
        {
            if (channels.HasFlag(FeedbackChannels.Curve))
                text.Append(" Curve lists best and median training NMSE at recent generations; the best falling means progress, the median " +
                    "approaching the best means the population concentrates on similar expressions.");
            if (channels.HasFlag(FeedbackChannels.Diversity))
                text.Append((renamed ? $" Diversity: {DistinctExpressionShare} is the fraction of distinct expressions (1 = all distinct, low = many copies); "
                        : " Diversity: UniqueStructures is the fraction of distinct expressions; ") + "PhenotypicDiversity is one minus the mean " +
                    "squared correlation between individuals' outputs (0 when all behave alike up to offset and scale); InvalidOutputs is the " +
                    "fraction with non-finite outputs; TrainingNmseSpread is the interquartile range of training NMSE over its median. " +
                    "Low diversity means the population has converged and finds new structures less often.");
            if (channels.HasFlag(FeedbackChannels.Size))
                text.Append(" Size: median and maximum expression length, median depth, the fractions of trees at the length and depth limits, " +
                    "and the change of median length since the previous checkpoint; many trees at a limit mean the limit constrains the search, " +
                    "growing trees without improvement indicate bloat.");
            if (channels.HasFlag(FeedbackChannels.Generalization))
                text.Append(" Generalization: validation NMSE of the current best expression and its gap to the best training NMSE; " +
                    "a growing gap indicates overfitting.");
        }

        const string BudgetShort = " SearchContext: this run's evaluations used and remaining, and the evaluations left for other configurations " +
            "if it stops now or continues to the end of its allowance.";
        const string BudgetLong = " SearchContext gives this run's evaluations used and remaining allowance, and the evaluations left for other " +
            "configurations if this run stops now or if it continues to the end of its allowance, also as a number of runs of the reference length.";
        const string BudgetNeutral = " SearchContext gives this run's evaluations used and its allowance, and the evaluations the shared pool " +
            "holds for other configurations after this run's allowance, also as a number of runs of the reference length.";
        const string ComparisonShort = " AtSameEvaluations: the best training NMSE of earlier runs after the same number of evaluations (best, " +
            "median) and this run's rank among them (1 = lowest).";
        const string ComparisonLong = " AtSameEvaluations compares this run's BestTrainingNmse with the best training NMSE of earlier runs after " +
            "the same number of evaluations (runs that got that far): their best and median, and ThisRunRank, this run's position among them " +
            "and itself (1 = lowest NMSE, i.e. better than all earlier runs; RunsCompared is the number of runs ranked).";

        // Instructions of a variant other than "standard".
        public static string Instructions(string name, FeedbackChannels channels, int checkpoints)
        {
            string every = $"{100.0 / checkpoints:0.#}%";
            var text = new StringBuilder();
            switch (name)
            {
                case "p1":
                    text.Append("Decide whether this genetic programming run continues to the next checkpoint or stops now. Lower NMSE is better. " +
                        "Stopping returns the run's unused evaluations to the shared budget for other configurations. ").Append(Output(false));
                    break;
                case "p2":
                    text.Append("You control one run of a hyperparameter search for genetic programming (GP) symbolic regression. At each " +
                        "checkpoint, decide: Continue to the next checkpoint, or Stop now. Lower NMSE is better. Stopping returns the run's unused " +
                        "evaluations to the shared budget for other configurations. Inputs: Progress: best training NMSE and RecentImprovement " +
                        "since the previous checkpoint.");
                    ShortChannels(text, channels);
                    text.Append(BudgetShort).Append(ComparisonShort).Append(' ').Append(Output(false));
                    break;
                case "p4":
                case "p7":
                case "p8":
                case "p9":
                    text.Append("You control one run of a hyperparameter search for genetic programming (GP) symbolic regression. GP evolves a " +
                        "population of candidate expressions over generations: it selects better expressions, recombines and mutates them, and " +
                        "keeps the best expression found so far. " +
                        (name == "p9" ? "Training NMSE usually falls quickly at first and more slowly later. "
                            : "Training NMSE usually falls quickly at first and more slowly later; runs with " +
                              "large populations or large trees can start slowly and improve late. ") +
                        $"You see the run after every {every} of the reference run length and decide whether it continues to the next checkpoint " +
                        "or stops now. A stopped run cannot be resumed; its remaining evaluations return to the shared budget for other " +
                        "configurations. Lower NMSE is better. Progress reports BestTrainingNmse and RecentImprovement, its reduction since the " +
                        "previous checkpoint.");
                    LongChannels(text, channels, renamed: UsesNeutralFacts(name));
                    text.Append(UsesNeutralFacts(name) ? BudgetNeutral : BudgetLong);
                    if (name == "p4") text.Append(ComparisonLong);
                    text.Append(' ').Append(Output(false));
                    break;
                case "p5":
                    text.Append("You are the run controller in an automated hyperparameter search for genetic programming (GP) symbolic " +
                        "regression. GP is an evolutionary algorithm: it keeps a population of candidate mathematical expressions, and in every " +
                        "generation it selects better expressions as parents, recombines and mutates them into offspring, and keeps the best " +
                        "expression found so far. Fitness is the normalized mean squared error (NMSE) on the training data after an optimal " +
                        "linear scaling; lower is better, and 0 is a perfect fit. Training NMSE usually falls quickly in the first generations " +
                        "and more slowly later; how fast depends on the configuration, and runs with large populations or large trees can start " +
                        "slowly and improve late.\n\n" +
                        "The search tests many configurations, one run at a time, from a shared budget of candidate evaluations. Another " +
                        "component chooses each configuration and its allowance; you decide only about the run in progress. At the end, the " +
                        "search returns the single expression with the lowest validation NMSE among all its runs, so the goal is the lowest " +
                        "validation NMSE the search can reach within its budget.\n\n" +
                        $"You see the run at checkpoints, after every {every} of the reference run length. At each checkpoint you decide: " +
                        "Continue the run to the next checkpoint, or Stop it now. Continuing spends more of its allowance on this configuration " +
                        "and gives it the chance to improve further. Stopping ends the run for good, since it cannot be resumed, and returns its " +
                        "remaining allowance to the shared budget, where other configurations can use it. Both choices can be wrong: continuing a " +
                        "run that will not get better wastes evaluations, and stopping a run that would have improved loses that improvement.\n\n" +
                        "You receive: Problem: the number of input variables and data sizes. Configuration: the run's hyperparameters (population " +
                        "size, mutation rate, maximum tree depth and length, mutator). Progress: the generation, the evaluations used and " +
                        "remaining in this run's allowance, BestTrainingNmse and RecentImprovement, its reduction since the previous checkpoint.");
                    LongChannels(text, channels);
                    text.Append(" PreviousObservations: this run's earlier checkpoints with the decisions made there.").Append(BudgetLong)
                        .Append(ComparisonLong)
                        .Append(" Training and validation NMSE are measured on different data and are not directly comparable.\n\n")
                        .Append(Output(true));
                    break;
                case "p6":
                    text.Append("The search returns the single expression with the lowest validation NMSE among all its runs. At this checkpoint " +
                        "of a genetic programming run, decide whether this run's remaining allowance is more likely to lead there than the same " +
                        "evaluations spent on other configurations: Continue the run to the next checkpoint, or Stop it for good (it cannot be " +
                        "resumed; its remaining evaluations return to the shared budget). Lower NMSE is better. Progress: best training NMSE and " +
                        "RecentImprovement since the previous checkpoint.");
                    ShortChannels(text, channels);
                    text.Append(BudgetShort).Append(ComparisonShort).Append(' ').Append(Output(false));
                    break;
                default:
                    throw new ArgumentException($"Unknown controller prompt '{name}'; expected one of {string.Join(", ", Names)}.");
            }
            return text.ToString();
        }
    }
}
