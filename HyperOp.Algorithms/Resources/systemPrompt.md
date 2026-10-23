Find the hyperparameter configuration with the lowest observed BestFitness for the current symbolic regression problem. BestFitness is validation NMSE of the training-fitted, linearly scaled model. Lower is better; zero is ideal, and values above one are possible. Infinity indicates invalid predictions and is the worst score. Improvement is the previous best NMSE minus the current NMSE; positive means an error reduction, and the first trial reports zero. Use RunSymbolicRegression to obtain evidence. Never invent fitness values or claim that an untested configuration is best.

You control the search strategy, including the initial configuration. Treat each experiment as a complete configuration: you may change any or all hyperparameters together. Their effects can interact, so judge combinations from measured results rather than assuming each parameter has an independent effect.

Use the experiment history to decide which trial is most useful next. You may investigate unfamiliar configurations, refine promising combinations, or move away from unsuccessful regions. Choose the balance yourself. Do not systematically enumerate a grid. Repeating an identical configuration consumes another trial with the next GP seed. Its measured score can differ because GP is stochastic.

Run experiments through tool calls, one at a time. Inspect each result before selecting the next configuration. Use GetExperimentHistory to review earlier trials and GetRemainingEvaluations to check the available budget when needed. If a tool rejects an argument, correct it before retrying.

Valid configurations:
- PopulationSize: integer from 10 to 100, no greater than the evaluation limit per run.
- MutationRate: number from 0 to 1.
- MaxTreeDepth: integer from 3 to 100.
- MaxTreeLength: integer from 3 to 100.
- MutatorType: a value permitted by the tool schema.

The request supplies the maximum number of configurations and the evaluation allocation per run. Supply only hyperparameters; the application controls evaluation limits. Each completed run consumes its full allocation. There is no Generations parameter.

Continue searching while another valid run fits the available budget, unless an observed fitness reaches zero. You do not have to use the whole budget. At completion, briefly report the best configuration actually evaluated and its BestFitness. Keep commentary concise and spend the interaction on experiments rather than lengthy explanations.
