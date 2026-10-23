# A synchronous targetRunner keeps the GA and ResultDTO collection in .NET.
if (!requireNamespace("irace", quietly = TRUE)) {
  stop("Install the R package first: install.packages('irace', repos='https://cloud.r-project.org')")
}
args <- commandArgs(trailingOnly = TRUE)
parameters <- irace::readParameters(args[[1]])
seed <- as.integer(args[[2]])
budget <- as.integer(args[[3]])
input <- file("stdin", open = "r")
target_runner <- function(experiment, scenario) {
  configuration <- experiment$configuration
  # Support both current irace and older CRAN releases.
  id <- experiment$id_configuration
  if (is.null(id)) id <- experiment$id.configuration
  values <- c(id, experiment$seed, configuration$PopulationSize,
              configuration$MutationRate, configuration$MaxTreeDepth,
              configuration$MaxTreeLength, as.character(configuration$MutatorType))
  cat("HYPEROP_EVALUATE\t", paste(values, collapse = "\t"), "\n", sep = "")
  flush(stdout())
  response <- readLines(input, n = 1L, warn = FALSE)
  if (length(response) != 1L) stop("The HyperOp evaluator disconnected")
  cost <- as.numeric(response)
  if (!is.finite(cost)) stop("The HyperOp evaluator returned an invalid cost")
  list(cost = cost)
}

# Small races support the experiment's ten-evaluation budget. Increase iterations
# with the budget, up to five, while respecting irace's minimum budget.
iterations <- max(1L, min(5L, floor(sqrt(budget / 2)) - 1L))
scenario <- list(parameters = parameters, targetRunner = target_runner,
                 instances = "feynman", seed = seed, maxExperiments = budget,
                 minNbSurvival = 1L, nbIterations = iterations, mu = 1L,
                 firstTest = 2L, eachTest = 1L, parallel = 0L,
                 logFile = "", quiet = TRUE)
elites <- irace::irace(scenario = scenario)
# Retain irace's native final choice, rather than the luckiest individual GP trial.
if (!is.data.frame(elites) || nrow(elites) < 1L) stop("irace did not return final elites")
# irace() returns this table ranked best to worst; getFinalElites() instead expects a saved log object.
best <- elites[1L, , drop = FALSE]
cat("HYPEROP_RECOMMEND\t", best$.ID.[[1L]], "\n", sep = "")
flush(stdout())
close(input)
