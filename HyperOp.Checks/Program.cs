using HEAL.HeuristicLib.Encodings.SymbolicExpressions;
using HEAL.HeuristicLib.MachineLearning;
using HEAL.HeuristicLib.Problems.MachineLearning;
using HyperOp.Algorithms;
using HyperOp.Algorithms.Benchmarks;
using HyperOp.Algorithms.Feynman;
using HyperOp.Algorithms.Feynman.Instances;
using HyperOp.Algorithms.HyperParameterOptimization;
using HyperOp.Experiments;
using System.Reflection;
using System.Text.Json;

// Dependency-free regression checks. The equations/domains are the audited specification
// in data/Feynman/README.md, rather than values inferred from the generator under test.
var specs = new (FeynmanDescriptor Descriptor, string Id, string[] Inputs, double[] Min, double[] Max, Func<double[], double> Target)[] {
    (new Feynman2(42), "I.6.20", ["sigma", "theta"], [1,1], [3,3],
        x => Math.Exp(-Math.Log(x[0]) - Math.Log(2 * Math.PI) / 2 - x[1] * x[1] / (2 * x[0] * x[0]))),
    (new Feynman5(42), "I.9.18", ["m1","m2","G","x1","x2","y1","y2","z1","z2"],
        [1,1,1,3,1,3,1,3,1], [2,2,2,4,2,4,2,4,2],
        x => x[0]*x[1]*x[2] / Enumerable.Range(0,3).Sum(i => Math.Pow(x[3+2*i]-x[4+2*i],2))),
    (new Feynman17(42), "I.15.3.x", ["x","u","c","t"], [5,1,3,1], [10,2,20,2],
        x => (x[0]-x[1]*x[3])*x[2]/Math.Sqrt(x[2]*x[2]-x[1]*x[1])),
    (new Feynman31(42), "I.30.5", ["lambd","d","n"], [1,2,1], [2,5,5],
        x => Math.Atan2(x[0], Math.Sqrt(Math.Pow(x[1]*x[2],2)-x[0]*x[0]))),
    (new Feynman33(42), "I.32.17", ["epsilon","c","Ef","r","omega","omega_0"],
        [1,1,1,1,1,3], [2,2,2,2,2,5],
        x => 4*Math.PI/3*x[0]*x[1]*Math.Pow(x[2]*x[3],2)/Math.Pow(1-Math.Pow(x[5]/x[4],2),2)),
    (new Feynman44(42), "I.41.16", ["omega","T","h","kb","c"], [1,1,1,1,1], [5,5,5,5,5],
        x => x[2]*Math.Pow(x[0],3)/Math.Pow(Math.PI*x[4],2)*Math.Exp(-x[2]*x[0]/(x[3]*x[1]))/(1-Math.Exp(-x[2]*x[0]/(x[3]*x[1])))),
    (new Feynman50(42), "I.48.20", ["m","v","c"], [1,1,3], [5,2,20],
        x => x[0]*Math.Pow(x[2],3)/Math.Sqrt(x[2]*x[2]-x[1]*x[1])),
    (new Feynman56(42), "II.6.15a", ["epsilon","p_d","r","x","y","z"], [1,1,1,1,1,1], [3,3,3,3,3,3],
        x => 3*x[1]*x[5]*Math.Sqrt(x[3]*x[3]+x[4]*x[4])/(4*Math.PI*x[0]*Math.Pow(x[2],5))),
    (new Feynman64(42), "II.11.27", ["n","alpha","epsilon","Ef"], [0,0,1,1], [1,1,2,2],
        x => 3*x[0]*x[1]*x[2]*x[3]/(3-x[0]*x[1])),
    (new Feynman65(42), "II.11.28", ["n","alpha"], [0,0], [1,1],
        x => (3+2*x[0]*x[1])/(3-x[0]*x[1])),
    (new Feynman81(42), "II.35.21", ["n_rho","mom","B","kb","T"], [1,1,1,1,1], [5,5,5,5,5],
        x => { double e=Math.Exp(-2*x[1]*x[2]/(x[3]*x[4])); return x[0]*x[1]*(1-e)/(1+e); }),
    (new Feynman90(42), "III.9.52", ["p_d","Ef","t","h","omega","omega_0"], [1,1,1,1,1,1], [3,3,3,3,5,5],
        x => { double z=(x[4]-x[5])*x[2]/2; double s=Math.Abs(z)<0.001 ? 1-z*z/3+2*Math.Pow(z,4)/45 : (1-Math.Cos(2*z))/(2*z*z); return x[0]*x[1]*x[2]/x[3]*s; }),
    (new Feynman91(42), "III.10.19", ["mom","Bx","By","Bz"], [1,1,1,1], [5,5,5,5],
        x => Math.Sqrt(x.Skip(1).Sum(v => Math.Pow(x[0]*v,2))))
};
int checkedTargets = 0;
foreach (var (descriptor, id, inputs, min, max, target) in specs)
{
    Check(descriptor.Name.StartsWith(id + " "), $"Identifier: {id}");
    Check(descriptor.AllowedInputVariables.SequenceEqual(inputs), $"Input order: {id}");
    var columns = descriptor.GenerateValues();
    var repeated = descriptor.GenerateValues();
    Check(columns.Count == descriptor.VariableNames.Length && columns.All(c => c.Count == 20000), $"Shape: {id}");
    Check(columns.Zip(repeated).All(pair => pair.First.SequenceEqual(pair.Second)), $"Fixed generation: {id}");
    int[] inputIndices = inputs.Select(name => Array.IndexOf(descriptor.VariableNames,name)).ToArray();
    int targetIndex = Array.IndexOf(descriptor.VariableNames,descriptor.TargetVariable);
    for (int j=0; j<inputs.Length; j++)
    {
        var values=columns[inputIndices[j]];
        Check(values.All(v => v>=min[j] && v<=max[j]), $"Domain: {id}/{inputs[j]}");
        // Detect silently retained narrower ranges as well as out-of-domain samples.
        Check(values.Min()<min[j]+0.02*(max[j]-min[j]) && values.Max()>max[j]-0.02*(max[j]-min[j]), $"Coverage: {id}/{inputs[j]}");
    }
    for (int i=0; i<20000; i++)
    {
        double expected=target(inputIndices.Select(j => columns[j][i]).ToArray());
        double actual=columns[targetIndex][i];
        Check(double.IsFinite(actual) && Math.Abs(expected-actual)<=1e-10*Math.Max(1,Math.Abs(expected)), $"Equation: {id}, row {i}");
        checkedTargets++;
    }
    var data=descriptor.PreparedData;
    Check(ReferenceEquals(data,descriptor.PreparedData), $"Frozen snapshot: {id}");
    Check(data.TrainingRows.Count==8000 && data.ValidationRows.Count==2000 && data.TestRows.Count==10000, $"Partitions: {id}");
    Check(data.TrainingRows.Concat(data.ValidationRows).Concat(data.TestRows).Distinct().Count()==20000, $"Disjoint partitions: {id}");
    Console.WriteLine($"PASS {id}: all 20,000 targets, names, domains, repeatability and partitions");
}

var toy = new LinearDescriptor();
var evaluator = new TrialProbe();
foreach (var (population, allowance) in new[] { (100,1000), (550,1010), (910,10000), (1000,1000), (1000,1010) })
{
    var parameter=new AlgorithmParameter { PopulationSize=population, MaxTreeDepth=6, MaxTreeLength=20 };
    var result=await evaluator.Run(toy,parameter,42,allowance);
    Check(result.EvaluationsUsed==allowance && result.EvaluationsLimit==allowance, "Exact evaluation cap, including partial generation");
    Check(result.GenerationsCompleted==result.QualityCurve!.Count-1, "Initialization is not an offspring generation");
    Check(result.GenerationsCompleted==(allowance+population-1)/population-1, "Expected number of complete/partial generations");
    Check(result.GPSeed==42 && double.IsFinite(result.BestFitness) && result.TestFitness is null, "Trial seed and deferred test scoring");
    Check(result.BestIndividualsSnapshot.Any(s => double.IsFinite(s.ObjectiveValue)), "Finite evaluated model survives partial generation");
    var again=await evaluator.Run(toy,parameter,42,allowance);
    Check(result.SelectedExpression==again.SelectedExpression && result.BestFitness==again.BestFitness, "GP repeatability");
    result.EvaluateTest(toy.PreparedData);
    Check(double.IsFinite(result.TestFitness!.Value), "Held-out model scoring");
    Console.WriteLine($"PASS exact evaluator: population={population}, evaluations={result.EvaluationsUsed}, offspring generations={result.GenerationsCompleted}");
}

foreach (HyperParameterOptimizationAlgorithm method in new HyperParameterOptimizationAlgorithm[] { new RandomSearch(),new GridSearch(),new HillClimbingSearch() })
{
    var results=await method.Execute(toy,142,3,1010);
    Check(results.Count>0 && results.Count<=3 && results.All(r => r.EvaluationsUsed==1010), method.GetType().Name+" trial/work cap");
    Check(results.Select(r => r.ConfigurationIndex).SequenceEqual(Enumerable.Range(0,results.Count)), "Trial sequence includes every neighbor");
    Check(results.Select(r => r.GPSeed!.Value).SequenceEqual(Enumerable.Range(0,results.Count).Select(i=>HyperParameterOptimizationAlgorithm.TrialSeed(142,i))), "Common local-search seed schedule");
    await Throws<ArgumentOutOfRangeException>(() => method.Execute(toy,142,0,1000));
    await Throws<ArgumentOutOfRangeException>(() => method.Execute(toy,142,1,99));
    Console.WriteLine($"PASS {method.GetType().Name}: {results.Count} recorded GP trials");
}

// Freeze a recommendation using validation only, then assess its fixed configuration.
var nominator = new RandomSearch();
var tuning = await nominator.Execute(toy,142,3,1010);
var nominated = nominator.NominateConfiguration(tuning);
Check(nominated.BestFitness == tuning.Min(r=>r.BestFitness), "Validation-based nomination");
var native = new IRaceSearch();
typeof(IRaceSearch).GetField("recommendedConfigurationId",BindingFlags.Instance|BindingFlags.NonPublic)!
    .SetValue(native,tuning[^1].ConfigurationIndex);
Check(ReferenceEquals(native.NominateConfiguration(tuning),tuning[^1]), "Native irace recommendation overrides a luckiest-trial shortcut");
var beforeAssessment = JsonSerializer.Serialize(tuning,JsonOptions());
var assessed = await nominator.AssessConfiguration(toy,nominated.Configuration,tuning,142,3,1010);
Check(assessed.Count==3 && assessed.All(r=>r.Configuration==nominated.Configuration && r.EvaluationsUsed==1010 && r.TestFitness is not null), "Fixed configuration, separate assessment budget, and test scores");
Check(assessed.Select(r=>r.GPSeed).Distinct().Count()==3 && !assessed.Any(r=>tuning.Any(t=>t.GPSeed==r.GPSeed)), "Fresh unique assessment seeds");
Check(JsonSerializer.Serialize(tuning,JsonOptions())==beforeAssessment, "Assessment does not change tuning history or its selection scores");
var repeatedAssessment = await nominator.AssessConfiguration(toy,nominated.Configuration,tuning,142,3,1010);
Check(assessed.Zip(repeatedAssessment).All(pair=>pair.First.GPSeed==pair.Second.GPSeed && pair.First.SelectedExpression==pair.Second.SelectedExpression && pair.First.TestFitness==pair.Second.TestFitness), "Deterministic assessment seed stream and models");
int? originalSeed=tuning[0].GPSeed;
tuning[0].GPSeed=assessed[0].GPSeed;
var collisionCheck = await nominator.AssessConfiguration(toy,nominated.Configuration,tuning,142,1,1010);
Check(!tuning.Any(t=>t.GPSeed==collisionCheck[0].GPSeed), "Assessment skips a seed already used in tuning");
tuning[0].GPSeed=originalSeed;
var combined=new AggregatedResultDTO {
    AlgorithmName="RandomSearch",InstanceName="linear-check",Repetition=1,Seed=142,
    NumConfigurations=3,EvaluationsPerConfiguration=1010,TotalConfigurationsEvaluated=tuning.Count,
    ConfigurationResults=tuning,RecommendedConfiguration=nominated.Configuration,
    RecommendedConfigurationIndex=nominated.ConfigurationIndex,RecommendationRule=nominator.RecommendationRule,
    ConfigurationAssessmentResults=assessed };
var restored=JsonSerializer.Deserialize<AggregatedResultDTO>(JsonSerializer.Serialize(combined,JsonOptions()),JsonOptions())!;
Check(restored.ConfigurationResults.Count==3 && restored.ConfigurationAssessmentResults!.Count==3 && restored.RecommendedConfiguration==nominated.Configuration, "Separate outcome records round-trip");
// Shared-directory publication and aggregation: concurrent writers own distinct jobs.
var storeDirectory = Path.Combine(Environment.CurrentDirectory, ".artifacts", "result-store-checks-" + Guid.NewGuid().ToString("N"));
Directory.CreateDirectory(storeDirectory);
try
{
    restored.TrainingRows = toy.PreparedData.TrainingRows.Count;
    restored.ValidationRows = toy.PreparedData.ValidationRows.Count;
    restored.TestRows = toy.PreparedData.TestRows.Count;
    restored.SelectedTrialIndex = restored.ConfigurationResults.IndexOf(
        restored.ConfigurationResults.MinBy(trial => trial.BestFitness)!);
    restored.ConfigurationResults[restored.SelectedTrialIndex.Value].TestFitness = 0.25;
    var completedJson = JsonSerializer.Serialize(restored, JsonOptions());
    Parallel.For(1, 5, repetition => {
        var job = JsonSerializer.Deserialize<AggregatedResultDTO>(completedJson, JsonOptions())!;
        job.Repetition = repetition;
        job.Seed = 42 + 100 * repetition;
        ExperimentResultStore.Save(storeDirectory, job);
    });
    Check(Directory.GetFiles(storeDirectory, "*.result.json").Length == 4 &&
        Directory.GetFiles(storeDirectory, "*.tmp").Length == 0, "Independent concurrent result publication");
    var originalPath = Path.Combine(storeDirectory, ExperimentResultStore.FileName(restored));
    var originalText = File.ReadAllText(originalPath);
    await Throws<IOException>(() => Task.Run(() => ExperimentResultStore.Save(storeDirectory, restored)));
    Check(File.ReadAllText(originalPath) == originalText, "Completed results cannot be overwritten");
    File.WriteAllText(Path.Combine(storeDirectory, "interrupted.result.json.partial.tmp"), "{");
    Check(ExperimentResultStore.Aggregate(storeDirectory) == 4, "Aggregation ignores interrupted temporary files");
    var aggregatePath = Path.Combine(storeDirectory, "all_results.json");
    var aggregateText = File.ReadAllText(aggregatePath);
    Check(JsonSerializer.Deserialize<List<AggregatedResultDTO>>(aggregateText, JsonOptions())!.Count == 4,
        "Aggregate contains each complete job once");
    var duplicatePath = Path.Combine(storeDirectory, "duplicate.result.json");
    File.Copy(originalPath, duplicatePath);
    await Throws<InvalidDataException>(() => Task.Run(() => ExperimentResultStore.Aggregate(storeDirectory)));
    Check(File.ReadAllText(aggregatePath) == aggregateText, "Duplicate input does not replace a valid aggregate");
    File.Delete(duplicatePath);
    var brokenPath = Path.Combine(storeDirectory, "broken.result.json");
    File.WriteAllText(brokenPath, "{");
    await Throws<JsonException>(() => Task.Run(() => ExperimentResultStore.Aggregate(storeDirectory)));
    Check(File.ReadAllText(aggregatePath) == aggregateText, "Malformed input does not replace a valid aggregate");
    File.Delete(brokenPath);
    restored.AlgorithmName = "AgentSearch";
    restored.ModelName = "publisher/model:8b";
    var modelFile = ExperimentResultStore.FileName(restored);
    Check(modelFile.Contains("__model=publisher%2Fmodel%3A8b__instance="), "Reversible portable model filename");
    ExperimentResultStore.Save(storeDirectory, restored);
    restored.ModelName = "publisher_model:8b";
    Check(ExperimentResultStore.FileName(restored) != modelFile, "Distinct model names do not merge under sanitization");
    ExperimentResultStore.Save(storeDirectory, restored);
    Check(ExperimentResultStore.Aggregate(storeDirectory) == 6, "Models remain separate in aggregation");
    restored.ModelName = "incomplete:fixture";
    restored.SelectedTrialIndex = null;
    ExperimentResultStore.Save(storeDirectory, restored);
    await Throws<InvalidDataException>(() => Task.Run(() => ExperimentResultStore.Load(
        Path.Combine(storeDirectory, ExperimentResultStore.FileName(restored)))));
    Console.WriteLine("PASS shared results: atomic publication, concurrent jobs, model names, incomplete/duplicate rejection and aggregation");
}
finally
{
    foreach (var path in Directory.EnumerateFiles(storeDirectory)) File.Delete(path);
    Directory.Delete(storeDirectory);
}
await Throws<ArgumentOutOfRangeException>(()=>nominator.AssessConfiguration(toy,nominated.Configuration,tuning,142,0,1010));
Console.WriteLine("PASS holdout nomination and fresh-seed configuration assessment");

// Exercise search control independently from GP quality, including non-improving neighbors,
// a cap in the middle of a neighborhood, and exhaustion of the fixed grid.
var hill=new HillProbe();
var hillResults=await hill.Execute(toy,42,5,1010);
Check(hillResults.Count==5 && hill.Calls==5, "All unsuccessful neighbor trials retained and charged");
Check((await hill.Execute(toy,42,1,1000)).Count==1, "One-trial hill cap");
var grid=new GridProbe();
var gridResults=await grid.Execute(toy,42,5001,1010);
Check(gridResults.Count==5000 && gridResults.All(r => r.EvaluationsLimit==1010), "Grid exhaustion and per-trial allocation");
Check(gridResults.Select(r => r.Configuration).Distinct().Count()==5000, "Grid has no duplicate points");
var shortGrid=await grid.Execute(toy,42,7,1010);
var hillRestarts=await new HillProbe().Execute(toy,42,60,1010);
Check(hillRestarts.Count==60 && hillRestarts.Select(r=>r.Configuration).Distinct().Count()==60,
    "Hill climbing restarts at local optima until the trial cap, without repeating configurations");
Check(hillRestarts[0].Configuration!=(await new HillProbe().Execute(toy,142,1,1010))[0].Configuration, "Hill climbing start depends on the HPO seed");
var seedsRep1=Enumerable.Range(0,1000).Select(i=>HyperParameterOptimizationAlgorithm.TrialSeed(142,i)).ToHashSet();
var seedsRep2=Enumerable.Range(0,1000).Select(i=>HyperParameterOptimizationAlgorithm.TrialSeed(242,i)).ToHashSet();
Check(seedsRep1.Count==1000 && seedsRep2.Count==1000 && !seedsRep1.Overlaps(seedsRep2) && seedsRep1.All(s=>s>=0),
    "Trial seeds are non-negative and not shared between consecutive repetitions");
Check(shortGrid.Select(r=>r.Configuration).SequenceEqual(gridResults.Take(7).Select(r=>r.Configuration)), "Stable grid prefix");
var random=new RandomProbe();
var randomResults=await random.Execute(toy,42,2000,1010);
foreach (var (name, select) in new (string, Func<AlgorithmParameter,int>)[] { (nameof(AlgorithmParameter.PopulationSize), p=>p.PopulationSize),
    (nameof(AlgorithmParameter.MaxTreeDepth), p=>p.MaxTreeDepth), (nameof(AlgorithmParameter.MaxTreeLength), p=>p.MaxTreeLength) })
{
    int maximum = Convert.ToInt32(typeof(AlgorithmParameter).GetProperty(name)!.GetCustomAttribute<System.ComponentModel.DataAnnotations.RangeAttribute>()!.Maximum);
    Check(randomResults.Any(r=>select(r.Configuration)==maximum), "Random integer upper endpoint reachable: " + name);
}
Console.WriteLine("PASS search-control regression checks");

// Real GP checkpoint execution with deterministic controllers, without a model service.
foreach (var behavior in new[] { "stop", "continue", "failure", "timeout", "invalid" })
{
    var controlled = new ControllerProbe(behavior) { FeynmanInstance = toy, Seed = 142 };
    foreach (var (name, value) in new[] { ("_numConfigurations", 1), ("_totalEvaluations", 3010), ("_evaluationsPerConfiguration", 3010) })
        typeof(AgentSearch).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(controlled, value);
    await controlled.RunSymbolicRegression(new AlgorithmParameter { PopulationSize = 550, MaxTreeDepth = 6, MaxTreeLength = 20 }, 3010);
    var record = controlled.Results.Single();
    Check(record.ControlObservations!.Count > 0, "Checkpoint observations persisted");
    Check(record.ControlObservations.All(o => o.Progress.Generation > 0 && o.Progress.EvaluationsRemaining > 0 &&
        o.Progress.BestTrainingNmse is double b && double.IsFinite(b)), "Training checkpoint summary and no terminal calls");
    // The probe requests every channel: each checkpoint carries all sections with values in their ranges.
    Check(record.ControlObservations.All(o => o.Progress.Curve is { } c && c.Generations.Length is > 0 and <= 10 &&
            c.Generations.Zip(c.Generations.Skip(1)).All(p => p.First < p.Second) && c.Generations[^1] == o.Progress.Generation &&
            c.BestTrainingNmse.Zip(c.MedianTrainingNmse).All(p => p.First <= p.Second) &&
        o.Progress.Diversity is { } d && d.UniqueStructures is > 0 and <= 1 && d.PhenotypicDiversity is >= 0 and <= 1 &&
            d.InvalidOutputs is >= 0 and <= 1 &&
        o.Progress.Size is { } z && z.AtLengthLimit is >= 0 and <= 1 && z.AtDepthLimit is >= 0 and <= 1 && z.MaxLength <= 20 &&
            z.MedianLength <= z.MaxLength &&
        o.Progress.Generalization is { } g && double.IsFinite(g.ValidationNmseOfBest) &&
            Math.Abs(g.Gap - (g.ValidationNmseOfBest - o.Progress.BestTrainingNmse!.Value)) < 1e-12),
        "Curve, diversity, size and generalization feedback are present and consistent");
    Check(controlled.BudgetWasCurrent, "Active GP evaluations are included in agent budget");
    Check(record.EvaluationsUsed == (behavior == "stop" ? 1100 : 3010), "Stop early or resume to hard cap");
    Check(record.TerminationReason == (behavior == "stop" ? "controller_stop" : "evaluation_limit"), "Explicit termination reason");
    Check(record.ControlObservations.All(o => o.UsedFallback == (behavior is "failure" or "timeout" or "invalid")), "Controller failure/timeout falls back to continue");
    if (behavior == "stop")
    {
        record.EvaluateTest(toy.PreparedData);
        // Every nominee, stopped early or not, is assessed at E and 4E with the same seeds at both lengths.
        int[] assessmentLengths = HyperParameterOptimizationAlgorithm.AssessmentRunLengths(3010);
        var assessments = await controlled.AssessConfiguration(toy, record.Configuration, controlled.Results, 142, 2, assessmentLengths);
        Check(assessmentLengths.SequenceEqual([3010, 12040]) && assessments.Select(a => a.EvaluationsUsed).SequenceEqual([3010, 3010, 12040, 12040]) &&
            assessments.Take(2).Select(a => a.GPSeed).SequenceEqual(assessments.Skip(2).Select(a => a.GPSeed)) &&
            assessments.All(a => a.TerminationReason == "evaluation_limit" && a.TestFitness is not null),
            "Assessment runs every seed at E and 4E, without a controller, whatever the nominee's run used");
        var stored = new AggregatedResultDTO {
            AlgorithmName = "AgentSearch", ModelName = "fixture", InstanceName = "linear-check", Repetition = 1, Seed = 142,
            NumConfigurations = 1, EvaluationsPerConfiguration = 3010, TotalConfigurationsEvaluated = 1,
            AssessmentRunLengths = assessmentLengths,
            ConfigurationResults = controlled.Results, SelectedTrialIndex = 0,
            RecommendedConfiguration = record.Configuration, RecommendedConfigurationIndex = 0,
            RecommendationRule = controlled.RecommendationRule, ConfigurationAssessmentResults = assessments,
            TrainingRows = toy.PreparedData.TrainingRows.Count, ValidationRows = toy.PreparedData.ValidationRows.Count,
            TestRows = toy.PreparedData.TestRows.Count
        };
        var folder = Directory.CreateTempSubdirectory("hyperop-control-check-");
        try {
            ExperimentResultStore.Save(folder.FullName, stored);
            string path = Directory.GetFiles(folder.FullName, "*.result.json").Single();
            Check(ExperimentResultStore.Load(path).ConfigurationResults[0].TerminationReason == "controller_stop", "Early-stopped result is cache-compatible");
            Check(ExperimentResultStore.Aggregate(folder.FullName) == 1, "Early-stopped result aggregates");
            // Pilots without assessment (K = 0) are complete results that resume and aggregate.
            var pilot = new AggregatedResultDTO {
                AlgorithmName = stored.AlgorithmName, ModelName = stored.ModelName, InstanceName = stored.InstanceName, Repetition = 2, Seed = 242,
                NumConfigurations = 1, EvaluationsPerConfiguration = 3010, TotalConfigurationsEvaluated = 1, AssessmentRunLengths = assessmentLengths,
                ConfigurationResults = controlled.Results, SelectedTrialIndex = 0, RecommendedConfiguration = record.Configuration,
                RecommendedConfigurationIndex = 0, RecommendationRule = controlled.RecommendationRule, ConfigurationAssessmentResults = [],
                TrainingRows = stored.TrainingRows, ValidationRows = stored.ValidationRows, TestRows = stored.TestRows,
                Provenance = HyperOp.Algorithms.Logging.RunProvenance.Capture(HyperParameterOptimizationAlgorithm.ConstantFitting)
            };
            ExperimentResultStore.Save(folder.FullName, pilot);
            string pilotPath = Directory.GetFiles(folder.FullName, "*rep=2*K=0.result.json").Single();
            Check(ExperimentResultStore.Load(pilotPath).ConfigurationAssessmentResults!.Count == 0 && ExperimentResultStore.Aggregate(folder.FullName) == 2 &&
                ExperimentResultStore.Load(pilotPath).Provenance!.SameFitting(pilot.Provenance), "Results without assessment (K = 0) resume and aggregate, with their provenance");
            stored.ConfigurationResults[0].ControlObservations = [];
            File.WriteAllText(path, JsonSerializer.Serialize(stored, JsonOptions()));
            await Throws<InvalidDataException>(() => Task.Run(() => ExperimentResultStore.Load(path)));
        }
        finally { folder.Delete(true); }
    }
    if (behavior != "stop")
    {
        var uninterrupted = await new TrialProbe().Run(toy, record.Configuration, HyperParameterOptimizationAlgorithm.TrialSeed(142, 0), 3010);
        Check(record.SelectedExpression == uninterrupted.SelectedExpression && record.BestFitness == uninterrupted.BestFitness,
            "Checkpoint continuation preserves population/RNG trajectory");
    }
}
Console.WriteLine("PASS asynchronous run control: stop, continue, failures, timeouts, budgets and GP trajectory");

// Evaluate the real agent tool without contacting a model; initialize its run context directly.
// The deterministic controller stops each run at its first checkpoint (generation 1). The first call may use the
// whole pool; the controller's stop leaves 92 evaluations, which the second call requests.
var agent=new ControllerProbe("stop") { FeynmanInstance=toy, Seed=142 };
foreach (var (name,value) in new[] { ("_numConfigurations",2),("_totalEvaluations",2020),("_evaluationsPerConfiguration",1010) })
    typeof(AgentSearch).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(agent,value);
var agentParameter=new AlgorithmParameter { PopulationSize=550,MaxTreeDepth=6,MaxTreeLength=20 };
await Throws<System.ComponentModel.DataAnnotations.ValidationException>(() => agent.RunSymbolicRegression(agentParameter with { PopulationSize=99 },2020));
await Throws<System.ComponentModel.DataAnnotations.ValidationException>(() => agent.RunSymbolicRegression(agentParameter with { MutationRate=double.NaN },2020));
await Throws<ArgumentException>(() => agent.RunSymbolicRegression(agentParameter with { MutatorType=(MutatorType)99 },2020));
await Throws<System.ComponentModel.DataAnnotations.ValidationException>(() => agent.RunSymbolicRegression(agentParameter with { MaxTreeDepth=25 },2020));
Check(agent.Results.Count==0 && agent.GetRemainingEvaluations().UsedEvaluations==0 &&
    typeof(AgentSearch).GetField("_trialFailure",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(agent) is null,
    "Shared validation rejects invalid agent proposals before spending trials, without ending the job");
// The second request asks for more than the 92 remaining evaluations and receives the remainder.
await Task.WhenAll(agent.RunSymbolicRegression(agentParameter,2020),agent.RunSymbolicRegression(agentParameter,5000));
await Throws<InvalidOperationException>(()=>agent.RunSymbolicRegression(agentParameter,92));
// First trial: stopped after 1100 of the 2020 shared evaluations; the repeat receives the remaining 920.
Check(agent.Results.Count==2 && agent.GetRemainingEvaluations().UsedEvaluations==2020 && agent.GetRemainingEvaluations().RemainingEvaluations==0, "Concurrent agent tools honor trial/work caps");
Check(agent.Results.Select(r=>r.EvaluationsUsed).SequenceEqual(new[] {1100,920}), "Shared pool: controller stop frees budget for the next trial");
Check(agent.Results.Select(r=>r.GPSeed).SequenceEqual(new int?[] {HyperParameterOptimizationAlgorithm.TrialSeed(142,0),HyperParameterOptimizationAlgorithm.TrialSeed(142,1)}) && agent.ShownTrials().Count==2,
    "Repeated agent configuration is a new charged trial with a new seed, not a cached result");
var agentNominee=agent.NominateConfiguration(agent.Results);
var assessedAgent=await agent.AssessConfiguration(toy,agentNominee.Configuration,agent.Results,142,2,1010);
Check(assessedAgent.Count==2 && assessedAgent.All(r=>r.EvaluationsUsed==1010) && agent.Results.Count==2 &&
    agent.ShownTrials().Count==2 && agent.GetRemainingEvaluations().UsedEvaluations==2020,
    "Post-tuning assessment neither consumes tuning budget nor feeds back into history");
var json=JsonSerializer.Serialize(agent.Results,new JsonSerializerOptions { NumberHandling=System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals });
Check(JsonSerializer.Deserialize<List<ResultDTO>>(json,new JsonSerializerOptions { NumberHandling=System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals })!
    .Select(r=>(r.GPSeed,r.EvaluationsUsed)).SequenceEqual(new (int?,int)[] {(HyperParameterOptimizationAlgorithm.TrialSeed(142,0),1100),(HyperParameterOptimizationAlgorithm.TrialSeed(142,1),920)}), "Result count/seed JSON round-trip");
Console.WriteLine($"PASS agent tools and JSON; {checkedTargets:N0} benchmark targets verified");

// Baselines for the agent: the untuned default and the rule-controlled shared-pool search.
var defaults=new DefaultConfigurationSearch();
var defaultRuns=await defaults.Execute(toy,142,3,1010);
Check(defaultRuns.Count==3 && defaultRuns.All(r=>r.Configuration==new AlgorithmParameter() && r.EvaluationsUsed==1010) &&
    defaults.NominateConfiguration(defaultRuns).Configuration==new AlgorithmParameter(), "Default baseline repeats the default configuration");
Check(defaultRuns.Select(r=>r.GPSeed).Distinct().Count()==3, "Default baseline uses a new GP seed per trial");

static RunProgress Progress(int evaluations,double best)=>new(evaluations/10-1,evaluations,1000-evaluations,best,null);
static RunControlObservation Seen(RunProgress progress)=>new(progress,new RunDecision(RunAction.Continue,"x",5),0,false);
static ResultDTO Curve(double best)=>new() { Configuration=new AlgorithmParameter { PopulationSize=10 },ConfigurationIndex=0,BestFitness=1,
    MeanPopulationFitness=1,WorstPopulationFitness=1,ExecutionTimeMs=0,EvaluationsUsed=200,EvaluationsLimit=200,GenerationsCompleted=19,
    PopulationSize=10,BestIndividualsSnapshot=[],
    QualityCurve=Enumerable.Range(0,20).Select(g=>new ResultDTO.GenerationQualitySnapshot { GenerationNumber=g,BestQuality=best+1.0/(g+1),MedianQuality=1,WorstQuality=1 }).ToList() };
Check(RuleControlledSearch.DecideRun(Progress(100,0.5),[],[],5).Action==RunAction.Continue, "Rules continue at the first checkpoint");
Check(RuleControlledSearch.DecideRun(Progress(150,0.499),[Seen(Progress(100,0.5))],[],5).Action==RunAction.Stop, "Stagnation rule stops below 1% improvement");
Check(RuleControlledSearch.DecideRun(Progress(150,0.4),[Seen(Progress(100,0.5))],[],5).Action==RunAction.Continue, "Improving run continues");
Check(RuleControlledSearch.BestTrainingAt(Curve(0),100)==0.1 && RuleControlledSearch.BestTrainingAt(Curve(0),500) is null, "Best training NMSE at an evaluation count");
var references=new[] { Curve(0), Curve(0), Curve(0) };
Check(RuleControlledSearch.DecideRun(Progress(100,0.5),[],references,5).Action==RunAction.Stop, "Median rule stops a run worse than earlier runs");
Check(RuleControlledSearch.DecideRun(Progress(100,0.05),[],references,5).Action==RunAction.Continue, "Median rule keeps a run better than earlier runs");
Check(RuleControlledSearch.DecideRun(Progress(100,0.5),[],references[..2],5).Action==RunAction.Continue, "Median rule needs three earlier runs");
var rules=new RuleControlledSearch(5);
var ruleRuns=await rules.Execute(toy,142,4,3010);
Check(ruleRuns.Count>=1 && ruleRuns.Count<=HyperParameterOptimizationAlgorithm.MaximumSharedPoolTrials(4,3010,100) && ruleRuns.Sum(r=>r.EvaluationsUsed)<=4*3010,
    "Rule baseline honors the shared pool and the loose trial cap");
// The pool, not T, is the rule baseline's budget: a run stopped early frees evaluations for further trials.
Check(ruleRuns.Count<=4 || ruleRuns.Take(4).Any(r=>r.TerminationReason=="controller_stop"), "Trials beyond T only use evaluations freed by stopped runs");
Check(CheckpointSchedule.Interval(30000,10)==3000 && CheckpointSchedule.Interval(1010,3)==337 && CheckpointSchedule.Interval(5,10)==1 &&
    CheckpointSchedule.Name(10).IndexOfAny(Path.GetInvalidFileNameChars())<0, "Checkpoint interval is E/k evaluations, rounded up; its name is file-name safe");
int ruleInterval=CheckpointSchedule.Interval(3010,rules.Checkpoints);
// Each checkpoint is the first generation boundary at least E/5 evaluations after the previous one (or the run start).
Check(ruleRuns.All(r=>r.ControlObservations!.Select((o,i)=>o.Progress.EvaluationsUsed-(i==0 ? 0 : r.ControlObservations![i-1].Progress.EvaluationsUsed))
        .All(step=>step>=ruleInterval && step<ruleInterval+2*r.PopulationSize)) &&
    ruleRuns.Where(r=>r.TerminationReason=="controller_stop").All(r=>r.ControlObservations![^1].Decision.Action==RunAction.Stop),
    "Rule baseline checkpoints every E/5 evaluations and records its stop decisions");
Check(ruleRuns.Select(r=>r.GPSeed).SequenceEqual(Enumerable.Range(0,ruleRuns.Count).Select(i=>(int?)HyperParameterOptimizationAlgorithm.TrialSeed(142,i))),
    "Rule baseline uses the common trial seeds");
Check(ruleRuns.All(r=>r.EvaluationsLimit<=HyperParameterOptimizationAlgorithm.MaximumRunEvaluations(3010)), "Rule baseline runs are granted at most 4E");

// Per-run bounds of the methods that choose run length: E/10 to 4E, never beyond the remaining pool.
Check(HyperParameterOptimizationAlgorithm.MinimumRunEvaluations(30000)==3000 && HyperParameterOptimizationAlgorithm.MaximumRunEvaluations(30000)==120000 &&
    HyperParameterOptimizationAlgorithm.BoundRunAllowance(500,30000,900000)==3000 && HyperParameterOptimizationAlgorithm.BoundRunAllowance(250000,30000,900000)==120000 &&
    HyperParameterOptimizationAlgorithm.BoundRunAllowance(50000,30000,900000)==50000 && HyperParameterOptimizationAlgorithm.BoundRunAllowance(250000,30000,40000)==40000,
    "Run allowances are bounded to E/10..4E and the remaining pool");
// Random run length: random configurations with log-uniform run lengths from the shared pool, no controller.
var lengthRuns=await new RandomRunLengthSearch().Execute(toy,142,4,3010);
var lengthRunsAgain=await new RandomRunLengthSearch().Execute(toy,142,4,3010);
Check(lengthRuns.Count>=1 && lengthRuns.Count<=HyperParameterOptimizationAlgorithm.MaximumSharedPoolTrials(4,3010,301) && lengthRuns.Sum(r=>r.EvaluationsUsed)<=4*3010 &&
    lengthRuns.All(r=>r.TerminationReason=="evaluation_limit" && r.EvaluationsLimit<=HyperParameterOptimizationAlgorithm.MaximumRunEvaluations(3010) &&
        (r.EvaluationsLimit>=HyperParameterOptimizationAlgorithm.MinimumRunEvaluations(3010) || r==lengthRuns[^1])) &&
    lengthRuns.Select(r=>r.EvaluationsLimit).Distinct().Count()>1 &&
    lengthRuns.Select(r=>(r.EvaluationsLimit,r.BestFitness)).SequenceEqual(lengthRunsAgain.Select(r=>(r.EvaluationsLimit,r.BestFitness))) &&
    lengthRuns.Select(r=>r.GPSeed).SequenceEqual(Enumerable.Range(0,lengthRuns.Count).Select(i=>(int?)HyperParameterOptimizationAlgorithm.TrialSeed(142,i))),
    $"Random run length: bounded, varied, deterministic lengths from the shared pool ({string.Join(", ", lengthRuns.Select(r=>r.EvaluationsLimit))})");
// ASHA: rungs at E/10 x 3^k below 4E; a run continues past a rung only within the best third of the runs that reached it.
Check(AshaSearch.Rungs(30000,3).SequenceEqual([3000,9000,27000,81000]) && AshaSearch.Rungs(3010,3).SequenceEqual([301,903,2709,8127]),
    "ASHA rungs lie at E/10 x 3^k below the longest allowed run");
Check(!AshaSearch.Promote(0.1,[0.1],3) && !AshaSearch.Promote(0.1,[0.2,0.1],3) && AshaSearch.Promote(0.1,[0.5,0.2,0.1],3) && !AshaSearch.Promote(0.2,[0.5,0.2,0.1],3) &&
    AshaSearch.Promote(0.2,[0.9,0.8,0.7,0.6,0.5,0.2],3) && AshaSearch.Promote(0.5,[0.9,0.8,0.7,0.6,0.5,0.2],3) && !AshaSearch.Promote(0.6,[0.9,0.8,0.7,0.6,0.5,0.2],3),
    "ASHA promotes the best floor(n/3) of the runs recorded at a rung, none before 3 runs reached it");
var ashaRuns=await new AshaSearch().Execute(toy,142,4,3010);
var ashaAgain=await new AshaSearch().Execute(toy,142,4,3010);
Check(ashaRuns.Count>=2 && ashaRuns.Sum(r=>r.EvaluationsUsed)<=4*3010 && ashaRuns.Sum(r=>r.EvaluationsUsed)>=4*3010-100 &&
    ashaRuns.All(r=>r.EvaluationsLimit<=HyperParameterOptimizationAlgorithm.MaximumRunEvaluations(3010)) &&
    ashaRuns.Any(r=>r.TerminationReason=="controller_stop") &&
    ashaRuns.Where(r=>r.TerminationReason=="controller_stop").All(r=>r.ControlObservations![^1].Decision.Reason.Contains("not promoted")) &&
    ashaRuns.All(r=>r.ControlObservations!.Select((o,i)=>o.Progress.EvaluationsUsed>=AshaSearch.Rungs(3010,3)[i]).All(x=>x)) &&
    // Rung values: the best training NMSE after at most the rung's evaluations, from the run's own quality curve.
    ashaRuns.All(r=>r.ControlObservations!.Select((o,i)=>(o,i)).All(x=>x.o.Decision.Reason.StartsWith($"rung {x.i} (")) ) &&
    ashaRuns.Take(2).All(r=>r.ControlObservations!.Count==1 && r.ControlObservations![0].Decision.Reason.Contains("not promoted")) &&
    ashaRuns.Select(r=>(r.EvaluationsUsed,r.BestFitness)).SequenceEqual(ashaAgain.Select(r=>(r.EvaluationsUsed,r.BestFitness))),
    $"ASHA stops runs at rungs, spends the pool and is deterministic ({ashaRuns.Count} runs: {string.Join(", ", ashaRuns.Select(r=>r.EvaluationsUsed))})");

// Snapshots every E/10 evaluations in every tuning run (controlled or not), closed by the final state; none in assessment runs.
int snapshotInterval=CheckpointSchedule.Interval(3010,HyperParameterOptimizationAlgorithm.SnapshotsPerConfiguration);
foreach (var tuned in lengthRuns.Concat(ruleRuns))
{
    var s=tuned.Snapshots!;
    Check(s.Count>=1 && s[^1].Final && s[^1].Evaluations==tuned.EvaluationsUsed && s[^1].ValidationNmse==tuned.BestFitness && s.Count(x=>x.Final)==1 &&
        s.Zip(s.Skip(1)).All(p=>p.First.Evaluations<p.Second.Evaluations && p.First.Evaluations/snapshotInterval<p.Second.Evaluations/snapshotInterval || p.Second.Final) &&
        s.Where(x=>!x.Final).All(x=>x.Evaluations>=snapshotInterval) && s.All(x=>double.IsFinite(x.TestNmse) && x.Length>0 && x.Expression.Length>0),
        $"Tuning runs record a snapshot at every multiple of E/10 and at the end ({s.Count} snapshots for {tuned.EvaluationsUsed} evaluations)");
}
// Recording snapshots streams the run; the trajectory equals an unrecorded run with the same configuration and seed.
var unrecorded=await new TrialProbe().Run(toy,lengthRuns[0].Configuration,HyperParameterOptimizationAlgorithm.TrialSeed(142,0),lengthRuns[0].EvaluationsLimit);
Check(unrecorded.Snapshots is null && unrecorded.BestFitness==lengthRuns[0].BestFitness && unrecorded.SelectedExpression==lengthRuns[0].SelectedExpression,
    "Recording snapshots does not change the GP run");
var lengthAssessment=await new RandomRunLengthSearch().AssessConfiguration(toy,lengthRuns[0].Configuration,lengthRuns,142,1,3010);
Check(lengthAssessment.All(a=>a.Snapshots is null), "Assessment runs record no snapshots");

// Selection: lowest validation NMSE, values below 1e-10 equal, then the shortest expression, then the earliest trial and count.
ResultDTO WithSnapshots(int index, params (int Evaluations,double Validation,int Length)[] points)=>new() {
    Configuration=new AlgorithmParameter(), ConfigurationIndex=index, BestFitness=points[^1].Validation, MeanPopulationFitness=1, WorstPopulationFitness=1,
    ExecutionTimeMs=0, EvaluationsUsed=points[^1].Evaluations, EvaluationsLimit=points[^1].Evaluations, GenerationsCompleted=1, PopulationSize=100,
    BestIndividualsSnapshot=[], SelectedExpressionLength=points[^1].Length,
    Snapshots=points.Select((p,i)=>new ResultDTO.RunSnapshot { Evaluations=p.Evaluations, Generation=i, ValidationNmse=p.Validation, TestNmse=p.Validation*2,
        Length=p.Length, Expression=$"e{index}_{i}", Final=i==points.Length-1 }).ToList() };
var candidates=new List<ResultDTO> { WithSnapshots(0,(3000,1e-3,9),(6000,1e-12,15)), WithSnapshots(1,(3000,1e-20,11),(6000,1e-25,11)), WithSnapshots(2,(3000,5e-11,11)) };
var chosen=HyperParameterOptimizationAlgorithm.SelectExpression(candidates)!;
Check(chosen.TrialIndex==1 && chosen.Evaluations==3000 && chosen.Length==11 && chosen.TestNmse==2e-20,
    $"Below 1e-10 the shortest, then earliest expression is selected, not the smallest NMSE ({chosen})");
candidates.Add(WithSnapshots(3,(3000,2e-10,3)));
Check(HyperParameterOptimizationAlgorithm.SelectExpression(candidates)!.TrialIndex==1, "Above 1e-10 the lower validation NMSE wins over a shorter expression");
Check(new RandomSearch().NominateConfiguration(candidates)==candidates[1] && HyperParameterOptimizationAlgorithm.BestValidationTrial(candidates)==candidates[1],
    "The nominee is the trial of the selected snapshot; the final-only rule applies the same floor and tie-break");
Check(HyperParameterOptimizationAlgorithm.SelectExpression([unrecorded]) is null && new RandomSearch().NominateConfiguration([unrecorded])==unrecorded,
    "Without snapshots, nomination falls back to the final results");
Console.WriteLine($"PASS agent baselines: default configuration, rule-controlled shared pool ({ruleRuns.Count} trials, {ruleRuns.Sum(r=>r.EvaluationsUsed)} evaluations), run-length bounds, random run length, ASHA, snapshots, selection");

// Stateless configurator against a stub model: replies that are no valid proposal are rejected, the next request says
// why and uses another sampling seed (an identical request with a fixed seed returns an identical reply), and after
// MaxInvalidReplies rejections in a row the search ends; requests are timed at the HTTP layer with token usage.
string Reply(string action,int population=100,string depth="8",int evaluations=1010,string analysis="a",string plan="p")=>
    $"{{\"analysis\":\"{analysis}\",\"plan\":\"{plan}\",\"action\":\"{action}\",\"populationSize\":{population},\"mutationRate\":0.2,\"maxTreeDepth\":{depth},\"maxTreeLength\":20,\"mutatorType\":\"Combined\",\"evaluations\":{evaluations}}}";
AgentSearch StubAgent(StubModel model,FeedbackChannels feedback=FeedbackChannels.None,int checkpoints=10,string controller="llm",double[]? hazard=null)=>new(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
    Provider=LlmTornado.Code.LLmProviders.Custom, ModelName="stub", OllamaBaseUrl=model.Url, Streaming=false, Feedback=feedback, Checkpoints=checkpoints,
    Controller=controller, StopHazard=hazard });
using (var silent=new StubModel((n,body)=>"No JSON."))
{
    var idle=StubAgent(silent);
    await Throws<InvalidOperationException>(()=>idle.Execute(toy,142,3,1010));
    var bodies=silent.Bodies.ToArray();
    Check(silent.Requests==AgentSearch.MaxInvalidReplies && !bodies[0].Contains("PREVIOUS REPLY REJECTED") &&
        bodies.Skip(1).All(b=>b.Contains("PREVIOUS REPLY REJECTED")) &&
        Enumerable.Range(0,AgentSearch.MaxInvalidReplies).All(i=>bodies[i].Contains($"\"seed\":{142+i}")),
        $"Invalid replies are retried with the rejection and another seed, then end the search ({silent.Requests} requests)");
    Check(idle.Summary is { Exit: "configurator_invalid", Trials: 0, RemainingEvaluations: 3030 } summary && summary.InvalidReplies==AgentSearch.MaxInvalidReplies,
        "A search without a valid configuration still records its summary (the job fails and the analysis counts it)");
    Check(idle.LlmCalls.All(c=>c.Role=="configurator" && c.Outcome=="response" && c.InputTokens==11 && c.OutputTokens==3 && c.DurationMs>=0),
        "Model requests are timed at the HTTP layer with token usage");
}
Console.WriteLine("PASS agent guard: invalid configurator replies are retried with the rejection and another seed, then end the search; HTTP timing and tokens recorded");

// A string for an integer (as Qwen3.5 4B sent with tool calls) is rejected; an out-of-range value is moved to the nearest
// bound and the next request says so; valid proposals run until the pool is spent, and each later request shows the
// trials with the model's reasons and its plan. There is no way to end the search early: a reply that asks to finish is
// still read as a proposal.
using (var model=new StubModel((n,body)=>n switch {
    1=>Reply("run",depth:"\"8.5\""), 2=>Reply("run",depth:"25",analysis:"try combined",plan:"explore then exploit"),
    _=>Reply("finish",analysis:"second") }))
{
    var proposing=StubAgent(model);
    var runs=await proposing.Execute(toy,142,3,1010);
    var bodies=model.Bodies.ToArray();
    Check(runs.Count==3 && runs.Sum(r=>r.EvaluationsUsed)==3030 && model.Requests==4 && runs[0].Configuration.MaxTreeDepth==20,
        $"Rejected and clamped replies do not fail the job, and the search spends the whole pool ({runs.Count} trials, {model.Requests} requests)");
    Check(bodies[1].Contains("PREVIOUS REPLY REJECTED: unreadable reply") && !bodies[1].Contains("ADJUSTED") &&
        bodies[2].Contains("PREVIOUS PROPOSAL ADJUSTED") && bodies[2].Contains("MaxTreeDepth 25 -> 20") &&
        bodies[2].Contains("Trials so far: 1") && bodies[2].Contains("try combined") && bodies[2].Contains("YOUR PLAN (written before trial 0): explore then exploit") &&
        !bodies[2].Contains("PREVIOUS REPLY REJECTED") && !bodies[3].Contains("ADJUSTED") && bodies.All(b=>!b.Contains("finish")),
        "Each request states the last rejection or adjustment; later requests show the trials with their reasons and the plan; finishing is never offered");
    Check(proposing.Summary is { Exit: "insufficient_evaluation_budget", Trials: 3, RemainingEvaluations: 0, InvalidReplies: 1, ClampedProposals: 1, ConfiguratorRequests: 4 },
        $"The search summary records exit, rejected replies and clamped proposals ({proposing.Summary})");
}
// A remainder below the proposed population is spent with a smaller population, as the baselines spend it, instead of
// ending the search as configurator_invalid; the request states the limit.
using (var model=new StubModel((n,body)=>Reply("run",population:900,evaluations:1100)))
{
    var rest=StubAgent(model);
    var runs=await rest.Execute(toy,142,3,1010);
    var bodies=model.Bodies.ToArray();
    Check(runs.Count==3 && runs.Sum(r=>r.EvaluationsUsed)==3030 && runs[2].Configuration.PopulationSize==830 && runs[2].EvaluationsUsed==830 &&
        bodies[2].Contains("PopulationSize at most 830") && !bodies[1].Contains("PopulationSize at most") &&
        rest.Summary is { Exit: "insufficient_evaluation_budget", InvalidReplies: 0, ClampedProposals: 1 },
        $"A remainder below the proposed population is spent with a clamped population ({string.Join(", ", runs.Select(r=>r.EvaluationsUsed))})");
}
// Exact names only: Enum.TryParse would read numbers and comma-separated flag combinations.
Check(AgentSearch.ExactEnum<MutatorType>("combined")==MutatorType.Combined && AgentSearch.ExactEnum<MutatorType>(" Subtree ")==MutatorType.Subtree &&
    AgentSearch.ExactEnum<MutatorType>("Subtree, LocalPerturbation") is null && AgentSearch.ExactEnum<MutatorType>("3") is null &&
    AgentSearch.ExactEnum<MutatorType>(null) is null && AgentSearch.ExactEnum<RunAction>("1") is null && AgentSearch.ExactEnum<RunAction>("stop")==RunAction.Stop,
    "Mutator and controller action are read by exact name only");
{
    var (inside, none)=AgentSearch.ClampToSearchSpace(new AgentSearch.ConfiguratorReply("a","p",500,0.2,8,50,"Combined",3000),MutatorType.Combined,900000);
    var (outside, changes)=AgentSearch.ClampToSearchSpace(new AgentSearch.ConfiguratorReply("a","p",1500,0.7,2,500,"Combined",3000),MutatorType.Combined,900000);
    var (rest, restChanges)=AgentSearch.ClampToSearchSpace(new AgentSearch.ConfiguratorReply("a","p",500,0.2,8,50,"Combined",3000),MutatorType.Combined,300);
    Check(none.Count==0 && inside.PopulationSize==500 && inside.MutationRate==0.2 &&
        outside is { PopulationSize: 1000, MutationRate: 0.5, MaxTreeDepth: 5, MaxTreeLength: 100 } && changes.Count==4 &&
        rest.PopulationSize==300 && restChanges.Single().Contains("PopulationSize 500 -> 300"),
        "Proposals are moved into the search space and the pool's remainder, each change described");
}
// A server that fails is no invalid reply: the job fails, as before.
using (var down=new StubModel((n,body)=>throw new InvalidOperationException("stub down")))
    await Throws<InvalidOperationException>(()=>StubAgent(down).Execute(toy,142,3,1010));
Console.WriteLine("PASS agent guard: unreadable replies are rejected and out-of-range values clamped without failing the job; a remainder is spent; exact enum names; the pool is spent; reasons and plan carry over; search summary; server failures fail the job");

// Context usage: the largest prompt per role, the largest prompt plus reply, and requests that fill 90% of the window.
var contextUse = new HyperOp.Algorithms.Logging.TimingBreakdown { ContextLength = 1000 };
contextUse.RecordContextUsage([
    new("configurator", 0, 1, "response", 750, 150, null, null), new("configurator", 0, 1, "response", 850, 10, null, null),
    new("controller", 0, 1, "response", 400, 50, null, null), new("controller", 0, 1, "error", null, null, null, "down")]);
Check(contextUse.MaxConfiguratorInputTokens == 850 && contextUse.MaxControllerInputTokens == 400 && contextUse.MaxRequestTokens == 900 &&
      contextUse.RequestsNearContextLimit == 1, "Context usage: largest prompt per role, prompt plus reply, requests near the window");
var noContext = new HyperOp.Algorithms.Logging.TimingBreakdown();
noContext.RecordContextUsage([]);
Check(noContext.MaxRequestTokens == 0 && noContext.RequestsNearContextLimit == 0, "Context usage of a job without model requests is zero");
// Fallback rate: checkpoints kept without a decision are not decisions and must not dilute it.
{
    RunControlObservation Obs(string reason, bool fallback) => new(new RunProgress(1, 100, 900, 0.5, 0.1), new RunDecision(RunAction.Continue, reason), 0, fallback);
    var decided = new List<RunControlObservation> { Obs("Controller unavailable", true) };
    decided.AddRange(Enumerable.Range(0, 5).Select(_ => Obs("ok", false)));
    var plain = new HyperOp.Algorithms.Logging.TimingBreakdown();
    plain.RecordControllerDecisions(decided, [new("controller", 0, 1, "response", 10, 1, null, null)]);
    var padded = new HyperOp.Algorithms.Logging.TimingBreakdown();
    padded.RecordControllerDecisions(decided.Concat(Enumerable.Range(0, 120).Select(_ => Obs(AgentSearch.NoDecisionReason, false))),
        [new("controller", 0, 1, "response", 10, 1, null, null), new("configurator", 0, 1, "response", 10, 1, null, null)]);
    Check(plain.ControllerRequests == 6 && plain.ControllerFallbacks == 1 && padded.ControllerRequests == 6 && padded.ControllerFallbacks == 1 &&
        padded.ControllerObservationsWithoutDecision == 120 && padded.ControllerModelRequests == 1,
        "Fallback rate counts decisions only: 120 kept checkpoints without a decision leave 1 of 6");
}
Console.WriteLine("PASS context usage recorded per job");

// A first trial may use the whole pool, which ends the search without a further request; controller answers that are
// not decisions fall back to Continue. Requests state the problem facts but no variable names (the toy's input is "x").
using (var model=new StubModel((n,body)=>Reply("run",population:550,depth:"6",evaluations:3030)))
{
    var eager=StubAgent(model,FeedbackChannels.Progress,checkpoints:100000);
    var eagerRuns=await eager.Execute(toy,142,3,1010);
    var bodies=model.Bodies.ToArray();
    var eagerCalls=eager.LlmCalls;
    Check(eagerRuns.Count==1 && eagerRuns[0].EvaluationsUsed==3030, "The first trial may use the whole shared pool");
    Check(eagerCalls.Count(c=>c.Role=="configurator")==1, "An exhausted pool ends the search without another configurator request");
    Check(eagerRuns[0].ControlObservations!.All(o=>o.UsedFallback), "Controller answers that are not decisions fall back to Continue");
    Check(bodies.Where(b=>b.Contains("FACTS")).All(b=>b.Contains("input variables") && b.Contains("training rows")) &&
        bodies.Where(b=>!b.Contains("FACTS")).All(b=>b.Contains("InputVariables") && !b.Contains("TRIALS")) &&
        bodies.Any(b=>b.Contains("GP algorithm is fixed")) && !bodies.Any(b=>b.Contains("\"x\"") || b.Contains("\\\"x\\\"")),
        "Configurator and controller requests state the problem facts without variable names");
    Check(bodies.All(b=>b.Contains("\"seed\":142") && b.Contains("max_tokens")), "Every model request carries the repetition's sampling seed and an output-token cap");
    Check(bodies.All(b=>b.Contains("response_format")), "Both roles request JSON under a schema");
    Check(eagerRuns.All(r=>r.TrialCpuTimeMs is > 0 && r.TrialCpuTimeMs <= 4 * r.TrialWallTimeMs!.Value + 1000), "Trial CPU time is recorded");
}
Console.WriteLine("PASS agent guard: an exhausted pool ends the search; schema replies, problem facts, seeds and output caps");

// RQ1 controls: the LLM configurator with a rule or random controller. Neither asks the model at checkpoints; the rule
// stops exactly when the best training NMSE improved by less than 1% since the previous checkpoint; the random controller
// stops with the given probability per decision, reproducibly; no decision is requested with less than one checkpoint left.
using (var model=new StubModel((n,body)=>Reply("run",population:100,evaluations:1010)))
{
    var ruled=StubAgent(model,FeedbackChannels.Progress,controller:"rule");
    var runs=await ruled.Execute(toy,142,3,1010);
    var observations=runs.SelectMany(r=>r.ControlObservations!).ToList();
    bool RuleWouldStop(IReadOnlyList<RunControlObservation> o,int i)=>i>0 && o[i-1].Progress.BestTrainingNmse is double before && before>0 &&
        o[i].Progress.BestTrainingNmse is double now && (before-now)/before<RuleControlledSearch.MinRelativeImprovement;
    Check(model.Bodies.All(b=>b.Contains("FACTS")) && observations.Count>0 && observations.All(o=>!o.UsedFallback) &&
        runs.All(r=>r.ControlObservations!.Select((o,i)=>o.Decision.Reason==AgentSearch.NoDecisionReason ||
            (o.Decision.Action==RunAction.Stop)==RuleWouldStop(r.ControlObservations!,i)).All(x=>x)) &&
        observations.Where(o=>o.Decision.Reason!=AgentSearch.NoDecisionReason).All(o=>o.Progress.EvaluationsRemaining>=101) &&
        observations.Where(o=>o.Decision.Reason==AgentSearch.NoDecisionReason).All(o=>o.Progress.EvaluationsRemaining<101 && o.Decision.Action==RunAction.Continue) &&
        ruled.Variant.EndsWith("__controller=rule") &&
        runs.Sum(r=>r.EvaluationsUsed)==3030,
        $"Rule controller: no model requests at checkpoints, stops exactly on <1% improvement, none with less than one checkpoint left ({observations.Count} decisions, {runs.Count} trials)");
}
foreach (var (hazard,expectStops) in new[] { (new[] {1.0},true), (new[] {0.0},false) })
    using (var model=new StubModel((n,body)=>Reply("run",population:100,evaluations:1010)))
    {
        var randomly=StubAgent(model,FeedbackChannels.Progress,controller:"random",hazard:hazard);
        var runs=await randomly.Execute(toy,142,3,1010);
        Check(model.Bodies.All(b=>b.Contains("FACTS")) && runs.Sum(r=>r.EvaluationsUsed)==3030 &&
            (expectStops ? runs.Take(runs.Count-1).All(r=>r.TerminationReason=="controller_stop" && r.ControlObservations!.Count==1)
                         : runs.All(r=>r.TerminationReason=="evaluation_limit" && r.ControlObservations!.All(o=>o.Decision.Action==RunAction.Continue &&
                               (o.Progress.EvaluationsRemaining>=101 || o.Decision.Reason==AgentSearch.NoDecisionReason))) &&
                           // The last checkpoint of a run, with less than one checkpoint left, is kept for the configurator without a decision.
                           runs.Any(r=>r.ControlObservations!.Any(o=>o.Decision.Reason==AgentSearch.NoDecisionReason)) &&
                           randomly.ShownTrials().Any(t=>t.CheckpointCount>0)) &&
            randomly.Variant.EndsWith($"__controller=random__stop-hazard={AgentSearch.FormatHazard(hazard[0])}"),
            $"Random controller with stop probability {hazard[0]}: {runs.Count} trials, {string.Join(",", runs.Select(r=>r.EvaluationsUsed))} evaluations");
    }
{
    using var model=new StubModel((n,body)=>"No JSON.");
    var drawn=StubAgent(model,FeedbackChannels.Progress,controller:"random",hazard:[0.5,0.25]);
    drawn.Seed=142;
    var first=Enumerable.Range(1,40).Select(k=>drawn.RandomStopDecision(3,k).Action).ToArray();
    var again=Enumerable.Range(1,40).Select(k=>drawn.RandomStopDecision(3,k).Action).ToArray();
    Check(first.SequenceEqual(again) && first.Contains(RunAction.Stop) && first.Contains(RunAction.Continue) &&
        drawn.RandomStopDecision(3,1).Reason.Contains("p = 0.5") && drawn.RandomStopDecision(3,7).Reason.Contains("p = 0.25") &&
        StubAgent(model,FeedbackChannels.Progress,controller:"random",hazard:[0.12345]).Variant.EndsWith("__stop-hazard=0.123"),
        "Random stops are reproducible per (seed, trial, decision); the last probability repeats; the job name states the rounded hazard");
    await Throws<ArgumentException>(()=>Task.Run(()=>StubAgent(model,FeedbackChannels.Progress,controller:"random")));
    await Throws<ArgumentException>(()=>Task.Run(()=>StubAgent(model,FeedbackChannels.Progress,controller:"random",hazard:[1.5])));
    await Throws<ArgumentException>(()=>Task.Run(()=>StubAgent(model,FeedbackChannels.None,controller:"rule")));
    await Throws<ArgumentException>(()=>Task.Run(()=>StubAgent(model,FeedbackChannels.Progress,hazard:[0.5])));
    await Throws<ArgumentException>(()=>Task.Run(()=>StubAgent(model,FeedbackChannels.Progress,controller:"oracle")));
}
Console.WriteLine("PASS RQ1 controls: rule and random controllers with the LLM configurator, reproducible random stops, argument checks");

// Text state: numbers to three significant digits in plain decimals; all trials up to the row limit, beyond it the best,
// the most recent and every new best in trial order plus mutator aggregates; channel columns only where they apply.
{
    Check(ConfiguratorState.Number(0.000123456)=="0.000123" && ConfiguratorState.Number(12345.6)=="12346" && ConfiguratorState.Number(0.15,2)=="0.15" &&
        ConfiguratorState.Number(2.0)=="2" && ConfiguratorState.Number(null)=="-" && ConfiguratorState.Number(double.PositiveInfinity)=="inf" &&
        ConfiguratorState.Number(1.5e-12)=="0.0000000000015" && ConfiguratorState.Thousands(3250)=="3.25k" && ConfiguratorState.Thousands(120000)=="120k",
        "Numbers: three significant digits, plain decimals, evaluations in thousands");
    var view=new CheckpointView(new RunProgress(1,1000,1000,0.5,0.1) { Diversity=new DiversityFeedback(0.9,0.4,0,null) },RunAction.Stop,"PhenotypicDiversity=0.4 low");
    var many=Enumerable.Range(0,70).Select(i=>new Experiment { Trial=i, HyperParameters=new AlgorithmParameter { MutatorType=(MutatorType)(i%4) },
        BestFitness=i==3 ? double.PositiveInfinity : (i*37%70)/10.0+0.01, Improvement=0, IsNewBest=i is 0 or 5 or 12, GrantedEvaluations=3000, EvaluationsUsed=3000,
        TerminationReason="controller_stop", ControlObservations=[view], CheckpointCount=1, LastImprovementCheckpoint=1, Reason=$"reason {i}" }).ToList();
    var shown=ConfiguratorState.ShownTrials(many);
    var bestTrials=many.Where(e=>double.IsFinite(e.BestFitness)).OrderBy(e=>e.BestFitness).ThenBy(e=>e.Trial).Take(ConfiguratorState.TableBest).Select(e=>e.Trial);
    Check(ConfiguratorState.ShownTrials(many.Take(ConfiguratorState.TableRows).ToList()).Count==ConfiguratorState.TableRows &&
        shown.Select(e=>e.Trial).SequenceEqual(shown.Select(e=>e.Trial).Order()) && bestTrials.All(t=>shown.Any(e=>e.Trial==t)) &&
        Enumerable.Range(70-ConfiguratorState.TableRecent,ConfiguratorState.TableRecent).All(t=>shown.Any(e=>e.Trial==t)) &&
        new[] {0,5,12}.All(t=>shown.Any(e=>e.Trial==t)) && shown.Count<=ConfiguratorState.TableBest+ConfiguratorState.TableRecent+3,
        $"Row limit: all trials up to {ConfiguratorState.TableRows}, beyond it the best, the most recent and every new best, in trial order ({shown.Count} of 70)");
    string plain=ConfiguratorState.Table(many,FeedbackChannels.None,withReason:true), rich=ConfiguratorState.Table(many,FeedbackChannels.All,withReason:true);
    Check(plain.Split('\n')[0]=="trial|pop|mut|depth|len|mutator|granted|used|end|valNMSE|rank|your reason" &&
        plain.Contains("trials not shown") && Enum.GetValues<MutatorType>().All(m=>plain.Contains($"\n{m}|")) && !plain.Contains("trainNMSE") &&
        rich.Contains("trainNMSE first>last") && rich.Contains("0.5>0.5 imp@1/1") && rich.Contains("0.9>0.9/0.4") &&
        !rich.Contains("PhenotypicDiversity=0.4 low") && !rich.Contains("E-"),
        "Table: final-result columns everywhere, channel cells only in conditions with checkpoints, no controller reasons");
    Check(!ConfiguratorState.Legend(FeedbackChannels.None,true).Contains("trainNMSE") && ConfiguratorState.Legend(FeedbackChannels.Diversity|FeedbackChannels.Progress,true).Contains("phenotypic") &&
        !ConfiguratorState.Legend(FeedbackChannels.Diversity|FeedbackChannels.Progress,true).Contains("medLen"), "The legend names only the columns of the view");
    // The controller's comparison at the same evaluation count uses the rule baseline's references and median.
    ResultDTO Run(int population, int used, params double[] bestByGeneration) => new() {
        Configuration=new AlgorithmParameter(), ConfigurationIndex=0, GPSeed=1, BestFitness=1, MeanPopulationFitness=1, WorstPopulationFitness=1,
        ExecutionTimeMs=0, EvaluationsUsed=used, EvaluationsLimit=used, GenerationsCompleted=bestByGeneration.Length-1, PopulationSize=population,
        BestIndividualsSnapshot=[], QualityCurve=bestByGeneration.Select((q,g)=>new ResultDTO.GenerationQualitySnapshot { GenerationNumber=g, BestQuality=q,
            MedianQuality=q, WorstQuality=q }).ToList() };
    var compared=new ControllerProbe("stop");
    Check(compared.CompareAtSameEvaluations(new RunProgress(2,3000,7000,0.5,0.1)) is null, "No comparison before any run has completed");
    // At 3000 evaluations: 0.4 (population 1000, generation 2), 0.2 (population 500, generation 5), 0.95 (generation 0 at 2000);
    // the fourth run stopped at 2000 evaluations and is not a reference.
    compared.Results.AddRange([Run(1000,30000,0.9,0.6,0.4,0.1), Run(500,30000,0.8,0.7,0.5,0.4,0.3,0.2,0.1), Run(2000,30000,0.95,0.9,0.5), Run(1000,2000,0.01,0.01)]);
    var comparison=compared.CompareAtSameEvaluations(new RunProgress(2,3000,7000,0.5,0.1))!;
    var ruleReferences=RuleControlledSearch.TrainingReferences(compared.Results,3000);
    Check(comparison.Evaluations==3000 && comparison.EarlierRuns==3 && ruleReferences.SequenceEqual([0.2,0.4,0.95]) && comparison.Best==0.2 &&
        comparison.Median==RuleControlledSearch.Median(ruleReferences) && comparison.Median==0.4 && comparison.ThisRunRank==3 && comparison.RunsCompared==4,
        $"Comparison at the same evaluation count matches the rule baseline's references ({comparison})");

    // The controller request: p7 sends no comparison and nothing of earlier trials but the budget; the standard prompt
    // sends the comparison.
    var standard=new AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
        Provider=LlmTornado.Code.LLmProviders.Custom, ModelName="fixture", OllamaBaseUrl="http://localhost:11434", Feedback=FeedbackChannels.All,
        ControllerPrompt="standard" });
    standard.Results.AddRange(compared.Results);
    string p7Request=WithProblem(compared).ControllerRequest(new AlgorithmParameter(), new RunProgress(2,3000,7000,0.5,0.1), []);
    string standardRequest=WithProblem(standard).ControllerRequest(new AlgorithmParameter(), new RunProgress(2,3000,7000,0.5,0.1), []);
    Check(compared.ControllerPrompt==ControllerPrompts.Study && !p7Request.Contains("AtSameEvaluations") && !p7Request.Contains("0.95") && !p7Request.Contains("Median") &&
        p7Request.Contains("SearchContext") && p7Request.Contains("Progress") &&
        standardRequest.Contains("AtSameEvaluations") && standardRequest.Contains("ThisRunRank"),
        "Controller request: the study prompt omits the comparison with earlier runs, the standard prompt sends it");
    // p8: neutral budget facts and the renamed diversity field; p7 keeps the fields of its pilot.
    var diverse=new RunProgress(2,3000,7000,0.5,0.1) { Diversity=new DiversityFeedback(1,0.5,0,0.2) };
    var p7Agent=WithProblem(new AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
        Provider=LlmTornado.Code.LLmProviders.Custom, ModelName="fixture", OllamaBaseUrl="http://localhost:11434", Feedback=FeedbackChannels.All, ControllerPrompt="p7" }));
    string p8Diverse=compared.ControllerRequest(new AlgorithmParameter(), diverse, []);
    string p7Diverse=p7Agent.ControllerRequest(new AlgorithmParameter(), diverse, []);
    Check(p8Diverse.Contains("\"DistinctExpressionShare\":1") && !p8Diverse.Contains("UniqueStructures") && p8Diverse.Contains("SharedPoolAfterThisRunsAllowance") &&
        p8Diverse.Contains("\"ThisRunAllowance\":10000") && !p8Diverse.Contains("IfStoppedNow") && p8Diverse.Contains("PhenotypicDiversity") &&
        p7Diverse.Contains("UniqueStructures") && p7Diverse.Contains("IfStoppedNow") && p7Agent.Variant.EndsWith("__cprompt=p7"),
        "Controller request p8: neutral budget facts and DistinctExpressionShare; p7 keeps its pilot fields");

    // Budget fields of the controller, for a pool of 10 x 30000 evaluations with the four runs above completed.
    typeof(AgentSearch).GetField("_totalEvaluations",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(compared,300000);
    typeof(AgentSearch).GetField("_numConfigurations",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(compared,10);
    typeof(AgentSearch).GetField("_evaluationsPerConfiguration",BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(compared,30000);
    var budget=compared.ControllerBudget(new RunProgress(5,12000,38000,0.5,0.1));
    int spent=compared.Results.Sum(r=>r.EvaluationsUsed);
    Check(budget.ThisRunEvaluationsUsed==12000 && budget.ThisRunAllowanceRemaining==38000 &&
        budget.EvaluationsForOtherConfigurationsIfStoppedNow==300000-spent-12000 &&
        budget.EvaluationsForOtherConfigurationsIfContinuedToAllowance==300000-spent-12000-38000 &&
        budget.ReferenceRunsIfStoppedNow==Math.Round((300000.0-spent-12000)/30000,1) &&
        budget.ReferenceRunsIfContinuedToAllowance==Math.Round((300000.0-spent-12000-38000)/30000,1) &&
        // The pool, not a trial cap, ends a search: stopping every run at the first checkpoint still spends the budget.
        HyperParameterOptimizationAlgorithm.MaximumSharedPoolTrials(30,30000,3000)==301 && 300L*3000>=30L*30000,
        $"Controller budget fields: unused allowance returns to other configurations when stopped ({budget})");
    var whole=compared.ControllerBudget(new RunProgress(5,12000,300000,0.5,0.1));
    Check(whole.EvaluationsForOtherConfigurationsIfContinuedToAllowance==0, "A run granted the rest of the pool leaves nothing for others if it continues");
}
Console.WriteLine("PASS text state: numbers, row limit, table columns, legend; controller comparison at the same evaluation count; controller budget fields");

// Feedback conditions: names, the unobserved "final" mode, and distinct job names per condition.
Check(FeedbackChannelNames.Parse("final")==FeedbackChannels.None && FeedbackChannelNames.Parse("all")==FeedbackChannels.All &&
    FeedbackChannelNames.Parse("size,curve")==(FeedbackChannels.Progress|FeedbackChannels.Curve|FeedbackChannels.Size) &&
    FeedbackChannelNames.Format(FeedbackChannelNames.Parse("generalization+curve"))=="progress+curve+generalization" &&
    FeedbackChannelNames.Format(FeedbackChannels.All)=="progress+curve+diversity+size+generalization", "Feedback condition names");
await Throws<ArgumentException>(()=>Task.Run(()=>FeedbackChannelNames.Parse("noise")));
var unobserved=new ControllerProbe("final") { FeynmanInstance=toy, Seed=142 };
foreach (var (name,value) in new[] { ("_numConfigurations",2),("_totalEvaluations",4000),("_evaluationsPerConfiguration",2000) })
    typeof(AgentSearch).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(unobserved,value);
var finalExperiment=await unobserved.RunSymbolicRegression(agentParameter,2500);
Check(finalExperiment.EvaluationsUsed==2500 && finalExperiment.ControlObservations.Count==0 && finalExperiment.CheckpointCount==0 &&
    unobserved.ControllerCalls==0 && unobserved.Results[0].TerminationReason=="evaluation_limit",
    "Final-result mode: the run uses its requested evaluations with no controller and no checkpoints");
Check(unobserved.Variant=="feedback=final" && new ControllerProbe("stop").Variant=="feedback=progress+curve+diversity+size+generalization__checkpoints-per-E=100000" &&
    new RuleControlledSearch(7).Variant=="checkpoints-per-E=7", "Method variants name the feedback condition and interval");
{
    string longVariant="feedback=progress+curve+diversity+size+generalization__checkpoints-per-E=10__checkpoints=hidden__placebo__compare=off__first-decision=cp2__cprompt=standard";
    string longName=ExperimentResultStore.JobName("AgentSearch","II.11.27",10,30,30000,5,"qwen3.5:9b-q4_K_M",longVariant,0.1,200);
    string shortName=ExperimentResultStore.JobName("AgentSearch","II.11.27",10,30,30000,5,"qwen3.5:9b-q4_K_M","feedback=final");
    Check(longName.Length<=ExperimentResultStore.MaxJobNameLength && longName.Contains("__variant="+ExperimentResultStore.VariantHash(longVariant)+"__instance=") &&
        longName.Length+".result.json".Length+".xxxxxxxx.tmp".Length<=255 && longName.Length+"__attempt=xxxxxxxx.protocol.jsonl".Length<=255 &&
        shortName.Contains("__feedback=final__") && ExperimentResultStore.VariantHash(longVariant)!=ExperimentResultStore.VariantHash(longVariant+"x"),
        $"Long job names carry a hash of the variant and stay below the 255-character file name limit ({longName.Length})");
}
Check(ExperimentResultStore.JobName("AgentSearch","I.6.20",1,10,1000,5,"m:1","feedback=final")!=
    ExperimentResultStore.JobName("AgentSearch","I.6.20",1,10,1000,5,"m:1","feedback=progress__checkpoints-per-E=10"), "Feedback conditions have distinct job names");
// Controller-only condition: the controller acts, the configurator sees only final results.
var hidden=new ControllerProbe("stop", hideCheckpoints: true) { FeynmanInstance=toy, Seed=142 };
foreach (var (name,value) in new[] { ("_numConfigurations",2),("_totalEvaluations",4000),("_evaluationsPerConfiguration",2000) })
    typeof(AgentSearch).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(hidden,value);
var hiddenExperiment=await hidden.RunSymbolicRegression(agentParameter,2500);
Check(hidden.ControllerCalls==1 && hiddenExperiment.TerminationReason=="controller_stop" && hiddenExperiment.EvaluationsUsed==1100 &&
    hiddenExperiment.ControlObservations.Count==0 && hiddenExperiment.CheckpointCount==0 &&
    hidden.ShownTrials().All(e=>e.ControlObservations.Count==0) && hidden.Results[0].ControlObservations!.Count==1 &&
    WithProblem(hidden).ConfiguratorRequest(null,0,null) is var hiddenRequest && !hiddenRequest.Contains("Deterministic check") && !hiddenRequest.Contains("trainNMSE") &&
    WithProblem(agent).ConfiguratorRequest(null,0,null) is var richRequest && richRequest.Contains("Deterministic check") && richRequest.Contains("trainNMSE"),
    "Hidden checkpoints: the controller stops the run, the configurator sees only the final result (no channel columns, no controller reasons), the result keeps the checkpoint");
Check(hidden.Variant=="feedback=progress+curve+diversity+size+generalization__checkpoints-per-E=100000__checkpoints=hidden", "Hidden checkpoints are part of the job name");
Check(new AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter { Provider=LlmTornado.Code.LLmProviders.Custom,
        ModelName="fixture", OllamaBaseUrl="http://localhost:11434", ControllerComparison=false, FirstControllerCheckpoint=2 }).Variant==
    "feedback=progress__checkpoints-per-E=10__compare=off__first-decision=cp2", "Pilot switches are part of the job name");
Check(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter().ControllerPrompt==ControllerPrompts.Study && ControllerPrompts.Study=="p8" &&
    !ControllerPrompts.ShowsComparison(ControllerPrompts.Study) && ControllerPrompts.UsesNeutralFacts("p8") && !ControllerPrompts.UsesNeutralFacts("p7"),
    "The study's controller prompt is p8, without the comparison with earlier runs");
{
    string p7Text=ControllerPrompts.Instructions("p7",FeedbackChannels.All,10), p8Text=ControllerPrompts.Instructions("p8",FeedbackChannels.All,10),
        p9Text=ControllerPrompts.Instructions("p9",FeedbackChannels.All,10);
    Check(p8Text.Contains("DistinctExpressionShare") && !p8Text.Contains("UniqueStructures") && !p8Text.Contains("if this run stops now") && p8Text.Contains("improve late") &&
        p7Text.Contains("UniqueStructures") && p7Text.Contains("if this run stops now") && !p9Text.Contains("improve late") &&
        p9Text==p8Text.Replace("; runs with large populations or large trees can start slowly and improve late.", "."),
        "p8 names the renamed field and neutral budget; p9 is p8 without the late-improvement sentence; p7 unchanged");
}
foreach (var promptName in ControllerPrompts.Names.Where(n=>n!="standard"))
{
    var text=ControllerPrompts.Instructions(promptName,FeedbackChannels.All,10);
    Check(text.Contains("Continue") && text.Contains("Stop") && text.Contains("Return JSON") &&
        text.Contains("AtSameEvaluations")==(promptName is not ("p1" or "p7" or "p8" or "p9")) && !ControllerPrompts.Instructions(promptName,FeedbackChannels.Progress,10).Contains("PhenotypicDiversity"),
        $"Controller prompt {promptName}: both actions, JSON reply, comparison only where shown, channels only where present");
}
Check(ControllerPrompts.Names.Where(n=>n is "p1" or "p2" or "p4" or "p5").Select(n=>ControllerPrompts.Instructions(n,FeedbackChannels.All,10).Length).Zip(
        ControllerPrompts.Names.Where(n=>n is "p2" or "p4" or "p5").Select(n=>ControllerPrompts.Instructions(n,FeedbackChannels.All,10).Length)).All(p=>p.First<p.Second),
    "Controller prompts p1, p2, p4, p5 grow in length");
// Channel placebo: progress stays real; the channel sections come from the first run's true checkpoints (frozen at its
// first checkpoint while no earlier run exists); the variant name marks it.
var placeboAgent=new ControllerProbe("continue", placebo: true) { FeynmanInstance=toy, Seed=142 };
foreach (var (name,value) in new[] { ("_numConfigurations",2),("_totalEvaluations",6000),("_evaluationsPerConfiguration",3000) })
    typeof(AgentSearch).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(placeboAgent,value);
var placeboParameter=new AlgorithmParameter { PopulationSize=150, MaxTreeDepth=6, MaxTreeLength=20 };
await placeboAgent.RunSymbolicRegression(placeboParameter,3000);
var donors=((List<RunProgress>)typeof(AgentSearch).GetField("_placeboDonors",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(placeboAgent)!).ToList();
var firstTrial=placeboAgent.Received.Where(r=>r.Trial==0).Select(r=>r.Progress).ToList();
await placeboAgent.RunSymbolicRegression(placeboParameter with { MutationRate=0.4 },3000);
var secondTrial=placeboAgent.Received.Where(r=>r.Trial==1).Select(r=>r.Progress).ToList();
Check(firstTrial.Count>=2 && donors.Count==firstTrial.Count && firstTrial.All(p=>ReferenceEquals(p.Diversity,donors[0].Diversity) &&
        p.Curve!.BestTrainingNmse.SequenceEqual(donors[0].Curve!.BestTrainingNmse)) &&
    firstTrial.Zip(donors).All(x=>x.First.EvaluationsUsed==x.Second.EvaluationsUsed && x.First.BestTrainingNmse==x.Second.BestTrainingNmse) &&
    secondTrial.Count>=2 && secondTrial.All(p=>donors.Any(d=>ReferenceEquals(d.Diversity,p.Diversity) && d.Curve!.BestTrainingNmse.SequenceEqual(p.Curve!.BestTrainingNmse) &&
        d.Generalization!.ValidationNmseOfBest==p.Generalization!.ValidationNmseOfBest)) &&
    // Donor values made consistent with the current run: the curve ends at its generation, sizes within its limits, the
    // gap against its own best training NMSE.
    firstTrial.Concat(secondTrial).All(p=>p.Curve!.Generations[^1]==p.Generation && p.Size!.MaxLength<=placeboParameter.MaxTreeLength &&
        p.Size.MedianDepth<=placeboParameter.MaxTreeDepth && p.Generalization!.Gap==p.Generalization.ValidationNmseOfBest-p.BestTrainingNmse) &&
    // The donor choice depends only on (seed, trial, checkpoint), not on the process.
    ReferenceEquals(placeboAgent.WithPlaceboChannels(secondTrial[0],firstTrial[0],1,0).Diversity, placeboAgent.WithPlaceboChannels(secondTrial[1],firstTrial[0],1,0).Diversity) &&
    secondTrial.Zip(secondTrial.Skip(1)).All(x=>x.First.EvaluationsUsed<x.Second.EvaluationsUsed) &&
    placeboAgent.Variant.EndsWith("__placebo"),
    $"Placebo: real progress, channel values from another run's checkpoints ({firstTrial.Count}+{secondTrial.Count} decisions)");
await Throws<ArgumentException>(()=>Task.Run(()=>new AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
    Provider=LlmTornado.Code.LLmProviders.Custom, ModelName="fixture", OllamaBaseUrl="http://localhost:11434", Feedback=FeedbackChannels.Progress, ChannelPlacebo=true })));
Console.WriteLine("PASS feedback conditions: channel names, final-result mode, controller-only mode, channel placebo, all channels on real checkpoints, distinct job names");

// Feedback measures against hand-computed values.
double[] baseOutput = [1, 2, 4, 8, 3, 5];
Check(RunFeedbackBuilder.PhenotypicDiversity([baseOutput, baseOutput.Select(v => 2 * v + 5).ToArray(), baseOutput.Select(v => -v).ToArray()]) < 1e-12,
    "Phenotypic diversity ignores offset, scale and sign, as linear scaling does");
Check(Math.Abs(RunFeedbackBuilder.PhenotypicDiversity([[1, -1, 1, -1], [1, 1, -1, -1]]) - 1) < 1e-12, "Uncorrelated outputs have diversity one");
double[] partner = [1, 2, 3, 4, 5, 6];
double r = (baseOutput.Zip(partner).Sum(p => (p.First - baseOutput.Average()) * (p.Second - partner.Average())))
    / Math.Sqrt(baseOutput.Sum(v => Math.Pow(v - baseOutput.Average(), 2)) * partner.Sum(v => Math.Pow(v - partner.Average(), 2)));
Check(Math.Abs(RunFeedbackBuilder.PhenotypicDiversity([baseOutput, partner]) - (1 - r * r)) < 1e-12, "Diversity of two outputs is one minus their squared correlation");
// Scale invariance also at extreme magnitudes: tiny and huge outputs of the same shape are perfectly correlated, a
// large offset does not make a varying output constant, and huge values do not overflow.
Check(RunFeedbackBuilder.PhenotypicDiversity([baseOutput.Select(v => v * 1e-14).ToArray(), baseOutput]) < 1e-12 &&
      RunFeedbackBuilder.PhenotypicDiversity([baseOutput.Select(v => v * 1e200).ToArray(), baseOutput]) < 1e-12 &&
      RunFeedbackBuilder.PhenotypicDiversity([baseOutput.Select(v => v + 1e6).ToArray(), baseOutput]) < 1e-9,
    "Phenotypic diversity is invariant to extreme scales and offsets");
Check(RunFeedbackBuilder.PhenotypicDiversity([[3, 3, 3], [7, 7, 7]]) == 0 && RunFeedbackBuilder.PhenotypicDiversity([[3, 3, 3], [1, 2, 3]]) == 1 &&
    RunFeedbackBuilder.PhenotypicDiversity([[1, 2, 3]]) == 0, "Constant outputs and single individuals");
var sampled = RunFeedbackBuilder.SampleEvenly(Enumerable.Range(11, 25).ToArray(), 10);
Check(sampled.Length == 10 && sampled[0] == 11 && sampled[^1] == 35 && sampled.Zip(sampled.Skip(1)).All(p => p.First < p.Second) &&
    RunFeedbackBuilder.SampleEvenly([1, 2, 3], 10).Length == 3, "Curve keeps at most ten evenly spaced points including the latest");
Check(RunFeedbackBuilder.Spread([1, 2, 3, 4, 5]) == (4 - 2) / 3.0 && RunFeedbackBuilder.Spread([1, 2, 3]) is null && RunFeedbackBuilder.Spread([0, 0, 0, 0]) is null,
    "Training NMSE spread is IQR over median");
var size = RunFeedbackBuilder.Size([5, 20, 20, 7], [3, 6, 4, 6], 20, 6, 10);
Check(size.MedianLength == 13.5 && size.MaxLength == 20 && size.MedianDepth == 5 && size.AtLengthLimit == 0.5 && size.AtDepthLimit == 0.5 &&
    size.MedianLengthChange == 3.5 && RunFeedbackBuilder.Size([4], [2], 20, 6, null).MedianLengthChange is null, "Size and limit fractions");
Console.WriteLine("PASS feedback measures: phenotypic diversity invariances, curve sampling, spread and size");

// Constant fitting: Levenberg-Marquardt on the evolvable constants, for the linearly scaled objective, before evaluation.
{
    var expData = new ExponentialDescriptor();
    var space = new ExpressionTreeSearchSpace(30, 8, Symbols.DefaultOperations, expData.AllowedInputVariables);
    var problem = new SymbolicRegressionProblem(expData.PreparedData.Training, Metrics.NMSE, space, useLinearScaling: true);
    var constant = (EvolvableConstantSymbol)space.ResolveUniqueSymbol(s => s is EvolvableConstantSymbol)!;
    var variable = (VariableSymbol)space.ResolveUniqueSymbol(s => s is VariableSymbol)!;
    ExpressionTree Exp(double k) => new(new UnaryExpressionNode(Symbols.Exponential,
        new BinaryExpressionNode(Symbols.Multiplication, new NumericConstantExpressionNode(constant, k), new VariableExpressionNode(variable, "x"))));
    double Nmse(ExpressionTree tree) => problem.Evaluate(tree, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(0))[0];
    double[] Constants(ExpressionTree tree) => tree.TraversePreOrder().OfType<NumericConstantExpressionNode>().Select(n => n.Value).ToArray();
    var refiner = new ScaledConstantFittingRefiner();
    ExpressionTree Refine(ExpressionTree tree) => refiner.RefineCandidate(tree, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(1), space, problem);

    var start = Exp(0.5);
    var fitted = Refine(start);
    // A few iterations per call; offspring inherit fitted constants (Lamarckian), so later generations continue the fit.
    Check(Nmse(fitted) < Nmse(start) && Constants(fitted).Single() is > 0.5 and < 0.7,
        $"One constant fit improves a constant that linear scaling cannot absorb (k {Constants(fitted).Single():G6}, NMSE {Nmse(start):G3} -> {Nmse(fitted):G3})");
    var iterated = fitted;
    for (int i = 0; i < 30 && Nmse(iterated) >= 1e-10; i++)
    {
        var next = Refine(iterated);
        Check(!(Nmse(next) > Nmse(iterated)), "Repeated constant fits never get worse");
        iterated = next;
    }
    Check(Math.Abs(Constants(iterated).Single() - 0.7) < 1e-4 && Nmse(iterated) < 1e-10,
        $"Repeated fits converge to the true constant (k {Constants(iterated).Single():G8}, NMSE {Nmse(iterated):G3})");
    Check(fitted.Length == start.Length && fitted.Depth == start.Depth && Constants(start).Single() == 0.5, "Fitting keeps the structure and leaves the input tree unchanged");
    Check(Constants(Refine(start)).Single() == Constants(fitted).Single(), "Constant fitting is deterministic");
    {
        // The fitter's sums must not depend on where the arrays lie in memory (TensorPrimitives.Dot does): the same
        // 250 values at every offset within a 64-byte line give bit-identical dot products.
        var values = new System.Random(7);
        double[] xs = Enumerable.Range(0, 250).Select(_ => values.NextDouble() * 2e3 - 1e3).ToArray();
        double[] ys = Enumerable.Range(0, 250).Select(_ => values.NextDouble() * 2 - 1).ToArray();
        var dots = Enumerable.Range(0, 8).Select(offset =>
        {
            var bx = new double[260]; var by = new double[260];
            xs.CopyTo(bx, offset); ys.CopyTo(by, offset);
            return BitConverter.DoubleToInt64Bits(ScaledConstantFittingRefiner.Dot(bx.AsSpan(offset, 250), by.AsSpan(offset, 250)));
        }).Distinct().Count();
        Check(dots == 1, $"Fitter dot products do not depend on memory alignment ({dots} distinct results over 8 offsets)");
        // The fitter's own linear scaling: HeuristicLib's results and conventions, in a fixed summation order.
        double[] ys10 = ys.Take(10).ToArray(), constantX = Enumerable.Repeat(3.0, 10).ToArray(), nanX = xs.Take(9).Append(double.NaN).ToArray();
        bool Same((double Slope, double Intercept) own, LinearScalingParameters lib) =>
            Math.Abs(own.Slope - lib.Slope) <= 1e-12 * Math.Abs(lib.Slope) && Math.Abs(own.Intercept - lib.Intercept) <= 1e-12 * Math.Abs(lib.Intercept);
        Check(Same(ScaledConstantFittingRefiner.FitLine(xs, ys), LinearScaling.Fit(xs, ys)) &&
              Same(ScaledConstantFittingRefiner.FitLine(constantX, ys10), LinearScaling.Fit(constantX, ys10)) &&
              Same(ScaledConstantFittingRefiner.FitLine(nanX, ys10), LinearScaling.Fit(nanX, ys10)),
            "The fitter's linear scaling matches HeuristicLib's, including constant and non-finite inputs");
        var lines = Enumerable.Range(0, 8).Select(offset =>
        {
            var bx = new double[260]; var by = new double[260];
            xs.CopyTo(bx, offset); ys.CopyTo(by, offset);
            var (s, i) = ScaledConstantFittingRefiner.FitLine(bx.AsSpan(offset, 250), by.AsSpan(offset, 250));
            return (BitConverter.DoubleToInt64Bits(s), BitConverter.DoubleToInt64Bits(i));
        }).Distinct().Count();
        Check(lines == 1, $"The fitter's linear scaling does not depend on memory alignment ({lines} distinct results over 8 offsets)");
    }
    var noConstants = new ExpressionTree(new VariableExpressionNode(variable, "x"));
    Check(ReferenceEquals(Refine(noConstants), noConstants), "Trees without evolvable constants are returned as they are");
    var undefined = new ExpressionTree(new UnaryExpressionNode(Symbols.Logarithm,
        new BinaryExpressionNode(Symbols.Multiplication, new NumericConstantExpressionNode(constant, -1), new VariableExpressionNode(variable, "x"))));
    Check(ReferenceEquals(Refine(undefined), undefined), "Trees with non-finite training predictions are returned as they are");
    foreach (var k in new[] { 0.5, 2.0, 7.3, 30.0 })
    {
        var refined = Refine(Exp(k));
        Check(Constants(refined).All(double.IsFinite) && !(Nmse(refined) > Nmse(Exp(k))), $"Fitted constants are finite and never worse: exp({k}x)");
    }

    // a + b·exp(50x) on x in [1, 5] made MathNet's covariance SVD loop forever before the fit was normalized.
    var feynman = new Feynman44(42);
    var feynmanSpace = new ExpressionTreeSearchSpace(30, 8, Symbols.DefaultOperations, feynman.AllowedInputVariables);
    var feynmanProblem = new SymbolicRegressionProblem(feynman.PreparedData.Training, Metrics.NMSE, feynmanSpace, useLinearScaling: true);
    var feynmanConstant = (EvolvableConstantSymbol)feynmanSpace.ResolveUniqueSymbol(s => s is EvolvableConstantSymbol)!;
    var feynmanVariable = (VariableSymbol)feynmanSpace.ResolveUniqueSymbol(s => s is VariableSymbol)!;
    foreach (var k in new[] { 50.0, 60.0 })
    {
        var steep = new ExpressionTree(new UnaryExpressionNode(Symbols.Exponential, new BinaryExpressionNode(Symbols.Multiplication,
            new NumericConstantExpressionNode(feynmanConstant, k), new VariableExpressionNode(feynmanVariable, "omega"))));
        var fit = Task.Run(() => refiner.RefineCandidate(steep, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(1), feynmanSpace, feynmanProblem));
        Check(fit.Wait(TimeSpan.FromSeconds(20)) && Constants(fit.Result).All(double.IsFinite), $"Badly scaled fit terminates: exp({k}·omega)");
    }

    // The GP candidate on which HeuristicLib's fitter (MathNet's covariance SVD) never returned, in a development run on
    // I.13.12: inner constants from 1e-9 to exp(24.7), well beyond what scaling the output can balance.
    var gravity = new Feynman14(42);
    var gravitySpace = new ExpressionTreeSearchSpace(90, 18, Symbols.DefaultOperations, gravity.AllowedInputVariables);
    var gravityProblem = new SymbolicRegressionProblem(gravity.PreparedData.Training, Metrics.NMSE, gravitySpace, useLinearScaling: true);
    var gravityConstant = (EvolvableConstantSymbol)gravitySpace.ResolveUniqueSymbol(s => s is EvolvableConstantSymbol)!;
    var gravityVariable = (VariableSymbol)gravitySpace.ResolveUniqueSymbol(s => s is VariableSymbol)!;
    ExpressionNode V(string name) => new VariableExpressionNode(gravityVariable, name);
    ExpressionNode C(double value) => new NumericConstantExpressionNode(gravityConstant, value);
    ExpressionNode Add(ExpressionNode l, ExpressionNode r) => new BinaryExpressionNode(Symbols.Addition, l, r);
    ExpressionNode Sub(ExpressionNode l, ExpressionNode r) => new BinaryExpressionNode(Symbols.Subtraction, l, r);
    ExpressionNode Mul(ExpressionNode l, ExpressionNode r) => new BinaryExpressionNode(Symbols.Multiplication, l, r);
    ExpressionNode Div(ExpressionNode l, ExpressionNode r) => new BinaryExpressionNode(Symbols.Division, l, r);
    ExpressionNode ExpN(ExpressionNode a) => new UnaryExpressionNode(Symbols.Exponential, a);
    ExpressionNode SqrtN(ExpressionNode a) => new UnaryExpressionNode(Symbols.SquareRoot, a);
    ExpressionNode SquareN(ExpressionNode a) => new UnaryExpressionNode(Symbols.Square, a);
    const double c1 = 24.661417685263753, c2 = 4.3460358157364549, c3 = -1.0140708423559498E-09;
    var hung = new ExpressionTree(Div(V("G"), Div(Mul(ExpN(Div(V("r1"), Sub(
            Mul(Sub(ExpN(C(c1)), ExpN(C(c1))), Div(Mul(
                Div(SqrtN(SquareN(V("r1"))), Mul(Mul(ExpN(V("G")), Sub(V("m2"), Div(Div(C(c2), V("m1")), V("m2")))), C(c3))),
                Div(Mul(ExpN(Div(V("m2"), Sub(Sub(V("r2"), V("r2")), Sub(V("r2"), Div(V("r1"), V("G")))))), Div(V("r1"), V("m1"))), V("m2"))),
                C(c2))),
            Add(Mul(Mul(Div(SqrtN(V("G")), V("r1")), Sub(ExpN(V("G")),
                Div(SqrtN(SquareN(ExpN(C(c1)))), Mul(Mul(ExpN(V("G")), Sub(V("r1"), Div(C(c2), V("m2")))), C(c3))))), V("r2")),
                Div(V("r1"), V("m2")))))),
            Div(V("r1"), V("m1"))), Div(V("m2"), Div(V("r2"), Sub(V("r1"), V("r2")))))));
    var sampleRefiner = new ScaledConstantFittingRefiner { FittingData = gravity.PreparedData.FittingSample(ConstantFittingSettings.DefaultRows) };
    var hungFit = Task.Run(() => sampleRefiner.RefineCandidate(hung, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(1), gravitySpace, gravityProblem));
    Check(hungFit.Wait(TimeSpan.FromSeconds(20)) && Constants(hungFit.Result).All(double.IsFinite) && Constants(hung).Length == 8,
        "The GP candidate that hung HeuristicLib's fitter is fitted in bounded time");

    // Random trees on the row sample: scaled error on the fitted rows never rises, training predictions stay finite,
    // constants stay finite, and the result does not depend on the generator passed in.
    var samplingProblem = new SymbolicRegressionProblem(sampleRefiner.FittingData!, Metrics.NMSE, gravitySpace, useLinearScaling: true);
    var treeRandom = HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(7);
    ExpressionNode Grow(int depth)
    {
        var symbol = depth <= 1 || treeRandom.NextDouble() < 0.25 ? gravitySpace.SelectSymbol(0, treeRandom) : gravitySpace.SelectSymbol(1, 2, treeRandom);
        int arity = symbol is OperationSymbol ? (gravitySpace.GetSymbols(1).Contains(symbol) ? 1 : 2) : 0;
        return symbol.CreateNode(treeRandom, [.. Enumerable.Range(0, arity).Select(_ => Grow(depth - 1))]);
    }
    int changedTrees = 0;
    for (int i = 0; i < 400; i++)
    {
        var tree = new ExpressionTree(Grow(2 + (int)(treeRandom.NextDouble() * 7)));
        double before = samplingProblem.Evaluate(tree, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(0))[0];
        if (!double.IsFinite(before)) continue;
        var refined = sampleRefiner.RefineCandidate(tree, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(1), gravitySpace, gravityProblem);
        var again = sampleRefiner.RefineCandidate(tree, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(99), gravitySpace, gravityProblem);
        if (ReferenceEquals(refined, tree)) continue;
        changedTrees++;
        double after = samplingProblem.Evaluate(refined, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(0))[0];
        Check(after <= before * (1 + 1e-9) + 1e-15, $"Fitting never raises the scaled error on its rows ({before:G6} -> {after:G6}): {tree.ToInfixString()}");
        Check(double.IsFinite(gravityProblem.Evaluate(refined, HEAL.HeuristicLib.Random.RandomNumberGenerator.Create(0))[0]) &&
            Constants(refined).All(double.IsFinite), $"Fitted trees stay finite on all training rows: {tree.ToInfixString()}");
        Check(refined.ToInfixString() == again.ToInfixString() && refined.Length == tree.Length, "Fitting is deterministic and keeps the structure");
    }
    Check(changedTrees > 50, $"Random trees are fitted ({changedTrees} of 400 changed)");

    // Target noise: training and validation targets get Gaussian noise of 10% of the target's standard deviation from a
    // fixed stream; inputs and the clean test targets are unchanged; noisy test targets give the secondary score.
    var clean = new Feynman14(42).PreparedData;
    var noisyDescriptor = new Feynman14(42) { TargetNoise = 0.1 };
    var noisy = noisyDescriptor.PreparedData;
    var noisyAgain = new Feynman14(42) { TargetNoise = 0.1 }.PreparedData;
    double[] Targets(HEAL.HeuristicLib.MachineLearning.RegressionData data) => data.Target.Values.ToArray();
    var trainingNoise = Targets(noisy.Training).Zip(Targets(clean.Training), (a, b) => a - b).ToArray();
    double cleanSd = Math.Sqrt(Targets(clean.Training).Select(v => v - Targets(clean.Training).Average()).Sum(d => d * d) / clean.Training.RowCount);
    double noiseSd = Math.Sqrt(trainingNoise.Sum(d => d * d) / trainingNoise.Length);
    Check(trainingNoise.All(d => d != 0) && Math.Abs(noiseSd / cleanSd - 0.1) < 0.01 && Math.Abs(trainingNoise.Average()) < 0.05 * noiseSd &&
        Targets(noisy.Training).SequenceEqual(Targets(noisyAgain.Training)) &&
        noisy.Training.Inputs.Get<double>("m1").Values.ToArray().SequenceEqual(clean.Training.Inputs.Get<double>("m1").Values.ToArray()) &&
        noisy.TargetNoise == 0.1 && clean.TargetNoise == 0,
        $"Training targets carry deterministic noise of 10% of the target's standard deviation (measured {noiseSd / cleanSd:P2})");
    var probeModel = new SymbolicRegressor(new ExpressionTree(new VariableExpressionNode(gravityVariable, "m1")), "prediction").FitLinearScaling(noisy.Training);
    Check(noisy.ValidationFitness(probeModel) != clean.ValidationFitness(probeModel) && noisy.TestFitness(probeModel) == clean.TestFitness(probeModel) &&
        noisy.NoisyTestFitness(probeModel) is double noisyTest && noisyTest != noisy.TestFitness(probeModel) && clean.NoisyTestFitness(probeModel) is null,
        "Validation is noisy, test is scored against clean targets, and noisy test targets give a second score");
    await Throws<InvalidOperationException>(() => Task.Run(() => noisyDescriptor.TargetNoise = 0.2));
    // Small-data variant: 200 evenly spaced training and validation rows with the same noise per row; test unchanged.
    var small = new Feynman14(42) { TargetNoise = 0.1, SampleRows = 200 }.PreparedData;
    var keptRows = Enumerable.Range(0, 200).Select(i => (int)((long)i * noisy.TrainingRows.Count / 200)).ToArray();
    var validationKept = Enumerable.Range(0, 200).Select(i => noisy.ValidationRows[(int)((long)i * noisy.ValidationRows.Count / 200)]).ToArray();
    // The 200 validation rows are split in alternating order: 100 select, 100 feed the generalization channel.
    Check(small.ValidationRows.SequenceEqual(validationKept.Where((_, i) => i % 2 == 0)) && small.MonitoringRows.SequenceEqual(validationKept.Where((_, i) => i % 2 == 1)) &&
        !small.ValidationRows.Intersect(small.MonitoringRows).Any() && noisy.MonitoringRows.SequenceEqual(noisy.ValidationRows) &&
        small.MonitoringFitness(probeModel) != small.ValidationFitness(probeModel) && noisy.MonitoringFitness(probeModel) == noisy.ValidationFitness(probeModel),
        "Small data: selection and the generalization feedback use disjoint halves of the validation rows; otherwise the same rows");
    Check(small.Training.RowCount == 200 && small.TestRows.SequenceEqual(noisy.TestRows) &&
        small.TrainingRows.SequenceEqual(keptRows.Select(i => noisy.TrainingRows[i])) &&
        Targets(small.Training).SequenceEqual(keptRows.Select(i => Targets(noisy.Training)[i])) &&
        small.TestFitness(probeModel) == clean.TestFitness(probeModel),
        "The small-data variant keeps 200 evenly spaced training rows with their noise; test is unchanged");
    Check(ExperimentResultStore.JobName("RandomSearch", "I.6.20", 1, 30, 30000, 5, targetNoise: 0.1, sampleRows: 200) ==
        "RandomSearch__instance=I.6.20__noise=0.1__rows=200__rep=1__T=30__E=30000__K=5", "Small-data jobs have their own names");
    Check(ExperimentResultStore.JobName("RandomSearch", "I.6.20", 1, 30, 30000, 5, targetNoise: 0.1) ==
        "RandomSearch__instance=I.6.20__noise=0.1__rep=1__T=30__E=30000__K=5" &&
        ExperimentResultStore.JobName("RandomSearch", "I.6.20", 1, 30, 30000, 5) == "RandomSearch__instance=I.6.20__rep=1__T=30__E=30000__K=5",
        "Noisy jobs have their own names; noiseless names are unchanged");

    // The runner's stall watchdog reads the time of the latest protocol record.
    string protocolPath = Path.Combine(Path.GetTempPath(), $"hyperop-check-{Guid.NewGuid():N}.protocol.jsonl");
    using (var protocolLog = new HyperOp.Algorithms.Logging.JsonlProtocolLog(protocolPath))
    {
        var opened = protocolLog.LastWriteUtc;
        await Task.Delay(20);
        protocolLog.Write("check");
        Check(protocolLog.LastWriteUtc > opened && DateTime.UtcNow - protocolLog.LastWriteUtc < TimeSpan.FromMinutes(1), "Protocol records update the stall watchdog's timestamp");
    }
    File.Delete(protocolPath);

    // Provenance: a cached result is reused only with the same fitting implementation and settings.
    var here = HyperOp.Algorithms.Logging.RunProvenance.Capture(new ConstantFittingSettings());
    Check(here.FitterVersion == ScaledConstantFittingRefiner.Version && here.SameFitting(here with { GitCommit = "other" }) &&
        !here.SameFitting(here with { ConstantFitting = new ConstantFittingSettings { Iterations = 10 } }) &&
        !here.SameFitting(here with { FitterVersion = "old" }) && !here.SameFitting(null),
        $"Provenance compares fitting implementation and settings (commit {here.GitCommit ?? "unknown"}, dirty {here.GitDirty}, AVX2 {here.Vector256Accelerated}, AVX-512 {here.Vector512Accelerated})");
    // Code fingerprint: by content, not history. The same files give the same value whatever their line endings and commit;
    // a changed source file changes it; files in excluded directories do not; cache reuse follows it.
    {
        var tree = Path.Combine(Path.GetTempPath(), "fingerprint-" + Guid.NewGuid().ToString("N"));
        string Write(string relative, string text) { var path = Path.Combine(tree, relative); Directory.CreateDirectory(Path.GetDirectoryName(path)!); File.WriteAllText(path, text); return path; }
        Write("HyperOp.Algorithms/A.cs", "class A {}\r\nclass B {}\r\n"); Write("HyperOp.Experiments/P.cs", "x"); Write("docker/c.yml", "a: 1");
        string first = HyperOp.Algorithms.Logging.CodeFingerprint.Compute(tree);
        Write("HyperOp.Algorithms/A.cs", "class A {}\nclass B {}\n");
        string lf = HyperOp.Algorithms.Logging.CodeFingerprint.Compute(tree);
        Write("HyperOp.Algorithms/bin/Release/x.dll", "binary"); Write("HyperOp.Experiments/Properties/launchSettings.json", "{}"); Write("NOTES.md", "notes");
        string excluded = HyperOp.Algorithms.Logging.CodeFingerprint.Compute(tree);
        Write("HyperOp.Algorithms/A.cs", "class A {}\nclass C {}\n");
        string changed = HyperOp.Algorithms.Logging.CodeFingerprint.Compute(tree);
        Directory.Delete(tree, true);
        var a = here with { GitCommit = "private" }; a = a with { CodeFingerprint = first };
        var b = here with { GitCommit = "public", GitDirty = true }; b = b with { CodeFingerprint = first };
        Check(first == lf && lf == excluded && changed != first && first.Length == 64 &&
            a.CacheConflict(b) is null && a.CacheConflict(b with { CodeFingerprint = changed }) is not null &&
            here.CodeFingerprint is { Length: 64 } && HyperOp.Algorithms.Logging.CodeFingerprint.FindRoot() is not null,
            $"Code fingerprint by content: line endings, commits and excluded files do not matter, a source change does (this code {here.CodeFingerprint?[..12]})");
    }


    // A GP run that fails after it started ends the agent's job instead of counting as a rejected proposal.
    var failing = new FailingTrialProbe { FeynmanInstance = expData, Seed = 142 };
    foreach (var (name, value) in new[] { ("_numConfigurations", 2), ("_totalEvaluations", 8000), ("_evaluationsPerConfiguration", 4000) })
        typeof(AgentSearch).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(failing, value);
    await Throws<InvalidOperationException>(() => failing.RunSymbolicRegression(new AlgorithmParameter { PopulationSize = 200, MaxTreeDepth = 6, MaxTreeLength = 20 }, 4000));
    Check(failing.Results.Count == 0 && typeof(AgentSearch).GetField("_trialFailure", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(failing) is InvalidOperationException,
        "A failed GP trial is recorded as a trial failure that ends the job");

    var parameter = new AlgorithmParameter { PopulationSize = 200, MaxTreeDepth = 6, MaxTreeLength = 20 };
    var run = await new TrialProbe().Run(expData, parameter, 42, 4000);
    Check(run.ConstantFittingCandidates > 0 && run.ConstantFittingMs is > 0 && run.ConstantFittingMs <= run.ExecutionTimeMs && run.EvaluationsUsed == 4000,
        "GP runs fit constants within GP time and outside the evaluation budget");
    // Trial points: at most MaximumIterations per fitted candidate, counted per GP run.
    Check(run.ConstantFittingTrialPoints is > 0 && run.ConstantFittingTrialPoints <= (long)ConstantFittingSettings.DefaultIterations * run.ConstantFittingCandidates!.Value,
        $"Constant-fitting trial points are counted per run ({run.ConstantFittingTrialPoints} for {run.ConstantFittingCandidates} candidates)");

    // Fitting a share of candidates: the choice comes from the run's seeded generator.
    var settings = HyperParameterOptimizationAlgorithm.ConstantFitting;
    try
    {
        HyperParameterOptimizationAlgorithm.ConstantFitting = settings with { Probability = 0.25 };
        var partial = await new TrialProbe().Run(expData, parameter, 42, 4000);
        var partialAgain = await new TrialProbe().Run(expData, parameter, 42, 4000);
        Check(partial.ConstantFittingCandidates > 0 && partial.ConstantFittingCandidates < 0.5 * run.ConstantFittingCandidates &&
            partial.EvaluationsUsed == 4000 && partial.ConstantFitting!.Probability == 0.25,
            $"A fitting probability of 0.25 fits about a quarter of the candidates ({partial.ConstantFittingCandidates} of {run.ConstantFittingCandidates})");
        Check(partial.SelectedExpression == partialAgain.SelectedExpression && partial.BestFitness == partialAgain.BestFitness &&
            partial.ConstantFittingCandidates == partialAgain.ConstantFittingCandidates, "Partial constant fitting is deterministic");
    }
    finally { HyperParameterOptimizationAlgorithm.ConstantFitting = settings; }
}
Console.WriteLine("PASS constant fitting: nonlinear recovery, structure, determinism, no-op cases, finiteness, badly scaled fits, the hung GP candidate, random trees, GP integration, partial fitting, stall timestamp, provenance, trial points, failed trials, target noise");

static void Check(bool condition,string message) { if(!condition) throw new Exception(message); }
static JsonSerializerOptions JsonOptions()=>new() { NumberHandling=System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals };
static async Task Throws<T>(Func<Task> action) where T:Exception
{
    try { await action(); } catch(T) { return; }
    throw new Exception($"Expected {typeof(T).Name}");
}

// Sets the problem facts that Execute would set, so a probe can render its configurator request.
static AgentSearch WithProblem(AgentSearch search)
{
    typeof(AgentSearch).GetField("_problem", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(search, new AgentSearch.ProblemFacts(1, 100, 50));
    return search;
}

class LinearDescriptor:RegressionDescriptor
{
    public override string Name=>"linear-check";
    public override string Description=>"Small deterministic execution check";
    public override string TargetVariable=>"y";
    public override string[] VariableNames=>["x","y"];
    public override string[] AllowedInputVariables=>["x"];
    public override List<List<double>> GenerateValues()=>[Enumerable.Range(1,30).Select(i=>i/10.0).ToList(),Enumerable.Range(1,30).Select(i=>2+3*i/10.0).ToList()];
}
class ExponentialDescriptor:RegressionDescriptor
{
    public override string Name=>"exponential-check";
    public override string Description=>"y = exp(0.7 x): a constant that linear scaling cannot absorb";
    public override string TargetVariable=>"y";
    public override string[] VariableNames=>["x","y"];
    public override string[] AllowedInputVariables=>["x"];
    public override List<List<double>> GenerateValues()=>[Enumerable.Range(0,60).Select(i=>1+i/15.0).ToList(),Enumerable.Range(0,60).Select(i=>Math.Exp(0.7*(1+i/15.0))).ToList()];
}
class TrialProbe:HyperParameterOptimizationAlgorithm
{
    public Task<ResultDTO> Run(RegressionDescriptor d,AlgorithmParameter p,int seed,int limit)=>EvaluateConfiguration(d,p,0,seed,limit);
    public override Task<List<ResultDTO>> Execute(RegressionDescriptor d,int s,int n,int e)=>throw new NotSupportedException();
    public static Task<ResultDTO> Observation(AlgorithmParameter p,int id,int seed,int limit)=>Task.FromResult(new ResultDTO {
        Configuration=p,ConfigurationIndex=id,GPSeed=seed,BestFitness=1,MeanPopulationFitness=1,WorstPopulationFitness=1,
        ExecutionTimeMs=0,EvaluationsUsed=limit,EvaluationsLimit=limit,GenerationsCompleted=0,PopulationSize=p.PopulationSize,BestIndividualsSnapshot=[] });
}
class FailingTrialProbe() : AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
    Provider = LlmTornado.Code.LLmProviders.Custom, ModelName = "fixture", OllamaBaseUrl = "http://localhost:11434", Feedback = FeedbackChannels.None })
{
    protected override Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor d, AlgorithmParameter p, int id, int seed, int limit) =>
        throw new InvalidOperationException("GP produced no finite training objective.");
}
class HillProbe:HillClimbingSearch
{
    public int Calls;
    protected override Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor d,AlgorithmParameter p,int id,int seed,int limit)
    { Calls++; return TrialProbe.Observation(p,id,seed,limit); }
}
class GridProbe:GridSearch
{
    protected override Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor d,AlgorithmParameter p,int id,int seed,int limit)=>TrialProbe.Observation(p,id,seed,limit);
}
class RandomProbe:RandomSearch
{
    protected override Task<ResultDTO> EvaluateConfiguration(RegressionDescriptor d,AlgorithmParameter p,int id,int seed,int limit)=>TrialProbe.Observation(p,id,seed,limit);
}

class ControllerProbe(string behavior, bool hideCheckpoints = false, bool placebo = false) : AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
    ConfiguratorSeesCheckpoints = !hideCheckpoints, ChannelPlacebo = placebo,
    Provider = LlmTornado.Code.LLmProviders.Custom, ModelName = "fixture", OllamaBaseUrl = "http://localhost:11434",
    DecisionTimeout = TimeSpan.FromMilliseconds(100), Checkpoints = 100000,
    Feedback = behavior == "final" ? FeedbackChannels.None : FeedbackChannels.All
})
{
    public bool BudgetWasCurrent = true;
    public int ControllerCalls;
    // What the controller received, per call (with the trial index).
    public readonly List<(int Trial, RunProgress Progress)> Received = [];
    protected override async Task<RunDecision> RequestRunDecisionAsync(AlgorithmParameter configuration, RunProgress progress,
        IReadOnlyList<RunControlObservation> previousObservations, CancellationToken token)
    {
        // Completed trials plus the active run's checkpoint count.
        ControllerCalls++;
        Received.Add((Results.Count, progress));
        BudgetWasCurrent &= GetRemainingEvaluations().UsedEvaluations == Results.Sum(r => r.EvaluationsUsed) + progress.EvaluationsUsed;
        await Task.Yield();
        if (behavior == "invalid") return new RunDecision((RunAction)99, "Bad action");
        if (behavior == "failure") throw new InvalidOperationException("Simulated inference failure");
        if (behavior == "timeout") await Task.Delay(Timeout.Infinite, token);
        return behavior == "stop" ? new RunDecision(RunAction.Stop, "Deterministic check")
            : new RunDecision(RunAction.Continue, "Deterministic check");
    }
}

// A local stand-in for the model server: answers each chat request (numbered from 1) with the given assistant content,
// or with HTTP 500 when the content function throws.
sealed class StubModel : IDisposable
{
    private readonly System.Net.HttpListener _listener = new();
    public string Url { get; }
    public int Requests;
    public readonly System.Collections.Concurrent.ConcurrentQueue<string> Bodies = new();
    public StubModel(Func<int, string, string> content)
    {
        var probe = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        probe.Start(); int port = ((System.Net.IPEndPoint)probe.LocalEndpoint).Port; probe.Stop();
        Url = $"http://localhost:{port}";
        _listener.Prefixes.Add(Url + "/");
        _listener.Start();
        Task.Run(async () => {
            while (_listener.IsListening)
            {
                System.Net.HttpListenerContext context;
                try { context = await _listener.GetContextAsync(); } catch { return; }
                int n = Interlocked.Increment(ref Requests);
                string body; using (var reader = new StreamReader(context.Request.InputStream)) body = await reader.ReadToEndAsync();
                Bodies.Enqueue(body);
                string? text = null;
                try { text = content(n, body); } catch { }
                context.Response.ContentType = "application/json";
                if (text is null) context.Response.StatusCode = 500;
                else await context.Response.OutputStream.WriteAsync(System.Text.Encoding.UTF8.GetBytes(
                    "{\"id\":\"s" + n + "\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"stub\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" +
                    JsonSerializer.Serialize(text) + "},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":3,\"total_tokens\":14}}"));
                context.Response.Close();
            }
        });
    }
    public void Dispose() => _listener.Stop();
}
