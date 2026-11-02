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
foreach (var (population, allowance) in new[] { (10,100), (55,101), (91,1000), (100,100), (100,101) })
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
    var results=await method.Execute(toy,142,3,101);
    Check(results.Count>0 && results.Count<=3 && results.All(r => r.EvaluationsUsed==101), method.GetType().Name+" trial/work cap");
    Check(results.Select(r => r.ConfigurationIndex).SequenceEqual(Enumerable.Range(0,results.Count)), "Trial sequence includes every neighbor");
    Check(results.Select(r => r.GPSeed!.Value).SequenceEqual(Enumerable.Range(0,results.Count).Select(i=>HyperParameterOptimizationAlgorithm.TrialSeed(142,i))), "Common local-search seed schedule");
    await Throws<ArgumentOutOfRangeException>(() => method.Execute(toy,142,0,100));
    await Throws<ArgumentOutOfRangeException>(() => method.Execute(toy,142,1,99));
    Console.WriteLine($"PASS {method.GetType().Name}: {results.Count} recorded GP trials");
}

// Freeze a recommendation using validation only, then assess its fixed configuration.
var nominator = new RandomSearch();
var tuning = await nominator.Execute(toy,142,3,101);
var nominated = nominator.NominateConfiguration(tuning);
Check(nominated.BestFitness == tuning.Min(r=>r.BestFitness), "Validation-based nomination");
var native = new IRaceSearch();
typeof(IRaceSearch).GetField("recommendedConfigurationId",BindingFlags.Instance|BindingFlags.NonPublic)!
    .SetValue(native,tuning[^1].ConfigurationIndex);
Check(ReferenceEquals(native.NominateConfiguration(tuning),tuning[^1]), "Native irace recommendation overrides a luckiest-trial shortcut");
var beforeAssessment = JsonSerializer.Serialize(tuning,JsonOptions());
var assessed = await nominator.AssessConfiguration(toy,nominated.Configuration,tuning,142,3,101);
Check(assessed.Count==3 && assessed.All(r=>r.Configuration==nominated.Configuration && r.EvaluationsUsed==101 && r.TestFitness is not null), "Fixed configuration, separate assessment budget, and test scores");
Check(assessed.Select(r=>r.GPSeed).Distinct().Count()==3 && !assessed.Any(r=>tuning.Any(t=>t.GPSeed==r.GPSeed)), "Fresh unique assessment seeds");
Check(JsonSerializer.Serialize(tuning,JsonOptions())==beforeAssessment, "Assessment does not change tuning history or its selection scores");
var repeatedAssessment = await nominator.AssessConfiguration(toy,nominated.Configuration,tuning,142,3,101);
Check(assessed.Zip(repeatedAssessment).All(pair=>pair.First.GPSeed==pair.Second.GPSeed && pair.First.SelectedExpression==pair.Second.SelectedExpression && pair.First.TestFitness==pair.Second.TestFitness), "Deterministic assessment seed stream and models");
int? originalSeed=tuning[0].GPSeed;
tuning[0].GPSeed=assessed[0].GPSeed;
var collisionCheck = await nominator.AssessConfiguration(toy,nominated.Configuration,tuning,142,1,101);
Check(!tuning.Any(t=>t.GPSeed==collisionCheck[0].GPSeed), "Assessment skips a seed already used in tuning");
tuning[0].GPSeed=originalSeed;
var combined=new AggregatedResultDTO {
    AlgorithmName="RandomSearch",InstanceName="linear-check",Repetition=1,Seed=142,
    NumConfigurations=3,EvaluationsPerConfiguration=101,TotalConfigurationsEvaluated=tuning.Count,
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
await Throws<ArgumentOutOfRangeException>(()=>nominator.AssessConfiguration(toy,nominated.Configuration,tuning,142,0,101));
Console.WriteLine("PASS holdout nomination and fresh-seed configuration assessment");

// Exercise search control independently from GP quality, including non-improving neighbors,
// a cap in the middle of a neighborhood, and exhaustion of the fixed grid.
var hill=new HillProbe();
var hillResults=await hill.Execute(toy,42,5,101);
Check(hillResults.Count==5 && hill.Calls==5, "All unsuccessful neighbor trials retained and charged");
Check((await hill.Execute(toy,42,1,100)).Count==1, "One-trial hill cap");
var grid=new GridProbe();
var gridResults=await grid.Execute(toy,42,5001,101);
Check(gridResults.Count==5000 && gridResults.All(r => r.EvaluationsLimit==101), "Grid exhaustion and per-trial allocation");
Check(gridResults.Select(r => r.Configuration).Distinct().Count()==5000, "Grid has no duplicate points");
var shortGrid=await grid.Execute(toy,42,7,101);
var hillRestarts=await new HillProbe().Execute(toy,42,60,101);
Check(hillRestarts.Count==60 && hillRestarts.Select(r=>r.Configuration).Distinct().Count()==60,
    "Hill climbing restarts at local optima until the trial cap, without repeating configurations");
Check(hillRestarts[0].Configuration!=(await new HillProbe().Execute(toy,142,1,101))[0].Configuration, "Hill climbing start depends on the HPO seed");
var seedsRep1=Enumerable.Range(0,1000).Select(i=>HyperParameterOptimizationAlgorithm.TrialSeed(142,i)).ToHashSet();
var seedsRep2=Enumerable.Range(0,1000).Select(i=>HyperParameterOptimizationAlgorithm.TrialSeed(242,i)).ToHashSet();
Check(seedsRep1.Count==1000 && seedsRep2.Count==1000 && !seedsRep1.Overlaps(seedsRep2) && seedsRep1.All(s=>s>=0),
    "Trial seeds are non-negative and not shared between consecutive repetitions");
Check(shortGrid.Select(r=>r.Configuration).SequenceEqual(gridResults.Take(7).Select(r=>r.Configuration)), "Stable grid prefix");
var random=new RandomProbe();
var randomResults=await random.Execute(toy,42,2000,101);
foreach (var select in new Func<AlgorithmParameter,int>[] { p=>p.PopulationSize,p=>p.MaxTreeDepth,p=>p.MaxTreeLength })
    Check(randomResults.Any(r=>select(r.Configuration)==100), "Random integer upper endpoint reachable");
Console.WriteLine("PASS search-control regression checks");

// Real GP checkpoint execution with deterministic controllers, without a model service.
foreach (var behavior in new[] { "stop", "continue", "failure", "timeout", "invalid" })
{
    var controlled = new ControllerProbe(behavior) { FeynmanInstance = toy, Seed = 142 };
    foreach (var (name, value) in new[] { ("_numConfigurations", 1), ("_totalEvaluations", 301) })
        typeof(AgentSearch).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(controlled, value);
    await controlled.RunSymbolicRegression(new AlgorithmParameter { PopulationSize = 55, MaxTreeDepth = 6, MaxTreeLength = 20 }, 301);
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
    Check(record.EvaluationsUsed == (behavior == "stop" ? 110 : 301), "Stop early or resume to hard cap");
    Check(record.TerminationReason == (behavior == "stop" ? "controller_stop" : "evaluation_limit"), "Explicit termination reason");
    Check(record.ControlObservations.All(o => o.UsedFallback == (behavior is "failure" or "timeout" or "invalid")), "Controller failure/timeout falls back to continue");
    if (behavior == "stop")
    {
        record.EvaluateTest(toy.PreparedData);
        int assessmentEvaluations = controlled.AssessmentEvaluations(record, 301);
        Check(assessmentEvaluations == 110, "Agent nominee is assessed at the run length it used");
        var assessments = await controlled.AssessConfiguration(toy, record.Configuration, controlled.Results, 142, 1, assessmentEvaluations);
        Check(assessments.Single().EvaluationsUsed == 110 && assessments.Single().GenerationsCompleted == record.GenerationsCompleted,
            "Assessment replays the nominee's fidelity without a controller");
        var stored = new AggregatedResultDTO {
            AlgorithmName = "AgentSearch", ModelName = "fixture", InstanceName = "linear-check", Repetition = 1, Seed = 142,
            NumConfigurations = 1, EvaluationsPerConfiguration = 301, TotalConfigurationsEvaluated = 1,
            AssessmentEvaluationsPerRun = assessmentEvaluations,
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
            stored.ConfigurationResults[0].ControlObservations = [];
            File.WriteAllText(path, JsonSerializer.Serialize(stored, JsonOptions()));
            await Throws<InvalidDataException>(() => Task.Run(() => ExperimentResultStore.Load(path)));
        }
        finally { folder.Delete(true); }
    }
    if (behavior != "stop")
    {
        var uninterrupted = await new TrialProbe().Run(toy, record.Configuration, HyperParameterOptimizationAlgorithm.TrialSeed(142, 0), 301);
        Check(record.SelectedExpression == uninterrupted.SelectedExpression && record.BestFitness == uninterrupted.BestFitness,
            "Checkpoint continuation preserves population/RNG trajectory");
    }
}
Console.WriteLine("PASS asynchronous run control: stop, continue, failures, timeouts, budgets and GP trajectory");

// Evaluate the real agent tool without contacting a model; initialize its run context directly.
// The deterministic controller stops each run at its first checkpoint (generation 1). The first call may use the
// whole pool; the controller's stop leaves 92 evaluations, which the second call requests.
var agent=new ControllerProbe("stop") { FeynmanInstance=toy, Seed=142 };
foreach (var (name,value) in new[] { ("_numConfigurations",2),("_totalEvaluations",202) })
    typeof(AgentSearch).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(agent,value);
var agentParameter=new AlgorithmParameter { PopulationSize=55,MaxTreeDepth=6,MaxTreeLength=20 };
await Throws<System.ComponentModel.DataAnnotations.ValidationException>(() => agent.RunSymbolicRegression(agentParameter with { PopulationSize=9 },202));
await Throws<System.ComponentModel.DataAnnotations.ValidationException>(() => agent.RunSymbolicRegression(agentParameter with { MutationRate=double.NaN },202));
await Throws<ArgumentException>(() => agent.RunSymbolicRegression(agentParameter with { MutatorType=(MutatorType)99 },202));
await Throws<ArgumentOutOfRangeException>(() => agent.RunSymbolicRegression(agentParameter,54));
Check(agent.Results.Count==0 && agent.GetRemainingEvaluations().UsedEvaluations==0, "Shared validation rejects invalid agent proposals before spending trials");
// The second request asks for more than the 92 remaining evaluations and receives the remainder.
await Task.WhenAll(agent.RunSymbolicRegression(agentParameter,202),agent.RunSymbolicRegression(agentParameter,500));
await Throws<InvalidOperationException>(()=>agent.RunSymbolicRegression(agentParameter,92));
// First trial: stopped after 110 of the 202 shared evaluations; the repeat receives the remaining 92.
Check(agent.Results.Count==2 && agent.GetRemainingEvaluations().UsedEvaluations==202 && agent.GetRemainingEvaluations().RemainingEvaluations==0, "Concurrent agent tools honor trial/work caps");
Check(agent.Results.Select(r=>r.EvaluationsUsed).SequenceEqual(new[] {110,92}), "Shared pool: controller stop frees budget for the next trial");
Check(agent.Results.Select(r=>r.GPSeed).SequenceEqual(new int?[] {HyperParameterOptimizationAlgorithm.TrialSeed(142,0),HyperParameterOptimizationAlgorithm.TrialSeed(142,1)}) && agent.GetExperimentHistory().Count==2,
    "Repeated agent configuration is a new charged trial with a new seed, not a cached result");
var agentNominee=agent.NominateConfiguration(agent.Results);
var assessedAgent=await agent.AssessConfiguration(toy,agentNominee.Configuration,agent.Results,142,2,agent.AssessmentEvaluations(agentNominee,101));
Check(assessedAgent.Count==2 && assessedAgent.All(r=>r.EvaluationsUsed==agentNominee.EvaluationsUsed) && agent.Results.Count==2 &&
    agent.GetExperimentHistory().Count==2 && agent.GetRemainingEvaluations().UsedEvaluations==202,
    "Post-tuning assessment uses the nominee's fidelity and neither consumes tuning budget nor feeds back into history");
var json=JsonSerializer.Serialize(agent.Results,new JsonSerializerOptions { NumberHandling=System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals });
Check(JsonSerializer.Deserialize<List<ResultDTO>>(json,new JsonSerializerOptions { NumberHandling=System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals })!
    .Select(r=>(r.GPSeed,r.EvaluationsUsed)).SequenceEqual(new (int?,int)[] {(HyperParameterOptimizationAlgorithm.TrialSeed(142,0),110),(HyperParameterOptimizationAlgorithm.TrialSeed(142,1),92)}), "Result count/seed JSON round-trip");
Console.WriteLine($"PASS agent tools and JSON; {checkedTargets:N0} benchmark targets verified");

// Baselines for the agent: the untuned default and the rule-controlled shared-pool search.
var defaults=new DefaultConfigurationSearch();
var defaultRuns=await defaults.Execute(toy,142,3,101);
Check(defaultRuns.Count==3 && defaultRuns.All(r=>r.Configuration==new AlgorithmParameter() && r.EvaluationsUsed==101) &&
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
var ruleRuns=await rules.Execute(toy,142,4,301);
Check(ruleRuns.Count>=1 && ruleRuns.Count<=4 && ruleRuns.Sum(r=>r.EvaluationsUsed)<=4*301, "Rule baseline honors the shared pool and trial cap");
Check(ruleRuns.All(r=>r.ControlObservations!.All(o=>o.Progress.Generation%rules.CheckpointInterval==0)) &&
    ruleRuns.Where(r=>r.TerminationReason=="controller_stop").All(r=>r.ControlObservations![^1].Decision.Action==RunAction.Stop),
    "Rule baseline checkpoints every five generations and records its stop decisions");
Check(ruleRuns.Select(r=>r.GPSeed).SequenceEqual(Enumerable.Range(0,ruleRuns.Count).Select(i=>(int?)HyperParameterOptimizationAlgorithm.TrialSeed(142,i))),
    "Rule baseline uses the common trial seeds");
var ruleNominee=rules.NominateConfiguration(ruleRuns);
Check(rules.AssessmentEvaluations(ruleNominee,301)==ruleNominee.EvaluationsUsed, "Rule baseline nominee is assessed at its run length");
Console.WriteLine($"PASS agent baselines: default configuration and rule-controlled shared pool ({ruleRuns.Count} trials, {ruleRuns.Sum(r=>r.EvaluationsUsed)} evaluations)");

// A configurator that answers without calling tools must not loop forever; the stub also exercises HTTP timing.
var stubListener=new System.Net.HttpListener();
var portProbe=new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback,0);
portProbe.Start();
int stubPort=((System.Net.IPEndPoint)portProbe.LocalEndpoint).Port;
portProbe.Stop();
stubListener.Prefixes.Add($"http://localhost:{stubPort}/");
stubListener.Start();
int stubRequests=0;
var stub=Task.Run(async()=>{
    while(stubListener.IsListening)
    {
        System.Net.HttpListenerContext context;
        try { context=await stubListener.GetContextAsync(); } catch { return; }
        Interlocked.Increment(ref stubRequests);
        var body=System.Text.Encoding.UTF8.GetBytes("{\"id\":\"x\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"stub\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"No tool needed.\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":3,\"total_tokens\":14}}");
        context.Response.ContentType="application/json";
        await context.Response.OutputStream.WriteAsync(body);
        context.Response.Close();
    }
});
var idle=new AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
    Provider=LlmTornado.Code.LLmProviders.Custom, ModelName="stub", OllamaBaseUrl=$"http://localhost:{stubPort}", Streaming=false });
await Throws<InvalidOperationException>(()=>idle.Execute(toy,142,3,101));
stubListener.Stop();
var idleCalls=idle.LlmCalls;
Check(stubRequests==AgentSearch.MaxIdleConversations && idleCalls.Count==AgentSearch.MaxIdleConversations,
    "Idle configurator conversations end the search");
Check(idleCalls.All(c=>c.Role=="configurator" && c.Outcome=="response" && c.InputTokens==11 && c.OutputTokens==3 && c.DurationMs>=0),
    "Model requests are timed at the HTTP layer with token usage");
Console.WriteLine("PASS agent guard: idle configurator conversations end the search; HTTP request timing and tokens recorded");

// A configurator that keeps proposing invalid trials: each conversation ends after the rejection limit, a new one can
// start after the cancellation, and the idle limit ends the search.
var rejectPort=new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback,0);
rejectPort.Start(); int rejectingPort=((System.Net.IPEndPoint)rejectPort.LocalEndpoint).Port; rejectPort.Stop();
var rejectListener=new System.Net.HttpListener();
rejectListener.Prefixes.Add($"http://localhost:{rejectingPort}/");
rejectListener.Start();
int rejectRequests=0;
var rejectStub=Task.Run(async()=>{
    while(rejectListener.IsListening)
    {
        System.Net.HttpListenerContext context;
        try { context=await rejectListener.GetContextAsync(); } catch { return; }
        int n=Interlocked.Increment(ref rejectRequests);
        string arguments="{\\\"populationSize\\\":55,\\\"mutationRate\\\":0.2,\\\"maxTreeDepth\\\":6,\\\"maxTreeLength\\\":20,\\\"mutatorType\\\":\\\"Combined\\\",\\\"evaluations\\\":10}";
        var body=System.Text.Encoding.UTF8.GetBytes("{\"id\":\"r"+n+"\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"stub\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call"+n+"\",\"type\":\"function\",\"function\":{\"name\":\"RunSymbolicRegression\",\"arguments\":\""+arguments+"\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":5,\"total_tokens\":25}}");
        context.Response.ContentType="application/json";
        await context.Response.OutputStream.WriteAsync(body);
        context.Response.Close();
    }
});
var rejecting=new AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
    Provider=LlmTornado.Code.LLmProviders.Custom, ModelName="stub", OllamaBaseUrl=$"http://localhost:{rejectingPort}", Streaming=false });
await Throws<InvalidOperationException>(()=>rejecting.Execute(toy,142,3,101));
rejectListener.Stop();
Check(rejectRequests==AgentSearch.MaxIdleConversations*AgentSearch.MaxConsecutiveRejections,
    $"Rejected proposals end each conversation after the limit, and new conversations start after it ({rejectRequests} requests)");
Console.WriteLine("PASS agent guard: repeated rejected proposals end the conversation; the idle limit ends the search");

// A configurator that keeps proposing trials after the pool is exhausted: the tool ends the conversation.
var probe2=new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback,0);
probe2.Start(); int toolPort=((System.Net.IPEndPoint)probe2.LocalEndpoint).Port; probe2.Stop();
var toolListener=new System.Net.HttpListener();
toolListener.Prefixes.Add($"http://localhost:{toolPort}/");
toolListener.Start();
int toolRequests=0;
var toolBodies=new System.Collections.Concurrent.ConcurrentBag<string>();
var toolStub=Task.Run(async()=>{
    while(toolListener.IsListening)
    {
        System.Net.HttpListenerContext context;
        try { context=await toolListener.GetContextAsync(); } catch { return; }
        int n=Interlocked.Increment(ref toolRequests);
        using (var reader=new StreamReader(context.Request.InputStream)) toolBodies.Add(await reader.ReadToEndAsync());
        string arguments="{\\\"populationSize\\\":55,\\\"mutationRate\\\":0.2,\\\"maxTreeDepth\\\":6,\\\"maxTreeLength\\\":20,\\\"mutatorType\\\":\\\"Combined\\\",\\\"evaluations\\\":303}";
        var body=System.Text.Encoding.UTF8.GetBytes("{\"id\":\"x"+n+"\",\"object\":\"chat.completion\",\"created\":0,\"model\":\"stub\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call"+n+"\",\"type\":\"function\",\"function\":{\"name\":\"RunSymbolicRegression\",\"arguments\":\""+arguments+"\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":5,\"total_tokens\":25}}");
        context.Response.ContentType="application/json";
        await context.Response.OutputStream.WriteAsync(body);
        context.Response.Close();
    }
});
var eager=new AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
    Provider=LlmTornado.Code.LLmProviders.Custom, ModelName="stub", OllamaBaseUrl=$"http://localhost:{toolPort}", Streaming=false,
    CheckpointInterval=1 });
var eagerRuns=await eager.Execute(toy,142,3,101);
toolListener.Stop();
var eagerCalls=eager.LlmCalls;
Check(eagerRuns.Count==1 && eagerRuns[0].EvaluationsUsed==303, "The first trial may use the whole shared pool");
Check(eagerCalls.Count(c=>c.Role=="configurator")==1 && eagerCalls.Where(c=>c.Role=="configurator").All(c=>c.Outcome=="tool_call:RunSymbolicRegression"),
    "An exhausted pool ends the configurator conversation instead of further rejected proposals");
Check(eagerRuns[0].ControlObservations!.All(o=>o.UsedFallback), "Controller answers that are not decisions fall back to Continue");
// Requests carry the problem facts (input count, data sizes) but never variable names; the toy's input is named "x".
Check(toolBodies.Count==toolRequests && toolBodies.All(b=>b.Contains("InputVariables") && b.Contains("TrainingRows")) &&
    toolBodies.Any(b=>b.Contains("GP algorithm is fixed")) && !toolBodies.Any(b=>b.Contains("\"x\"")),
    "Configurator and controller requests state the problem facts without variable names");
Check(toolBodies.All(b=>b.Contains("\"seed\":142") && b.Contains("max_tokens")), "Every model request carries the repetition's sampling seed and an output-token cap");
Check(eagerRuns.All(r=>r.TrialCpuTimeMs is > 0 && r.TrialCpuTimeMs <= 4 * r.TrialWallTimeMs!.Value + 1000), "Trial CPU time is recorded");
Console.WriteLine($"PASS agent guard: exhausted pool ends the conversation ({eagerCalls.Count} model requests)");

// Feedback conditions: names, the unobserved "final" mode, and distinct job names per condition.
Check(FeedbackChannelNames.Parse("final")==FeedbackChannels.None && FeedbackChannelNames.Parse("all")==FeedbackChannels.All &&
    FeedbackChannelNames.Parse("size,curve")==(FeedbackChannels.Progress|FeedbackChannels.Curve|FeedbackChannels.Size) &&
    FeedbackChannelNames.Format(FeedbackChannelNames.Parse("generalization+curve"))=="progress+curve+generalization" &&
    FeedbackChannelNames.Format(FeedbackChannels.All)=="progress+curve+diversity+size+generalization", "Feedback condition names");
await Throws<ArgumentException>(()=>Task.Run(()=>FeedbackChannelNames.Parse("noise")));
var unobserved=new ControllerProbe("final") { FeynmanInstance=toy, Seed=142 };
foreach (var (name,value) in new[] { ("_numConfigurations",2),("_totalEvaluations",400) })
    typeof(AgentSearch).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(unobserved,value);
var finalExperiment=await unobserved.RunSymbolicRegression(agentParameter,250);
Check(finalExperiment.EvaluationsUsed==250 && finalExperiment.ControlObservations.Count==0 && finalExperiment.CheckpointCount==0 &&
    unobserved.ControllerCalls==0 && unobserved.Results[0].TerminationReason=="evaluation_limit",
    "Final-result mode: the run uses its requested evaluations with no controller and no checkpoints");
Check(unobserved.Variant=="feedback=final" && new ControllerProbe("stop").Variant=="feedback=progress+curve+diversity+size+generalization__interval=1" &&
    new RuleControlledSearch(7).Variant=="interval=7", "Method variants name the feedback condition and interval");
Check(ExperimentResultStore.JobName("AgentSearch","I.6.20",1,10,1000,5,"m:1","feedback=final")!=
    ExperimentResultStore.JobName("AgentSearch","I.6.20",1,10,1000,5,"m:1","feedback=progress__interval=10"), "Feedback conditions have distinct job names");
// Controller-only condition: the controller acts, the configurator sees only final results.
var hidden=new ControllerProbe("stop", hideCheckpoints: true) { FeynmanInstance=toy, Seed=142 };
foreach (var (name,value) in new[] { ("_numConfigurations",2),("_totalEvaluations",400) })
    typeof(AgentSearch).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.SetValue(hidden,value);
var hiddenExperiment=await hidden.RunSymbolicRegression(agentParameter,250);
Check(hidden.ControllerCalls==1 && hiddenExperiment.TerminationReason=="controller_stop" && hiddenExperiment.EvaluationsUsed==110 &&
    hiddenExperiment.ControlObservations.Count==0 && hiddenExperiment.CheckpointCount==0 &&
    hidden.GetExperimentHistory().All(e=>e.ControlObservations.Count==0) && hidden.Results[0].ControlObservations!.Count==1,
    "Hidden checkpoints: the controller stops the run, the configurator sees only the final result, the result keeps the checkpoint");
Check(hidden.Variant=="feedback=progress+curve+diversity+size+generalization__interval=1__checkpoints=hidden", "Hidden checkpoints are part of the job name");
Console.WriteLine("PASS feedback conditions: channel names, final-result mode, controller-only mode, all channels on real checkpoints, distinct job names");

// Feedback measures against hand-computed values.
double[] baseOutput = [1, 2, 4, 8, 3, 5];
Check(RunFeedbackBuilder.PhenotypicDiversity([baseOutput, baseOutput.Select(v => 2 * v + 5).ToArray(), baseOutput.Select(v => -v).ToArray()]) < 1e-12,
    "Phenotypic diversity ignores offset, scale and sign, as linear scaling does");
Check(Math.Abs(RunFeedbackBuilder.PhenotypicDiversity([[1, -1, 1, -1], [1, 1, -1, -1]]) - 1) < 1e-12, "Uncorrelated outputs have diversity one");
double[] partner = [1, 2, 3, 4, 5, 6];
double r = (baseOutput.Zip(partner).Sum(p => (p.First - baseOutput.Average()) * (p.Second - partner.Average())))
    / Math.Sqrt(baseOutput.Sum(v => Math.Pow(v - baseOutput.Average(), 2)) * partner.Sum(v => Math.Pow(v - partner.Average(), 2)));
Check(Math.Abs(RunFeedbackBuilder.PhenotypicDiversity([baseOutput, partner]) - (1 - r * r)) < 1e-12, "Diversity of two outputs is one minus their squared correlation");
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

static void Check(bool condition,string message) { if(!condition) throw new Exception(message); }
static JsonSerializerOptions JsonOptions()=>new() { NumberHandling=System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals };
static async Task Throws<T>(Func<Task> action) where T:Exception
{
    try { await action(); } catch(T) { return; }
    throw new Exception($"Expected {typeof(T).Name}");
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
class TrialProbe:HyperParameterOptimizationAlgorithm
{
    public Task<ResultDTO> Run(RegressionDescriptor d,AlgorithmParameter p,int seed,int limit)=>EvaluateConfiguration(d,p,0,seed,limit);
    public override Task<List<ResultDTO>> Execute(RegressionDescriptor d,int s,int n,int e)=>throw new NotSupportedException();
    public static Task<ResultDTO> Observation(AlgorithmParameter p,int id,int seed,int limit)=>Task.FromResult(new ResultDTO {
        Configuration=p,ConfigurationIndex=id,GPSeed=seed,BestFitness=1,MeanPopulationFitness=1,WorstPopulationFitness=1,
        ExecutionTimeMs=0,EvaluationsUsed=limit,EvaluationsLimit=limit,GenerationsCompleted=0,PopulationSize=p.PopulationSize,BestIndividualsSnapshot=[] });
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

class ControllerProbe(string behavior, bool hideCheckpoints = false) : AgentSearch(new HyperOp.Algorithms.HyperParameterOptimization.AgentComponents.AgentParameter {
    ConfiguratorSeesCheckpoints = !hideCheckpoints,
    Provider = LlmTornado.Code.LLmProviders.Custom, ModelName = "fixture", OllamaBaseUrl = "http://localhost:11434",
    DecisionTimeout = TimeSpan.FromMilliseconds(100), CheckpointInterval = 1,
    Feedback = behavior == "final" ? FeedbackChannels.None : FeedbackChannels.All
})
{
    public bool BudgetWasCurrent = true;
    public int ControllerCalls;
    protected override async Task<RunDecision> RequestRunDecisionAsync(AlgorithmParameter configuration, RunProgress progress,
        IReadOnlyList<RunControlObservation> previousObservations, CancellationToken token)
    {
        // Completed trials plus the active run's checkpoint count.
        ControllerCalls++;
        BudgetWasCurrent &= GetRemainingEvaluations().UsedEvaluations == Results.Sum(r => r.EvaluationsUsed) + progress.EvaluationsUsed;
        await Task.Yield();
        if (behavior == "invalid") return new RunDecision((RunAction)99, "Bad action");
        if (behavior == "failure") throw new InvalidOperationException("Simulated inference failure");
        if (behavior == "timeout") await Task.Delay(Timeout.Infinite, token);
        return behavior == "stop" ? new RunDecision(RunAction.Stop, "Deterministic check")
            : new RunDecision(RunAction.Continue, "Deterministic check");
    }
}
