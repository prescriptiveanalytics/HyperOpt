#requires -Version 7.0
[CmdletBinding()]
param(
    [string[]] $Algorithms = @('RandomSearch', 'GridSearch', 'HillClimbingSearch', 'IRaceSearch'),
    [string[]] $Instances = @('I.6.20'),
    [ValidateRange('Positive')] [int] $RepetitionStart = 1,
    [ValidateRange('Positive')] [int] $Repetitions = 1,
    [ValidateNotNullOrEmpty()] [ValidateRange('Positive')] [int[]] $Configurations = @(100),
    [ValidateNotNullOrEmpty()] [ValidateRange(100, [int]::MaxValue)] [int[]] $Evaluations = @(1000),
    # The study's local models: one family and quantization at three sizes.
    [string[]] $Models = @('qwen3.5:2b-q4_K_M', 'qwen3.5:4b-q4_K_M', 'qwen3.5:9b-q4_K_M'),
    [ValidateSet('ollama', 'openrouter')] [string] $Provider = 'ollama',
    # Agent feedback conditions (one job per condition): final, progress, all, or e.g. progress+curve+size.
    [string[]] $Feedback = @('progress'),
    # Generations between checkpoints for AgentSearch and RuleControlledSearch.
    [ValidateRange('Positive')] [int] $CheckpointInterval = 10,
    # Controller-only condition: the controller acts at checkpoints, the configurator sees final results only.
    [switch] $HideConfiguratorCheckpoints,
    [string] $OllamaUrl = 'http://localhost:11434',
    [string] $Output = './results',
    [ValidateRange('NonNegative')] [int] $MaxProcesses = 0,
    # Concurrent agent jobs against one Ollama server; match OLLAMA_NUM_PARALLEL on the server.
    [ValidateRange('Positive')] [int] $MaxAgentProcesses = 1,
    [switch] $PlanOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repository = Split-Path $PSScriptRoot -Parent
$project = Join-Path $repository 'HyperOp.Experiments/HyperOp.Experiments.csproj'
$assembly = Join-Path $repository 'HyperOp.Experiments/bin/Release/net10.0/HyperOp.Experiments.dll'
$outputDirectory = [IO.Path]::GetFullPath($Output)
$workers = if ($MaxProcesses -gt 0) { $MaxProcesses } else { [Math]::Max(1, [Math]::Floor([Environment]::ProcessorCount / 2)) }
$lastRepetition = $RepetitionStart + $Repetitions - 1
if (!$Algorithms.Count -or !$Instances.Count -or !$Models.Count) { throw 'Algorithm, instance and model lists cannot be empty.' }

$availableAlgorithms = @('RandomSearch', 'GridSearch', 'HillClimbingSearch', 'IRaceSearch', 'DefaultConfigurationSearch', 'RuleControlledSearch', 'AgentSearch')
$availableInstances = @('I.6.20', 'I.9.18', 'I.15.3.x', 'I.30.5', 'I.32.17', 'I.41.16', 'I.48.20',
    'II.6.15a', 'II.11.27', 'II.11.28', 'II.35.21', 'III.9.52', 'III.10.19')
$canonicalAlgorithms = @($Algorithms | ForEach-Object {
    $requested = $_
    $match = @($availableAlgorithms | Where-Object { $_ -ieq $requested })
    if (!$match.Count) { throw "Unknown algorithm '$requested'. Use: $($availableAlgorithms -join ', ')." }
    $match[0]
} | Select-Object -Unique)
$canonicalInstances = @($Instances | ForEach-Object {
    $requested = $_
    $match = @($availableInstances | Where-Object { $_ -ieq $requested })
    if (!$match.Count) { throw "Unknown instance '$requested'. Use: $($availableInstances -join ', ')." }
    $match[0]
} | Select-Object -Unique)

# Canonical feedback name, as FeedbackChannelNames.Format writes it.
function Format-Feedback([string] $Text) {
    $order = @('progress', 'curve', 'diversity', 'size', 'generalization')
    $parts = @($Text.ToLowerInvariant() -split '[,+]' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    if ($parts.Count -eq 1 -and $parts[0] -eq 'final') { return 'final' }
    if ($parts.Count -eq 1 -and $parts[0] -eq 'all') { return $order -join '+' }
    foreach ($part in $parts) { if ($part -notin $order) { throw "Unknown feedback '$part'." } }
    return (@($order | Where-Object { $_ -eq 'progress' -or $_ -in $parts }) -join '+')
}

$jobs = [Collections.Generic.List[object]]::new()
foreach ($algorithm in $canonicalAlgorithms) {
    $jobModels = if ($algorithm -eq 'AgentSearch') { @($Models | Select-Object -Unique) } else { @('') }
    # Variants must match the method's Variant property, which ExperimentResultStore.JobName appends.
    $variants = switch ($algorithm) {
        'AgentSearch' { @($Feedback | ForEach-Object { $canonical = Format-Feedback $_
            if ($canonical -eq 'final') { 'feedback=final' } else { "feedback=$canonical" + "__interval=$CheckpointInterval" +
                $(if ($HideConfiguratorCheckpoints) { '__checkpoints=hidden' } else { '' }) } } | Select-Object -Unique) }
        'RuleControlledSearch' { @("interval=$CheckpointInterval") }
        default { @('') }
    }
    foreach ($model in $jobModels) { foreach ($variant in $variants) {
        if ($algorithm -eq 'AgentSearch' -and [string]::IsNullOrWhiteSpace($model)) { throw 'Model names cannot be empty.' }
        foreach ($instance in $canonicalInstances) {
            foreach ($trials in ($Configurations | Select-Object -Unique)) {
                foreach ($evaluationLimit in ($Evaluations | Select-Object -Unique)) {
                    for ($repetition = $RepetitionStart; $repetition -le $lastRepetition; $repetition++) {
                        # Keep this convention identical to ExperimentResultStore.JobName.
                        $name = [Uri]::EscapeDataString($algorithm)
                        if ($model) { $name += '__model=' + [Uri]::EscapeDataString($model) }
                        if ($variant) { $name += '__' + $variant }
                        $name += "__instance=$([Uri]::EscapeDataString($instance))__rep=$repetition" +
                            "__T=${trials}__E=${evaluationLimit}__K=5"
                        $arguments = @($assembly, '--algorithm', $algorithm, '--instance', $instance,
                            '--configurations', "$trials", '--evaluations', "$evaluationLimit",
                            '--repetition-start', "$repetition", '--repetitions', '1', '--output', $outputDirectory)
                        if ($model) {
                            $arguments += @('--provider', $Provider, '--model', $model, '--feedback', ($variant -replace '^feedback=([^_]+).*$', '$1'))
                            if ($Provider -eq 'ollama') { $arguments += @('--ollama-url', $OllamaUrl) }
                        }
                        if ($algorithm -in 'AgentSearch', 'RuleControlledSearch') { $arguments += @('--checkpoint-interval', "$CheckpointInterval") }
                        if ($algorithm -eq 'AgentSearch' -and $HideConfiguratorCheckpoints) { $arguments += @('--configurator-checkpoints', 'hidden') }
                        $jobs.Add([pscustomobject]@{ Name = $name; Agent = ($algorithm -eq 'AgentSearch'); Arguments = $arguments })
                    }
                }
            }
        }
    } }
}

Write-Host "Available logical processors: $([Environment]::ProcessorCount); CPU worker limit: $workers"
Write-Host "Prepared $($jobs.Count) jobs. Output: $outputDirectory"
if ($PlanOnly) {
    foreach ($job in @($jobs | Where-Object { !$_.Agent }) + @($jobs | Where-Object { $_.Agent })) {
        [pscustomobject]@{ Phase = $(if ($job.Agent) { "Agent: up to $MaxAgentProcesses at a time" } else { 'CPU: parallel' });
            Result = $job.Name + '.result.json'; Arguments = $job.Arguments }
    }
    return
}

$dotnet = (Get-Command dotnet -ErrorAction Stop).Source
& $dotnet build $project --configuration Release --nologo
if ($LASTEXITCODE -ne 0) { throw 'Build failed; no experiments were launched.' }
[void][IO.Directory]::CreateDirectory($outputDirectory)
$failures = [Collections.Generic.List[string]]::new()

function Invoke-ExperimentBatch([object[]] $Batch, [int] $Limit) {
    $pending = [Collections.Generic.Queue[object]]::new()
    foreach ($job in $Batch) { $pending.Enqueue($job) }
    $active = [Collections.Generic.List[object]]::new()
    try {
        while ($pending.Count -or $active.Count) {
            while ($pending.Count -and $active.Count -lt $Limit) {
                $job = $pending.Dequeue()
                $start = [Diagnostics.ProcessStartInfo]::new($dotnet)
                $start.UseShellExecute = $false
                $start.CreateNoWindow = $true
                $start.WorkingDirectory = $repository
                $start.RedirectStandardOutput = $true
                $start.RedirectStandardError = $true
                foreach ($argument in $job.Arguments) { $start.ArgumentList.Add($argument) }
                $process = [Diagnostics.Process]::new()
                $process.StartInfo = $start
                $stdout = [IO.File]::Create((Join-Path $outputDirectory ($job.Name + '.stdout.log')))
                $stderr = [IO.File]::Create((Join-Path $outputDirectory ($job.Name + '.stderr.log')))
                try {
                    [void]$process.Start()
                    $active.Add([pscustomobject]@{ Job = $job; Process = $process; Stdout = $stdout; Stderr = $stderr;
                        OutputCopy = $process.StandardOutput.BaseStream.CopyToAsync($stdout);
                        ErrorCopy = $process.StandardError.BaseStream.CopyToAsync($stderr) })
                }
                catch {
                    $stdout.Dispose(); $stderr.Dispose(); $process.Dispose()
                    throw
                }
                Write-Host "Started $($job.Name) (PID $($process.Id))"
            }
            foreach ($worker in $active.ToArray()) {
                if (!$worker.Process.HasExited) { continue }
                $worker.Process.WaitForExit()
                [void]$worker.OutputCopy.GetAwaiter().GetResult()
                [void]$worker.ErrorCopy.GetAwaiter().GetResult()
                $code = $worker.Process.ExitCode
                if ($code -ne 0) { $failures.Add($worker.Job.Name) }
                Write-Host "Finished $($worker.Job.Name) (exit $code)"
                $worker.Stdout.Dispose(); $worker.Stderr.Dispose(); $worker.Process.Dispose()
                [void]$active.Remove($worker)
            }
            if ($active.Count) { Start-Sleep -Milliseconds 200 }
        }
    }
    finally {
        foreach ($worker in $active) {
            if (!$worker.Process.HasExited) { $worker.Process.Kill($true) }
            $worker.Process.WaitForExit()
            try {
                [void]$worker.OutputCopy.GetAwaiter().GetResult()
                [void]$worker.ErrorCopy.GetAwaiter().GetResult()
            }
            finally { $worker.Stdout.Dispose(); $worker.Stderr.Dispose(); $worker.Process.Dispose() }
        }
    }
}

Invoke-ExperimentBatch -Batch @($jobs | Where-Object { !$_.Agent }) -Limit $workers
# Finish the entire CPU batch first, so GP timings of agent jobs are not distorted by CPU contention.
Invoke-ExperimentBatch -Batch @($jobs | Where-Object { $_.Agent }) -Limit $MaxAgentProcesses
& $dotnet $assembly --aggregate --output $outputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Aggregation failed; individual completed results remain available.' }
if ($failures.Count) { throw "$($failures.Count) experiments failed. Completed results were aggregated; inspect the per-job logs: $($failures -join ', ')" }
