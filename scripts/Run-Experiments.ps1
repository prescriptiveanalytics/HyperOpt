#requires -Version 7.0
[CmdletBinding()]
param(
    [string[]] $Algorithms = @('RandomSearch', 'GridSearch', 'HillClimbingSearch', 'IRaceSearch'),
    [string[]] $Instances = @('I.6.20'),
    [ValidateRange('Positive')] [int] $RepetitionStart = 1,
    [ValidateRange('Positive')] [int] $Repetitions = 1,
    [ValidateNotNullOrEmpty()] [ValidateRange('Positive')] [int[]] $Configurations = @(30),
    [ValidateNotNullOrEmpty()] [ValidateRange(1000, [int]::MaxValue)] [int[]] $Evaluations = @(30000),
    # The study's local models: one family and quantization at three sizes.
    [string[]] $Models = @('qwen3.5:2b-q4_K_M', 'qwen3.5:4b-q4_K_M', 'qwen3.5:9b-q4_K_M'),
    [ValidateSet('ollama', 'openrouter')] [string] $Provider = 'ollama',
    # Agent feedback conditions (one job per condition): final, progress, all, or e.g. progress+curve+size; a suffix
    # '/hidden' makes that condition controller-only (e.g. 'final','progress/hidden','progress' in one call), and '/placebo'
    # shows channel values from another run (length-matched placebo, e.g. 'all/placebo').
    [string[]] $Feedback = @('progress'),
    # Checkpoints per per-configuration budget E (one every E/k evaluations) for AgentSearch and RuleControlledSearch.
    [ValidateRange('Positive')] [int] $Checkpoints = 10,
    # Controller-only condition: the controller acts at checkpoints, the configurator sees final results only.
    [switch] $HideConfiguratorCheckpoints,
    # RQ1 controls for all agent jobs of this call (not for 'final'): 'rule' stops a run below 1% improvement since the previous
    # checkpoint, 'random' stops with probability -StopHazard[k-1] at the k-th decision (e.g. '0.3,0.2,0.1'); the LLM
    # configurator is the same. Run one call per model and condition when the hazards differ.
    [ValidateSet('llm', 'rule', 'random')] [string] $Controller = 'llm',
    # Controller instructions for the LLM controller (ControllerPrompts.cs); 'p8' is the study prompt, 'p9' its sensitivity
    # variant without the late-improvement sentence.
    [ValidateSet('p8', 'p9', 'p7', 'standard', 'p1', 'p2', 'p4', 'p5', 'p6')] [string] $ControllerPrompt = 'p8',
    # Typed in PowerShell, 0.1,0.2 is a list; quoted '0.1,0.2' is one string; both are accepted.
    [string[]] $StopHazard,
    [string] $OllamaUrl = 'http://localhost:11434',
    [string] $Output = './results',
    [ValidateRange('NonNegative')] [int] $MaxProcesses = 0,
    # Concurrent agent jobs against one Ollama server (the study runs 4). Ollama 0.40.3 serves Qwen3.5 with one slot,
    # so their requests queue and only GP overlaps.
    [ValidateRange('Positive')] [int] $MaxAgentProcesses = 4,
    # Stop starting agent jobs once a finished job's controller fallback rate exceeds this share (0 disables).
    [ValidateRange(0.0, 1.0)] [double] $MaxFallbackRate = 0.05,
    # Fresh-seed assessment runs of the nominated configuration (at E and 4E); 0 skips them (pilots only).
    [ValidateRange('NonNegative')] [int] $AssessmentRuns = 5,
    # Constant fitting; omitted values use the runner's defaults. Results record the settings in their provenance.
    [Nullable[int]] $FittingIterations, [Nullable[int]] $FittingRows, [Nullable[double]] $FittingProbability,
    # Minutes without protocol records before a job is treated as hung (0 disables the watchdog).
    [ValidateRange('NonNegative')] [int] $StallMinutes = 60,
    # Target noise on training and validation (standard deviation as a fraction of the target's); 0 is noiseless.
    [ValidateRange(0.0, [double]::MaxValue)] [double] $Noise = 0,
    # Small-data variant: training and validation reduced to this many rows each; 0 uses all rows.
    [ValidateRange('NonNegative')] [int] $Rows = 0,
    # Docker container of the Ollama server; its settings are recorded in campaign-manifest.json (empty: not recorded).
    [string] $OllamaContainer = 'hyperop-agent-ollama-1',
    # Development only: reuse cached results from other or uncommitted code (runner --allow-mixed-code).
    [switch] $AllowMixedCode,
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

$availableAlgorithms = @('RandomSearch', 'GridSearch', 'HillClimbingSearch', 'IRaceSearch', 'DefaultConfigurationSearch', 'RuleControlledSearch', 'RandomRunLengthSearch', 'AshaSearch', 'AgentSearch')
$availableInstances = @('I.6.20', 'I.9.18', 'I.15.3.x', 'I.30.5', 'I.32.17', 'I.41.16', 'I.48.20',
    'II.6.15a', 'II.11.27', 'II.11.28', 'II.35.21', 'III.9.52', 'III.10.19',
    # Development equations (pilots only; never part of the evaluation).
    'I.8.14', 'I.13.12', 'I.16.6', 'I.18.4', 'I.27.6', 'I.29.16', 'I.34.14', 'I.40.1',
    'II.6.11', 'II.11.3', 'II.24.17', 'II.35.18', 'II.36.38', 'III.14.14', 'III.15.12')
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
# Same convention as AgentSearch.Variant: hazards rounded to three decimals, joined by '+'.
$controllerSuffix = ''
if ($Controller -eq 'llm' -and $ControllerPrompt -ne 'p8') { $controllerSuffix = "__cprompt=$ControllerPrompt" }
elseif ($Controller -ne 'llm' -and $ControllerPrompt -ne 'p8') { throw "-ControllerPrompt applies to the LLM controller only." }
if ($Controller -ne 'llm') {
    $controllerSuffix = "__controller=$Controller"
    if ($Controller -eq 'random') {
        if (!$StopHazard) { throw "-Controller random needs -StopHazard." }
        $hazards = @(($StopHazard -join ',') -split '[,+]' | Where-Object { $_.Trim() } | ForEach-Object {
            ([double]::Parse($_.Trim(), [Globalization.CultureInfo]::InvariantCulture)).ToString('0.###', [Globalization.CultureInfo]::InvariantCulture) })
        $controllerSuffix += '__stop-hazard=' + ($hazards -join '+')
    }
} elseif ($StopHazard) { throw "-StopHazard applies to -Controller random only." }
foreach ($algorithm in $canonicalAlgorithms) {
    $jobModels = if ($algorithm -eq 'AgentSearch') { @($Models | Select-Object -Unique) } else { @('') }
    # Variants must match the method's Variant property, which ExperimentResultStore.JobName appends.
    $variants = switch ($algorithm) {
        'AgentSearch' { @($Feedback | ForEach-Object { $condition, $visibility = $_ -split '/', 2
            if ($visibility -and $visibility -notin 'hidden', 'placebo') { throw "Unknown suffix '$visibility' in '$_'; use '/hidden' or '/placebo'." }
            $canonical = Format-Feedback $condition
            if ($canonical -eq 'final') { 'feedback=final' } else { "feedback=$canonical" + "__checkpoints-per-E=$Checkpoints" +
                $(if ($HideConfiguratorCheckpoints -or $visibility -eq 'hidden') { '__checkpoints=hidden' } else { '' }) +
                $(if ($visibility -eq 'placebo') { '__placebo' } else { '' }) + $controllerSuffix } } | Select-Object -Unique) }
        'RuleControlledSearch' { @("checkpoints-per-E=$Checkpoints") }
        'AshaSearch' { @('eta=3') }
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
                        $name += "__instance=$([Uri]::EscapeDataString($instance))" +
                            $(if ($Noise -ne 0) { "__noise=$($Noise.ToString('R', [Globalization.CultureInfo]::InvariantCulture))" } else { '' }) +
                            $(if ($Rows -ne 0) { "__rows=$Rows" } else { '' }) +
                            "__rep=$repetition" +
                            "__T=${trials}__E=${evaluationLimit}__K=$AssessmentRuns"
                        # ExperimentResultStore.MaxJobNameLength: longer names carry a hash of the variant instead.
                        if ($name.Length -gt 200 -and $variant) {
                            $sha = [Security.Cryptography.SHA256]::Create()
                            $hash = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($variant))) -replace '-', '').Substring(0, 12).ToLowerInvariant()
                            $sha.Dispose()
                            $name = $name.Replace("__$($variant)__instance=", "__variant=$($hash)__instance=")
                        }
                        $arguments = @($assembly, '--algorithm', $algorithm, '--instance', $instance,
                            '--configurations', "$trials", '--evaluations', "$evaluationLimit",
                            '--repetition-start', "$repetition", '--repetitions', '1', '--output', $outputDirectory,
                            '--assessment-runs', "$AssessmentRuns", '--stall-minutes', "$StallMinutes",
                            '--noise', $Noise.ToString('R', [Globalization.CultureInfo]::InvariantCulture), '--rows', "$Rows")
                        if ($AllowMixedCode) { $arguments += '--allow-mixed-code' }
                        if ($null -ne $FittingIterations) { $arguments += @('--fitting-iterations', "$FittingIterations") }
                        if ($null -ne $FittingRows) { $arguments += @('--fitting-rows', "$FittingRows") }
                        if ($null -ne $FittingProbability) { $arguments += @('--fitting-probability', $FittingProbability.ToString([Globalization.CultureInfo]::InvariantCulture)) }
                        if ($model) {
                            $arguments += @('--provider', $Provider, '--model', $model, '--feedback', ($variant -replace '^feedback=([^_]+).*$', '$1'))
                            if ($Provider -eq 'ollama') { $arguments += @('--ollama-url', $OllamaUrl) }
                        }
                        if ($algorithm -in 'AgentSearch', 'RuleControlledSearch') { $arguments += @('--checkpoints', "$Checkpoints") }
                        if ($algorithm -eq 'AgentSearch' -and $variant -like '*__checkpoints=hidden*') { $arguments += @('--configurator-checkpoints', 'hidden') }
                        if ($algorithm -eq 'AgentSearch' -and $variant -like '*__placebo*') { $arguments += @('--placebo') }
                        if ($algorithm -eq 'AgentSearch' -and $variant -like '*__cprompt=*') { $arguments += @('--controller-prompt', $ControllerPrompt) }
                        if ($algorithm -eq 'AgentSearch' -and $variant -like '*__controller=*') { $arguments += @('--controller', $Controller) }
                        if ($algorithm -eq 'AgentSearch' -and $variant -like '*__stop-hazard=*') { $arguments += @('--stop-hazard', ($hazards -join ',')) }
                        $jobs.Add([pscustomobject]@{ Name = $name; Agent = ($algorithm -eq 'AgentSearch'); Model = $model; Arguments = $arguments })
                    }
                }
            }
        }
    } }
}

Write-Host "Available logical processors: $([Environment]::ProcessorCount); CPU worker limit: $workers"
Write-Host "Prepared $($jobs.Count) jobs. Output: $outputDirectory"
if ($PlanOnly) {
    foreach ($job in @($jobs | Where-Object { !$_.Agent }) + @($jobs | Where-Object { $_.Agent } | Sort-Object Model -Stable)) {
        # One line per job for Start-Study.ps1's expected-job list.
        Write-Host "JOB $($job.Name)"
        [pscustomobject]@{ Phase = $(if ($job.Agent) { "Agent ($($job.Model)): up to $MaxAgentProcesses at a time" } else { 'CPU: parallel' });
            Result = $job.Name + '.result.json'; Arguments = $job.Arguments }
    }
    return
}

$dotnet = (Get-Command dotnet -ErrorAction Stop).Source
& $dotnet build $project --configuration Release --nologo
if ($LASTEXITCODE -ne 0) { throw 'Build failed; no experiments were launched.' }
[void][IO.Directory]::CreateDirectory($outputDirectory)
$failures = [Collections.Generic.List[string]]::new()

# Controller fallbacks (failed, invalid or timed-out decisions that continue the run) distort the agent's results
# without stopping the job, so they are reported per job and collected in fallbacks.csv.
$script:fallbackAbort = $null
function Write-FallbackRate([object] $Job, [datetime] $Started) {
    $resultPath = Join-Path $outputDirectory ($Job.Name + '.result.json')
    # Only results this attempt wrote; a job that found its result cached is not counted again.
    if (!(Test-Path $resultPath) -or (Get-Item $resultPath).LastWriteTime -lt $Started) { return }
    $timing = (Get-Content $resultPath -Raw | ConvertFrom-Json).Timing
    $requests = [int]$timing.ControllerRequests; $fallbacks = [int]$timing.ControllerFallbacks
    $rate = if ($requests) { $fallbacks / $requests } else { 0 }
    $csv = Join-Path $outputDirectory 'fallbacks.csv'
    if (!(Test-Path $csv)) { 'job,controller_requests,fallbacks' | Set-Content $csv -Encoding utf8 }
    "$($Job.Name),$requests,$fallbacks" | Add-Content $csv -Encoding utf8
    if ($fallbacks) { Write-Warning ("{0}: {1} of {2} controller decisions fell back to Continue ({3:P1})" -f $Job.Name, $fallbacks, $requests, $rate) }
    if ($MaxFallbackRate -gt 0 -and $rate -gt $MaxFallbackRate -and !$script:fallbackAbort) {
        $script:fallbackAbort = "{0}: fallback rate {1:P1} above {2:P1}" -f $Job.Name, $rate, $MaxFallbackRate
        Write-Warning "No further agent jobs are started: $script:fallbackAbort"
    }
}

# One model size at a time: the server keeps one model resident (OLLAMA_MAX_LOADED_MODELS=1, KEEP_ALIVE=-1), and jobs of
# different sizes would block each other, so loaded models are unloaded before each batch of one size.
function Clear-OllamaModels {
    if ($Provider -ne 'ollama') { return }
    $loaded = @((Invoke-RestMethod -Uri "$OllamaUrl/api/ps" -Method Get).models)
    foreach ($model in $loaded) {
        Write-Host "Unloading $($model.name)"
        $body = @{ model = $model.name; keep_alive = 0 } | ConvertTo-Json
        [void](Invoke-RestMethod -Uri "$OllamaUrl/api/generate" -Method Post -Body $body -ContentType 'application/json')
    }
}

# Graceful stop: a file named STOP in the output directory (scripts/Stop-Study.ps1) ends the campaign after the running
# jobs; no new job starts. Completed jobs are cached, so the same command later resumes with the missing ones.
$stopFile = Join-Path $outputDirectory 'STOP'
$script:stopNoticed = $false
function Invoke-ExperimentBatch([object[]] $Batch, [int] $Limit) {
    $pending = [Collections.Generic.Queue[object]]::new()
    foreach ($job in $Batch) { $pending.Enqueue($job) }
    $active = [Collections.Generic.List[object]]::new()
    try {
        while ($pending.Count -or $active.Count) {
            if ($pending.Count -and (Test-Path $stopFile)) {
                if (!$script:stopNoticed) { Write-Host "Stop requested: no new jobs; waiting for $($active.Count) running"; $script:stopNoticed = $true }
                while ($pending.Count) { [void]$pending.Dequeue() }
            }
            # After a fallback abort, the agent jobs still waiting are dropped (they had kept the loop running forever).
            while ($script:fallbackAbort -and $pending.Count -and $pending.Peek().Agent) {
                Write-Warning "Not started (fallback abort): $($pending.Dequeue().Name)"
            }
            while ($pending.Count -and $active.Count -lt $Limit -and !($script:fallbackAbort -and $pending.Peek().Agent)) {
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
                # Appended, so the logs of earlier attempts of the same job are kept.
                $stdout = [IO.File]::Open((Join-Path $outputDirectory ($job.Name + '.stdout.log')), [IO.FileMode]::Append)
                $stderr = [IO.File]::Open((Join-Path $outputDirectory ($job.Name + '.stderr.log')), [IO.FileMode]::Append)
                try {
                    [void]$process.Start()
                    $active.Add([pscustomobject]@{ Job = $job; Process = $process; Stdout = $stdout; Stderr = $stderr; Started = Get-Date;
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
                if ($worker.Job.Agent) { Write-FallbackRate $worker.Job $worker.Started }
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

# Campaign manifest: server version, loaded models and the Ollama container's settings, which results cannot record.
$manifest = [ordered]@{ Started = [DateTime]::UtcNow.ToString('o'); Arguments = $PSBoundParameters; OllamaVersion = $null; OllamaEnvironment = $null; Gpu = $null }
# The GPU that serves the model (name, memory, driver), which the results do not record.
try { $manifest.Gpu = @(& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null) } catch { }
if ($Provider -eq 'ollama' -and @($jobs | Where-Object { $_.Agent }).Count) {
    try { $manifest.OllamaVersion = (Invoke-RestMethod -Uri "$OllamaUrl/api/version").version } catch { Write-Warning "Ollama version unavailable: $_" }
    if ($OllamaContainer) {
        $environment = & docker exec $OllamaContainer printenv 2>$null
        if ($LASTEXITCODE -eq 0) { $manifest.OllamaEnvironment = @($environment | Where-Object { $_ -like 'OLLAMA_*' }) }
        else { Write-Warning "Ollama container '$OllamaContainer' not found; its settings are not recorded." }
    }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputDirectory ("campaign-manifest-{0:yyyyMMdd-HHmmss}Z.json" -f [DateTime]::UtcNow)) -Encoding utf8

Invoke-ExperimentBatch -Batch @($jobs | Where-Object { !$_.Agent }) -Limit $workers
# Finish the entire CPU batch first, so GP timings of agent jobs are not distorted by CPU contention.
foreach ($group in @($jobs | Where-Object { $_.Agent } | Group-Object Model | Sort-Object Name)) {
    if (Test-Path $stopFile) { $script:stopNoticed = $true; break }
    Clear-OllamaModels
    Write-Host "Agent batch: $($group.Count) jobs with $($group.Name)"
    Invoke-ExperimentBatch -Batch @($group.Group) -Limit $MaxAgentProcesses
}
& $dotnet $assembly --aggregate --output $outputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Aggregation failed; individual completed results remain available.' }
if ($script:stopNoticed) { Write-Host 'Stopped on request; run the same command again to continue.' }
if ($script:fallbackAbort) { throw "Agent jobs were stopped early: $script:fallbackAbort. Check the model server before rerunning." }
if ($failures.Count) { throw "$($failures.Count) experiments failed. Completed results were aggregated; inspect the per-job logs: $($failures -join ', ')" }
