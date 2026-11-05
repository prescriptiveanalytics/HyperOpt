#requires -Version 7.0
<#
Runs the re-pilot or the campaign (the plans below; conditions per research question in the paper's Table 3) and can be
stopped and started repeatedly:
completed jobs are cached in the output directory, so starting again continues with the missing ones.

    ./scripts/Start-Study.ps1 -Plan repilot
    ./scripts/Start-Study.ps1 -Plan campaign -StopHazard 0.018,0.696,0.713,0.662,0.556,0.5,0.545
    ./scripts/Stop-Study.ps1 -Plan campaign          # finish the running jobs, then stop
    ./scripts/Stop-Study.ps1 -Plan campaign -Now     # stop at once (the running jobs are redone next time)

Resuming needs the same code: the runner reuses cached results only with the same code fingerprint (a content hash of the
code, CodeFingerprint.cs; notes and the paper may change in between). The code must be committed (-Development passes
--allow-mixed-code for tests only). Written to the output directory:
  study-progress.log       job i/N, durations, elapsed, ETA, warnings and errors (follow with Get-Content -Wait)
  system-monitor.csv       every minute: GPU utilization and memory, CPU load, free RAM, running jobs
  ollama-<time>.log        the Ollama server log of each step (docker logs)
  campaign-manifest-*.json arguments, Ollama version and settings, GPU (one per step, from Run-Experiments.ps1)
  fallbacks.csv, <job>.stdout.log, <job>.stderr.log, <job>.result.json, <job>__attempt=<id>.protocol.jsonl
#>
param(
    # smoke: 8 short 2B jobs (T = 10, E = 10,000) to try starting, stopping and resuming.
    [ValidateSet('repilot', 'campaign', 'smoke')] [string] $Plan = 'campaign',
    [string] $Output,
    # Campaign only: stop probability per decision for the random-stopping cell (controller_stops.py --hazard on the
    # re-pilot's 4B controller-only cell); without it the cell is skipped.
    # Typed in PowerShell, 0.1,0.2 is a list; quoted '0.1,0.2' is one string; both are accepted.
    [string[]] $StopHazard,
    # Campaign only: also run the four single channels (exploratory, RQ2).
    [switch] $IncludeChannels,
    [string] $OllamaContainer = 'hyperop-agent-ollama-1',
    [string] $OllamaUrl = 'http://localhost:11434',
    [switch] $Development,
    [switch] $PlanOnly
)
$ErrorActionPreference = 'Stop'
$repository = Split-Path $PSScriptRoot -Parent
$driver = Join-Path $PSScriptRoot 'Run-Experiments.ps1'
if (!$Output) { $Output = Join-Path $repository "results/$Plan-2026-11" }
$Output = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force $Output | Out-Null
$log = Join-Path $Output 'study-progress.log'
$stopFile = Join-Path $Output 'STOP'
function Format-Span([TimeSpan] $span) { '{0}h{1:00}m{2:00}s' -f [int][math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds }
function Write-Log([string] $text) {
    # UTC, like the protocols: local times would reveal the time zone (anonymous artifact).
    $line = "[{0:yyyy-MM-dd HH:mm:ss}Z] {1}" -f [DateTime]::UtcNow, $text
    Add-Content $log $line -Encoding utf8; Write-Host $line
}

# --- Plans --------------------------------------------------------------------------------------------------------------
$m4 = 'qwen3.5:4b-q4_K_M'; $m9 = 'qwen3.5:9b-q4_K_M'; $m2 = 'qwen3.5:2b-q4_K_M'
if ($Plan -eq 'smoke') {
    $base = @{ Instances = 'I.29.16', 'I.8.14'; Configurations = 10; Evaluations = 10000; AssessmentRuns = 0; Repetitions = 2 }
    $steps = @(
        @{ Name = '2B final'; Args = @{ Algorithms = 'AgentSearch'; Models = $m2; Feedback = 'final' } },
        @{ Name = '2B progress without stopping'; Args = @{ Algorithms = 'AgentSearch'; Models = $m2; Feedback = 'progress'; Controller = 'random'; StopHazard = '0' } }
    )
}
elseif ($Plan -eq 'repilot') {
    $base = @{ Instances = 'I.29.16', 'I.8.14', 'II.6.11', 'III.15.12'; Configurations = 30; Evaluations = 30000; AssessmentRuns = 0; Repetitions = 2 }
    $steps = @(
        @{ Name = 'CPU baselines'; Args = @{ Algorithms = 'RandomSearch', 'RuleControlledSearch', 'RandomRunLengthSearch', 'AshaSearch' } },
        @{ Name = '4B conditions'; Args = @{ Algorithms = 'AgentSearch'; Models = $m4; Feedback = 'final', 'progress', 'progress/hidden', 'all', 'all/placebo' } },
        @{ Name = 'progress without stopping (4B, 9B, 2B)'; Args = @{ Algorithms = 'AgentSearch'; Models = $m4, $m9, $m2; Feedback = 'progress'; Controller = 'random'; StopHazard = '0' } }
    )
}
else {
    $tasks = 'I.6.20', 'I.9.18', 'I.15.3.x', 'I.30.5', 'I.32.17', 'I.41.16', 'I.48.20', 'II.6.15a', 'II.11.27', 'II.11.28', 'II.35.21', 'III.9.52', 'III.10.19'
    $base = @{ Instances = $tasks; Configurations = 30; Evaluations = 30000; AssessmentRuns = 5 }
    $steps = @(
        @{ Name = 'CPU references'; Args = @{ Repetitions = 10; Algorithms = 'RandomSearch', 'IRaceSearch', 'DefaultConfigurationSearch', 'RuleControlledSearch', 'RandomRunLengthSearch', 'AshaSearch', 'GridSearch', 'HillClimbingSearch' } },
        @{ Name = 'RQ1 4B: final, progress, controller-only'; Args = @{ Repetitions = 10; Algorithms = 'AgentSearch'; Models = $m4; Feedback = 'final', 'progress', 'progress/hidden' } },
        @{ Name = 'RQ1 4B: progress without stopping'; Args = @{ Repetitions = 10; Algorithms = 'AgentSearch'; Models = $m4; Feedback = 'progress'; Controller = 'random'; StopHazard = '0' } },
        @{ Name = 'RQ1 4B: random stopping'; Args = @{ Repetitions = 10; Algorithms = 'AgentSearch'; Models = $m4; Feedback = 'progress/hidden'; Controller = 'random'; StopHazard = $StopHazard } },
        @{ Name = 'RQ3 9B and 2B: final'; Args = @{ Repetitions = 5; Algorithms = 'AgentSearch'; Models = $m9, $m2; Feedback = 'final' } },
        @{ Name = 'RQ3 9B and 2B: progress without stopping'; Args = @{ Repetitions = 5; Algorithms = 'AgentSearch'; Models = $m9, $m2; Feedback = 'progress'; Controller = 'random'; StopHazard = '0' } },
        @{ Name = 'RQ3 9B and 2B: controller-only'; Args = @{ Repetitions = 5; Algorithms = 'AgentSearch'; Models = $m9, $m2; Feedback = 'progress/hidden' } },
        @{ Name = 'RQ2 4B: all, placebo'; Args = @{ Repetitions = 5; Algorithms = 'AgentSearch'; Models = $m4; Feedback = 'all', 'all/placebo' } }
    )
    if ($IncludeChannels) {
        $steps += @{ Name = 'RQ2 4B: single channels (exploratory)'; Args = @{ Repetitions = 5; Algorithms = 'AgentSearch'; Models = $m4; Feedback = 'progress+curve', 'progress+diversity', 'progress+size', 'progress+generalization' } }
    }
    # The random-stopping cell is part of the frozen design; the campaign does not start without its hazard.
    if (!$StopHazard -and !$PlanOnly) { throw '-StopHazard is required for the campaign: the re-pilot''s 4B controller-only stop rates (scripts/controller_stops.py --hazard; Supplement S4).' }
    if (!$StopHazard) { $steps = @($steps | Where-Object { $_.Name -ne 'RQ1 4B: random stopping' }) }
}
foreach ($step in $steps) {
    $step.Args = $base + $step.Args + @{ Output = $Output; OllamaContainer = $OllamaContainer; OllamaUrl = $OllamaUrl }
    if ($Development) { $step.Args.AllowMixedCode = $true }
}

# --- Checks before starting ---------------------------------------------------------------------------------------------
if (Test-Path $stopFile) { Remove-Item $stopFile; Write-Log 'Removed the STOP file of an earlier stop; continuing.' }
if (!$PlanOnly) {
    # Only the code that the fingerprint covers must be committed (so it is kept); notes and the paper may be edited.
    $dirty = @(& git -C $repository status --porcelain --untracked-files=no -- HyperOp.Algorithms HyperOp.Experiments data docker)
    if ($dirty.Count -and !$Development) {
        throw "Uncommitted code changes: commit them first, so the code behind the results is kept.`n$($dirty -join "`n")"
    }
    try { [void](Invoke-RestMethod -Uri "$OllamaUrl/api/version" -TimeoutSec 10) }
    catch { throw "The Ollama server at $OllamaUrl is not reachable; start Docker Desktop and the ollama service first." }
}

# Job counts per step (the driver's plan), for progress and ETA.
$total = 0
foreach ($step in $steps) {
    $stepArgs = $step.Args
    $preview = & $driver @stepArgs -PlanOnly *>&1 | Out-String
    $step.Jobs = if ($preview -match 'Prepared (\d+) jobs') { [int]$Matches[1] } else { 0 }
    $step.Names = @([regex]::Matches($preview, '(?m)^JOB (\S+)') | ForEach-Object { $_.Groups[1].Value })
    if ($step.Names.Count -ne $step.Jobs) { throw "Step '$($step.Name)': the plan lists $($step.Names.Count) of $($step.Jobs) jobs." }
    $total += $step.Jobs
}
$expected = @($steps | ForEach-Object { $_.Names } | Sort-Object -Unique)
if ($expected.Count -ne $total) { throw "The plan names $total jobs but only $($expected.Count) distinct ones." }
# The content fingerprint of the code this study runs (the runner prints it); builds first, as the plan does not.
$fingerprint = $null
if (!$PlanOnly) {
    & dotnet build (Join-Path $repository 'HyperOp.Experiments/HyperOp.Experiments.csproj') --configuration Release --nologo -v q *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
    $fingerprint = (& dotnet (Join-Path $repository 'HyperOp.Experiments/bin/Release/net10.0/HyperOp.Experiments.dll') --code-fingerprint | Select-Object -Last 1).Trim()
    if ($fingerprint -notmatch '^[0-9a-f]{64}$') { throw "No code fingerprint: $fingerprint" }
}
Write-Log "=== $Plan, $total jobs in $($steps.Count) steps, code $(if ($fingerprint) { $fingerprint.Substring(0, 12) } else { '-' })$(if ($Development) { ' (development: mixed code allowed)' }), output $Output"
if ($Plan -eq 'campaign' -and !$StopHazard) { Write-Log 'Plan only without -StopHazard: the random-stopping cell is not listed.' }
foreach ($step in $steps) { Write-Log ("    {0}: {1} jobs" -f $step.Name, $step.Jobs) }
if ($PlanOnly) { return }

# --- Study identity -----------------------------------------------------------------------------------------------------
# Everything that must stay the same while the study is stopped and started again: code, models (by digest: tags can change
# after a pull), server, its settings, the plan and its job list. The first start writes study-identity.json; every later
# start must match it (development runs ignore the code fingerprint). No commit, host name or local time is recorded. The expected-job list is the denominator for the check.
$sha = [Security.Cryptography.SHA256]::Create()
$jobsText = ($expected -join "`n")
$jobsHash = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($jobsText))) -replace '-', '').ToLowerInvariant()
$sha.Dispose()
$tags = @((Invoke-RestMethod -Uri "$OllamaUrl/api/tags" -TimeoutSec 30).models)
$planModels = @($steps | ForEach-Object { $_.Args.Models } | Where-Object { $_ } | Sort-Object -Unique)
$digests = [ordered]@{}
foreach ($model in $planModels) {
    $entry = @($tags | Where-Object { $_.name -eq $model })
    if (!$entry.Count) { throw "Model '$model' is not available on the Ollama server." }
    $digests[$model] = $entry[0].digest
}
$serverSettings = @()
if ($OllamaContainer) {
    $environment = & docker exec $OllamaContainer printenv 2>$null
    if ($LASTEXITCODE -eq 0) { $serverSettings = @($environment | Where-Object { $_ -match '^OLLAMA_' } | Sort-Object) }
}
$identity = [ordered]@{
    Plan = $Plan
    CodeFingerprint = $fingerprint
    OllamaVersion = (Invoke-RestMethod -Uri "$OllamaUrl/api/version" -TimeoutSec 10).version
    ModelDigests = $digests
    OllamaSettings = $serverSettings
    Base = [ordered]@{ T = $base.Configurations; E = $base.Evaluations; K = $base.AssessmentRuns; Instances = @($base.Instances) }
    Jobs = $expected.Count
    JobsSha256 = $jobsHash
}
$identityFile = Join-Path $Output 'study-identity.json'
$identityJson = $identity | ConvertTo-Json -Depth 5
if (Test-Path $identityFile) {
    $recorded = Get-Content $identityFile -Raw | ConvertFrom-Json -AsHashtable
    $differences = @()
    foreach ($key in $identity.Keys) {
        if ($Development -and $key -eq 'CodeFingerprint') { continue }
        $now = $identity[$key] | ConvertTo-Json -Depth 5 -Compress
        $before = $recorded[$key] | ConvertTo-Json -Depth 5 -Compress
        if ($now -ne $before) { $differences += "${key}: recorded $before, now $now" }
    }
    if ($differences.Count) {
        throw "This output directory belongs to a study with another identity; use a new directory or restore the setup:`n$($differences -join "`n")"
    }
    Write-Log 'Identity matches study-identity.json; resuming.'
}
else {
    Set-Content $identityFile $identityJson -Encoding utf8
    Write-Log "Wrote study-identity.json (code $($fingerprint.Substring(0, 12)), $($digests.Count) model digests, Ollama $($identity.OllamaVersion))."
}
Set-Content (Join-Path $Output 'expected-jobs.txt') $jobsText -Encoding utf8

# --- System monitor (every minute, in a background thread) --------------------------------------------------------------
$monitorFile = Join-Path $Output 'system-monitor.csv'
if (!(Test-Path $monitorFile)) { 'time,gpu_util_percent,gpu_memory_used_mb,gpu_memory_total_mb,cpu_load_percent,free_ram_mb,running_jobs' | Set-Content $monitorFile -Encoding utf8 }
$monitor = Start-ThreadJob -ArgumentList $monitorFile, $stopFile -ScriptBlock {
    param($file, $stop)
    while ($true) {
        $gpu = try { (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null) -replace ' ', '' } catch { ',,' }
        $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        $ram = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
        $jobs = @(Get-CimInstance Win32_Process -Filter "Name='dotnet.exe'" | Where-Object CommandLine -like '*HyperOp.Experiments.dll*').Count
        Add-Content $file ("{0:yyyy-MM-ddTHH:mm:ssZ},{1},{2},{3},{4}" -f [DateTime]::UtcNow, $gpu, $cpu, $ram, $jobs) -Encoding utf8
        Start-Sleep -Seconds 60
    }
}

# --- Steps --------------------------------------------------------------------------------------------------------------
$start = Get-Date; $done = 0; $failed = 0; $fresh = 0; $freshSeconds = 0.0; $started = @{}
try {
    $script:aborted = $false
    foreach ($step in $steps) {
        if (Test-Path $stopFile) { Write-Log "Stop requested; not starting '$($step.Name)'."; break }
        if ($script:aborted) { Write-Log "Fallback abort in an earlier step; not starting '$($step.Name)'. Check the model server, then start again."; break }
        Write-Log "--- $($step.Name) ($($step.Jobs) jobs)"
        $stepStart = (Get-Date).ToUniversalTime()
        try {
            $stepArgs = $step.Args
            & $driver @stepArgs *>&1 | ForEach-Object {
                $text = "$_"
                if ($text -match '^Started (\S+) \(PID') { $started[$Matches[1]] = Get-Date }
                elseif ($text -match '^Finished (\S+) \(exit (-?\d+)\)') {
                    $name = $Matches[1]; $code = $Matches[2]; $script:done++
                    if ($code -ne '0') { $script:failed++ }
                    $seconds = if ($started.ContainsKey($name)) { ((Get-Date) - $started[$name]).TotalSeconds } else { 0 }
                    # Jobs that load a cached result finish within seconds and do not count for the ETA.
                    if ($seconds -gt 15) { $script:fresh++; $script:freshSeconds = ((Get-Date) - $start).TotalSeconds }
                    $remaining = $total - $script:done
                    $eta = if ($script:fresh) { [TimeSpan]::FromSeconds($script:freshSeconds / $script:fresh * $remaining) } else { $null }
                    Write-Log ("{0}/{1} {2} in {3:N0} s (exit {4}); elapsed {5}, ETA {6}" -f $script:done, $total, $name, $seconds, $code,
                        (Format-Span ((Get-Date) - $start)), $(if ($eta) { Format-Span $eta } else { '-' }))
                }
                # Build summaries ("0 Warnung(en)", "0 Error(s)") are not problems.
                elseif ($text -notmatch '^\s*\d+ (Warn|Fehler|Error)' -and
                    $text -match 'Warning|WARNUNG|failed|Error|Fehler|Not started|Stop requested|Stopped on request|Unloading|Agent batch') { Write-Log $text }
            }
        }
        catch {
            Write-Log "Step '$($step.Name)' ended with an error: $_"
            # A fallback abort means the model server misbehaves: the whole study stops, not only this step.
            if ("$_" -match 'stopped early') { $script:aborted = $true }
        }
        finally {
            # Server log of this step, for errors the jobs cannot see (out of memory, truncation, restarts).
            try {
                $ollamaLog = Join-Path $Output ("ollama-{0:yyyyMMdd-HHmmss}Z.log" -f [DateTime]::UtcNow)
                & docker logs --since $stepStart.ToString('yyyy-MM-ddTHH:mm:ssZ') $OllamaContainer *> $ollamaLog
            } catch { Write-Log "Ollama log not saved: $_" }
        }
    }
}
finally {
    Stop-Job $monitor -ErrorAction SilentlyContinue; Remove-Job $monitor -Force -ErrorAction SilentlyContinue
    $stopped = Test-Path $stopFile
    # The completeness check against expected-jobs.txt, once nothing was stopped or aborted.
    if (!$stopped -and !$script:aborted) {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if ($python) {
            $check = & $python.Source (Join-Path $PSScriptRoot 'repilot_check.py') $Output 2>&1
            Write-Log ("Check (exit {0}): {1}" -f $LASTEXITCODE, (($check | Select-Object -Last 4) -join ' | '))
        }
        else { Write-Log 'Python not found; run scripts/repilot_check.py on the output directory.' }
    }
    Write-Log ("=== {0}: {1} of {2} jobs ended in this session ({3} computed, {6} without result: failed or stopped), {4} elapsed{5}" -f $Plan, $done, $total, $fresh,
        (Format-Span ((Get-Date) - $start)), $(if ($stopped) { '; stopped on request, start again to continue' } else { '' }), $failed)
}
