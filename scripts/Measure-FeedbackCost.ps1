#requires -Version 5.1
# Measures the controller cost of each feedback condition (paper Table 2, Supplement Table S5):
# one AgentSearch job per condition on I.6.20 with three trials at the study's E = 30,000. Prompts grow
# with the checkpoints shown to the controller, so conditions are compared on the first controller
# request (first checkpoint of the first trial, no earlier checkpoints); means over all requests depend
# on how many checkpoints a run reached. Token counts are reproducible for a given code version,
# model digest and Ollama version; seconds depend on the hardware and the server's prompt cache.
[CmdletBinding()]
param(
    [string] $Model = 'qwen3.5:2b-q4_K_M',
    [string] $OllamaUrl = 'http://localhost:11434',
    [string] $Output = './results/feedback-cost',
    # Summarize existing results without running jobs.
    [switch] $SummaryOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repository = Split-Path $PSScriptRoot -Parent
$project = Join-Path $repository 'HyperOp.Experiments/HyperOp.Experiments.csproj'
$outputDirectory = [IO.Path]::GetFullPath($Output)

$conditions = [ordered]@{
    'progress'         = @('--feedback', 'progress')
    'progress, hidden' = @('--feedback', 'progress', '--configurator-checkpoints', 'hidden')
    '+ curve'          = @('--feedback', 'progress+curve')
    '+ diversity'      = @('--feedback', 'progress+diversity')
    '+ size'           = @('--feedback', 'progress+size')
    '+ generalization' = @('--feedback', 'progress+generalization')
    'all'              = @('--feedback', 'all')
}

if (!$SummaryOnly) {
    dotnet build $project -c Release -nologo -v q
    if ($LASTEXITCODE) { throw 'Build failed.' }
    foreach ($condition in $conditions.GetEnumerator()) {
        Write-Host "Running $($condition.Key)"
        dotnet run --project $project -c Release --no-build -- --algorithm AgentSearch --model $Model `
            --ollama-url $OllamaUrl --instance I.6.20 --configurations 3 --evaluations 30000 --assessment-runs 0 `
            --output $outputDirectory @($condition.Value)
        if ($LASTEXITCODE) { throw "Job for $($condition.Key) failed." }
    }
}

function Get-Mean([double[]] $values) { if ($values.Count) { ($values | Measure-Object -Average).Average } else { [double]::NaN } }

$rows = foreach ($condition in $conditions.GetEnumerator()) {
    # Job names carry the condition, e.g. feedback=progress+curve__checkpoints-per-E=10 (plus __checkpoints=hidden);
    # 'all' is written as the full channel list.
    $feedback = if ($condition.Value[1] -eq 'all') { 'progress+curve+diversity+size+generalization' } else { $condition.Value[1] }
    $hidden = $condition.Value -contains 'hidden'
    $protocol = Get-ChildItem $outputDirectory -Recurse -Filter '*.protocol.jsonl' |
        Where-Object { $_.Name -like "*feedback=$feedback`__checkpoints-per-E=*" -and ($_.Name -like '*checkpoints=hidden*') -eq $hidden } |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if (!$protocol) { Write-Warning "No protocol for $($condition.Key)"; continue }
    $events = Get-Content $protocol.FullName | ForEach-Object { $_ | ConvertFrom-Json }
    $requests = @($events | Where-Object Type -eq 'controller_request')
    $calls = @($events | Where-Object { $_.Type -eq 'llm_call' -and $_.Data.Role -eq 'controller' })
    $checkpoints = @($events | Where-Object Type -eq 'checkpoint')
    $trials = @($events | Where-Object { $_.Type -eq 'gp_run' -and $_.Data.Phase -eq 'tuning' })
    # Trial overhead per checkpoint: trial wall time minus GP time and controller inference, i.e. feedback
    # computation plus trial setup and validation scoring, divided by the number of checkpoints.
    $overheadMs = ($trials | ForEach-Object { $_.Data.TrialWallTimeMs - $_.Data.GpMs } | Measure-Object -Sum).Sum -
        ($calls | ForEach-Object { $_.Data.DurationMs } | Measure-Object -Sum).Sum
    [pscustomobject]@{
        Condition       = $condition.Key
        FirstCharacters = $requests[0].Data.Input.Length
        FirstTokens     = $calls[0].Data.InputTokens
        FirstSeconds    = [Math]::Round($calls[0].Data.DurationMs / 1000, 1)
        Requests        = $calls.Count
        MeanTokens      = [Math]::Round((Get-Mean ($calls | ForEach-Object { $_.Data.InputTokens })))
        MeanSeconds     = [Math]::Round((Get-Mean ($calls | ForEach-Object { $_.Data.DurationMs })) / 1000, 1)
        FeedbackMs      = if ($checkpoints.Count) { [Math]::Round($overheadMs / $checkpoints.Count) } else { $null }
    }
}
$rows | Format-Table -AutoSize
