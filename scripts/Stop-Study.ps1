#requires -Version 7.0
<#
Stops a run of Start-Study.ps1. By default the running jobs finish and no new job starts (a STOP file in the output
directory); -Now also ends the running jobs at once (they are redone when the study is started again).

    ./scripts/Stop-Study.ps1 -Plan campaign
    ./scripts/Stop-Study.ps1 -Plan campaign -Now
#>
param(
    [ValidateSet('repilot', 'campaign', 'smoke')] [string] $Plan = 'campaign',
    [string] $Output,
    [switch] $Now
)
$repository = Split-Path $PSScriptRoot -Parent
if (!$Output) { $Output = Join-Path $repository "results/$Plan-2026-11" }
$Output = [IO.Path]::GetFullPath($Output)
if (!(Test-Path $Output)) { throw "No study output at $Output." }
Set-Content (Join-Path $Output 'STOP') "stop requested $([DateTime]::UtcNow.ToString('o'))" -Encoding utf8
$line = "[{0:yyyy-MM-dd HH:mm:ss}Z] Stop requested{1}" -f [DateTime]::UtcNow, $(if ($Now) { ' (now: ending running jobs)' } else { ': running jobs finish, no new job starts' })
Add-Content (Join-Path $Output 'study-progress.log') $line -Encoding utf8
Write-Host $line
if ($Now) {
    $jobs = @(Get-CimInstance Win32_Process -Filter "Name='dotnet.exe'" | Where-Object CommandLine -like '*HyperOp.Experiments.dll*')
    foreach ($job in $jobs) { Stop-Process -Id $job.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "Ended $($jobs.Count) running jobs."
}
