#requires -Version 7.0
# The re-pilot; kept as a shortcut for Start-Study.ps1 -Plan repilot (stop with Stop-Study.ps1 -Plan repilot).
param([string] $Output, [switch] $PlanOnly)
$arguments = @{ Plan = 'repilot'; PlanOnly = $PlanOnly }
if ($Output) { $arguments.Output = $Output }
& (Join-Path $PSScriptRoot 'Start-Study.ps1') @arguments
