param(
    [int]$MinSyn = 5,
    [int]$StimulusCount = 8,
    [int]$Trials = 1,
    [double]$DurationMs = 20.0,
    [double]$StimulusRateHz = 205.0,
    [double]$Reward = 1.0,
    [int]$Seed = 7,
    [switch]$Download,
    [switch]$NoDownload
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
}

$argsList = @(
    "-m", "drosomath.malecns.mushroom_body_demo",
    "--data-dir", (Join-Path $PSScriptRoot "data\malecns_v1"),
    "--min-syn", "$MinSyn",
    "--stimulus-count", "$StimulusCount",
    "--trials", "$Trials",
    "--duration-ms", "$DurationMs",
    "--stimulus-rate-hz", "$StimulusRateHz",
    "--reward", "$Reward",
    "--seed", "$Seed"
)

if ($Download -and -not $NoDownload) {
    $argsList += "--download"
}

Write-Host "=== DrosoMath connectome mushroom-body trial ==="
Write-Host "Python: $python"
& $python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Mushroom-body trial failed with exit code $LASTEXITCODE"
}
