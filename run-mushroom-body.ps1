param(
    [int]$MinSyn = 5,
    [int]$StimulusCount = 8,
    [int]$Trials = 1,
    [double]$DurationMs = 20.0,
    [double]$StimulusRateHz = 205.0,
    [double]$Reward = 1.0,
    [int]$Seed = 7,
    [switch]$Download,
    [switch]$NoDownload,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
}

$resultRelative = "results/latest_mushroom_body_trial.json"
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

if ($NoPush) {
    Write-Host "Result push skipped (-NoPush)."
    exit 0
}

$currentBranch = (& git branch --show-current).Trim()
if ([string]::IsNullOrWhiteSpace($currentBranch)) {
    throw "Cannot auto-push from a detached HEAD. Check out the experiment branch first."
}

& git add -- $resultRelative
if ($LASTEXITCODE -ne 0) {
    throw "git add failed for $resultRelative"
}

& git diff --cached --quiet -- $resultRelative
if ($LASTEXITCODE -eq 0) {
    Write-Host "Result unchanged; nothing to push."
    exit 0
}

& git commit -m "Update latest mushroom body trial result" -- $resultRelative
if ($LASTEXITCODE -ne 0) {
    throw "git commit failed"
}

& git push origin "HEAD:$currentBranch"
if ($LASTEXITCODE -ne 0) {
    throw "git push failed for branch $currentBranch"
}
Write-Host "Result pushed to origin/$currentBranch. Codex can read it from GitHub now."
