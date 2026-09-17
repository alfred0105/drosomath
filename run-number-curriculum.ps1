param(
    [int]$MinSyn = 5,
    [int]$StageTrials = 256,
    [int]$DecoderEpochs = 8,
    [int]$ValidationTrials = 8,
    [int]$CheckpointEvery = 64,
    [int]$Seed = 7,
    [switch]$Download,
    [switch]$NoDownload,
    [switch]$Fresh,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = "python"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
}

$resultRelative = "results/latest_malecns_v1_curriculum.json"
$argsList = @(
    "-m", "drosomath.malecns.curriculum_v1",
    "--data-dir", (Join-Path $PSScriptRoot "data\malecns_v1"),
    "--min-syn", "$MinSyn",
    "--stage-trials", "$StageTrials",
    "--decoder-epochs", "$DecoderEpochs",
    "--validation-trials", "$ValidationTrials",
    "--checkpoint-every", "$CheckpointEvery",
    "--seed", "$Seed",
    "--numeric-first"
)

if ($Download -and -not $NoDownload) {
    $argsList += "--download"
}
if ($Fresh) {
    $argsList += "--fresh"
}

Write-Host "=== DrosoMath number-concept curriculum ==="
Write-Host "Python: $python"
Write-Host "Stages: numerosity 1-4 -> comparison -> addition 1-3"
& $python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Number curriculum failed with exit code $LASTEXITCODE"
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

& git commit -m "Update number curriculum result" -- $resultRelative
if ($LASTEXITCODE -ne 0) {
    throw "git commit failed"
}

& git push origin "HEAD:$currentBranch"
if ($LASTEXITCODE -ne 0) {
    throw "git push failed for branch $currentBranch"
}
Write-Host "Result pushed to origin/$currentBranch. Codex can read it from GitHub now."
