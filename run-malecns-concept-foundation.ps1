param(
    [int]$MinSyn = 5,
    [int]$StageTrials = 512,
    [int]$ValidationTrials = 32,
    [int]$DecoderEpochs = 8,
    [int]$CheckpointEvery = 64,
    [int]$Seed = 7,
    [double]$MinLearningGain = 0.03,
    [switch]$NoDownload,
    [switch]$NoOpen,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath Concept Foundation ==="
Write-Host "object presence -> single/multiple -> latent quantity A/B/C"
Write-Host "No number symbols, comparison operators, or arithmetic are used."
Write-Host "seed=$Seed | trials/stage=$StageTrials | validation/class=$ValidationTrials"
Write-Host "strict gate requires held-out learning gain >= $MinLearningGain"

python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.concept_foundation_strict",
    "--min-syn", "$MinSyn",
    "--stage-trials", "$StageTrials",
    "--validation-trials", "$ValidationTrials",
    "--decoder-epochs", "$DecoderEpochs",
    "--checkpoint-every", "$CheckpointEvery",
    "--seed", "$Seed",
    "--min-learning-gain", "$MinLearningGain"
)
if (-not $NoDownload) { $argsList += "--download" }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "concept foundation failed with exit code $LASTEXITCODE" }

if (-not $NoPush) {
    git add .\results\latest_malecns_concept_foundation.json .\results\latest_malecns_concept_foundation.html
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Update MaleCNS concept foundation result"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        git push
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    } else {
        Write-Host "Concept foundation results unchanged; nothing to push."
    }
}

if (-not $NoOpen) {
    Start-Process (Resolve-Path ".\results\latest_malecns_concept_foundation.html")
}

Write-Host "=== Concept foundation complete ==="
Write-Host "Result: results\latest_malecns_concept_foundation.json"
Write-Host "Dashboard: results\latest_malecns_concept_foundation.html"
