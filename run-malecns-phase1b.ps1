param(
    [int]$MinSyn = 5,
    [int]$StageTrials = 512,
    [int]$ValidationTrials = 32,
    [int]$DecoderEpochs = 8,
    [int]$CheckpointEvery = 64,
    [string]$Seeds = "7,17,27",
    [switch]$Fresh,
    [switch]$NoDownload,
    [switch]$NoOpen,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath Phase-1B adaptive memory benchmark ==="
Write-Host "v2 baseline vs v2.1 | seeds=$Seeds | trials/stage=$StageTrials | validation/class=$ValidationTrials"
Write-Host "Reuses Phase-1A v2 checkpoints when available."

python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.phase1b_benchmark",
    "--min-syn", "$MinSyn",
    "--stage-trials", "$StageTrials",
    "--validation-trials", "$ValidationTrials",
    "--decoder-epochs", "$DecoderEpochs",
    "--checkpoint-every", "$CheckpointEvery",
    "--seeds", "$Seeds"
)
if (-not $NoDownload) { $argsList += "--download" }
if ($Fresh) { $argsList += "--fresh" }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "Phase-1B benchmark failed with exit code $LASTEXITCODE" }

if (-not $NoPush) {
    git add .\results\latest_malecns_phase1b_stat.json .\results\latest_malecns_phase1b_stat.html .\results\phase1b_stat
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Update MaleCNS Phase-1B adaptive memory benchmark"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        git push
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    } else {
        Write-Host "Phase-1B results unchanged; nothing to push."
    }
}

if (-not $NoOpen) {
    Start-Process (Resolve-Path ".\results\latest_malecns_phase1b_stat.html")
}

Write-Host "=== Phase-1B complete ==="
Write-Host "Summary: results\latest_malecns_phase1b_stat.json"
Write-Host "Dashboard: results\latest_malecns_phase1b_stat.html"
Write-Host "Per-seed v2.1: results\phase1b_stat\seed_*\v21.json"
