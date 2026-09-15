param(
    [int]$MinSyn = 5,
    [int]$StageTrials = 256,
    [int]$DecoderEpochs = 8,
    [int]$ValidationTrials = 8,
    [int]$CheckpointEvery = 64,
    [int]$ReplayInterval = 4,
    [int]$Seed = 17,
    [int]$Port = 8772,
    [switch]$Fresh,
    [switch]$NoDownload,
    [switch]$NoBrowser,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath Phase-1 Memory v2 ==="
Write-Host "Protected plasticity + replay + consolidation + 3-D telemetry"
python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.live_memory_v2",
    "--min-syn", "$MinSyn",
    "--stage-trials", "$StageTrials",
    "--decoder-epochs", "$DecoderEpochs",
    "--validation-trials", "$ValidationTrials",
    "--checkpoint-every", "$CheckpointEvery",
    "--replay-interval", "$ReplayInterval",
    "--seed", "$Seed",
    "--port", "$Port"
)
if (-not $NoDownload) { $argsList += "--download" }
if ($Fresh) { $argsList += "--fresh" }
if ($NoBrowser) { $argsList += "--no-browser" }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "MaleCNS memory v2 failed with exit code $LASTEXITCODE" }

if (-not $NoPush) {
    git add .\results\latest_malecns_v2_memory.json .\results\latest_malecns_v2_memory.html
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Update latest MaleCNS memory v2 result"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        git push
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    } else {
        Write-Host "Memory v2 result unchanged; nothing to push."
    }
}

if (-not $NoBrowser) {
    Start-Process (Resolve-Path ".\results\latest_malecns_v2_memory.html")
}

Write-Host "=== Phase-1 Memory v2 complete ==="
Write-Host "Result: results\latest_malecns_v2_memory.json"
Write-Host "Dashboard: results\latest_malecns_v2_memory.html"
Write-Host "Checkpoint: checkpoints\malecns_v2_memory_brain.npz"
Write-Host "Progress: results\malecns_v2_memory_progress.json"
