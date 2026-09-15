param(
    [int]$MinSyn = 5,
    [int]$StageTrials = 256,
    [int]$DecoderEpochs = 8,
    [int]$ValidationTrials = 8,
    [int]$CheckpointEvery = 64,
    [int]$Seed = 7,
    [switch]$Fresh,
    [switch]$NoDownload,
    [switch]$NoBrowser,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath MaleCNS v1 curriculum ==="
python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.curriculum_v1",
    "--min-syn", "$MinSyn",
    "--stage-trials", "$StageTrials",
    "--decoder-epochs", "$DecoderEpochs",
    "--validation-trials", "$ValidationTrials",
    "--checkpoint-every", "$CheckpointEvery",
    "--seed", "$Seed"
)
if (-not $NoDownload) { $argsList += "--download" }
if ($Fresh) { $argsList += "--fresh" }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "MaleCNS v1 curriculum failed with exit code $LASTEXITCODE" }

python -m drosomath.malecns.visualize_curriculum
if ($LASTEXITCODE -ne 0) { throw "curriculum visualization failed" }

if (-not $NoPush) {
    git add .\results\latest_malecns_v1_curriculum.json .\results\latest_malecns_v1_curriculum.html
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Update latest MaleCNS v1 curriculum result"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        git push
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    } else {
        Write-Host "Curriculum result unchanged; nothing to push."
    }
}

if (-not $NoBrowser) {
    Start-Process (Resolve-Path ".\results\latest_malecns_v1_curriculum.html")
}

Write-Host "=== MaleCNS v1 curriculum complete ==="
Write-Host "Result: results\latest_malecns_v1_curriculum.json"
Write-Host "Dashboard: results\latest_malecns_v1_curriculum.html"
Write-Host "Brain checkpoint: checkpoints\malecns_v1_brain.npz"
Write-Host "Task readouts: checkpoints\malecns_v1_readouts\"
