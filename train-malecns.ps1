param(
    [int]$MinSyn = 5,
    [int]$InputPerSide = 32,
    [int]$OutputSize = 512,
    [int]$DecoderEpochs = 8,
    [int]$BrainTrials = 64,
    [int]$ValidationTrials = 4,
    [double]$DurationMs = 20.0,
    [double]$PlasticFraction = 0.05,
    [int]$Seed = 0,
    [switch]$NoDownload,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath: first MaleCNS learning curriculum ==="
Write-Host "Installing MaleCNS dependencies..."
python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.first_training",
    "--min-syn", "$MinSyn",
    "--input-per-side", "$InputPerSide",
    "--output-size", "$OutputSize",
    "--decoder-epochs", "$DecoderEpochs",
    "--brain-trials", "$BrainTrials",
    "--validation-trials", "$ValidationTrials",
    "--duration-ms", "$DurationMs",
    "--plastic-fraction", "$PlasticFraction",
    "--seed", "$Seed"
)
if (-not $NoDownload) {
    $argsList += "--download"
}

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MaleCNS training failed with exit code $LASTEXITCODE"
}

if (-not $NoPush) {
    if (Test-Path ".\results\latest_malecns_training.json") {
        git add .\results\latest_malecns_training.json
        git diff --cached --quiet
        if ($LASTEXITCODE -ne 0) {
            git commit -m "Update latest MaleCNS training result"
            if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
            git push
            if ($LASTEXITCODE -ne 0) { throw "git push failed" }
        } else {
            Write-Host "Training result unchanged; nothing to push."
        }
    }
}

Write-Host "=== MaleCNS training complete ==="
Write-Host "Summary: results\latest_malecns_training.json"
Write-Host "Checkpoint: checkpoints\latest_malecns_training.npz"
