param(
    [int]$MinSyn = 5,
    [int]$InputPerSide = 32,
    [int]$OutputSize = 512,
    [int]$DecoderEpochs = 8,
    [int]$BrainTrials = 192,
    [int]$ValidationTrials = 16,
    [double]$DurationMs = 20.0,
    [double]$PlasticFraction = 0.05,
    [int]$Seed = 1,
    [switch]$NoDownload,
    [switch]$NoOpen,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath: adaptive MaleCNS learning ==="
Write-Host "Installing MaleCNS dependencies..."
python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.adaptive_training",
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
if (-not $NoDownload) { $argsList += "--download" }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "adaptive MaleCNS training failed with exit code $LASTEXITCODE" }

python -m drosomath.malecns.visualize_training
if ($LASTEXITCODE -ne 0) { throw "visualization generation failed" }

if (-not $NoOpen -and (Test-Path ".\results\latest_malecns_adaptive_training.html")) {
    Start-Process ".\results\latest_malecns_adaptive_training.html"
}

if (-not $NoPush) {
    git add .\results\latest_malecns_adaptive_training.json .\results\latest_malecns_adaptive_training.html
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Update latest adaptive MaleCNS training result"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        git push
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    } else {
        Write-Host "Adaptive training result unchanged; nothing to push."
    }
}

Write-Host "=== Adaptive MaleCNS training complete ==="
Write-Host "Summary:       results\latest_malecns_adaptive_training.json"
Write-Host "Visualization: results\latest_malecns_adaptive_training.html"
Write-Host "Checkpoint:    checkpoints\latest_malecns_adaptive_training.npz"
