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
    [int]$Port = 8770,
    [switch]$NoDownload,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath MaleCNS Live Training ==="
Write-Host "Installing MaleCNS dependencies..."
python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.live_training",
    "--min-syn", "$MinSyn",
    "--input-per-side", "$InputPerSide",
    "--output-size", "$OutputSize",
    "--decoder-epochs", "$DecoderEpochs",
    "--brain-trials", "$BrainTrials",
    "--validation-trials", "$ValidationTrials",
    "--duration-ms", "$DurationMs",
    "--plastic-fraction", "$PlasticFraction",
    "--seed", "$Seed",
    "--port", "$Port"
)
if (-not $NoDownload) { $argsList += "--download" }
if ($NoBrowser) { $argsList += "--no-browser" }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "MaleCNS live training failed with exit code $LASTEXITCODE" }
