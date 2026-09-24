param(
    [int]$MinSyn = 5,
    [int]$StageTrials = 256,
    [int]$DecoderEpochs = 8,
    [int]$ValidationTrials = 8,
    [int]$CheckpointEvery = 64,
    [int]$Seed = 7,
    [int]$Port = 8771,
    [switch]$Fresh,
    [switch]$NoDownload,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath MaleCNS v1 LIVE curriculum ==="
python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.live_curriculum",
    "--min-syn", "$MinSyn",
    "--stage-trials", "$StageTrials",
    "--decoder-epochs", "$DecoderEpochs",
    "--validation-trials", "$ValidationTrials",
    "--checkpoint-every", "$CheckpointEvery",
    "--seed", "$Seed",
    "--port", "$Port"
)
if (-not $NoDownload) { $argsList += "--download" }
if ($Fresh) { $argsList += "--fresh" }
if ($NoBrowser) { $argsList += "--no-browser" }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "MaleCNS v1 live curriculum failed with exit code $LASTEXITCODE" }
