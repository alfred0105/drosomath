param(
    [int]$MinSyn = 5,
    [double]$DurationMs = 10.0,
    [double]$StimulusRateHz = 300.0,
    [double]$PlasticFraction = 0.05,
    [double]$Reward = 1.0,
    [switch]$NoDownload
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath MaleCNS v1.0 ==="
Write-Host "Installing whole-brain dependencies..."
python -m pip install -e ".[malecns]"

$argsList = @(
    "-m", "drosomath.malecns.brain",
    "--min-connection-synapses", "$MinSyn",
    "--duration-ms", "$DurationMs",
    "--stimulus-rate-hz", "$StimulusRateHz",
    "--plastic-fraction", "$PlasticFraction",
    "--reward", "$Reward"
)
if (-not $NoDownload) {
    $argsList += "--download"
}

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MaleCNS run failed with exit code $LASTEXITCODE"
}
