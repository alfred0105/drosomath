param(
    [double]$DurationMs = 20.0,
    [double]$StimulusRateHz = 100.0,
    [int]$StimulusCount = 3,
    [int]$MinConnectionSynapses = 1,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($env:VIRTUAL_ENV) {
    $windowsPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    $unixPython = Join-Path $env:VIRTUAL_ENV "bin/python"
    if (Test-Path $windowsPython) { $python = $windowsPython }
    elseif (Test-Path $unixPython) { $python = $unixPython }
    else { $python = "python" }
} else {
    $python = "python"
}

$dataDir = Join-Path $PSScriptRoot "data\flywire_v783"
$resultDir = Join-Path $PSScriptRoot "results"
$resultPath = Join-Path $resultDir "latest_flywire_brain.json"
$resultRelative = "results/latest_flywire_brain.json"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

$files = @(
    @{
        Name = "Completeness_783.csv"
        Url = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv"
    },
    @{
        Name = "Connectivity_783.parquet"
        Url = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet"
    }
)

function Download-File([string]$Url, [string]$Destination) {
    if (Test-Path $Destination) { return }
    Write-Host ("Downloading real FlyWire v783 data: {0}" -f (Split-Path $Destination -Leaf))
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --retry 3 --progress-bar -o $Destination $Url
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
    }
}

foreach ($file in $files) {
    Download-File $file.Url (Join-Path $dataDir $file.Name)
}

& $python -c "import numpy, pyarrow" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing FlyWire runtime dependencies (NumPy + PyArrow)..."
    & $python -m pip install -e ".[flywire]"
    if ($LASTEXITCODE -ne 0) { throw "FlyWire dependency installation failed" }
}

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$output = & $python -m drosomath.flywire_real `
    --data-dir $dataDir `
    --duration-ms $DurationMs `
    --stimulus-rate-hz $StimulusRateHz `
    --auto-stimuli $StimulusCount `
    --min-connection-synapses $MinConnectionSynapses
$exitCode = $LASTEXITCODE
$stopwatch.Stop()
if ($exitCode -ne 0) { throw "Real FlyWire brain simulation failed with exit code $exitCode" }

$json = ($output -join [Environment]::NewLine).Trim()
try { $parsed = $json | ConvertFrom-Json }
catch { throw "FlyWire runner output was not valid JSON." }

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($resultPath, $json + [Environment]::NewLine, $utf8NoBom)

Write-Host ""
Write-Host "=== DrosoMath REAL FlyWire v783 brain ==="
Write-Host ("Neurons: {0:N0}" -f [double]$parsed.neuron_count)
Write-Host ("Loaded edges: {0:N0}" -f [double]$parsed.edge_count)
Write-Host ("Loaded |synapse count|: {0:N0}" -f [double]$parsed.absolute_synapse_count_in_loaded_edges)
Write-Host ("Active neurons: {0:N0}" -f [double]$parsed.active_neurons)
Write-Host ("Total spikes: {0:N0}" -f [double]$parsed.total_spikes)
Write-Host ("Elapsed: {0:N3}s" -f $stopwatch.Elapsed.TotalSeconds)
Write-Host ("Saved: {0}" -f $resultRelative)

if ($NoPush) {
    Write-Host "Push skipped (-NoPush)."
    exit 0
}

& git add -- $resultRelative
if ($LASTEXITCODE -ne 0) { throw "git add failed" }
& git diff --cached --quiet -- $resultRelative
$hasChanges = $LASTEXITCODE -ne 0
if ($hasChanges) {
    & git commit -m "Update latest real FlyWire brain result" -- $resultRelative
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
    & git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    Write-Host "Result pushed. ChatGPT can read it from GitHub."
} else {
    Write-Host "Result unchanged; nothing to push."
}
