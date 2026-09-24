param(
    [int]$Runs = 100,
    [int]$SeedStart = 0,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($env:VIRTUAL_ENV) {
    $windowsPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    $unixPython = Join-Path $env:VIRTUAL_ENV "bin/python"
    if (Test-Path $windowsPython) {
        $python = $windowsPython
    } elseif (Test-Path $unixPython) {
        $python = $unixPython
    } else {
        $python = "python"
    }
} else {
    $python = "python"
}

$resultDir = Join-Path $PSScriptRoot "results"
$resultPath = Join-Path $resultDir "latest_interference.json"
$resultRelative = "results/latest_interference.json"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$output = & $python -m drosomath.interference_v2 --runs $Runs --seed-start $SeedStart
$exitCode = $LASTEXITCODE
$stopwatch.Stop()
if ($exitCode -ne 0) {
    throw "Interference benchmark failed with exit code $exitCode"
}

$json = ($output -join [Environment]::NewLine).Trim()
try {
    $parsed = $json | ConvertFrom-Json
} catch {
    throw "Interference benchmark output was not valid JSON."
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($resultPath, $json + [Environment]::NewLine, $utf8NoBom)

Write-Host ""
Write-Host "=== DrosoMath catastrophic interference v2 ==="
Write-Host ("Model runs: {0} ({1} variants x {2} seeds)" -f $parsed.total_model_runs, $parsed.variants, $parsed.runs_per_variant)
Write-Host ("Elapsed: {0:N3}s" -f $stopwatch.Elapsed.TotalSeconds)
foreach ($variant in $parsed.results) {
    $line = "{0,-34} acquire={1:N3} final={2:N3} retain={3:N3} forget={4:N3} cat={5:P1}" -f $variant.name, [double]$variant.acquisition_accuracy.mean, [double]$variant.final_accuracy.mean, [double]$variant.learned_task_retention.mean, [double]$variant.mean_forgetting.mean, [double]$variant.catastrophic_forgetting_rate
    Write-Host $line
}
Write-Host "Saved: $resultRelative"

if ($NoPush) {
    Write-Host "Push skipped (-NoPush)."
    exit 0
}

& git add -- $resultRelative
if ($LASTEXITCODE -ne 0) { throw "git add failed" }
& git diff --cached --quiet -- $resultRelative
$hasChanges = $LASTEXITCODE -ne 0
if ($hasChanges) {
    & git commit -m "Update latest interference result" -- $resultRelative
    if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
    & git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    Write-Host "Result pushed. It can now be inspected from GitHub."
} else {
    Write-Host "Result unchanged; nothing to push."
}
