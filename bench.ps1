param(
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
$resultPath = Join-Path $resultDir "latest_ablation.json"
$resultRelative = "results/latest_ablation.json"

New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$output = & $python -m drosomath.ablation_demo
$exitCode = $LASTEXITCODE
$stopwatch.Stop()

if ($exitCode -ne 0) {
    throw "Ablation benchmark failed with exit code $exitCode"
}

$json = ($output -join [Environment]::NewLine).Trim()
try {
    $parsed = $json | ConvertFrom-Json
} catch {
    throw "Benchmark output was not valid JSON."
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    $resultPath,
    $json + [Environment]::NewLine,
    $utf8NoBom
)

Write-Host ""
Write-Host "=== DrosoMath ablation benchmark ==="
Write-Host ("Elapsed: {0:N3}s" -f $stopwatch.Elapsed.TotalSeconds)
foreach ($variant in $parsed.variants) {
    $line = "{0,-42} retention={1:N3} forgetting={2:N3}" -f $variant.name, [double]$variant.mean_retention, [double]$variant.mean_forgetting
    Write-Host $line
}
Write-Host "Saved: $resultRelative"

if ($NoPush) {
    Write-Host "Push skipped (-NoPush)."
    exit 0
}

& git add -- $resultRelative
if ($LASTEXITCODE -ne 0) {
    throw "git add failed"
}

& git diff --cached --quiet -- $resultRelative
$hasChanges = $LASTEXITCODE -ne 0

if ($hasChanges) {
    & git commit -m "Update latest ablation result" -- $resultRelative
    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed"
    }

    & git push origin HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed"
    }
    Write-Host "Result pushed. I can read it from GitHub now."
} else {
    Write-Host "Result unchanged; nothing to push."
}
