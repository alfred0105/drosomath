param(
    [int]$MinSyn = 5,
    [int]$FixedTrials = 2048,
    [int]$VariedTrials = 3072,
    [int]$DropoutTrials = 3072,
    [double]$DurationMs = 100.0,
    [double]$StimulusRateHz = 205.0,
    [int]$CalibrationTrials = 32,
    [double]$CalibrationMinSeparationHz = 0.10,
    [int]$RepresentationProbeSize = 512,
    [int]$RepresentationTrials = 16,
    [int]$ValidationTrials = 64,
    [double]$Mastery = 0.85,
    [int]$MaxAttempts = 3,
    [int]$CheckpointEvery = 512,
    [int]$StructuralEvery = 64,
    [int]$Seed = 7,
    [switch]$Structural,
    [switch]$NoStructural,
    [switch]$NoDownload,
    [switch]$NoOpen,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== DrosoMath Graded Presence Mastery ==="
Write-Host "Curriculum: fixed presence -> varied position -> dropout robustness"
Write-Host "Output: one fixed descending-neuron population with graded firing-rate levels"
Write-Host "External trainable decoder: NONE"
Write-Host "Trials: $FixedTrials + $VariedTrials + $DropoutTrials per complete education pass"
Write-Host "Mastery threshold: $Mastery | max attempts/phase: $MaxAttempts (1 + up to 2 retraining cycles by default)"
Write-Host "Rate calibration: $CalibrationTrials samples/class, minimum separation $CalibrationMinSeparationHz Hz"
Write-Host "Quantity 2/3 remains LOCKED until all presence phases pass."

if ($Structural -and -not $NoStructural) {
    $env:DROSOMATH_STRUCTURAL = "1"
    $env:DROSOMATH_STRUCTURAL_INTERVAL = "$StructuralEvery"
    Write-Host "structural plasticity: ON | rewiring every $StructuralEvery teaching episodes"
} else {
    Remove-Item Env:DROSOMATH_STRUCTURAL -ErrorAction SilentlyContinue
    Remove-Item Env:DROSOMATH_STRUCTURAL_INTERVAL -ErrorAction SilentlyContinue
    Write-Host "structural plasticity: OFF (reference functional-plasticity run)"
}

$python = "python"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) { $python = $venvPython }
& $python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.presence_graded",
    "--data-dir", (Join-Path $PSScriptRoot "data\malecns_v1"),
    "--min-syn", "$MinSyn",
    "--fixed-trials", "$FixedTrials",
    "--varied-trials", "$VariedTrials",
    "--dropout-trials", "$DropoutTrials",
    "--duration-ms", "$DurationMs",
    "--stimulus-rate-hz", "$StimulusRateHz",
    "--calibration-trials", "$CalibrationTrials",
    "--calibration-min-separation-hz", "$CalibrationMinSeparationHz",
    "--representation-probe-size", "$RepresentationProbeSize",
    "--representation-trials", "$RepresentationTrials",
    "--validation-trials", "$ValidationTrials",
    "--mastery", "$Mastery",
    "--max-attempts", "$MaxAttempts",
    "--checkpoint-every", "$CheckpointEvery",
    "--seed", "$Seed"
)
if (-not $NoDownload) { $argsList += "--download" }

try {
    & $python @argsList
    if ($LASTEXITCODE -ne 0) { throw "graded presence mastery failed with exit code $LASTEXITCODE" }
} finally {
    Remove-Item Env:DROSOMATH_STRUCTURAL -ErrorAction SilentlyContinue
    Remove-Item Env:DROSOMATH_STRUCTURAL_INTERVAL -ErrorAction SilentlyContinue
}

if (-not $NoPush) {
    git add .\results\latest_malecns_presence_graded.json .\results\latest_malecns_presence_graded.html
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        git commit -m "Update graded MaleCNS presence mastery result"
        if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
        $currentBranch = (& git branch --show-current).Trim()
        if ([string]::IsNullOrWhiteSpace($currentBranch)) { throw "detached HEAD; cannot auto-push result" }
        git push origin "HEAD:$currentBranch"
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
    } else {
        Write-Host "Presence mastery results unchanged; nothing to push."
    }
}

if (-not $NoOpen) {
    Start-Process (Resolve-Path ".\results\latest_malecns_presence_graded.html")
}

Write-Host "=== Graded presence mastery complete ==="
Write-Host "Result: results\latest_malecns_presence_graded.json"
Write-Host "Dashboard: results\latest_malecns_presence_graded.html"
