param(
    [int]$MinSyn = 5,
    [int]$FixedTrials = 2048,
    [int]$VariedTrials = 3072,
    [int]$DropoutTrials = 3072,
    [int]$ValidationTrials = 64,
    [double]$Mastery = 0.85,
    [int]$MaxAttempts = 3,
    [int]$CheckpointEvery = 512,
    [int]$StructuralEvery = 64,
    [int]$Seed = 7,
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
Write-Host "Quantity 2/3 remains LOCKED until all presence phases pass."

if ($NoStructural) {
    Write-Host "structural plasticity: OFF (ablation mode)"
    Remove-Item Env:DROSOMATH_STRUCTURAL -ErrorAction SilentlyContinue
} else {
    $env:DROSOMATH_STRUCTURAL = "1"
    $env:DROSOMATH_STRUCTURAL_INTERVAL = "$StructuralEvery"
    Write-Host "structural plasticity: ON | rewiring every $StructuralEvery teaching episodes"
}

python -m pip install -e ".[malecns]"
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

$argsList = @(
    "-m", "drosomath.malecns.presence_graded",
    "--min-syn", "$MinSyn",
    "--fixed-trials", "$FixedTrials",
    "--varied-trials", "$VariedTrials",
    "--dropout-trials", "$DropoutTrials",
    "--validation-trials", "$ValidationTrials",
    "--mastery", "$Mastery",
    "--max-attempts", "$MaxAttempts",
    "--checkpoint-every", "$CheckpointEvery",
    "--seed", "$Seed"
)
if (-not $NoDownload) { $argsList += "--download" }

try {
    python @argsList
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
        git push
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
