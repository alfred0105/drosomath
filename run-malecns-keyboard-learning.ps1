param(
  [switch]$Download,
  [int]$Trials = 20000,
  [int]$MinTrialsPerKey = 32,
  [double]$TargetAccuracy = 0.80,
  [double]$DurationMs = 100,
  [double]$ControlWindowMs = 20,
  [int]$MaxControlWindows = 30,
  [int]$StimulusRateHz = 205,
  [int]$MotorPopulationSize = 32,
  [int]$CheckpointEvery = 32,
  [int]$Seed = 7,
  [switch]$NoPush,
  [switch]$Open
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
$dataDir = Join-Path $root 'data\malecns_v1'
Write-Host "=== DrosoMath virtual keyboard matching ==="
Write-Host "Keys: Korean jamo + English A-Z + O/X + 0-9 (60 physical keys)"
Write-Host "Max trials: $Trials | target: $($TargetAccuracy * 100)% per key | minimum observations: $MinTrialsPerKey"
Write-Host "The Python process will update the current trial live below."
$args = @('-u', '-m', 'drosomath.malecns.keyboard_learning', '--data-dir', $dataDir,
  '--trials', $Trials, '--min-trials-per-key', $MinTrialsPerKey, '--target-accuracy', $TargetAccuracy,
  '--duration-ms', $DurationMs,
  '--control-window-ms', $ControlWindowMs, '--max-control-windows', $MaxControlWindows,
  '--stimulus-rate-hz', $StimulusRateHz, '--motor-population-size', $MotorPopulationSize,
  '--checkpoint-every', $CheckpointEvery, '--seed', $Seed)
if ($Download) { $args += '--download' }
Push-Location $root
try {
  & $python @args
  if ($LASTEXITCODE -ne 0) { throw "keyboard learning failed with exit code $LASTEXITCODE" }
  if (-not $NoPush) {
    git add .\results\latest_malecns_keyboard_matching.json .\results\latest_malecns_keyboard_matching.html
    if ($LASTEXITCODE -eq 0) {
      git diff --cached --quiet
      if ($LASTEXITCODE -ne 0) {
        git commit -m 'Update virtual keyboard matching result'
        $branch = (& git branch --show-current).Trim()
        git push origin "HEAD:$branch"
      }
    }
  }
  if ($Open) {
    Start-Process (Resolve-Path ".\results\latest_malecns_keyboard_matching.html")
  }
} finally {
  Pop-Location
}
