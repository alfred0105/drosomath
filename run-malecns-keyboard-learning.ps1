param(
  [switch]$Download,
  [int]$Trials = 128,
  [double]$DurationMs = 100,
  [double]$ControlWindowMs = 20,
  [int]$MaxControlWindows = 30,
  [int]$StimulusRateHz = 205,
  [int]$MotorPopulationSize = 32,
  [int]$CheckpointEvery = 32,
  [int]$Seed = 7,
  [switch]$NoPush
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
$dataDir = Join-Path $root 'data\malecns_v1'
$args = @('-u', '-m', 'drosomath.malecns.keyboard_learning', '--data-dir', $dataDir,
  '--trials', $Trials, '--duration-ms', $DurationMs,
  '--control-window-ms', $ControlWindowMs, '--max-control-windows', $MaxControlWindows,
  '--stimulus-rate-hz', $StimulusRateHz, '--motor-population-size', $MotorPopulationSize,
  '--checkpoint-every', $CheckpointEvery, '--seed', $Seed)
if ($Download) { $args += '--download' }
Push-Location $root
try {
  & $python @args
  if ($LASTEXITCODE -ne 0) { throw "keyboard learning failed with exit code $LASTEXITCODE" }
  if (-not $NoPush) {
    git add .\results\latest_malecns_keyboard_matching.json
    if ($LASTEXITCODE -eq 0) {
      git diff --cached --quiet
      if ($LASTEXITCODE -ne 0) {
        git commit -m 'Update virtual keyboard matching result'
        $branch = (& git branch --show-current).Trim()
        git push origin "HEAD:$branch"
      }
    }
  }
} finally {
  Pop-Location
}
