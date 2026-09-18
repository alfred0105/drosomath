param(
  [switch]$Download,
  [int]$Trials = 20000,
  [int]$MinTrialsPerKey = 5,
  [double]$TargetAccuracy = 0.80,
  [int]$CoverageInterval = 4,
  [double]$HardMiningFloor = 0.10,
  [double]$HardMiningPower = 2.0,
  [double]$ClickPenalty = 1.20,
  [double]$LowPeakClickPenaltyScale = 0.0,
  [double]$ClickTeacherLearningRate = 0.08,
  [double]$ClickTeacherCreditFloor = 0.05,
  [double]$PeakRegressionTriggerHz = 1.5,
  [double]$PeakRegressionFixedPenalty = 0.80,
  [double]$PeakRegressionEscalation = 0.50,
  [double]$PeakRegressionMaxPenalty = 2.00,
  [double]$ClickGateThresholdHz = 9.0,
  [int]$ClickIntegrationWindows = 1,
  [int]$ClickEvidenceWindows = 5,
  [double]$ClickMarginTargetHz = 15.0,
  [double]$ClickMarginRewardScale = 0.50,
  [double]$DistancePenaltyScale = 1.20,
  [double]$CorrectDistancePenaltyScale = 0.20,
  [double]$ConsecutiveCorrectBonus = 0.25,
  [double]$MaxConsecutiveBonus = 1.00,
  [double]$DurationMs = 100,
  [double]$ControlWindowMs = 20,
  [int]$MaxControlWindows = 30,
  [int]$StimulusRateHz = 205,
  [int]$MotorPopulationSize = 64,
  [ValidateSet('one_arm_fan', 'one_arm_circle', 'one_arm_circular', 'four_arm_grid')]
  [string]$BodyMode = 'one_arm_fan',
  [ValidateSet('click_gate', 'click_accuracy')]
  [string]$CurriculumStage = 'click_accuracy',
  [int]$CheckpointEvery = 32,
  [double]$DashboardUpdateIntervalSeconds = 0.5,
  [switch]$ProfileTiming,
  [switch]$Resume,
  [int]$Seed = 7,
  [switch]$NoPush,
  [switch]$Open
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
$dataDir = Join-Path $root 'data\malecns_v1'
$liveHtmlPath = Join-Path $root 'results\latest_malecns_keyboard_matching.html'
Write-Host "=== DrosoMath virtual keyboard matching ==="
Write-Host "Body: $BodyMode | Stage: $CurriculumStage | Korean jamo + English A-Z + O/X + 0-9 (60 physical keys)"
if ($Trials -eq 0) { $trialText = 'unlimited' } else { $trialText = [string]$Trials }
Write-Host "Training trials: $trialText | readiness: each key's recent 20 trials must be 20/20 correct | wrong/no click base: -$ClickPenalty"
Write-Host "Low peak click: global punishment disabled; active excitatory click inputs get targeted teaching (rate: $ClickTeacherLearningRate)"
Write-Host "Per-key peak regression: a drop larger than $PeakRegressionTriggerHz Hz gets a fixed local teacher correction (-$PeakRegressionFixedPenalty), escalating by $PeakRegressionEscalation up to -$PeakRegressionMaxPenalty"
Write-Host "Distance shaping: failure scale $DistancePenaltyScale | correct edge scale $CorrectDistancePenaltyScale"
Write-Host "Click curriculum: arm position locked; 20ms peak click threshold: $ClickGateThresholdHz Hz"
Write-Host "100ms click average is telemetry only ($ClickEvidenceWindows windows); strong peak clicks earn up to +$ClickMarginRewardScale at $ClickMarginTargetHz"
Write-Host "Per-key streak bonus: +$ConsecutiveCorrectBonus per extra correct, capped at +$MaxConsecutiveBonus"
Write-Host "Exam: randomized 60 keys x 20 = 1200 trials, learning frozen; pass only at 1200/1200"
Write-Host "Training priority: balanced warmup first, then coverage plus probabilistic low-accuracy mining"
Write-Host "Post-warmup scheduler: one shuffled all-key coverage trial every $CoverageInterval trials; remaining trials use probabilistic hard-example mining"
Write-Host "The Python process will update the current trial live below."
if ($Open -and (Test-Path -LiteralPath $liveHtmlPath)) {
  Write-Host "Opening live 2D dashboard: $liveHtmlPath"
  Start-Process -FilePath $liveHtmlPath
}
$args = @('-u', '-m', 'drosomath.malecns.keyboard_learning', '--data-dir', $dataDir,
  '--trials', $Trials, '--min-trials-per-key', $MinTrialsPerKey, '--target-accuracy', $TargetAccuracy,
  '--coverage-interval', $CoverageInterval, '--hard-mining-floor', $HardMiningFloor, '--hard-mining-power', $HardMiningPower,
  '--click-penalty', $ClickPenalty, '--distance-penalty-scale', $DistancePenaltyScale,
  '--low-peak-click-penalty-scale', $LowPeakClickPenaltyScale,
  '--click-teacher-learning-rate', $ClickTeacherLearningRate,
  '--click-teacher-credit-floor', $ClickTeacherCreditFloor,
  '--peak-regression-trigger-hz', $PeakRegressionTriggerHz,
  '--peak-regression-fixed-penalty', $PeakRegressionFixedPenalty,
  '--peak-regression-escalation', $PeakRegressionEscalation,
  '--peak-regression-max-penalty', $PeakRegressionMaxPenalty,
  '--click-gate-threshold-hz', $ClickGateThresholdHz,
  '--click-integration-windows', $ClickIntegrationWindows,
  '--click-evidence-windows', $ClickEvidenceWindows,
  '--click-margin-target-hz', $ClickMarginTargetHz,
  '--click-margin-reward-scale', $ClickMarginRewardScale,
  '--correct-distance-penalty-scale', $CorrectDistancePenaltyScale,
  '--consecutive-correct-bonus', $ConsecutiveCorrectBonus, '--max-consecutive-bonus', $MaxConsecutiveBonus,
  '--duration-ms', $DurationMs,
  '--control-window-ms', $ControlWindowMs, '--max-control-windows', $MaxControlWindows,
  '--stimulus-rate-hz', $StimulusRateHz, '--motor-population-size', $MotorPopulationSize,
  '--body-mode', $BodyMode,
  '--curriculum-stage', $CurriculumStage,
  '--checkpoint-every', $CheckpointEvery, '--seed', $Seed)
if ($ProfileTiming) { $args += '--profile-timing' }
$args += @('--dashboard-update-interval-seconds', $DashboardUpdateIntervalSeconds)
if ($Download) { $args += '--download' }
if ($Resume) { $args += '--resume' }
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
} finally {
  Pop-Location
}
