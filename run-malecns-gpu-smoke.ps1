param(
  [switch]$Download,
  [int]$DurationMs = 20,
  [int]$MinSyn = 5,
  [int]$StimulusRateHz = 300
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
  $python = 'python'
}
$dataDir = Join-Path $root 'data\malecns_v1'
$args = @('-u', '-m', 'drosomath.malecns.gpu', '--data-dir', $dataDir,
  '--duration-ms', $DurationMs, '--min-syn', $MinSyn,
  '--stimulus-rate-hz', $StimulusRateHz)
if ($Download) { $args += '--download' }
Push-Location $root
try {
  & $python @args
  if ($LASTEXITCODE -ne 0) { throw "GPU smoke test failed with exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}
