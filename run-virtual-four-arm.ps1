param([int]$Steps = 3)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
Push-Location $root
try {
  & $python -u -m drosomath.malecns.virtual_body --steps $Steps
  if ($LASTEXITCODE -ne 0) { throw "virtual four-arm smoke test failed with exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}
