param([string]$Prompt = 'O')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
Push-Location $root
try {
  & $python -u -m drosomath.malecns.virtual_keyboard --prompt $Prompt
  if ($LASTEXITCODE -ne 0) { throw "virtual keyboard smoke test failed with exit code $LASTEXITCODE" }
} finally {
  Pop-Location
}
