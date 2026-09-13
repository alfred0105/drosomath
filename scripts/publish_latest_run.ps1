$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runsDir = Join-Path $repoRoot "runs"

if (-not (Test-Path $runsDir)) {
    throw "runs directory not found: $runsDir"
}

$latest = Get-ChildItem $runsDir -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latest) {
    throw "No run directories found under $runsDir"
}

Push-Location $repoRoot
try {
    Write-Host "Publishing run: $($latest.Name)"
    git add -- "runs/$($latest.Name)"

    $staged = git diff --cached --name-only
    if (-not $staged) {
        Write-Host "Nothing new to commit for the latest run."
        exit 0
    }

    git commit -m "Add experiment run $($latest.Name)"
    git push
    Write-Host "Published. I can now inspect this run from GitHub."
}
finally {
    Pop-Location
}
