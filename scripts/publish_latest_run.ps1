param(
    [switch]$CompletedOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runsDir = Join-Path $repoRoot "runs"

if (-not (Test-Path $runsDir)) {
    throw "runs directory not found: $runsDir"
}

$runDirs = Get-ChildItem $runsDir -Directory |
    Sort-Object LastWriteTime -Descending

if (-not $runDirs) {
    throw "No run directories found under $runsDir"
}

$latest = $null
if ($CompletedOnly) {
    foreach ($run in $runDirs) {
        $summaryPath = Join-Path $run.FullName "summary.json"
        if (-not (Test-Path $summaryPath)) {
            continue
        }

        try {
            $summary = Get-Content $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            continue
        }

        if ($summary.status -eq "completed" -and $summary.experiment_complete -eq $true) {
            $latest = $run
            break
        }
    }

    if (-not $latest) {
        throw "No completed experiment run found. The newest run may still be running/interrupted."
    }
}
else {
    $latest = $runDirs | Select-Object -First 1
}

Push-Location $repoRoot
try {
    $summaryPath = Join-Path $latest.FullName "summary.json"
    if (Test-Path $summaryPath) {
        try {
            $summary = Get-Content $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            Write-Host "Publishing run: $($latest.Name) · status=$($summary.status) · complete=$($summary.experiment_complete) · experiment=$($summary.experiment)"
        }
        catch {
            Write-Host "Publishing run: $($latest.Name)"
        }
    }
    else {
        Write-Host "Publishing run: $($latest.Name)"
    }

    git add -- "runs/$($latest.Name)"

    $staged = git diff --cached --name-only
    if (-not $staged) {
        Write-Host "Nothing new to commit for the selected run."
        exit 0
    }

    git commit -m "Add experiment run $($latest.Name)"
    git push
    Write-Host "Published. I can now inspect this run from GitHub."
}
finally {
    Pop-Location
}
