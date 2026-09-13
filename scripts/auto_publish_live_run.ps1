param(
    [int]$IntervalSeconds = 60,
    [int]$MetricsEvery = 10
)

$ErrorActionPreference = "Stop"

if ($IntervalSeconds -lt 15) {
    throw "IntervalSeconds must be at least 15 seconds."
}

if ($MetricsEvery -lt 1) {
    throw "MetricsEvery must be at least 1."
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$runsDir = Join-Path $repoRoot "runs"

if (-not (Test-Path $runsDir)) {
    throw "runs directory not found: $runsDir"
}

function Get-LatestRunDirectory {
    Get-ChildItem $runsDir -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Invoke-Git([string[]]$Args, [switch]$AllowFailure) {
    & git @Args
    $code = $LASTEXITCODE
    if ($code -ne 0 -and -not $AllowFailure) {
        throw "git $($Args -join ' ') failed with exit code $code"
    }
    return $code
}

function Publish-LiveSnapshot([int]$Cycle) {
    $latest = Get-LatestRunDirectory
    if (-not $latest) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Waiting for a run directory..."
        return
    }

    $relativeRun = "runs/$($latest.Name)"
    $paths = @(
        "$relativeRun/config.json",
        "$relativeRun/summary.json"
    )

    $includeMetrics = ($Cycle % $MetricsEvery) -eq 0
    if ($includeMetrics) {
        $paths += "$relativeRun/metrics.csv"
    }

    Push-Location $repoRoot
    try {
        $existing = @()
        foreach ($path in $paths) {
            if (Test-Path (Join-Path $repoRoot $path)) {
                $existing += $path
            }
        }

        if (-not $existing) {
            return
        }

        Invoke-Git -Args (@("add", "--") + $existing) | Out-Null
        $staged = & git diff --cached --name-only -- $existing
        if (-not $staged) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] No new live snapshot changes."
            return
        }

        $kind = if ($includeMetrics) { "summary + metrics checkpoint" } else { "summary" }
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Invoke-Git -Args @("commit", "-m", "Live run snapshot $($latest.Name) [$kind] $stamp") | Out-Null

        $pushCode = Invoke-Git -Args @("push") -AllowFailure
        if ($pushCode -eq 0) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Published $kind for $($latest.Name)."
        }
        else {
            Write-Warning "Push failed. The snapshot is committed locally and will be retried on the next cycle. Stop this watcher before git pull/rebase if the branch changed remotely."
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "DrosoMath live GitHub publisher"
Write-Host "  summary interval : $IntervalSeconds sec"
Write-Host "  metrics checkpoint: every $MetricsEvery cycles (~$([math]::Round($IntervalSeconds * $MetricsEvery / 60, 1)) min)"
Write-Host "  Ctrl+C to stop"
Write-Host ""

$cycle = 1
while ($true) {
    try {
        Publish-LiveSnapshot -Cycle $cycle
    }
    catch {
        Write-Warning $_
    }

    $cycle += 1
    Start-Sleep -Seconds $IntervalSeconds
}
