param(
    [switch]$NoPull,
    [switch]$NoBrowser,
    [switch]$NoPublisher,
    [switch]$SelfTest
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$runtimeDir = Join-Path $repoRoot ".runtime"
$processFile = Join-Path $runtimeDir "processes.json"
$depsFile = Join-Path $runtimeDir "dependency_hashes.json"

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
Set-Location $repoRoot

function Stop-PreviousDrosoMathProcesses {
    if (-not (Test-Path $processFile)) {
        return
    }

    try {
        $state = Get-Content $processFile -Raw | ConvertFrom-Json
        foreach ($entry in @($state.processes)) {
            $pidValue = [int]$entry.pid
            $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "Stopping previous $($entry.name) (PID $pidValue)..."
                & taskkill.exe /PID $pidValue /T /F 2>$null | Out-Null
            }
        }
    }
    catch {
        Write-Warning "Could not fully read/stop the previous DrosoMath process state: $_"
    }
    finally {
        Remove-Item $processFile -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-GitChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$GitArgs
    )

    & git @GitArgs
    $gitCode = $LASTEXITCODE
    if ($gitCode -ne 0) {
        throw "git $($GitArgs -join ' ') failed with exit code $gitCode"
    }
}

function Invoke-GitPushBestEffort {
    # Windows PowerShell 5.1 can turn redirected native stderr into a
    # NativeCommandError even for Git's harmless "Everything up-to-date" text.
    # Do not redirect stderr here; use only git's process exit code.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & git push
        $pushCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($pushCode -ne 0) {
        Write-Warning "git push returned exit code $pushCode. The launcher will still continue after syncing from GitHub."
    }
}

function Get-FileSha256([string]$Path) {
    if (-not (Test-Path $Path)) {
        return $null
    }
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash
}

function Read-DependencyState {
    if (-not (Test-Path $depsFile)) {
        return [pscustomobject]@{ backend = $null; frontend = $null }
    }
    try {
        return Get-Content $depsFile -Raw | ConvertFrom-Json
    }
    catch {
        return [pscustomobject]@{ backend = $null; frontend = $null }
    }
}

function Save-DependencyState([string]$BackendHash, [string]$FrontendHash) {
    @{
        backend = $BackendHash
        frontend = $FrontendHash
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -Path $depsFile -Encoding UTF8
}

function Start-DrosoMathWindow {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [string]$Command
    )

    $escapedName = $Name.Replace("'", "''")
    $wrapped = "`$host.UI.RawUI.WindowTitle = '$escapedName'; $Command"
    $process = Start-Process powershell.exe `
        -WorkingDirectory $WorkingDirectory `
        -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $wrapped) `
        -PassThru

    return [pscustomobject]@{
        name = $Name
        pid = $process.Id
    }
}

if ($SelfTest) {
    Write-Host "Running DrosoMath launcher self-test..."
    Invoke-GitChecked -GitArgs @("--version")
    Write-Host "Launcher self-test passed."
    exit 0
}

Write-Host ""
Write-Host "=== DrosoMath one-command launcher ==="
Write-Host "Repository: $repoRoot"
Write-Host ""

# Stop only processes launched by an earlier run.ps1 invocation.
Stop-PreviousDrosoMathProcesses

if (-not $NoPull) {
    Write-Host "[1/4] Syncing GitHub..."

    # Pull/rebase first. This safely keeps local run commits while moving them
    # on top of any code changes pushed from ChatGPT/GitHub.
    Invoke-GitChecked -GitArgs @("pull", "--rebase", "--autostash")

    # If the live publisher left local commits ahead of origin, publish them now.
    # A failed push is non-fatal for launching the local experiment.
    Invoke-GitPushBestEffort
}
else {
    Write-Host "[1/4] Git pull skipped (-NoPull)."
}

Write-Host "[2/4] Checking dependencies..."
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"
$venvPython = Join-Path $backendDir ".venv\Scripts\python.exe"
$requirements = Join-Path $backendDir "requirements.txt"
$frontendManifest = if (Test-Path (Join-Path $frontendDir "package-lock.json")) {
    Join-Path $frontendDir "package-lock.json"
} else {
    Join-Path $frontendDir "package.json"
}

$dependencyState = Read-DependencyState
$backendHash = Get-FileSha256 $requirements
$frontendHash = Get-FileSha256 $frontendManifest

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating backend virtual environment..."
    & python -m venv (Join-Path $backendDir ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create backend virtual environment." }
}

if ($dependencyState.backend -ne $backendHash) {
    Write-Host "Installing/updating backend dependencies..."
    & $venvPython -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }
}

if (-not (Test-Path (Join-Path $frontendDir "node_modules")) -or $dependencyState.frontend -ne $frontendHash) {
    Write-Host "Installing/updating frontend dependencies..."
    Push-Location $frontendDir
    try {
        & npm install
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
    }
    finally {
        Pop-Location
    }
}

Save-DependencyState -BackendHash $backendHash -FrontendHash $frontendHash

Write-Host "[3/4] Starting services..."
$processes = @()
$processes += Start-DrosoMathWindow `
    -Name "DrosoMath Backend" `
    -WorkingDirectory $backendDir `
    -Command "& '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --reload --port 8000"

$processes += Start-DrosoMathWindow `
    -Name "DrosoMath Frontend" `
    -WorkingDirectory $frontendDir `
    -Command "npm run dev"

if (-not $NoPublisher) {
    $publisherScript = Join-Path $repoRoot "scripts\auto_publish_live_run.ps1"
    $processes += Start-DrosoMathWindow `
        -Name "DrosoMath GitHub Publisher" `
        -WorkingDirectory $repoRoot `
        -Command "& '$publisherScript'"
}

@{
    launched_at = (Get-Date).ToString("o")
    processes = $processes
} | ConvertTo-Json -Depth 4 | Set-Content -Path $processFile -Encoding UTF8

Write-Host "[4/4] Waiting for backend/frontend..."
$backendReady = $false
for ($attempt = 1; $attempt -le 40; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 1
        if ($health.status -eq "ok") {
            $backendReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

if ($backendReady) {
    Write-Host "Backend ready."
}
else {
    Write-Warning "Backend did not answer /health yet. Check the 'DrosoMath Backend' window."
}

if (-not $NoBrowser) {
    Start-Sleep -Seconds 1
    Start-Process "http://localhost:5173"
}

$branch = (& git branch --show-current).Trim()
$commit = (& git rev-parse --short HEAD).Trim()
Write-Host ""
Write-Host "DrosoMath launched."
Write-Host "  branch : $branch"
Write-Host "  commit : $commit"
Write-Host "  UI     : http://localhost:5173"
Write-Host ""
Write-Host "From now on, update + restart with the same command:"
Write-Host "  .\run.ps1"
