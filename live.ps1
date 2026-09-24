param(
    [int]$Port = 8765,
    [double]$Hz = 8.0,
    [switch]$NoBrowser,
    [switch]$Single
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($env:VIRTUAL_ENV) {
    $windowsPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    $unixPython = Join-Path $env:VIRTUAL_ENV "bin/python"
    if (Test-Path $windowsPython) {
        $python = $windowsPython
    } elseif (Test-Path $unixPython) {
        $python = $unixPython
    } else {
        $python = "python"
    }
} else {
    $python = "python"
}

$module = "drosomath.dual_live"
$label = "DrosoMath Dual Live"
if ($Single) {
    $module = "drosomath.live"
    $label = "DrosoMath Single Live"
}

$argsList = @("-m", $module, "--port", "$Port", "--hz", "$Hz")
if ($NoBrowser) {
    $argsList += "--no-browser"
}

Write-Host "Starting $label on http://127.0.0.1:$Port/"
Write-Host "Press Ctrl+C to stop."
& $python @argsList
exit $LASTEXITCODE
