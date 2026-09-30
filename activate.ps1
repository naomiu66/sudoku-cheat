if ($MyInvocation.InvocationName -ne '.') {
    Write-Error 'Run this script with: . .\activate.ps1'
    return
}

$activateScript = Join-Path $PSScriptRoot '.venv\Scripts\Activate.ps1'

if (-not (Test-Path $activateScript)) {
    throw "Virtual environment not found at '$activateScript'. Create it with: py -3.12 -m venv .venv"
}

. $activateScript