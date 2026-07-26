param([Parameter(Mandatory = $true)][int]$PipelinePid)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
try {
    Wait-Process -Id $PipelinePid -ErrorAction Stop
} catch {
}
& 'C:\Users\shish\anaconda3\envs\research-gpu\python.exe' 'scripts\repair_and_reaudit.py'
exit $LASTEXITCODE
