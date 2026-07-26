param([Parameter(Mandatory = $true)][int]$MatrixPid)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
try {
    Wait-Process -Id $MatrixPid -ErrorAction Stop
} catch {
    # If the process ended between scheduling and Wait-Process, continuation is
    # still safe because every downstream stage is resumable.
}
& 'C:\Users\shish\anaconda3\envs\research-gpu\python.exe' 'scripts\run_remaining_pipeline.py'
exit $LASTEXITCODE
