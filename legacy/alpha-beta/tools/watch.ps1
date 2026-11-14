# Windows default ExecutionPolicy (Restricted) will not load this file.
# Use instead:
#   python tools\watch.py --build
#   .\tools\watch.cmd
$ErrorActionPreference = "Stop"
$cmd = Join-Path $PSScriptRoot "watch.cmd"
if (-not (Test-Path -LiteralPath $cmd)) {
    Write-Host "missing $cmd" -ForegroundColor Red
    exit 2
}
& cmd.exe /c "`"$cmd`" $args"
exit $LASTEXITCODE
