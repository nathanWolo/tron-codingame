@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY where python3 >nul 2>&1 && set "PY=python3"
if not defined PY (
  echo python not found on PATH 1>&2
  exit /b 2
)

set "BASE=%SPRT_BASELINE%"
if "%BASE%"=="" set "BASE=%CD%\bin\tron-baseline"
if not exist "%BASE%" if exist "%BASE%.exe" set "BASE=%BASE%.exe"
if not exist "%BASE%" (
  echo No baseline at %BASE% 1>&2
  echo Run tools/save_baseline.sh first ^(or set SPRT_BASELINE^). 1>&2
  exit /b 2
)

set "DEV=%SPRT_DEV%"
if "%DEV%"=="" set "DEV=%CD%\target\release\tron"

"%PY%" "%CD%\tools\watch.py" --build --baseline "%BASE%" --dev "%DEV%" %*
exit /b %ERRORLEVEL%
