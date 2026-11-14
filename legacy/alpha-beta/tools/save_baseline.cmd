@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

cargo build --release
if errorlevel 1 exit /b %ERRORLEVEL%

if not exist bin mkdir bin

set "SRC=%CD%\target\release\tron.exe"
if not exist "%SRC%" set "SRC=%CD%\target\release\tron"
if not exist "%SRC%" (
  echo no release binary at target\release\tron[.exe] 1>&2
  exit /b 2
)

if /i "%SRC:~-4%"==".exe" (
  copy /Y "%SRC%" "%CD%\bin\tron-baseline.exe" >nul
  echo Saved baseline -^> %CD%\bin\tron-baseline.exe
) else (
  copy /Y "%SRC%" "%CD%\bin\tron-baseline" >nul
  echo Saved baseline -^> %CD%\bin\tron-baseline
)
exit /b 0
