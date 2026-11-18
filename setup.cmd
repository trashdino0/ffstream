@echo off
setlocal
cd /d "%~dp0"

rem Prefer the Python launcher, then fall back to python on PATH.
where py >nul 2>nul
if errorlevel 1 (
    set "PY=python"
) else (
    set "PY=py -3"
)

%PY% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    echo Install it from https://www.python.org/downloads/ and run this again.
    goto :fail
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 goto :fail
)

echo Installing dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -e ".[dev]"
if errorlevel 1 goto :fail

echo.
echo Setup complete. Next steps:
echo   weekly.cmd                       refresh, rebuild, and build the board
echo   .venv\Scripts\ffstream.exe --help
pause
exit /b 0

:fail
echo.
echo Setup failed. Read the messages above.
pause
exit /b 1
