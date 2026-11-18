@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\weekly.ps1" %*
if errorlevel 1 (
    echo.
    echo Something went wrong. Read the messages above.
    set "EXIT_CODE=1"
) else (
    echo.
    echo Finished. Open outputs\week_*_report.html for the board.
    set "EXIT_CODE=0"
)
pause
exit /b %EXIT_CODE%
