@echo off
setlocal
cd /d "%~dp0"

py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto setup_failed
)

echo Checking dependencies...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --quiet -r requirements.txt
if errorlevel 1 goto setup_failed

start "" ".venv\Scripts\pythonw.exe" "%~dp0main.py"
exit /b 0

:setup_failed
echo Setup failed. Check the error above and try again.
pause
exit /b 1
