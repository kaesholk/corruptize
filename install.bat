@echo off
setlocal

set VENV_DIR=.venv

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

if not exist requirements.txt (
    echo ERROR: requirements.txt not found.
    goto :EOF
)

echo Installing packages from requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete.
endlocal