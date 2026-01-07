@echo off
REM Check if the "venv" folder exists
if not exist "venv" (
    echo Error: No virtual environment found. Run setup.
    pause
    exit /b 1
)

echo Virtual environment found, launching flux image generation...

python flux-img-gen.py

