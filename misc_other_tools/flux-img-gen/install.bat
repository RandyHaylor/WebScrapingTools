@echo off

REM Check if the "venv" folder exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Install PyTorch with CUDA 11.8
echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other requirements
echo Installing requirements from requirements.txt...
pip install -r requirements.txt

REM Pause to display completion message
echo Installation complete.  BE SURE TO INSTALL CUDA 11.8 FOR WINDOWS SEPARATELY.  Press any key to continue.
pause

