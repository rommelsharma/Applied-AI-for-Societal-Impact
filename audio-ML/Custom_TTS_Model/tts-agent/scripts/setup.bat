@echo off
:: ============================================================================
:: TTS Agent — Windows Native Setup Script
:: Supports: Windows 10/11 with Python 3.10+ installed
:: GPU:       Automatically detects NVIDIA GPU (CUDA 12.1)
::            Falls back to CPU-only PyTorch if no NVIDIA GPU is found
:: ============================================================================

setlocal EnableDelayedExpansion

echo ========================================
echo  TTS Agent Setup — Windows Native
echo ========================================
echo.

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found on PATH.
    echo Download Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause & exit /b 1
)

:: Show Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo Python: %PYVER%
echo.

:: Create virtual environment
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment.
        pause & exit /b 1
    )
) else (
    echo Virtual environment already exists — skipping creation.
)

:: Activate venv
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment.
    pause & exit /b 1
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --quiet --upgrade pip wheel

:: Detect NVIDIA GPU and install appropriate PyTorch
echo Detecting GPU...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected — installing PyTorch with CUDA 12.1...
    pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No NVIDIA GPU detected — installing CPU-only PyTorch.
    echo NOTE: Voice synthesis will be slower without GPU acceleration.
    pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PyTorch installation failed.
    pause & exit /b 1
)

:: Install application dependencies
echo Installing application dependencies...
pip install --quiet -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install requirements.txt
    pause & exit /b 1
)

:: Install dev dependencies (optional — skip if not present)
if exist requirements-dev.txt (
    echo Installing dev dependencies...
    pip install --quiet -r requirements-dev.txt
)

:: espeak-ng — required by Kokoro for phonemisation
:: Check if already installed
python -c "import espeakng_loader; print('espeak-ng available')" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo NOTE: espeak-ng is required by Kokoro for non-English voices.
    echo Attempting install via Chocolatey...
    where choco >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        choco install espeak-ng -y
    ) else (
        echo Chocolatey not found. Install espeak-ng manually:
        echo   Option A: Install Chocolatey then run: choco install espeak-ng
        echo   Option B: Download from https://github.com/espeak-ng/espeak-ng/releases
        echo             and add its bin directory to PATH.
    )
) else (
    echo espeak-ng already available.
)

:: Create .env if missing
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo Created .env from .env.example — edit as needed.
    )
)

:: Create runtime directories
if not exist "models\kokoro"  mkdir models\kokoro
if not exist "models\xtts"    mkdir models\xtts
if not exist "voice_samples"  mkdir voice_samples
if not exist "input_samples"  mkdir input_samples
if not exist "outputs"        mkdir outputs

echo.
echo ========================================
echo  Setup complete!
echo ========================================
echo.
echo Next steps:
echo   1. Edit .env and add your HuggingFace token (optional, removes rate limits)
echo   2. Place test reference audio in input_samples\:
echo        cloning-voice-clip-male-hindi-1.wav
echo        cloning-voice-samples-JP.wav
echo   3. Pre-download models (optional, recommended for offline use):
echo        python scripts\download_models.py
echo   4. Start the server:
echo        venv\Scripts\uvicorn.exe backend.app:app --host 0.0.0.0 --port 8000 --reload
echo   5. Open browser: http://localhost:8000
echo.
pause
