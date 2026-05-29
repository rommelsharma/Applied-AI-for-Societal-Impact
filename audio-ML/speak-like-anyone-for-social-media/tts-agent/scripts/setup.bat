@echo off
:: ============================================================================
:: TTS Agent — Windows Native Setup Script
:: Supports: Windows 10/11 with Python 3.9, 3.10, or 3.11
::           XTTS v2 blocks Python 3.12+ — do NOT use a newer interpreter.
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
    echo Download Python 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo NOTE: XTTS v2 requires Python 3.9-3.11. Do NOT use Python 3.12 or later.
    pause & exit /b 1
)

:: Warn if Python 3.12+
for /f "tokens=2 delims=." %%v in ('python --version 2^>^&1') do set PYMINOR=%%v
if %PYMINOR% GEQ 12 (
    echo.
    echo WARNING: Python 3.1%PYMINOR% detected. XTTS v2 requires Python 3.9-3.11.
    echo          The setup may fail when installing the TTS package.
    echo          Please install Python 3.11 from https://www.python.org/downloads/
    echo.
    pause
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
:: RTX 50xx (Blackwell, sm_12x) requires CUDA 12.6+ — older cu121 wheels lack those kernels.
echo Detecting GPU...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    :: Read compute capability major version (e.g. "12.0" → "12")
    for /f "tokens=1 delims=." %%M in ('nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul') do set CC_MAJOR=%%M
    if not defined CC_MAJOR set CC_MAJOR=0

    if !CC_MAJOR! GEQ 12 (
        echo NVIDIA Blackwell GPU detected ^(compute capability !CC_MAJOR!.x^) — installing PyTorch with CUDA 12.6...
        echo NOTE: RTX 50xx requires CUDA 12.6+ wheels. Installing cu126 build.
        pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    ) else (
        echo NVIDIA GPU detected ^(compute capability !CC_MAJOR!.x^) — installing PyTorch with CUDA 12.1...
        pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
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

:: Kokoro CJK language support — optional extras that require a C++ compiler.
:: misaki[ja] → pyopenjtalk (Japanese phonemisation)
:: misaki[zh] → jieba + pypinyin (Chinese/Mandarin phonemisation)
:: Skipped gracefully if the build fails (e.g. Visual Studio Build Tools not installed).
echo Installing Kokoro Japanese support (misaki[ja])...
pip install --quiet "misaki[ja]"
if %ERRORLEVEL% EQU 0 (
    echo   misaki[ja] installed.
) else (
    echo   WARNING: misaki[ja] install failed -- Japanese Kokoro voices will be unavailable.
    echo            Fix: install Visual Studio Build Tools from
    echo            https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo            then run: pip install "misaki[ja]"
)

echo Installing Kokoro Chinese support (misaki[zh])...
pip install --quiet "misaki[zh]"
if %ERRORLEVEL% EQU 0 (
    echo   misaki[zh] installed.
) else (
    echo   WARNING: misaki[zh] install failed -- Chinese Kokoro voices will be unavailable.
)

:: Download UniDic dictionary for Japanese TTS support (~500 MB) — skip if already present
python -c "import unidic, os; assert os.path.isdir(unidic.DICDIR)" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo UniDic already installed -- skipping download.
) else (
    echo Downloading UniDic dictionary for Japanese TTS support...
    python -m unidic download
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
echo   2. Test suite files (already in input_samples\ if repo was cloned):
echo        cloning-voice-sample-english-male-1.wav
echo        cloning-voice-sample-hindi-male-1.wav
echo        cloning-voice-sample-japanese-female-1.wav
echo   3. Pre-download models (optional, recommended for offline use):
echo        python scripts\download_models.py
echo   4. Start the server:
echo        venv\Scripts\uvicorn.exe backend.app:app --host 0.0.0.0 --port 8000 --reload
echo   5. Open browser: http://localhost:8000
echo.
pause
