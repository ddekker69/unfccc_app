@echo off
REM Simple activation script for UNFCCC project (Windows)
REM Prioritizes system-wide Anaconda for managed computers
REM Usage: activate.bat

echo 🔧 Setting up UNFCCC environment...

REM Check if conda is available in PATH
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo 📦 Loading conda...
    
    REM Try conda installation paths in priority order (system-wide Anaconda first for managed systems)
    if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" (
        call "C:\ProgramData\Anaconda3\Scripts\activate.bat"
        echo ✅ Loaded conda from C:\ProgramData\Anaconda3 (system-wide Anaconda)
    ) else if exist "C:\Anaconda3\Scripts\activate.bat" (
        call "C:\Anaconda3\Scripts\activate.bat"
        echo ✅ Loaded conda from C:\Anaconda3 (system-wide Anaconda)
    ) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
        echo ✅ Loaded conda from %USERPROFILE%\anaconda3 (user Anaconda)
    ) else if exist "C:\ProgramData\Miniconda3\Scripts\activate.bat" (
        call "C:\ProgramData\Miniconda3\Scripts\activate.bat"
        echo ✅ Loaded conda from C:\ProgramData\Miniconda3 (system-wide Miniconda)
    ) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
        echo ✅ Loaded conda from %USERPROFILE%\miniconda3 (user Miniconda)
    ) else (
        echo ❌ Conda not found. For managed systems:
        echo    💡 Ask IT to install Anaconda system-wide (recommended)
        echo    📥 Or install Miniconda yourself: https://docs.conda.io/en/latest/miniconda.html
        echo    🔧 Or add existing conda to your PATH environment variable
        pause
        exit /b 1
    )
)

REM Verify conda is working
conda --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda loaded but not functional. Try running as Administrator.
    pause
    exit /b 1
)

REM Check if environment exists
conda env list | findstr "unfccc_env" >nul
if %errorlevel% equ 0 (
    echo ✅ Environment 'unfccc_env' found
) else (
    echo 📦 Environment 'unfccc_env' not found. Creating it...
    if exist "environment-cpu.yml" (
        echo 💻 Using CPU-optimized environment (environment-cpu.yml)
        conda env create -f environment-cpu.yml
        echo ✅ Environment created from environment-cpu.yml
    ) else if exist "environment.yml" (
        echo 🚀 Using full environment (environment.yml)
        conda env create -f environment.yml
        echo ✅ Environment created from environment.yml
    ) else (
        echo ❌ No environment file found. Please run this from the project root.
        echo 📁 Looking for: environment-cpu.yml or environment.yml
        pause
        exit /b 1
    )
)

REM Activate the environment with error handling
echo 🚀 Activating unfccc_env environment...
call conda activate unfccc_env
if %errorlevel% neq 0 (
    echo ❌ Failed to activate environment. This may be a permissions issue.
    echo.
    echo 🛠️  Troubleshooting steps:
    echo    1. Try running this script as Administrator
    echo    2. Contact IT for system-wide Anaconda installation
    echo    3. Try manual activation: 'conda activate unfccc_env'
    echo    4. Check if conda base environment works: 'conda activate base'
    echo.
    pause
    exit /b 1
)

REM Verify activation worked
if not defined CONDA_DEFAULT_ENV (
    echo ❌ Environment activation may have failed (CONDA_DEFAULT_ENV not set)
    echo 💡 Try manual activation or contact IT support
    pause
    exit /b 1
)

REM Set up project environment variables
for %%i in ("%cd%") do set "CURRENT_DIR=%%~nxi"
if /i "%CURRENT_DIR%"=="unfccc" (
    set "PROJECT_ROOT=%cd%"
) else (
    if exist "unfccc" (
        cd unfccc
        set "PROJECT_ROOT=%cd%"
        echo 📁 Changed to unfccc directory
    ) else (
        set "PROJECT_ROOT=%cd%"
        echo ⚠️  unfccc directory not found, using current directory
    )
)

set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

REM Set debug defaults (can be overridden)
if not defined DEBUG_MODE set DEBUG_MODE=false
if not defined DEBUG_RAG set DEBUG_RAG=false
if not defined DEBUG_PERFORMANCE set DEBUG_PERFORMANCE=false

REM Check for CUDA/GPU availability and set device accordingly
echo 🔍 Checking GPU availability...
python -c "import torch; print('✅ GPU available:', torch.cuda.is_available()); print('🔧 CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=*" %%a in ('python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"') do set DETECTED_DEVICE=%%a
) else (
    echo ⚠️  Could not detect GPU, defaulting to CPU
    set DETECTED_DEVICE=cpu
)

REM Set device defaults with GPU detection
if not defined EMBEDDING_DEVICE (
    if "%DETECTED_DEVICE%"=="cuda" (
        set EMBEDDING_DEVICE=auto
        echo 🚀 GPU detected - using GPU acceleration
    ) else (
        set EMBEDDING_DEVICE=cpu
        echo 💻 No GPU detected - using CPU mode (this is normal for systems without dedicated graphics cards)
    )
) else (
    echo 🔧 Using user-specified device: %EMBEDDING_DEVICE%
)

REM Set CPU-optimized settings for non-GPU systems
if "%DETECTED_DEVICE%"=="cpu" (
    echo 🛠️  Applying CPU optimizations...
    set PYTORCH_MKL_NUM_THREADS=1
    set OMP_NUM_THREADS=1
    set NUMEXPR_NUM_THREADS=1
    echo 💡 Tip: Consider using TinyLlama-1B or FLAN-T5-Small models for better performance on CPU
)

REM Set offline mode for HuggingFace (optional)
if not defined HF_HUB_OFFLINE set HF_HUB_OFFLINE=1

REM Display status
echo.
echo 🎉 UNFCCC environment ready!
echo 📁 Project root: %PROJECT_ROOT%
echo 🐍 Python: %CONDA_PREFIX%\python.exe
echo 🧠 Conda env: %CONDA_DEFAULT_ENV%
echo 🔍 Debug mode: %DEBUG_MODE%
echo 🎮 Device mode: %EMBEDDING_DEVICE% (detected: %DETECTED_DEVICE%)
echo.
echo 💡 Quick commands:
echo    streamlit run cluster_qa_app.py    # Start the main app
echo    python debug_demo.py               # Test debug system
echo    python automated_pipeline.py       # Run full pipeline
echo. 