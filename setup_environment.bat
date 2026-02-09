@echo off
REM Complete environment setup script for UNFCCC project (Windows)
REM Prioritizes system-wide Anaconda for managed computers
REM This script will:
REM 1. Create the conda environment
REM 2. Set up convenient batch files
REM 3. Test the environment

echo 🚀 UNFCCC Environment Setup (Windows)
echo =======================================

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda not found in PATH. Checking common installation locations...
    
    REM Try to find and temporarily add conda to PATH (system-wide Anaconda first)
    if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" (
        set PATH=C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3;%PATH%
        echo ✅ Found system-wide Anaconda at C:\ProgramData\Anaconda3
    ) else if exist "C:\Anaconda3\Scripts\conda.exe" (
        set PATH=C:\Anaconda3\Scripts;C:\Anaconda3;%PATH%
        echo ✅ Found system-wide Anaconda at C:\Anaconda3
    ) else if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
        set PATH=%USERPROFILE%\anaconda3\Scripts;%USERPROFILE%\anaconda3;%PATH%
        echo ✅ Found user Anaconda at %USERPROFILE%\anaconda3
    ) else if exist "C:\ProgramData\Miniconda3\Scripts\conda.exe" (
        set PATH=C:\ProgramData\Miniconda3\Scripts;C:\ProgramData\Miniconda3;%PATH%
        echo ✅ Found system-wide Miniconda at C:\ProgramData\Miniconda3
    ) else if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" (
        set PATH=%USERPROFILE%\miniconda3\Scripts;%USERPROFILE%\miniconda3;%PATH%
        echo ✅ Found user Miniconda at %USERPROFILE%\miniconda3
    ) else (
        echo ❌ No conda installation found. For managed systems:
        echo    💡 Ask IT to install Anaconda system-wide (recommended)
        echo    📥 Or download Miniconda: https://docs.conda.io/en/latest/miniconda.html
        echo    🔧 Or add existing conda to your PATH
        pause
        exit /b 1
    )
    
    REM Verify conda is now available
    where conda >nul 2>nul
    if %errorlevel% neq 0 (
        echo ❌ Found conda but couldn't add to PATH. Try running as Administrator.
        pause
        exit /b 1
    )
)

REM Get conda version
for /f "tokens=2" %%i in ('conda --version') do set CONDA_VERSION=%%i
echo ✅ Conda found: conda %CONDA_VERSION%

REM Ensure we're in the unfccc directory
for %%i in ("%cd%") do set "CURRENT_DIR=%%~nxi"
if /i not "%CURRENT_DIR%"=="unfccc" (
    if exist "unfccc" (
        cd unfccc
        echo 📁 Changed to unfccc directory
    ) else (
        echo ❌ Please run this script from the unfccc directory or its parent
        pause
        exit /b 1
    )
)

REM Detect NVIDIA GPU presence
echo.
echo 🔍 Detecting GPU hardware...
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    set ENV_FILE=environment.yml
    echo 🚀 NVIDIA GPU detected - using full environment with CUDA support
) else (
    set ENV_FILE=environment-cpu.yml
    echo 💻 No NVIDIA GPU detected - using CPU-optimized environment
    echo 💡 This will avoid CUDA dependency issues
)

REM Create environment from appropriate yml file
echo.
echo 📦 Creating conda environment from %ENV_FILE%...
conda env list | findstr "unfccc_env" >nul
if %errorlevel% equ 0 (
    echo ⚠️  Environment 'unfccc_env' already exists
    set /p "REPLY=Do you want to update it? (y/N): "
    if /i "!REPLY!"=="y" (
        echo 🔄 Updating environment...
        conda env update -f %ENV_FILE%
        if %errorlevel% neq 0 (
            echo ❌ Environment update failed. Try running as Administrator.
            pause
            exit /b 1
        )
    ) else (
        echo ✅ Using existing environment
    )
) else (
    echo 🆕 Creating new environment...
    conda env create -f %ENV_FILE%
    if %errorlevel% neq 0 (
        echo ❌ Environment creation failed. Common issues:
        echo    💡 Try running as Administrator
        echo    🔧 Check if you have write permissions to conda directory
        echo    📞 Contact IT if this is a managed computer
        pause
        exit /b 1
    )
)

REM Test the environment
echo.
echo 🧪 Testing environment...
call conda activate unfccc_env
if %errorlevel% neq 0 (
    echo ❌ Failed to activate environment. This may be a permissions issue.
    echo    💡 Try running as Administrator
    echo    📞 Contact IT for system-wide conda installation
    pause
    exit /b 1
)

echo 📦 Testing package imports...
python -c "import streamlit, sentence_transformers, torch; print('✅ Key packages imported successfully')" 2>nul
if %errorlevel% equ 0 (
    echo ✅ Package import test passed!
) else (
    echo ⚠️  Package import test failed - some packages may need manual installation
)

REM Test GPU/CUDA availability
echo.
echo 🔍 Testing GPU/CUDA availability...
python -c "import torch; print('✅ GPU available:', torch.cuda.is_available()); print('🔧 CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=*" %%a in ('python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"') do set DETECTED_DEVICE=%%a
    if "%DETECTED_DEVICE%"=="cuda" (
        echo 🚀 GPU detected - system will use GPU acceleration
        echo 💡 Recommended models: DeepSeek-R1-7B, DeepSeek-R1-14B
    ) else (
        echo 💻 No GPU detected - system will use CPU mode
        echo 💡 This is normal for systems without dedicated graphics cards
        echo 🎯 Recommended CPU models: TinyLlama-1B, FLAN-T5-Small, DistilGPT2
        echo 📝 Note: CPU models are optimized for systems with 8GB+ RAM
    )
) else (
    echo ⚠️  Could not test GPU availability, defaulting to CPU mode
    set DETECTED_DEVICE=cpu
)

echo ✅ Environment testing complete!

REM Create convenient shortcuts
echo.
echo ⚙️  Creating convenient shortcuts...

REM Create desktop shortcut batch file
echo @echo off > "%USERPROFILE%\Desktop\UNFCCC.bat"
echo cd /d "%cd%" >> "%USERPROFILE%\Desktop\UNFCCC.bat"
echo call activate.bat >> "%USERPROFILE%\Desktop\UNFCCC.bat"
echo pause >> "%USERPROFILE%\Desktop\UNFCCC.bat"

REM Create debug shortcut
echo @echo off > "%USERPROFILE%\Desktop\UNFCCC-Debug.bat"
echo cd /d "%cd%" >> "%USERPROFILE%\Desktop\UNFCCC-Debug.bat"
echo set DEBUG_MODE=true >> "%USERPROFILE%\Desktop\UNFCCC-Debug.bat"
echo set DEBUG_RAG=true >> "%USERPROFILE%\Desktop\UNFCCC-Debug.bat"
echo call activate.bat >> "%USERPROFILE%\Desktop\UNFCCC-Debug.bat"
echo pause >> "%USERPROFILE%\Desktop\UNFCCC-Debug.bat"

echo ✅ Created desktop shortcuts: UNFCCC.bat and UNFCCC-Debug.bat

REM Final instructions
echo.
echo 🎉 Setup Complete!
echo ==================
echo.
echo 🖥️  System Configuration:
echo    Device mode: %DETECTED_DEVICE%
if "%DETECTED_DEVICE%"=="cpu" (
    echo    💡 CPU optimizations will be applied automatically
    echo    🎯 Use lightweight models for best performance
)
echo.
echo 🔧 Activation Methods:
echo.
echo 1. 🖱️  Desktop Shortcuts:
echo    Double-click 'UNFCCC.bat' on your desktop
echo    Double-click 'UNFCCC-Debug.bat' for debug mode
echo.
echo 2. 💻 Command Line:
echo    cd "%cd%"
echo    activate.bat
echo.
echo 3. 🎯 PowerShell/Command Prompt:
echo    cd "%cd%"
echo    conda activate unfccc_env
echo    set PYTHONPATH=%cd%;%%PYTHONPATH%%
echo    set HF_HUB_OFFLINE=1
echo.
echo 💡 Quick Start:
echo    activate.bat                       # Activate environment
echo    streamlit run cluster_qa_app.py    # Start the app
echo.
echo 🔍 Debug Mode:
echo    Use UNFCCC-Debug.bat shortcut      # Auto-enable debugging
echo    python debug_demo.py               # Test debug system
echo.
if "%DETECTED_DEVICE%"=="cpu" (
    echo 🧠 Recommended Models for Your System:
    echo    • TinyLlama-1B ^(8GB RAM^) - Best balance for CPU systems
    echo    • FLAN-T5-Small ^(Ultra Light^) - Most memory efficient  
    echo    • DistilGPT2 ^(Low Memory^) - Ultra-lightweight fallback
    echo.
)
echo 📚 More info: See README.md in the docs folder
echo.
echo ✨ Happy coding!
pause 