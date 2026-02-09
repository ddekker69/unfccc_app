# Simple activation script for UNFCCC project (PowerShell)
# Usage: .\activate.ps1

Write-Host "🔧 Setting up UNFCCC environment..." -ForegroundColor Cyan

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Loading conda..." -ForegroundColor Yellow
    
    # Try common conda installation paths on Windows
    $condaPaths = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe", 
        "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    )
    
    $condaFound = $false
    foreach ($path in $condaPaths) {
        if (Test-Path $path) {
            # Add conda to PATH for this session
            $env:PATH = "$([System.IO.Path]::GetDirectoryName($path));$env:PATH"
            Write-Host "✅ Loaded conda from $path" -ForegroundColor Green
            $condaFound = $true
            break
        }
    }
    
    if (-not $condaFound) {
        Write-Host "❌ Conda not found. Please install conda first:" -ForegroundColor Red
        Write-Host "   https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Red
        Write-Host "   Or add conda to your PATH" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if environment exists
$envExists = (conda env list | Select-String "unfccc_env") -ne $null
if ($envExists) {
    Write-Host "✅ Environment 'unfccc_env' found" -ForegroundColor Green
} else {
    Write-Host "📦 Environment 'unfccc_env' not found. Creating it..." -ForegroundColor Yellow
    if (Test-Path "environment.yml") {
        conda env create -f environment.yml
        Write-Host "✅ Environment created from environment.yml" -ForegroundColor Green
    } else {
        Write-Host "❌ environment.yml not found. Please run this from the project root." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate the environment
Write-Host "🚀 Activating unfccc_env environment..." -ForegroundColor Cyan
conda activate unfccc_env

# Set up project environment variables
$currentDir = Split-Path -Leaf (Get-Location)
if ($currentDir -eq "unfccc") {
    $projectRoot = Get-Location
} else {
    if (Test-Path "unfccc") {
        Set-Location unfccc
        $projectRoot = Get-Location
        Write-Host "📁 Changed to unfccc directory" -ForegroundColor Yellow
    } else {
        $projectRoot = Get-Location
        Write-Host "⚠️  unfccc directory not found, using current directory" -ForegroundColor Yellow
    }
}

$env:PROJECT_ROOT = $projectRoot
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$projectRoot;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $projectRoot
}

# Set debug defaults (can be overridden)
if (-not $env:DEBUG_MODE) { $env:DEBUG_MODE = "false" }
if (-not $env:DEBUG_RAG) { $env:DEBUG_RAG = "false" }
if (-not $env:DEBUG_PERFORMANCE) { $env:DEBUG_PERFORMANCE = "false" }

# Set device defaults
if (-not $env:EMBEDDING_DEVICE) { $env:EMBEDDING_DEVICE = "auto" }

# Set offline mode for HuggingFace (optional)
if (-not $env:HF_HUB_OFFLINE) { $env:HF_HUB_OFFLINE = "1" }

# Display status
Write-Host ""
Write-Host "🎉 UNFCCC environment ready!" -ForegroundColor Green
Write-Host "📁 Project root: $env:PROJECT_ROOT" -ForegroundColor White
Write-Host "🐍 Python: $env:CONDA_PREFIX\python.exe" -ForegroundColor White
Write-Host "🧠 Conda env: $env:CONDA_DEFAULT_ENV" -ForegroundColor White
Write-Host "🔍 Debug mode: $env:DEBUG_MODE" -ForegroundColor White
Write-Host ""
Write-Host "💡 Quick commands:" -ForegroundColor Cyan
Write-Host "   streamlit run cluster_qa_app.py    # Start the main app" -ForegroundColor White
Write-Host "   python debug_demo.py               # Test debug system" -ForegroundColor White
Write-Host "   python automated_pipeline.py       # Run full pipeline" -ForegroundColor White
Write-Host "" 