#!/bin/bash
# Simple activation script for UNFCCC project
# Usage: source activate.sh

echo "🔧 Setting up UNFCCC environment..."

# Load conda if not already available
if ! command -v conda &> /dev/null; then
    echo "📦 Loading conda..."
    
    # Try common conda installation paths - prioritize the user's actual path
    if [[ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "/opt/anaconda3/etc/profile.d/conda.sh"
        echo "✅ Loaded conda from /opt/anaconda3"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        echo "✅ Loaded conda from ~/miniconda3"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
        echo "✅ Loaded conda from ~/anaconda3"
    elif [[ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "/opt/miniconda3/etc/profile.d/conda.sh"
        echo "✅ Loaded conda from /opt/miniconda3"
    else
        echo "❌ Conda not found. Please install conda first:"
        echo "   https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
fi

# Check if environment exists
if conda env list | grep -q "unfccc_env"; then
    echo "✅ Environment 'unfccc_env' found"
else
    echo "📦 Environment 'unfccc_env' not found. Creating it..."
    if [[ -f "environment.yml" ]]; then
        conda env create -f environment.yml
        echo "✅ Environment created from environment.yml"
    else
        echo "❌ environment.yml not found. Please run this from the project root."
        return 1
    fi
fi

# Activate the environment
echo "🚀 Activating unfccc_env environment..."
conda activate unfccc_env

# Set up project environment variables - always use unfccc as the working directory
if [[ $(basename "$(pwd)") == "unfccc" ]]; then
    export PROJECT_ROOT="$(pwd)"
else
    # We're in the parent directory, cd to unfccc
    if [[ -d "unfccc" ]]; then
        cd unfccc
        export PROJECT_ROOT="$(pwd)"
        echo "📁 Changed to unfccc directory"
    else
        export PROJECT_ROOT="$(pwd)"
        echo "⚠️  unfccc directory not found, using current directory"
    fi
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set debug defaults (can be overridden)
export DEBUG_MODE=${DEBUG_MODE:-false}
export DEBUG_RAG=${DEBUG_RAG:-false}
export DEBUG_PERFORMANCE=${DEBUG_PERFORMANCE:-false}

# Set device defaults
export EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-auto}

# Display status
echo ""
echo "🎉 UNFCCC environment ready!"
echo "📁 Project root: $PROJECT_ROOT"
echo "🐍 Python: $(which python)"
echo "🧠 Conda env: $CONDA_DEFAULT_ENV"
echo "🔍 Debug mode: $DEBUG_MODE"
echo ""
echo "💡 Quick commands:"
echo "   streamlit run cluster_qa_app.py    # Start the main app"
echo "   python debug_demo.py               # Test debug system"
echo "   python automated_pipeline.py       # Run full pipeline"
echo "" 