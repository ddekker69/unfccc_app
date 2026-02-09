#!/bin/bash
# Complete environment setup script for UNFCCC project
# This script will:
# 1. Create the conda environment
# 2. Set up automatic activation methods
# 3. Configure your shell for automatic activation

set -e  # Exit on any error

echo "🚀 UNFCCC Environment Setup"
echo "==========================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda first:"
    echo "   macOS: brew install conda"
    echo "   Or download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Ensure we're in the unfccc directory
if [[ $(basename "$(pwd)") != "unfccc" ]]; then
    if [[ -d "unfccc" ]]; then
        cd unfccc
        echo "📁 Changed to unfccc directory"
    else
        echo "❌ Please run this script from the unfccc directory or its parent"
        exit 1
    fi
fi

# Create environment from environment.yml
echo ""
echo "📦 Creating conda environment..."
if conda env list | grep -q "unfccc_env"; then
    echo "⚠️  Environment 'unfccc_env' already exists"
    read -p "Do you want to update it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Updating environment..."
        conda env update -f environment.yml
    else
        echo "✅ Using existing environment"
    fi
else
    echo "🆕 Creating new environment..."
    conda env create -f environment.yml
fi

# Make activation script executable
chmod +x activate.sh

# Setup direnv if available
echo ""
echo "🔧 Setting up automatic activation..."

if command -v direnv &> /dev/null; then
    echo "✅ direnv found - setting up automatic directory activation"
    direnv allow .
    echo "✅ Automatic activation configured with direnv"
else
    echo "📦 direnv not found. Installing it will enable automatic activation when entering this directory."
    echo ""
    echo "To install direnv:"
    echo "  macOS:   brew install direnv"
    echo "  Ubuntu:  sudo apt install direnv"
    echo "  Arch:    sudo pacman -S direnv"
    echo ""
    echo "After installing direnv, add this to your shell config:"
    echo "  ~/.bashrc or ~/.zshrc: eval \"\$(direnv hook bash)\" or eval \"\$(direnv hook zsh)\""
    echo ""
fi

# Setup shell aliases
echo "⚙️  Setting up convenient aliases..."

SHELL_CONFIG=""
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
fi

if [[ -n "$SHELL_CONFIG" && -f "$SHELL_CONFIG" ]]; then
    # Add alias if it doesn't exist
    if ! grep -q "alias unfccc=" "$SHELL_CONFIG"; then
        echo "" >> "$SHELL_CONFIG"
        echo "# UNFCCC project aliases" >> "$SHELL_CONFIG"
        echo "alias unfccc='cd $(pwd) && source activate.sh'" >> "$SHELL_CONFIG"
        echo "alias unfccc-debug='cd $(pwd) && DEBUG_MODE=true source activate.sh'" >> "$SHELL_CONFIG"
        echo "✅ Added 'unfccc' and 'unfccc-debug' aliases to $SHELL_CONFIG"
    else
        echo "✅ Aliases already exist in $SHELL_CONFIG"
    fi
fi

# Test the environment
echo ""
echo "🧪 Testing environment..."
source activate.sh > /dev/null 2>&1

if python -c "import streamlit, sentence_transformers, torch; print('✅ Key packages imported successfully')" 2>/dev/null; then
    echo "✅ Environment test passed!"
else
    echo "⚠️  Environment test failed - some packages may need manual installation"
fi

# Final instructions
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🔧 Automatic Activation Methods:"
echo ""
echo "1. 📁 Directory-based (with direnv):"
echo "   Just 'cd' into the unfccc directory and the environment activates!"
echo ""
echo "2. 🖥️  Shell aliases:"
echo "   unfccc        # Activate environment and cd to unfccc folder"
echo "   unfccc-debug  # Same but with debug mode enabled"
echo ""
echo "3. 💻 Manual activation:"
echo "   cd unfccc && source activate.sh"
echo ""
echo "4. 🎯 VS Code/Cursor:"
echo "   The environment will auto-activate in the integrated terminal"
echo "   Use Cmd/Ctrl+Shift+P → 'Terminal: Select Default Profile' → 'UNFCCC Environment'"
echo ""
echo "💡 Quick Start:"
echo "   unfccc                            # Activate environment"
echo "   streamlit run cluster_qa_app.py   # Start the app"
echo ""
echo "🔍 Debug Mode:"
echo "   unfccc-debug                      # Activate with debugging"
echo "   python debug_demo.py              # Test debug system"
echo ""
echo "📚 More info: See README.md and DEBUG_GUIDE.md"
echo ""

# Reload shell if possible
if [[ -n "$SHELL_CONFIG" ]]; then
    echo "💡 Run 'source $SHELL_CONFIG' or restart your terminal to use the new aliases"
fi

echo "✨ Happy coding!" 