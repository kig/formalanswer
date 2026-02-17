#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Starting dependency installation..."

# 1. Install Java (OpenJDK)
if command_exists java; then
    echo "Java is already installed."
else
    echo "Installing OpenJDK..."
    # Try Homebrew first
    if command_exists brew; then
        brew install openjdk
        
        # Link OpenJDK for the system to find it (Standard Homebrew path for macOS ARM64)
        sudo ln -sfn /opt/homebrew/opt/openjdk/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
        
        # Add to PATH
        export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
        echo 'export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"' >> ~/.zshrc
        echo 'export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"' >> ~/.bashrc
    else
        echo "Homebrew not found. Please install Homebrew or Java manually."
        exit 1
    fi
fi

# 2. Install Lean 4 (via Elan)
if command_exists lean; then
    echo "Lean is already installed."
else
    echo "Installing Elan (Lean Version Manager)..."
    # Attempt to install via brew if available, otherwise curl
    if command_exists brew; then
        brew install elan-init
    else
        curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
    fi
    
    # Add to PATH (Elan installs to ~/.elan/bin)
    export PATH="$HOME/.elan/bin:$PATH"
    echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.zshrc
    echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
    
    # Install stable toolchain
    elan toolchain install stable
    elan default stable
fi

# 3. Setup Python Virtual Environment and Install Dependencies
echo "Setting up Python virtual environment..."
if command_exists python3; then
    rm -rf venv
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip
    echo "Installing Python packages..."
    ./venv/bin/pip install google-genai python-dotenv openai jax jaxlib pytest mcp fastapi uvicorn "z3-solver==4.13.3.0" pyyaml
else
    echo "python3 not found. Please install Python 3.10+."
    exit 1
fi

# Verify Installations
echo "-----------------------------------"
java -version
lean --version
./venv/bin/python3 --version
./venv/bin/pip list | grep -E "jax|google-genai|openai|z3-solver"
echo "-----------------------------------"

echo "Dependency installation complete."
