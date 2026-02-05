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

# Verify Java
if command_exists java; then
    echo "Java installed successfully:"
    java -version
else
    echo "Failed to install Java."
    exit 1
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

# Verify Lean
if command_exists lean; then
    echo "Lean installed successfully:"
    lean --version
else
    echo "Failed to install Lean."
    exit 1
fi

echo "Dependency installation complete."
