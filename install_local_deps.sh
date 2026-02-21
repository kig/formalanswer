#!/bin/bash

WORK_DIR="$(pwd)/work"
mkdir -p "$WORK_DIR"

echo "Starting LOCAL dependency installation..."

# --- 1. Install Java (Local OpenJDK) ---
JAVA_DIR="$WORK_DIR/jdk"
if [ -d "$JAVA_DIR" ] && [ -f "$JAVA_DIR/bin/java" ]; then
    echo "Local JDK already present at $JAVA_DIR"
else
    echo "Downloading OpenJDK..."
    
    # Detect Platform
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    BASE_URL="https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.2%2B13"
    
    if [ "$OS" == "Darwin" ]; then
        if [ "$ARCH" == "arm64" ]; then
            JDK_URL="$BASE_URL/OpenJDK21U-jdk_aarch64_mac_hotspot_21.0.2_13.tar.gz"
        elif [ "$ARCH" == "x86_64" ]; then
            JDK_URL="$BASE_URL/OpenJDK21U-jdk_x64_mac_hotspot_21.0.2_13.tar.gz"
        else
            echo "Unsupported macOS architecture: $ARCH"
            exit 1
        fi
    elif [ "$OS" == "Linux" ]; then
        if [ "$ARCH" == "aarch64" ]; then
            JDK_URL="$BASE_URL/OpenJDK21U-jdk_aarch64_linux_hotspot_21.0.2_13.tar.gz"
        elif [ "$ARCH" == "x86_64" ]; then
            JDK_URL="$BASE_URL/OpenJDK21U-jdk_x64_linux_hotspot_21.0.2_13.tar.gz"
        else
            echo "Unsupported Linux architecture: $ARCH"
            exit 1
        fi
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi
    
    echo "Downloading OpenJDK from $JDK_URL..."
    curl -L -o "$WORK_DIR/openjdk.tar.gz" "$JDK_URL"
    
    echo "Extracting OpenJDK..."
    tar -xzf "$WORK_DIR/openjdk.tar.gz" -C "$WORK_DIR"
    
    # Identify the extracted folder name
    EXTRACTED_DIR=$(find "$WORK_DIR" -maxdepth 1 -type d -name "jdk-*" | head -n 1)
    if [ -z "$EXTRACTED_DIR" ]; then
        echo "Error: Could not find extracted JDK directory."
    else
        rm -rf "$JAVA_DIR"
        mv "$EXTRACTED_DIR" "$JAVA_DIR"
        rm "$WORK_DIR/openjdk.tar.gz"
    fi
fi

# Set JAVA_HOME and PATH for this session
export JAVA_HOME="$JAVA_DIR"
export PATH="$JAVA_HOME/bin:$PATH"

if "$JAVA_DIR/bin/java" -version >/dev/null 2>&1; then
    echo "Local Java verified."
else
    echo "Failed to verify local Java."
fi


# --- 2. Install Lean 4 (Local Elan) ---
export ELAN_HOME="$WORK_DIR/.elan"
export CARGO_HOME="$WORK_DIR/.cargo"
export RUSTUP_HOME="$WORK_DIR/.rustup"

if [ -f "$ELAN_HOME/bin/lean" ]; then
    echo "Local Lean already present at $ELAN_HOME"
else
    echo "Downloading Elan init script..."
    if [ ! -f "$WORK_DIR/elan-init.sh" ]; then
        curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf > "$WORK_DIR/elan-init.sh"
        chmod +x "$WORK_DIR/elan-init.sh"
    fi

    echo "Installing Elan to $ELAN_HOME..."
    "$WORK_DIR/elan-init.sh" -y --no-modify-path --default-toolchain stable
fi

# Add to PATH
export PATH="$ELAN_HOME/bin:$PATH"

if "$ELAN_HOME/bin/lean" --version >/dev/null 2>&1; then
    echo "Local Lean verified."
    # Initialize Mathlib cache
    if [ -f "$WORK_DIR/lakefile.toml" ]; then
        echo "Initializing Mathlib cache..."
        cd "$WORK_DIR" && ../env_wrapper.sh lake exe cache get
        cd ..
    fi
else
    echo "Failed to verify local Lean."
fi

# --- 3. TLA+ Tooling ---
if [ -f "$WORK_DIR/tla2tools.jar" ]; then
    echo "tla2tools.jar already present."
else
    echo "Downloading tla2tools.jar..."
    curl -L -o "$WORK_DIR/tla2tools.jar" https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
fi

# --- 4. Setup Python Virtual Environment ---
echo "Setting up Python virtual environment..."
VENV_DIR="venv"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python3" ]; then
    echo "Venv already exists."
else
    if python3 -m venv --without-pip "$VENV_DIR"; then
        echo "Bootstrapping pip..."
        curl -sS https://bootstrap.pypa.io/get-pip.py | ./"$VENV_DIR"/bin/python3 -
    else
        echo "Failed to create venv."
        exit 1
    fi
fi

echo "Ensuring Python packages are installed..."
./"$VENV_DIR"/bin/pip install google-genai python-dotenv openai jax jaxlib pytest mcp fastapi uvicorn "z3-solver==4.13.3.0" pyyaml rich

# Create a helper script to run commands with these paths
echo "#!/bin/bash" > env_wrapper.sh
echo "export JAVA_HOME=\"$JAVA_DIR\"" >> env_wrapper.sh
echo "export ELAN_HOME=\"$ELAN_HOME\"" >> env_wrapper.sh
echo "export PATH=\"\$JAVA_HOME/bin:\$ELAN_HOME/bin:\$PATH\"" >> env_wrapper.sh
echo "exec \"\$@\"" >> env_wrapper.sh
chmod +x env_wrapper.sh

echo "Installation complete. Use ./env_wrapper.sh <command> to run with tools."
