#!/bin/bash

WORK_DIR="$(pwd)/work"
mkdir -p "$WORK_DIR"

echo "Starting LOCAL dependency installation..."

# --- 1. Install Java (Local OpenJDK) ---
JAVA_DIR="$WORK_DIR/jdk"
if [ -d "$JAVA_DIR" ]; then
    echo "Local JDK seems to be present at $JAVA_DIR"
else
    echo "Downloading OpenJDK..."
    # URL for OpenJDK 21 (LTS) for macOS ARM64 (Eclipse Temurin)
    JDK_URL="https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.2%2B13/OpenJDK21U-jdk_aarch64_mac_hotspot_21.0.2_13.tar.gz"
    
    echo "Downloading OpenJDK from $JDK_URL..."
    curl -L -o "$WORK_DIR/openjdk.tar.gz" "$JDK_URL"
    
    echo "Extracting OpenJDK..."
    tar -xzf "$WORK_DIR/openjdk.tar.gz" -C "$WORK_DIR"
    
    # Identify the extracted folder name (it might vary)
    EXTRACTED_DIR=$(find "$WORK_DIR" -maxdepth 1 -type d -name "jdk-*" | head -n 1)
    if [ -z "$EXTRACTED_DIR" ]; then
        echo "Error: Could not find extracted JDK directory."
    else
        mv "$EXTRACTED_DIR" "$JAVA_DIR"
        rm "$WORK_DIR/openjdk.tar.gz"
    fi
fi

# Set JAVA_HOME and PATH for this session
export JAVA_HOME="$JAVA_DIR/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"

if "$JAVA_HOME/bin/java" -version >/dev/null 2>&1; then
    echo "Local Java installed successfully."
    "$JAVA_HOME/bin/java" -version
else
    echo "Failed to install/verify local Java."
fi


# --- 2. Install Lean 4 (Local Elan) ---
# We will install elan to the work directory if possible, 
# but elan defaults to ~/.elan. We'll try the standard install 
# but skip the profile update part to avoid permission errors.

export ELAN_HOME="$WORK_DIR/.elan"
export CARGO_HOME="$WORK_DIR/.cargo" # Elan might use this? 
export RUSTUP_HOME="$WORK_DIR/.rustup"

echo "Downloading Elan init script..."
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf > "$WORK_DIR/elan-init.sh"
chmod +x "$WORK_DIR/elan-init.sh"

echo "Installing Elan to $ELAN_HOME..."
# -y: Assume yes
# --no-modify-path: Don't try to edit .zshrc/.bashrc
"$WORK_DIR/elan-init.sh" -y --no-modify-path --default-toolchain stable

# Add to PATH
export PATH="$ELAN_HOME/bin:$PATH"

if "$ELAN_HOME/bin/lean" --version; then
    echo "Local Lean installed successfully."
else
    echo "Failed to verify local Lean."
fi

# Create a helper script to run commands with these paths
echo "#!/bin/bash" > env_wrapper.sh
echo "export JAVA_HOME=\"$JAVA_DIR/Contents/Home\"" >> env_wrapper.sh
echo "export ELAN_HOME=\"$ELAN_HOME\"" >> env_wrapper.sh
echo "export PATH=\"\$JAVA_HOME/bin:\$ELAN_HOME/bin:\$PATH\"" >> env_wrapper.sh
echo "exec \"\$@\"" >> env_wrapper.sh
chmod +x env_wrapper.sh

echo "Installation complete. Use ./env_wrapper.sh <command> to run with tools."
