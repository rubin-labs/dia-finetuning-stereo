#!/bin/bash
# Fix libpython3.11.so.1.0 missing issue for torch_xla
# Run this on your TPU VM

set -e

echo "=== Fixing libpython3.11.so.1.0 issue ==="
echo ""

# Step 1: Try to find libpython3.11.so* anywhere on the system
echo "Step 1: Searching for libpython3.11.so*..."
LIBPYTHON=$(find /usr/lib* /lib* $(conda info --base)/lib $(conda info --base)/envs/*/lib -name "libpython3.11.so*" 2>/dev/null | head -1)

if [ -z "$LIBPYTHON" ]; then
    echo "❌ libpython3.11.so* not found in standard locations"
    echo "Trying conda base installation..."
    CONDA_BASE=$(conda info --base)
    LIBPYTHON=$(find "$CONDA_BASE/lib" -name "libpython3.11.so*" 2>/dev/null | head -1)
fi

if [ -n "$LIBPYTHON" ]; then
    LIB_DIR=$(dirname "$LIBPYTHON")
    echo "✓ Found libpython: $LIBPYTHON"
    echo "✓ Library directory: $LIB_DIR"
    
    # Check if .1.0 version exists, if not create symlink
    if [ ! -f "$LIB_DIR/libpython3.11.so.1.0" ]; then
        if [ -f "$LIB_DIR/libpython3.11.so" ]; then
            echo "Creating symlink: libpython3.11.so.1.0 -> libpython3.11.so"
            sudo ln -sf "$LIB_DIR/libpython3.11.so" "$LIB_DIR/libpython3.11.so.1.0" 2>/dev/null || \
            ln -sf "$LIB_DIR/libpython3.11.so" "$LIB_DIR/libpython3.11.so.1.0"
        elif [ -f "$LIB_DIR/libpython3.11.so.1" ]; then
            echo "Creating symlink: libpython3.11.so.1.0 -> libpython3.11.so.1"
            sudo ln -sf "$LIB_DIR/libpython3.11.so.1" "$LIB_DIR/libpython3.11.so.1.0" 2>/dev/null || \
            ln -sf "$LIB_DIR/libpython3.11.so.1" "$LIB_DIR/libpython3.11.so.1.0"
        fi
    fi
    
    # Add to LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
    echo "export LD_LIBRARY_PATH=\"$LIB_DIR:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    echo "✓ Added $LIB_DIR to LD_LIBRARY_PATH (and ~/.bashrc)"
    
    # Also create symlink in conda environment
    CONDA_ENV_PATH=$(conda info --base)/envs/$(conda info --envs | grep '*' | awk '{print $1}' | sed 's/*//')
    if [ -d "$CONDA_ENV_PATH/lib" ] && [ ! -f "$CONDA_ENV_PATH/lib/libpython3.11.so.1.0" ]; then
        echo "Creating symlink in conda environment..."
        ln -sf "$LIBPYTHON" "$CONDA_ENV_PATH/lib/libpython3.11.so.1.0" 2>/dev/null || \
        sudo ln -sf "$LIBPYTHON" "$CONDA_ENV_PATH/lib/libpython3.11.so.1.0"
        echo "✓ Created symlink: $CONDA_ENV_PATH/lib/libpython3.11.so.1.0 -> $LIBPYTHON"
        export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
        echo "export LD_LIBRARY_PATH=\"$CONDA_ENV_PATH/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    fi
else
    echo "❌ Could not find any libpython3.11.so* library"
    echo ""
    echo "Trying to install from deadsnakes PPA (for Ubuntu 20.04)..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
    sudo apt-get update -qq
    sudo apt-get install -y python3.11-dev python3.11-venv python3.11-distutils || {
        echo ""
        echo "❌ Installation failed. Trying alternative: use conda's Python shared library"
        CONDA_BASE=$(conda info --base)
        PYTHON_EXE=$(which python)
        PYTHON_LIB_DIR=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null || echo "")
        if [ -n "$PYTHON_LIB_DIR" ] && [ -d "$PYTHON_LIB_DIR" ]; then
            echo "Using Python's LIBDIR: $PYTHON_LIB_DIR"
            export LD_LIBRARY_PATH="$PYTHON_LIB_DIR:$LD_LIBRARY_PATH"
            echo "export LD_LIBRARY_PATH=\"$PYTHON_LIB_DIR:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
        else
            echo "❌ All methods failed. Please check your Python installation."
            exit 1
        fi
    }
fi

# Step 2: Set other required environment variables
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export CUDA_VISIBLE_DEVICES=""
echo "export XRT_TPU_CONFIG=\"localservice;0;localhost:51011\"" >> ~/.bashrc
echo "export CUDA_VISIBLE_DEVICES=\"\"" >> ~/.bashrc

echo ""
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Step 3: Test
echo "Step 3: Testing torch_xla import..."
python -c "import torch_xla; print('✓ SUCCESS: torch_xla imported successfully!')" && {
    echo ""
    echo "=== Fix complete! You can now run training. ==="
} || {
    echo ""
    echo "❌ Import still failing. Try:"
    echo "   1. Restart your terminal session"
    echo "   2. Run: source ~/.bashrc"
    echo "   3. Try the import again"
    exit 1
}

