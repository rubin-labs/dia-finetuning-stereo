#!/bin/bash
# Setup script for TPU environment
# Fixes common issues with torch_xla and conda environments

set -e

echo "Setting up TPU environment..."

# Get conda environment path
CONDA_ENV_PATH=$(conda info --base)/envs/$(conda info --envs | grep '*' | awk '{print $1}' | sed 's/*//')

# Fix libpython path issue for torch_xla
if [ -d "$CONDA_ENV_PATH/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
    echo "Added $CONDA_ENV_PATH/lib to LD_LIBRARY_PATH"
fi

# Fix fsspec version conflict
echo "Fixing fsspec version conflict..."
pip install "fsspec[http]<=2025.10.0,>=2023.1.0" --force-reinstall

# Verify torch_xla installation
echo "Verifying torch_xla installation..."
python -c "import torch_xla; print('torch_xla version:', torch_xla.__version__)" || {
    echo "ERROR: torch_xla import failed. Trying to fix..."
    
    # Try to find and add Python library path
    PYTHON_LIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    if [ -d "$PYTHON_LIB" ]; then
        export LD_LIBRARY_PATH="$PYTHON_LIB:$LD_LIBRARY_PATH"
        echo "Added $PYTHON_LIB to LD_LIBRARY_PATH"
    fi
    
    # Also check conda's lib directory
    if [ -d "$CONDA_ENV_PATH/lib/python3.11" ]; then
        export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib/python3.11:$LD_LIBRARY_PATH"
        echo "Added $CONDA_ENV_PATH/lib/python3.11 to LD_LIBRARY_PATH"
    fi
}

# Set TPU environment variables
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export TPU_NAME="${TPU_NAME:-dia-training-tpu}"

echo "Environment setup complete!"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "XRT_TPU_CONFIG: $XRT_TPU_CONFIG"

# Test torch_xla import
python -c "
import torch
import torch_xla
import torch_xla.core.xla_model as xm
print('✓ torch:', torch.__version__)
print('✓ torch_xla:', torch_xla.__version__)
try:
    device = xm.xla_device()
    print('✓ TPU device:', device)
except Exception as e:
    print('⚠ TPU device check failed (this is OK if not on TPU):', e)
"

