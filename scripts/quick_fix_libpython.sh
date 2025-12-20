#!/bin/bash
# Quick fix for libpython3.11.so.1.0 error
# Run this if you get the ImportError when trying to use torch_xla

CONDA_ENV_PATH=$(conda info --base)/envs/$(conda info --envs | grep '*' | awk '{print $1}' | sed 's/*//')

# Find libpython
LIBPYTHON=$(find "$CONDA_ENV_PATH/lib" -name "libpython*.so*" 2>/dev/null | head -1)

if [ -n "$LIBPYTHON" ]; then
    LIB_DIR=$(dirname "$LIBPYTHON")
    export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"
    echo "export LD_LIBRARY_PATH=\"$LIB_DIR:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    echo "✓ Found libpython: $LIBPYTHON"
    echo "✓ Added $LIB_DIR to LD_LIBRARY_PATH"
else
    # Fallback: add the lib directory anyway
    export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
    echo "export LD_LIBRARY_PATH=\"$CONDA_ENV_PATH/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    echo "⚠ libpython not found, but added $CONDA_ENV_PATH/lib to LD_LIBRARY_PATH"
fi

# Set other required env vars
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export CUDA_VISIBLE_DEVICES=""
echo "export XRT_TPU_CONFIG=\"localservice;0;localhost:51011\"" >> ~/.bashrc
echo "export CUDA_VISIBLE_DEVICES=\"\"" >> ~/.bashrc

echo ""
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""
echo "Testing torch_xla import..."
python -c "import torch_xla; print('✓ torch_xla imported successfully!')" && echo "✓ Ready to run training!" || echo "❌ Import still failing - check the error above"

