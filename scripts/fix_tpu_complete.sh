#!/bin/bash
# Complete fix for TPU environment (libpython + GLIBC)
# Run this on your TPU VM

set -e

echo "=== Complete TPU Environment Fix ==="
echo ""

# Get conda environment path properly
CONDA_BASE=$(conda info --base)
ACTIVE_ENV=$(conda env list | grep '*' | awk '{print $1}' | sed 's/*//' | head -1)
CONDA_ENV_PATH="$CONDA_BASE/envs/$ACTIVE_ENV"

if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "❌ Could not find conda environment: $CONDA_ENV_PATH"
    exit 1
fi

echo "Conda environment: $CONDA_ENV_PATH"
echo ""

# Step 1: Ensure libpython3.11.so.1.0 exists
echo "Step 1: Checking libpython3.11.so.1.0..."
LIBPYTHON="$CONDA_ENV_PATH/lib/libpython3.11.so.1.0"
if [ ! -f "$LIBPYTHON" ]; then
    # Try to find any libpython3.11.so* and create symlink
    ALT_LIB=$(find "$CONDA_ENV_PATH/lib" -name "libpython3.11.so*" 2>/dev/null | head -1)
    if [ -n "$ALT_LIB" ]; then
        echo "Creating symlink: $LIBPYTHON -> $ALT_LIB"
        ln -sf "$ALT_LIB" "$LIBPYTHON"
    else
        echo "❌ libpython3.11.so* not found in $CONDA_ENV_PATH/lib"
        exit 1
    fi
fi
echo "✓ libpython3.11.so.1.0 found"

# Step 2: Install conda's libc (for GLIBC compatibility)
echo ""
echo "Step 2: Installing conda's libc (for GLIBC compatibility)..."
conda install -y -c conda-forge libcxx libstdcxx-ng 2>/dev/null || echo "⚠ libc already installed or installation failed"

# Step 3: Set LD_LIBRARY_PATH properly (conda libs first, then system)
echo ""
echo "Step 3: Setting LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$CONDA_BASE/lib:$LD_LIBRARY_PATH"

# Remove any weird entries and clean up
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '^$' | grep -v '#' | sort -u | tr '\n' ':' | sed 's/:$//')

# Add to bashrc (clean version)
{
    echo ""
    echo "# TPU environment fixes"
    echo "export LD_LIBRARY_PATH=\"$CONDA_ENV_PATH/lib:$CONDA_BASE/lib:\$LD_LIBRARY_PATH\""
    echo "export XRT_TPU_CONFIG=\"localservice;0;localhost:51011\""
    echo "export CUDA_VISIBLE_DEVICES=\"\""
} >> ~/.bashrc

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export CUDA_VISIBLE_DEVICES=""

echo "✓ LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
echo ""

# Step 4: Test
echo "Step 4: Testing torch_xla import..."
python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch_xla
import torch_xla.core.xla_model as xm
print('✓ SUCCESS: torch_xla imported successfully!')
print('✓ torch_xla version:', torch_xla.__version__)
" && {
    echo ""
    echo "=== Fix complete! You can now run training. ==="
} || {
    echo ""
    echo "❌ Import still failing. Error details:"
    python -c "import torch_xla" 2>&1 | head -5
    echo ""
    echo "Try:"
    echo "  1. Restart terminal: exit and reconnect"
    echo "  2. Run: source ~/.bashrc"
    echo "  3. Try again"
    exit 1
}

