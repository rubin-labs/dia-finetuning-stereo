#!/bin/bash
# Fix GLIBC version mismatch for torch_xla on Ubuntu 20.04
# torch_xla requires GLIBC 2.34, but Ubuntu 20.04 has 2.31
# Solution: Use conda's libc or install compatible torch_xla

set -e

echo "=== Fixing GLIBC version mismatch ==="
echo ""

# Check current GLIBC version
echo "Current system GLIBC version:"
ldd --version | head -1

# Option 1: Install conda's libc (recommended)
echo ""
echo "Installing conda's libc (compatible with torch_xla)..."
conda install -y -c conda-forge libcxx libstdcxx-ng

# Option 2: Add conda's lib directory to LD_LIBRARY_PATH
CONDA_ENV_PATH=$(conda info --base)/envs/$(conda info --envs | grep '*' | awk '{print $1}' | sed 's/*//')
CONDA_BASE=$(conda info --base)

# Find conda's lib directories
CONDA_LIB_PATHS=(
    "$CONDA_ENV_PATH/lib"
    "$CONDA_BASE/lib"
    "$CONDA_ENV_PATH/lib/python3.11/site-packages/torch/lib"
)

for lib_path in "${CONDA_LIB_PATHS[@]}"; do
    if [ -d "$lib_path" ]; then
        export LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH"
        echo "export LD_LIBRARY_PATH=\"$lib_path:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
        echo "✓ Added $lib_path to LD_LIBRARY_PATH"
    fi
done

# Also ensure libpython is still there
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
echo "export LD_LIBRARY_PATH=\"$CONDA_ENV_PATH/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc

export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export CUDA_VISIBLE_DEVICES=""
echo "export XRT_TPU_CONFIG=\"localservice;0;localhost:51011\"" >> ~/.bashrc
echo "export CUDA_VISIBLE_DEVICES=\"\"" >> ~/.bashrc

echo ""
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# Test
echo "Testing torch_xla import..."
python -c "import torch_xla; print('✓ SUCCESS: torch_xla imported!')" && {
    echo ""
    echo "=== Fix complete! ==="
} || {
    echo ""
    echo "❌ Still failing. Trying alternative: reinstall torch_xla from conda-forge"
    echo "This may take a few minutes..."
    pip uninstall -y torch_xla
    conda install -y -c conda-forge pytorch-cuda=12.1 pytorch torchvision torchaudio pytorch-cuda -c pytorch
    pip install torch_xla -f https://storage.googleapis.com/libtpu-releases/index.html || {
        echo ""
        echo "❌ All methods failed."
        echo "The torch_xla package may not be compatible with Ubuntu 20.04."
        echo "Consider using a newer Ubuntu version (22.04+) or a different torch_xla build."
        exit 1
    }
    python -c "import torch_xla; print('✓ SUCCESS after reinstall!')"
}

