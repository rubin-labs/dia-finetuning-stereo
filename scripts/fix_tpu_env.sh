#!/bin/bash
# Quick fix for TPU environment issues
# Run this on your TPU VM before training

set -e

echo "Fixing TPU environment..."

# 1. Fix fsspec version conflict
echo "Fixing fsspec version..."
pip install "fsspec[http]<=2025.10.0,>=2023.1.0" --force-reinstall

# 2. Reinstall NVIDIA packages (needed for PyTorch even on TPU, but won't be used)
echo "Reinstalling NVIDIA CUDA packages (required by PyTorch, but TPU uses XLA)..."
pip install --force-reinstall \
    nvidia-cublas-cu12==12.8.4.1 \
    nvidia-cuda-cupti-cu12==12.8.90 \
    nvidia-cuda-nvrtc-cu12==12.8.93 \
    nvidia-cuda-runtime-cu12==12.8.90 \
    nvidia-cudnn-cu12==9.10.2.21 \
    nvidia-cufft-cu12==11.3.3.83 \
    nvidia-cufile-cu12==1.13.1.3 \
    nvidia-curand-cu12==10.3.9.90 \
    nvidia-cusolver-cu12==11.7.3.90 \
    nvidia-cusparse-cu12==12.5.8.93 \
    nvidia-cusparselt-cu12==0.7.1 \
    nvidia-nccl-cu12==2.27.5 \
    nvidia-nvjitlink-cu12==12.8.93 \
    nvidia-nvshmem-cu12==3.3.20 \
    nvidia-nvtx-cu12==12.8.90

# 3. Fix libpython path for torch_xla
echo "Finding Python shared library..."
CONDA_ENV=$(conda info --envs | grep '*' | awk '{print $1}' | sed 's/*//')
CONDA_BASE=$(conda info --base)
CONDA_ENV_PATH="$CONDA_BASE/envs/$CONDA_ENV"

# Find Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.11")

# Try multiple potential locations for libpython
LIB_PATHS=(
    "$CONDA_ENV_PATH/lib"
    "$CONDA_ENV_PATH/lib/python${PYTHON_VERSION}"
    "$CONDA_BASE/lib"
    "$CONDA_BASE/lib/python${PYTHON_VERSION}"
)

# Also try Python's sysconfig
PYTHON_LIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null || echo "")
if [ -n "$PYTHON_LIB" ] && [ -d "$PYTHON_LIB" ]; then
    LIB_PATHS+=("$PYTHON_LIB")
fi

# Find the actual libpython file
LIBPYTHON_FILE=""
for lib_path in "${LIB_PATHS[@]}"; do
    if [ -d "$lib_path" ]; then
        # Look for libpython*.so* files
        found_lib=$(find "$lib_path" -name "libpython${PYTHON_VERSION}.so*" -o -name "libpython*.so*" 2>/dev/null | head -1)
        if [ -n "$found_lib" ] && [ -f "$found_lib" ]; then
            lib_dir=$(dirname "$found_lib")
            export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
            echo "export LD_LIBRARY_PATH=\"$lib_dir:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
            echo "✓ Found libpython at: $found_lib"
            echo "✓ Added $lib_dir to LD_LIBRARY_PATH"
            LIBPYTHON_FILE="$found_lib"
            break
        fi
    fi
done

# If not found, still add the main conda lib directory
if [ -z "$LIBPYTHON_FILE" ] && [ -d "$CONDA_ENV_PATH/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
    echo "export LD_LIBRARY_PATH=\"$CONDA_ENV_PATH/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    echo "✓ Added conda lib directory to LD_LIBRARY_PATH (libpython not found, but directory added)"
fi

# Show current LD_LIBRARY_PATH for debugging
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 4. Set TPU environment variables
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
echo "export XRT_TPU_CONFIG=\"localservice;0;localhost:51011\"" >> ~/.bashrc

# 5. Disable CUDA at runtime (TPU doesn't need it, but PyTorch checks for it)
export CUDA_VISIBLE_DEVICES=""
export TORCH_CUDA_ARCH_LIST=""
echo "export CUDA_VISIBLE_DEVICES=\"\"" >> ~/.bashrc

# 6. Test torch_xla
echo ""
echo "Testing torch_xla import..."
python -c "
import os
import sys

# Set environment variables BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

try:
    import torch
    print('✓ torch:', torch.__version__)
except Exception as e:
    print('❌ torch import failed:', e)
    sys.exit(1)

try:
    import torch_xla
    print('✓ torch_xla:', torch_xla.__version__)
except Exception as e:
    print('❌ torch_xla import failed:', e)
    print('  Error details:', str(e))
    sys.exit(1)

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print('✓ TPU device:', device)
except Exception as e:
    print('⚠ TPU device check failed:', e)
    print('  (This might be OK if TPU is not initialized yet)')
" || {
    echo ""
    echo "❌ torch_xla import failed!"
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check if libpython exists:"
    echo "   find \$(conda info --base)/envs/\$(conda info --envs | grep '*' | awk '{print \$1}' | sed 's/*//')/lib -name 'libpython*.so*'"
    echo ""
    echo "2. Manually set LD_LIBRARY_PATH:"
    echo "   export LD_LIBRARY_PATH=\$(conda info --base)/envs/\$(conda info --envs | grep '*' | awk '{print \$1}' | sed 's/*//')/lib:\$LD_LIBRARY_PATH"
    echo ""
    echo "3. Verify current LD_LIBRARY_PATH:"
    echo "   echo \$LD_LIBRARY_PATH"
    exit 1
}

echo ""
echo "✓ Environment fixed! You can now run training."

