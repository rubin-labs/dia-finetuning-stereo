#!/bin/bash
# Last resort: Try older torch_xla versions that might work with GLIBC 2.31
# WARNING: This may not work and older versions may have bugs

set -e

echo "=== Trying older torch_xla versions for GLIBC 2.31 compatibility ==="
echo ""

# Get current torch version
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "2.9.0")
echo "Current PyTorch version: $TORCH_VER"
echo ""

# Uninstall current torch_xla
echo "Uninstalling current torch_xla..."
pip uninstall -y torch_xla 2>/dev/null || true

# Try older versions (these might work with GLIBC 2.31)
OLD_VERSIONS=("2.8.0" "2.7.0" "2.6.0" "2.5.0")

for version in "${OLD_VERSIONS[@]}"; do
    echo ""
    echo "Trying torch_xla $version..."
    pip install "torch_xla==${version}" -f https://storage.googleapis.com/libtpu-releases/index.html || {
        echo "❌ torch_xla $version not available or installation failed"
        continue
    }
    
    echo "Testing import..."
    if python -c "import torch_xla; print('✓ SUCCESS with torch_xla', torch_xla.__version__)" 2>/dev/null; then
        echo ""
        echo "=== SUCCESS! torch_xla $version works! ==="
        echo "Note: This is an older version. Make sure it's compatible with your PyTorch version."
        exit 0
    else
        ERROR=$(python -c "import torch_xla" 2>&1 | head -1)
        echo "❌ Failed: $ERROR"
        pip uninstall -y torch_xla 2>/dev/null || true
    fi
done

echo ""
echo "❌ All older versions failed. You MUST use Ubuntu 22.04."
echo "See scripts/GLIBC_FIX_SOLUTION.md for instructions."


