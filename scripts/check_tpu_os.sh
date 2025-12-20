#!/bin/bash
# Check what OS version your TPU VM is running

echo "=== Checking TPU VM OS Version ==="
echo ""

# Check Ubuntu version
if command -v lsb_release >/dev/null 2>&1; then
    echo "Ubuntu Version:"
    lsb_release -a
else
    echo "lsb_release not found, checking /etc/os-release..."
    cat /etc/os-release | grep -E "VERSION|PRETTY_NAME"
fi

echo ""
echo "GLIBC Version:"
ldd --version | head -1

echo ""
echo "Kernel Version:"
uname -r

echo ""
echo "=== Compatibility Check ==="
GLIBC_VER=$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+')
echo "Your GLIBC version: $GLIBC_VER"

if (( $(echo "$GLIBC_VER >= 2.34" | bc -l 2>/dev/null || echo "0") )); then
    echo "✓ GLIBC version is compatible with torch_xla 2.9.0"
else
    echo "❌ GLIBC version is TOO OLD for torch_xla 2.9.0 (needs 2.34+)"
    echo ""
    echo "Your TPU VM image (tpu-vm-v4-pt-2.0) appears to be Ubuntu 20.04"
    echo "You need to recreate it with a newer image that uses Ubuntu 22.04"
fi


