# GLIBC 2.34 Fix for Ubuntu 20.04 TPU VMs

## The Problem

`torch_xla 2.9.0` requires GLIBC 2.34, but Ubuntu 20.04 only has GLIBC 2.31. This is a **binary compatibility issue** that cannot be easily worked around.

## The Solution: Use Ubuntu 22.04

**You need to recreate your TPU VM with Ubuntu 22.04**, which has GLIBC 2.35 and is compatible.

### Step 1: Delete Current TPU VM

```bash
# Get your TPU name and zone
TPU_NAME="your-tpu-name"  # Replace with actual name
ZONE="us-central2-b"      # Replace with your zone

# Delete the TPU VM
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE
```

### Step 2: Create New TPU VM with Ubuntu 22.04

**IMPORTANT**: `tpu-vm-v4-pt-2.0` uses Ubuntu 20.04 (GLIBC 2.31) - that's why it's not compatible!

Use the Ubuntu 22.04 base image instead:

```bash
# For TPU v4 with Ubuntu 22.04 (has GLIBC 2.35)
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base \
    --network=default \
    --subnetwork=default
```

**Note**: This image doesn't have PyTorch pre-installed, so you'll need to install it manually (which is fine - you're already doing that).

### Step 3: Verify Ubuntu Version

After SSH'ing into the new VM:

```bash
lsb_release -a
# Should show Ubuntu 22.04

ldd --version
# Should show GLIBC 2.35 or higher
```

### Step 4: Set Up Environment on New VM

```bash
# Run the complete fix script
bash scripts/fix_tpu_complete.sh
```

## Alternative: Try Older torch_xla Version (Not Recommended)

If you absolutely cannot recreate the VM, you could try an older torch_xla version, but this is **NOT RECOMMENDED** as it may not work with your PyTorch version:

```bash
# Uninstall current version
pip uninstall -y torch_xla

# Try older versions (may not work)
pip install torch_xla==2.8.0 -f https://storage.googleapis.com/libtpu-releases/index.html
# OR
pip install torch_xla==2.7.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

**Warning**: Older versions may not be compatible with PyTorch 2.9.0 and may have bugs.

## Why This Happens

- Ubuntu 20.04 (Focal): GLIBC 2.31
- Ubuntu 22.04 (Jammy): GLIBC 2.35
- torch_xla 2.9.0: Compiled against GLIBC 2.34+

The binary was compiled on a system with GLIBC 2.34+, so it requires that version or higher.

## Quick Reference

**Current Status**: ❌ Cannot run torch_xla 2.9.0 on Ubuntu 20.04

**Solution**: ✅ Recreate TPU VM (takes ~5 minutes)

**After Fix**: ✅ Everything should work

