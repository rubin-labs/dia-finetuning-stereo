# Mixed Precision Training Bug Fix - December 17, 2025

## Problem

Training crashed at step 400 with:
```
RuntimeError: expected dtype float for `end` but got dtype c10::BFloat16
```

The crash occurred in `torch.optim.adam._multi_tensor_adam` during `opt.step()`.

## Root Cause

The `Accelerator()` from HuggingFace Accelerate **auto-detects** the best mixed precision mode for your GPU. On modern NVIDIA GPUs (L40S, A100, RTX 30xx/40xx), it automatically selects **bfloat16 (bf16)**.

**The problem with bf16:**
- Model weights stay in float32
- `accelerator.autocast()` casts activations to bf16
- Gradients come back in bf16
- Adam optimizer has float32 states but tries to update with bf16 gradients
- `torch._foreach_lerp_()` fails on the dtype mismatch

Early warning sign in logs:
```
UserWarning: Mismatch dtype between input and weight: input dtype = c10::BFloat16, weight dtype = float
```

## Solutions Attempted

### 1. `accelerate launch --mixed_precision=fp16` ❌
- Would work but caused GPU access issues due to:
  - GPUs in "Exclusive Process" compute mode
  - Multi-GPU launch race condition (both processes tried `cuda:0`)
  - Zombie processes holding CUDA contexts

### 2. `mixed_precision="fp16"` in code ❌
- FP16 uses loss scaling (GradScaler) which should handle dtype conversion
- Still failed with: `"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`
- Indicates something in the pipeline was still producing bf16 tensors

### 3. `mixed_precision="no"` in code ✅
- Disables all automatic mixed precision
- Uses full float32 throughout
- Slightly slower but completely avoids dtype issues

## Final Fix

In `dia/train_acc_gpu.py`, line ~587:

```python
# Before (auto-detects bf16 on modern GPUs):
accelerator = Accelerator()

# After (disable mixed precision):
accelerator = Accelerator(mixed_precision="no")
```

## Mixed Precision Options Explained

| Setting | Speed | Memory | Stability | Notes |
|---------|-------|--------|-----------|-------|
| `"no"` | 1x | High | ✅ Safe | Full fp32, no dtype issues |
| `"fp16"` | ~2x | Low | ⚠️ Usually safe | Uses loss scaling, well-tested |
| `"bf16"` | ~2x | Low | ❌ Crashed | No loss scaling, caused our bug |

## Other Issues Encountered

### Zombie Processes
After crashes, Python processes can become zombies holding CUDA contexts:
```bash
ps aux | grep python  # Look for <defunct> or Tl status
pkill -9 -u $USER python  # Kill all your python processes
```

### Exclusive Process GPU Mode
The L40S GPUs were in "E. Process" (Exclusive Process) compute mode, meaning only one CUDA context per GPU. This caused race conditions with `accelerate launch --num_processes=2`.

### Port Conflicts
Default port 29500 was in use from crashed distributed training runs. Solution: `--main_process_port=29501` or kill zombie processes.

## Recommendations

1. **For stability**: Use `mixed_precision="no"` until training pipeline is verified working
2. **For speed later**: Try `mixed_precision="fp16"` (not bf16)
3. **For multi-GPU**: Need to fix code to use `LOCAL_RANK` environment variable for device assignment
4. **After crashes**: Always check `nvidia-smi` and `ps aux | grep python` for zombie processes

