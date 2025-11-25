# Comprehensive Bug Report - Dia Finetuning Stereo

**Generated:** November 24, 2025  
**Scope:** Full codebase scan for potential bugs, issues, and improvements

---

## Table of Contents
1. [Critical Bugs](#critical-bugs)
2. [High Severity Issues](#high-severity-issues)
3. [Medium Severity Issues](#medium-severity-issues)
4. [Low Severity Issues](#low-severity-issues)
5. [Code Quality & Best Practices](#code-quality--best-practices)

---

## Critical Bugs

### 1. ❌ Stereo Audio Speed Adjustment Fails for 2D Arrays
**File:** `app.py`, `app_local.py`  
**Lines:** ~193-166  
**Severity:** Critical

```python
# BUG: output_audio_np may be (2, N) for stereo but np.interp only works on 1D
original_len = len(output_audio_np)  # Returns 2 for stereo (2, N) instead of N
target_len = int(original_len / speed_factor)  # Wrong calculation
resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)  # Fails or wrong for 2D
```

**Problem:** When generating stereo audio, `output_audio_np` has shape `(2, N)` where `len()` returns `2` (number of channels), not the number of samples. The resampling/speed adjustment logic assumes 1D audio.

**Fix:**
```python
# Handle both mono (N,) and stereo (2, N) or (N, 2)
if output_audio_np.ndim == 2:
    if output_audio_np.shape[0] == 2:  # (2, N) format
        original_len = output_audio_np.shape[1]
        x_original = np.arange(original_len)
        x_resampled = np.linspace(0, original_len - 1, target_len)
        resampled_audio_np = np.array([
            np.interp(x_resampled, x_original, output_audio_np[0]),
            np.interp(x_resampled, x_original, output_audio_np[1])
        ])
    else:  # (N, 2) format
        original_len = output_audio_np.shape[0]
        # ... similar handling
else:
    original_len = len(output_audio_np)
    # existing 1D logic
```

---

### 2. ❌ Deprecated `torch.cuda.amp.autocast` Usage
**File:** `finetune_acc.py`  
**Lines:** 521, 590  
**Severity:** Critical (Will break in future PyTorch versions)

```python
# Deprecated pattern
with torch.cuda.amp.autocast(enabled=False):  # Line 590

# Should use
with torch.amp.autocast('cuda', enabled=False):
```

**Problem:** `torch.cuda.amp.autocast` is deprecated since PyTorch 2.0. The newer `torch.amp.autocast` API requires specifying device type.

**Note:** Line 955 uses the correct new API: `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`

---

### 3. ❌ Loss Not Properly Scaled Before Logging After Early Stop
**File:** `finetune_acc.py`  
**Lines:** ~1019-1027  
**Severity:** High

```python
loss = loss / train_cfg.grad_accum_steps  # Line 1001 - loss is divided
loss.backward()

# ...later...
return loss.item() * train_cfg.grad_accum_steps  # Line 1027 - scaled back up

# But if early stop triggers BEFORE the multiply-back...
if args.early_stop_loss is not None:
    trigger_local = loss <= args.early_stop_loss  # Comparing scaled-down loss!
```

**Problem:** The `early_stop_loss` comparison happens with the already-divided loss value (divided by `grad_accum_steps`). This means the actual threshold is different from what the user specifies.

**Fix:** Store the true loss before division:
```python
true_loss = loss.item()
loss = loss / train_cfg.grad_accum_steps
loss.backward()
# Use true_loss for comparison and logging
```

---

## High Severity Issues

### 4. ⚠️ `max_new_tokens` Parameter Ignored in Generation
**File:** `app.py`, `app_local.py`  
**Lines:** ~236-243  

```python
max_new_tokens = gr.Slider(
    label="Max New Tokens (Audio Length)",
    minimum=860,
    maximum=3072,
    value=model.config.data.audio_length,
    # ...
)
```

**Problem:** The `max_new_tokens` slider value is passed to `run_inference()` but never used - the `model.generate()` call doesn't accept this parameter. Generation always uses `config.data.audio_length`.

**Fix:** Either remove the slider or modify `Dia.generate()` to accept a `max_tokens` parameter.

---

### 5. ⚠️ Inconsistent Channel Count Between Config and Delay Pattern
**File:** `dia/config.json`  
**Severity:** High

```json
"data": {
    "channels": 18,
    "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15, 0, 8, 9, 10, 11, 12, 13, 14, 15]
}
```

**Problem:** The delay pattern has 18 values which is correct for stereo, but the code in `audio.py` and `model.py` sometimes assumes 9-channel mono patterns. The `codebook_to_audio` function passes `C=9` by default:

```python
# model.py line 442
audio = codebook_to_audio(
    generated_codes.transpose(1, 0), self.dac_model, delay_pattern, B=1, T=max_tokens, C=num_channels
)
```

This is actually correct since `num_channels` is from config, but hardcoded `C=9` appears elsewhere.

---

### 6. ⚠️ `seed_everything` Called Without Proper Rank Offset in Single GPU Path
**File:** `finetune_acc.py`  
**Lines:** 1199-1200  

```python
# Single GPU fallback
seed_everything(args.seed, rank=0)  # Correct
```

But in `run_ddp_worker`:
```python
seed_everything(args.seed, rank=rank)  # Line 1037
```

**Issue:** The seed function does `random.seed(seed + rank)` which is correct, but `np.random.seed()` only accepts 32-bit integers. If `seed + rank` exceeds `2^32 - 1`, it will wrap or error.

**Fix:**
```python
np.random.seed((seed + rank) % (2**32))
```

---

### 7. ⚠️ Global Variables Modified Without Thread Safety
**File:** `finetune_acc.py`  
**Lines:** 65-67, 1048-1051, 1207-1210  

```python
# Global declarations
TAG_SHUFFLE = True
TAG_DROPOUT = 0.0
TAG_LIMIT = None

# Modified in multiple places
global TAG_SHUFFLE, TAG_DROPOUT, TAG_LIMIT
TAG_SHUFFLE = False if getattr(args, 'tag_no_shuffle', False) else ...
```

**Problem:** These globals are modified at runtime. In multi-process DDP spawning, each process gets its own copy (which is fine), but this pattern is fragile and error-prone.

---

### 8. ⚠️ Potential Division by Zero in Loss Calculation
**File:** `finetune_acc.py`  
**Lines:** 555, 999  

```python
loss_e = loss_e / sum(weights_e)  # Line 555
loss = loss_c / sum(channel_weights)  # Line 999
```

**Problem:** If `weights_e` or `channel_weights` is empty (e.g., if `C == 0`), this causes division by zero.

**Fix:** Add guard:
```python
weight_sum = sum(channel_weights)
if weight_sum == 0:
    raise ValueError("Channel weights sum to zero")
loss = loss_c / weight_sum
```

---

### 9. ⚠️ Mask Creation Uses Wrong Length After Cropping
**File:** `finetune_acc.py`  
**Lines:** 343  

```python
seq_lens = [e.size(0) for e in encodings]  # Original lengths before padding
```

**Problem:** `seq_lens` captures lengths AFTER the cropping in lines 298-304 but BEFORE padding. This is technically correct, but the variable name and comment are misleading. The comment says "Original lengths before padding" but they're actually "lengths after cropping, before padding."

---

## Medium Severity Issues

### 10. 🔸 Encoder Freezing Threshold Too Low
**File:** `finetune_acc.py`  
**Lines:** 1308-1314  

```python
if train_cfg.unconditional_frac >= 0.9:
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("Frozen encoder parameters due to high unconditional_frac")
```

**Problem:** This freezing logic only appears in the single-GPU path, not in `run_ddp_worker`. DDP training won't freeze the encoder even with `unconditional_frac >= 0.9`.

---

### 11. 🔸 Missing Gradient Clipping Logging in DDP Worker
**File:** `finetune_acc_lora.py`  
**Line:** ~302  

The LoRA training imports `train()` from `finetune_acc` but doesn't have the same gradient norm logging that exists in `train_step_ddp`. This is an inconsistency.

---

### 12. 🔸 `steps_per_epoch` Calculation Redundant
**File:** `finetune_acc.py`  
**Lines:** 437-440  

```python
steps_per_epoch = len(train_loader)
train_loader.steps_per_epoch = steps_per_epoch  # Redundant - len() works fine
```

**Problem:** This adds an attribute that's already accessible via `len()`. Then in `setup_optimizer_and_scheduler`:

```python
try:
    steps_per_epoch = len(train_loader)
except TypeError:
    if hasattr(train_loader, 'steps_per_epoch'):
        steps_per_epoch = train_loader.steps_per_epoch
```

This fallback is only needed for iterators that don't support `len()`, which `DataLoader` always does.

---

### 13. 🔸 Checkpoint State Dict Reference After Delete
**File:** `finetune_acc.py`, `finetune_acc_lora.py`  
**Lines:** 1151-1152 (finetune_acc.py), 291-294 (finetune_acc_lora.py)

```python
del state
gc.collect()
# ...later in LoRA...
if any("lora_" in k for k in state.keys()):  # state was deleted!
```

**Problem:** In `finetune_acc.py`, `state` is deleted for memory. But in `finetune_acc_lora.py`, the code tries to access `state` after potential deletion for LoRA key checking.

**In finetune_acc_lora.py lines 291-294:**
```python
if any("lora_" in k for k in state.keys()):
    print("Detected LoRA weights in checkpoint, reloading after injection...")
    missing_lora, unexpected_lora = model.load_state_dict(state, strict=False)
```

This is in a different code path that loads its own `state`, so it's fine, but the pattern is confusing.

---

### 14. 🔸 `audio_demos` Directory Created with Relative Path
**File:** `finetune_acc.py`  
**Line:** 714  

```python
Path("./audio_demos").mkdir(exist_ok=True)
```

**Problem:** Uses relative path, which depends on the current working directory at runtime. Should use `train_cfg.output_dir / "audio_demos"` for consistency.

---

### 15. 🔸 `waveforms` Always `None` for PreEncodedDACDataset
**File:** `dataset.py`  
**Line:** 248  

```python
return text_prompt, encoded, None  # No waveform since we have encoded data
```

**File:** `finetune_acc.py`  
**Line:** 386  

```python
'waveforms': waveforms,
'raw_text': texts[0],
```

**Problem:** `waveforms` will be a tuple of `None` values. While not used in training, this could cause issues if logging or debugging code tries to use waveforms.

---

### 16. 🔸 LoRA `lora_alpha` Detection May Match Unintended Params
**File:** `finetune_acc.py`  
**Lines:** 471-472  

```python
lname = name.lower()
is_lora_alpha = ("lora" in lname and "alpha" in lname) or lname.endswith("lora_alpha")
```

**Problem:** This could match parameters like `exploration_alpha` or `color_alpha` if they existed. The check should be more specific.

---

### 17. 🔸 Stereo Warm-Start Only Copies First 9 Channels
**File:** `finetune_acc.py`  
**Lines:** 1139-1144  

```python
if hasattr(model.decoder, "embeddings") and len(model.decoder.embeddings) >= 18:
    for i in range(9, 18):
        model.decoder.embeddings[i].weight.data.copy_(model.decoder.embeddings[i - 9].weight.data)
```

**Problem:** This assumes exactly 18 channels (9+9). If config changes to different channel counts, this will silently do the wrong thing or fail.

---

## Low Severity Issues

### 18. 📝 Unused Import: `resource`
**File:** `finetune_acc.py`  
**Line:** 14  

```python
import resource
```

Only used once for logging RSS. Not available on Windows.

---

### 19. 📝 Commented Out Code Blocks
**File:** `app_local.py`  
**Lines:** 50-66  

Large block of commented-out model loading code. Should be removed for clarity.

---

### 20. 📝 Inconsistent Exception Handling
**File:** `finetune_acc.py`  
**Multiple locations**

```python
except Exception as e:
    logger.warning(f"...")  # Some places
except Exception:
    pass  # Other places silently swallow
```

Silent exception swallowing (bare `except` or `except Exception: pass`) can hide real bugs.

---

### 21. 📝 Magic Numbers Without Comments
**File:** `finetune_acc.py`  
**Lines:** 948, 596-597  

```python
gen_val = ((global_step * 997 + train_cfg.seed) % 10000) / 10000.0  # Why 997? Why 10000?

seeds = [int(train_cfg.seed), int(train_cfg.seed) + 1]  # Why seed + 1?
```

These magic numbers should have comments explaining their purpose.

---

### 22. 📝 `drop_last=True` May Lose Data
**File:** `finetune_acc.py`  
**Line:** 422  

```python
drop_last=True,
```

**Problem:** With small datasets, this could drop a significant percentage of data. Consider logging how much data is dropped.

---

### 23. 📝 Hardcoded Sample Rate
**File:** Multiple files  

```python
sf.write(audio_path, arr, 44100)  # Hardcoded
output_sr = 44100  # Hardcoded
```

Should use `config.data.sample_rate` if available, or define a constant.

---

### 24. 📝 `preencode_audio.py` Missing Stereo Validation
**File:** `scripts/preencode_audio.py`  
**Lines:** 90-99  

```python
if waveform.shape[0] >= 2:
    left = waveform[0:1, :]
    right = waveform[1:2, :]
```

**Problem:** No validation that the audio isn't > 2 channels. A 5.1 surround file would silently only use the first two channels.

---

### 25. 📝 `codebook_to_audio` Hardcoded `-30` Trim
**File:** `audio.py`  
**Line:** 264  

```python
reverted_codebook = reverted_codebook[:, :-30, :]
```

**Problem:** This magic number `-30` should be derived from the max delay pattern value or made configurable.

---

## Code Quality & Best Practices

### 26. 🔧 Type Hints Inconsistent
Many functions lack type hints or have incomplete hints. Consider adding comprehensive type hints for better IDE support and error catching.

### 27. 🔧 Duplicate Code Blocks
The stereo checkpoint expansion logic is duplicated between:
- `finetune_acc.py` lines 1099-1128
- `finetune_acc.py` lines 1250-1276
- `finetune_acc_lora.py` lines 230-252
- `finetune_acc_lora.py` lines 395-417

This should be refactored into a shared utility function.

### 28. 🔧 Error Messages Could Be More Informative
```python
raise ValueError("Must specify either --audio_folder or --preencoded_dir")
```

Could include suggestions or examples:
```python
raise ValueError(
    "Must specify either --audio_folder or --preencoded_dir\n"
    "Example: --audio_folder /path/to/audio --preencoded_dir /path/to/encoded"
)
```

### 29. 🔧 No Input Validation for `cfg_scale`
**File:** `model.py`  

The `cfg_scale` parameter can be negative or extremely large, potentially causing numerical issues. Consider adding bounds checking.

### 30. 🔧 Signal Handler Cleanup May Not Be Called
**File:** `finetune_acc.py`  
**Lines:** 178-182  

```python
def signal_handler(signum, frame):
    cleanup_ddp()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
```

**Problem:** This is set inside `setup_ddp()` which is called for each rank. In multi-process spawning, signal handlers may not work as expected.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 6 |
| Medium | 8 |
| Low | 8 |
| Code Quality | 5 |
| **Total** | **30** |

### Recommended Priority Order:
1. Fix stereo audio speed adjustment bug (Critical #1)
2. Fix deprecated autocast usage (Critical #2)
3. Fix early stop loss comparison (Critical #3)
4. Address unused `max_new_tokens` parameter (High #4)
5. Add division-by-zero guards (High #8)
6. Refactor duplicate stereo expansion code (Code Quality #27)

