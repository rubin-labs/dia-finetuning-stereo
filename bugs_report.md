# Bug Report & Analysis

Analysis of `dia/finetune_acc.py`, `dia/finetune_acc_lora.py`, and `dia/dataset.py`.

## Critical Issues

### 1. Random State Desynchronization in DDP
**Severity:** High
**Location:** `finetune_acc.py` (`train_step_ddp`, `collate_fn`, `seed_everything`)

- **Issue:** `seed_everything` sets the same `random` seed for all ranks (ignoring the `rank` argument).
- **Mechanism:** `collate_fn` calls `random.randint` (for cropping) and `random.random` (for tag dropout). Since `DistributedSampler` assigns different data samples to each rank, the number of `random` calls varies per rank (e.g., different audio lengths trigger different cropping logic, different tag counts). This causes the global `random` state to desynchronize across ranks immediately.
- **Consequence:** In `train_step_ddp`, the check `if random.random() < train_cfg.unconditional_frac:` relies on the global `random` state. Because of the desync, some ranks may choose to execute an "unconditional" step (masking text) while others execute a "conditional" step for the *same global step*. This leads to inconsistent gradients being averaged in DDP.
- **Fix:** Use a deterministic check for unconditional training, e.g., hashing the `global_step` or using a synchronized tensor broadcast.

### 2. Data Loading Performance Bottleneck
**Severity:** High
**Location:** `dataset.py` (`MusicDataset.__getitem__`, `_encode_mono_channel`)

- **Issue:** The `MusicDataset` performs the full DAC model encoding (forward pass) inside `__getitem__`.
- **Mechanism:** `finetune_acc.py` sets `num_workers=0` for the DataLoader. This means data loading happens in the main process, blocking the training loop. For every audio sample, the code loads the wav, resamples it, and runs the heavy DAC encoder on the GPU (or CPU if not moved properly) sequentially.
- **Consequence:** Training will be extremely slow as the GPU will starve waiting for the CPU/Single-threaded data loading and encoding.
- **Fix:** Pre-encode the dataset using `scripts/preencode_audio.py` (or similar) and use `PreEncodedDACDataset`, or implement a robust multi-process caching mechanism (difficult with CUDA in workers).

## Potential Bugs & Improvements

### 3. Hardcoded Audio Windowing
**Severity:** Medium
**Location:** `finetune_acc.py` (`collate_fn`)

- **Issue:** The window size calculation is hardcoded: `window_size = int(10 * 44100 / 512)`.
- **Consequence:** If the model configuration uses a different sample rate or hop length, or if the user wants a different context length (e.g., 15s or 5s), this hardcoded value will be incorrect.
- **Fix:** Use `config.data.audio_length` (tokens) directly or derive `window_size` from `DiaConfig` parameters.

### 4. Stereo/Mono Handling Consistency
**Severity:** Medium
**Location:** `dataset.py` (`MusicDataset`) vs `PreEncodedDACDataset`

- **Issue:**
    - `MusicDataset` returns `(text, encoded, waveform)`.
    - `PreEncodedDACDataset` returns `(text, encoded, None)`.
- **Consequence:** `collate_fn` expects 3 values. While current training logic doesn't seem to use the raw `waveforms` for loss calculation, any future logging or evaluation that expects `waveforms` in the batch will crash or behave unexpectedly when using pre-encoded data.

### 5. `PreEncodedDACDataset` Prompt Loading Fallback
**Severity:** Low
**Location:** `dataset.py` (`PreEncodedDACDataset`)

- **Issue:** If metadata is missing, it falls back to `filename.replace('_', ' ')`.
- **Consequence:** This heuristic is often insufficient for high-quality training (e.g., filenames like `track_01.pt` become prompt "track 01").
- **Fix:** Ensure metadata is robust or add a mechanism to load separate `.txt` prompt files matching the `.pt` files (similar to `MusicDataset`).

### 6. Inefficient Resampling
**Severity:** Low
**Location:** `dataset.py` (`MusicDataset`)

- **Issue:** `torchaudio.functional.resample` is called on every load if SR mismatches.
- **Consequence:** High CPU usage and latency.
- **Fix:** Resample once offline or use `torchaudio.io.StreamReader` / `sox` for faster on-the-fly resampling.

### 7. LoRA Layer Injection Assumptions
**Severity:** Medium
**Location:** `finetune_acc_lora.py` (`inject_lora_into_model`)

- **Issue:** The injection logic assumes specific attribute names (`wi_fused`, `wo`, `q_proj`, etc.) and module types (`Attention`, `MlpBlock`).
- **Consequence:** While it matches the current `layers.py`, any refactoring of the model definition will silently break LoRA injection (it simply won't find the targets) or cause attribute errors.

### 8. Evaluation Demo Generation in DDP
**Severity:** Low
**Location:** `finetune_acc.py` (`eval_step`)

- **Issue:** `eval_step` generates demos only on `rank == 0`.
- **Consequence:** This is generally correct, but `dist.barrier()` at the end of `eval_step` forces other ranks to wait while rank 0 generates audio (which can be slow). This introduces idle time for other GPUs.
- **Fix:** Not easily fixable without complex async logic, but something to be aware of for training efficiency.

### 9. Checkpoint Cleanup Logic
**Severity:** Low
**Location:** `finetune_acc.py` (`cleanup_old_checkpoints`)

- **Issue:** The regex `r'ckpt_step(\d+).pth'` assumes a specific naming convention.
- **Consequence:** If the saving format changes (e.g. `ckpt_step_{step}.pth`), cleanup will fail silently or delete wrong files.

