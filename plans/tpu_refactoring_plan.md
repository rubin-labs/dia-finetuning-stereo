# Iterative Refactoring Plan for `dia_tpu/finetune_acc_tpu.py`

This plan outlines the steps to refactor `dia_tpu/finetune_acc_tpu.py` to use **Hugging Face Accelerate**. This transformation will enable the script to run seamlessly on **TPUs**, GPUs (single or multi-node), and CPUs without changing the core logic.

The current script relies on manual PyTorch DDP (`torch.distributed`) with the `nccl` backend, which is **incompatible with TPUs**. TPUs require `torch_xla`, which Accelerate abstracts away.

---

## Phase 1: Setup & Initialization

### Step 1.1: Dependency Verification
Ensure the environment has `accelerate` installed.
- **Action**: Check `pyproject.toml` or `requirements.txt`.
- **Verify**: Run `accelerate config` in the terminal to set up the environment (optional for the script, but good for the user).

### Step 1.2: Clean Boilerplate Imports
Remove manual distributed training imports and setup functions that will be replaced.
- **Remove**:
  - `import torch.distributed as dist`
  - `import torch.multiprocessing as mp`
  - `from torch.nn.parallel import DistributedDataParallel as DDP`
  - `setup_ddp()` function
  - `cleanup_ddp()` function
  - `run_ddp_worker()` function (we will use `accelerate launch` instead of `mp.spawn`)
- **Add**:
  - `from accelerate import Accelerator`
  - `from accelerate.utils import set_seed`

### Step 1.3: Initialize Accelerator
Replace the manual device placement with the `Accelerator` object.
- **Action**:
  - Instantiate `accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)` at the start of `main()`.
  - Replace `device = torch.device(...)` with `accelerator.device`.
  - Replace manual seeding with `set_seed(args.seed)`.

---

## Phase 2: Core Training Loop Migration

### Step 2.1: Prepare Objects
Use `accelerator.prepare()` to handle device placement for all training objects.
- **Action**:
  - Pass `model`, `opt`, `train_loader`, `sched` to `accelerator.prepare()`.
  - **Critical**: Do this *before* the training loop starts.
  - **Note**: Do *not* manually move model/batch to `.to(device)`; Accelerate handles this (except for some complex dicts in `collate_fn`, checking `accelerator.device` is safer).

### Step 2.2: Refactor the Training Step
Update the `train_step` (or `train_step_ddp`) function to use Accelerate's backward pass and context managers.
- **Action**:
  - Remove `with torch.autocast(...)`: Accelerate handles mixed precision via `accelerator.autocast()` or automatically if configured.
  - **Replace**: `loss.backward()` with `accelerator.backward(loss)`.
  - **Replace**: Gradient accumulation logic. Accelerate simplifies this:
    ```python
    with accelerator.accumulate(model):
        # ... forward ...
        # ... loss ...
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()
        sched.step()
        opt.zero_grad()
    ```
  - Remove manual `loss / grad_accum_steps` division if using `accelerator.accumulate` (it handles it, or check documentation for specific version behavior).

### Step 2.3: Remove `mp.spawn`
The script currently uses `mp.spawn` to launch processes. Accelerate scripts are typically launched via CLI (`accelerate launch script.py`).
- **Action**:
  - Remove the `if world_size > 1: mp.spawn(...)` block in `main()`.
  - Refactor `main()` to just run the training logic directly (since `accelerate launch` handles the spawning).

---

## Phase 3: Evaluation & Checkpointing

### Step 3.1: Saving Checkpoints
DDP models need unwrapping before saving to avoid `module.` prefixes.
- **Action**:
  - Replace `model.module.state_dict()` logic with:
    ```python
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), save_path)
    ```
  - Ensure only the main process saves: `if accelerator.is_main_process: ...`

### Step 3.2: Logging
Ensure logs only print on the main process to avoid console spam.
- **Action**:
  - Wrap print statements and `wandb.log` calls with `if accelerator.is_main_process:`.
  - Initialize `wandb` through `accelerator.init_trackers` (optional but recommended) or keep existing `wandb.init` strictly on the main process.

### Step 3.3: Evaluation Loop
- **Action**:
  - Use `accelerator.gather_for_metrics(outputs)` if you need to aggregate validation metrics across TPUs.
  - Ensure validation loader is also prepared with `accelerator.prepare(val_loader)`.

---

## Phase 4: TPU Specific Optimizations (Optional but Recommended)

### Step 4.1: DataLoader Efficiency
TPUs can be starved if data loading is slow.
- **Action**:
  - Ensure `num_workers` in DataLoader is set appropriately (usually 4-8).
  - `pin_memory=True` is often good for GPU, but Accelerate handles `pin_memory` defaults.

### Step 4.2: Compilation
- **Action**:
  - `torch.compile` support on TPU is evolving. Ensure `args.compile` logic checks if the backend supports it or if `torch_xla` specific compilation is needed (Accelerate usually handles `torch.compile` wrapping well).

---

## Example Command to Run
Once refactored, the script can be launched on a TPU VM with:
```bash
accelerate launch --platform_type tpu dia_tpu/finetune_acc_tpu.py --config ...
```

