# TPU Training for Dia

This directory contains scripts for training Dia on Google Cloud TPUs using `torch_xla`.

## Prerequisites

Ensure you are in a TPU VM environment and have `torch_xla` installed.
Typically, this comes pre-installed in TPU VM images. If not:
```bash
pip install torch-xla
```

## Running Training

Run the training script using python directly. `torch_xla` handles multiprocessing via `xmp.spawn`.

```bash
python dia_tpu/finetune.py \
    --train_config configs/train/default.json \
    --audio_folder /path/to/audio \
    --output_dir ./outputs/tpu_test \
    --num_cores 8
```

### Key Differences from GPU Training

- **Fixed Padding**: To avoid frequent XLA graph recompilation, all batches are padded to the fixed `audio_length` specified in the config (default 1024 tokens + overhead). Dynamic batching is disabled.
- **Precision**: Uses XLA's default precision (often bfloat16 on TPUs). `autocast` is removed.
- **Optimization**: Uses standard `AdamW` instead of `bitsandbytes` 8-bit optimizers.
- **DDP**: Uses `torch_xla.distributed` instead of `torch.distributed`.

## Troubleshooting

- **Slow compilation**: The first few steps will be slow as XLA compiles the computation graph. It should speed up significantly after step 5-10.
- **OOM**: If you run out of memory, try reducing `batch_size` in your train config.

