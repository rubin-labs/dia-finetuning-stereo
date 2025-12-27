#!/bin/bash

# --- 1. CLEANUP ---
echo "Cleaning up old processes..."
sudo pkill -9 python3 2>/dev/null
sudo pkill -9 python 2>/dev/null
rm -f /tmp/libtpu_lockfile 
sleep 2

# --- 2. CREDENTIALS ---
export WANDB_API_KEY=2bdd33a710538780b0e66c62afd69104a3a22020

# --- 2b. INSTALL HUGGINGFACE DATASETS (if not already installed) ---
pip install datasets --quiet 2>/dev/null || true

# --- 3. SYSTEM SETTINGS ---
# Silence logs
export PYTHONWARNINGS="ignore"
# Fix the "item()" crash by making transfers safer
export XLA_TRANSFER_SEED_ASYNC=1

# --- 4. CONFIGURATION ---
HOST=$(hostname)
export NODE_RANK=$(echo $HOST | awk -F'-' '{print $NF}')
export MASTER_ADDR=$(echo $HOST | sed 's/[0-9]*$/0/')
export MASTER_PORT=12355

echo "------------------------------------------------"
echo "I am: $HOST (Rank $NODE_RANK)"
echo "Master is: $MASTER_ADDR"
echo "------------------------------------------------"

cd ~/dia-finetuning-stereo || exit

# --- 4a. MOUNT GCS BUCKETS ---
# Dataset: GCS bucket (pre-encoded .pt files)  
# Checkpoints: GCS bucket (persistent across restarts)

# Install gcsfuse if needed (do this once at the start)
if ! command -v gcsfuse &> /dev/null; then
    echo "Installing gcsfuse..."
    export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    sudo apt-get update
    sudo apt-get install -y gcsfuse
fi

# --- Dataset bucket ---
DATA_BUCKET_NAME="rubin-dia-dataset"
DATA_BUCKET_MOUNT="/home/olivercamp/dataset_bucket"

if ! mountpoint -q "$DATA_BUCKET_MOUNT" 2>/dev/null; then
    echo "Mounting dataset bucket: $DATA_BUCKET_NAME to $DATA_BUCKET_MOUNT"
    mkdir -p "$DATA_BUCKET_MOUNT"
    gcsfuse --implicit-dirs --file-mode=444 --dir-mode=555 "$DATA_BUCKET_NAME" "$DATA_BUCKET_MOUNT"
    echo "Dataset bucket mounted successfully"
else
    echo "Dataset bucket already mounted at $DATA_BUCKET_MOUNT"
fi

# --- Checkpoints bucket ---
CKPT_BUCKET_NAME="rubin-dia-checkpoints"
CKPT_BUCKET_MOUNT="/home/olivercamp/checkpoints_bucket"

if ! mountpoint -q "$CKPT_BUCKET_MOUNT" 2>/dev/null; then
    echo "Mounting checkpoints bucket: $CKPT_BUCKET_NAME to $CKPT_BUCKET_MOUNT"
    mkdir -p "$CKPT_BUCKET_MOUNT"
    gcsfuse --implicit-dirs "$CKPT_BUCKET_NAME" "$CKPT_BUCKET_MOUNT"
    echo "Checkpoints bucket mounted successfully"
else
    echo "Checkpoints bucket already mounted at $CKPT_BUCKET_MOUNT"
fi

# --- 4b. XLA PERSISTENT CACHE (FIXED) ---
# CRITICAL: Unset old flags that cause crashes
unset TF_XLA_FLAGS
unset XLA_FLAGS

CACHE_DIR=/home/olivercamp/xla_cache
mkdir -p "$CACHE_DIR"

# CORRECT WAY: Use the dedicated environment variable for caching
export XLA_PERSISTENT_CACHE_PATH="$CACHE_DIR"

echo "XLA Cache Path set to: $XLA_PERSISTENT_CACHE_PATH"

# --- 5. GENERATE TPU CONFIG (XLA) ---
cat <<YAML > tpu_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: XLA
downcast_bf16: 'no'
machine_rank: $NODE_RANK
main_process_ip: $MASTER_ADDR
main_process_port: $MASTER_PORT
main_training_function: main
mixed_precision: bf16
num_machines: 4
num_processes: 32
use_cpu: 'no'
YAML

# --- 5b. RESUME FROM CHECKPOINT ---
# Options:
#   RESUME_FROM=/path/to/ckpt_stepN  - Resume from specific checkpoint
#   AUTO_RESUME=1                     - Auto-find latest checkpoint
#   (default)                         - Start from scratch
AUTO_RESUME=${AUTO_RESUME:-0}
RESUME_FROM=${RESUME_FROM:-""}

RESUME_ARG=""
RESUME_PATH=""  # absolute path to checkpoint dir on filesystem (usually on gcsfuse mount)
if [ -n "$RESUME_FROM" ]; then
    # Explicit checkpoint path provided
    if [ -d "$RESUME_FROM" ]; then
        echo "RESUME_FROM=$RESUME_FROM: Resuming from specified checkpoint"
        RESUME_PATH="$RESUME_FROM"
        RESUME_ARG="--resume_from $RESUME_PATH"
    else
        echo "ERROR: RESUME_FROM path does not exist: $RESUME_FROM"
        exit 1
    fi
elif [ "$AUTO_RESUME" = "1" ]; then
    # Auto-find latest checkpoint
    # Prefer "complete" checkpoints (written marker file), fall back to newest dir
    LATEST_CKPT=""
    for ckpt in $(ls -d "$CKPT_BUCKET_MOUNT"/ckpt_step* 2>/dev/null | sort -V -r); do
        if [ -f "$ckpt/.complete" ]; then
            LATEST_CKPT="$ckpt"
            break
        fi
    done
    if [ -z "$LATEST_CKPT" ]; then
        LATEST_CKPT=$(ls -td "$CKPT_BUCKET_MOUNT"/ckpt_step* 2>/dev/null | head -n 1)
    fi
    if [ -n "$LATEST_CKPT" ]; then
        echo "AUTO_RESUME=1: Resuming from latest checkpoint: $LATEST_CKPT"
        RESUME_PATH="$LATEST_CKPT"
        RESUME_ARG="--resume_from $RESUME_PATH"
    else
        echo "AUTO_RESUME=1 but no checkpoint found; starting fresh."
    fi
else
    echo "Starting from scratch. Set RESUME_FROM=/path/to/ckpt or AUTO_RESUME=1 to resume."
fi

# --- 5c. FAST RESUME: Stage checkpoint locally (avoid slow/variable gcsfuse reads) ---
# NOTE: On TPU, Accelerate uses XLA (not FSDP). Checkpoints can be large; reading via gcsfuse across many
# processes causes huge variance. We download the checkpoint directory once per TPU VM with gsutil and
# point --resume_from at the local copy.
STAGE_RESUME_TO_LOCAL=${STAGE_RESUME_TO_LOCAL:-1}
LOCAL_CKPT_CACHE_BASE=${LOCAL_CKPT_CACHE_BASE:-/home/olivercamp/ckpt_cache}

if [ -n "$RESUME_PATH" ] && [ "$STAGE_RESUME_TO_LOCAL" = "1" ]; then
    if command -v gsutil >/dev/null 2>&1; then
        if [[ "$RESUME_PATH" == "$CKPT_BUCKET_MOUNT"* ]]; then
            REL_CKPT_PATH="${RESUME_PATH#${CKPT_BUCKET_MOUNT}/}"
            CKPT_GCS_URI="gs://${CKPT_BUCKET_NAME}/${REL_CKPT_PATH}"
            LOCAL_RESUME_PATH="${LOCAL_CKPT_CACHE_BASE}/${REL_CKPT_PATH}"

            echo "Staging checkpoint to local disk (worker $NODE_RANK)..."
            echo "  from: $CKPT_GCS_URI"
            echo "  to:   $LOCAL_RESUME_PATH"

            # Clear any partial staging from a previous attempt
            rm -rf "$LOCAL_RESUME_PATH"
            mkdir -p "$LOCAL_RESUME_PATH"

            # Parallel, robust download
            if gsutil -m rsync -r "$CKPT_GCS_URI" "$LOCAL_RESUME_PATH"; then
                echo "✓ Checkpoint staged locally (worker $NODE_RANK)"
                RESUME_ARG="--resume_from $LOCAL_RESUME_PATH"
            else
                echo "WARNING: Failed to stage checkpoint locally; falling back to gcsfuse path."
            fi
        else
            echo "Resume path is not under $CKPT_BUCKET_MOUNT; skipping local staging."
        fi
    else
        echo "WARNING: gsutil not found; cannot stage checkpoint locally. Falling back to gcsfuse path."
    fi
fi

# --- 6. RUN TRAINING ---
# ============================================================================
# TRAINING PARAMETER DESIGN (1.65B model, ~88.7K samples)
# ============================================================================
#
# KEY CALCULATIONS:
#   - Effective batch size: ~128 samples/step (8 × 16 data-parallel processes)
#   - Dataset: ~88,663 samples → ~693 steps/epoch (at batch_size=8, grad_accum=1)
#   - Total steps at 500 epochs: 215,500 steps
#   - Total steps at 1000 epochs: 431,000 steps
#
# LEARNING RATE SCALING (sqrt rule for transformers):
#   - Base LR for batch=256: ~3e-4
#   - For batch=128: ~2e-4 (conservative) to 3e-4 (aggressive)
#
# WARMUP (1-5% of total training):
#   - Conservative: 4,000 steps (~10 epochs, ~2% of 200K steps)
#   - Aggressive: 2,000 steps (~5 epochs, ~1% of 200K steps)
#
# SCHEDULE OPTIONS:
#   1. FAST TEST (50 epochs): ~21,550 steps - see if model learns
#   2. MEDIUM RUN (200 epochs): ~86,200 steps - solid baseline
#   3. FULL TRAINING (500+ epochs): 215K+ steps - frontier quality
#
# ============================================================================

# --- DATASET SOURCE SELECTION ---
# Using pre-converted .pt files from GCS bucket
# Created by: hf_to_gcs_fast.py (download) + convert_parquet_to_pt.py (convert)
# The bucket should contain: encoded_audio/*.pt + metadata.json
DATASET_ARG="--preencoded_dir $DATA_BUCKET_MOUNT"

# --- PRE-FLIGHT: Verify dataset bucket has data ---
echo "Checking dataset bucket..."
if [ -f "$DATA_BUCKET_MOUNT/metadata.json" ]; then
    SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$DATA_BUCKET_MOUNT/metadata.json'))))" 2>/dev/null || echo "0")
    echo "✓ Dataset bucket has metadata.json with $SAMPLE_COUNT samples"
else
    echo "ERROR: Dataset bucket missing metadata.json!"
    echo "Run the upload script first:"
    echo "  python scripts/hf_to_gcs.py --bucket gs://$DATA_BUCKET_NAME"
    exit 1
fi

# --- PRE-FLIGHT: Verify checkpoints bucket is responsive ---
# Use worker-specific test file to avoid race conditions
OUTPUT_DIR="$CKPT_BUCKET_MOUNT"
TEST_FILE="$OUTPUT_DIR/.write_test_worker_${NODE_RANK}"
echo "Testing checkpoints bucket write speed (worker $NODE_RANK)..."
if timeout 10 bash -c "touch $TEST_FILE && rm $TEST_FILE"; then
    echo "✓ Checkpoints bucket is responsive (worker $NODE_RANK)"
else
    echo "ERROR: Checkpoints bucket mount is hanging or unresponsive!"
    echo "1. Create the bucket: gsutil mb gs://$CKPT_BUCKET_NAME"
    echo "2. Remount: fusermount -u $CKPT_BUCKET_MOUNT && gcsfuse --implicit-dirs $CKPT_BUCKET_NAME $CKPT_BUCKET_MOUNT"
    exit 1
fi

python3 -m accelerate.commands.launch \
    --config_file tpu_config.yaml \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --machine_rank=$NODE_RANK \
    --num_machines=4 \
    --num_processes=32 \
    -m dia.train_acc_tpu \
    --config configs/architecture/experiments/20251127_dia_010_gpu_refactor_scratch_dataset_model.json \
    $DATASET_ARG \
    --output_dir $OUTPUT_DIR \
    $RESUME_ARG \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --warmup_steps 5000 \
    --unconditional_frac 0.15 \
    --epochs 50 \
    --wandb_project dia-tpu \
    --demo_every 244 \
    --eval_every 244 \
    --weight_decay 0.1 \
    --grad_clip_max_norm 1.0 \
    --save_step 244 \
    --keep_last_n 3 \
    --use_sliding_window \
    2>&1 | tee train_fsdp.log
