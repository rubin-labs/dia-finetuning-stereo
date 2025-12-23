#!/bin/bash

# --- 1. CLEANUP ---
echo "Cleaning up old processes..."
sudo pkill -9 python3 2>/dev/null
sudo pkill -9 python 2>/dev/null
rm -f /tmp/libtpu_lockfile 
sleep 2

# --- 2. CREDENTIALS ---
export WANDB_API_KEY=2bdd33a710538780b0e66c62afd69104a3a22020

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

# --- 6. RUN TRAINING ---
python3 -m accelerate.commands.launch \
    --config_file tpu_config.yaml \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --machine_rank=$NODE_RANK \
    --num_machines=4 \
    --num_processes=32 \
    -m dia.train_acc_tpu \
    --config configs/architecture/experiments/20251127_dia_010_gpu_refactor_scratch_dataset_model.json \
    --preencoded_dir /home/olivercamp/data_local/encoded_audio \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --epochs 1000 \
    --wandb_project dia-tpu \
    --demo_every 15 \
    --eval_every 15 \
    2>&1 | tee train_fsdp.log