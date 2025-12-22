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
# NOTE: XLA_USE_BF16 removed to allow manual FP32 casting for loss stability
# export XLA_USE_BF16=1
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

cd ~/dia-finetuning-stereo-main || exit

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
    --run_name dia_010_tpu_fsdp \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --epochs 1 \
    --warmup_steps 100 \
    --unconditional_frac 1 \
    --scratch \
    --tag_no_shuffle \
    --eval_step 20 \
    --demo_every 50 \
    --save_every 5000 \
    --wandb_project dia-tpu \
    --half 2>&1 | tee train_fsdp.log
