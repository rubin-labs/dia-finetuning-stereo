import os
import socket
import subprocess
import sys

# 1. Detect Worker Rank from Hostname
hostname = socket.gethostname()
try:
    parts = hostname.split('-')
    rank = int(parts[-1])
except ValueError:
    print(f"ERROR: Could not parse rank from hostname: {hostname}")
    sys.exit(1)

print(f"--- LAUNCHING WORKER {rank} ---")

# 2. Set Environment Variables
env = os.environ.copy()
env['PJRT_DEVICE'] = 'TPU'
env['WANDB_API_KEY'] = '2bdd33a710538780b0e66c62afd69104a3a22020'
env['PYTHONPATH'] = os.getcwd()

# Debugging
env['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
env['TORCH_CPP_LOG_LEVEL'] = 'INFO'

# 3. Construct torchrun Command
# We use torchrun directly which is more robust for TPU Pod rendezvous than accelerate launch
cmd = [
    "python3", "-m", "torch.distributed.run",
    "--nproc_per_node=8",
    "--nnodes=4",
    "--node_rank=" + str(rank),
    "--rdzv_id=dia_job",
    "--rdzv_backend=c10d",
    "--rdzv_endpoint=10.130.0.11:12355",
    "-m", "dia.train_acc_tpu",
    "--config", "configs/architecture/experiments/20251127_dia_010_gpu_refactor_scratch_dataset_model.json",
    "--preencoded_dir", "/home/olivercamp/data/preencoded/encoded_audio",
    "--output_dir", "./checkpoints",
    "--run_name", "dia_010_tpu_v4_32",
    "--batch_size", "32",
    "--grad_accum_steps", "1",
    "--learning_rate", "1e-4",
    "--epochs", "15",
    "--warmup_steps", "2000",
    "--unconditional_frac", "1",
    "--scratch",
    "--tag_no_shuffle",
    "--eval_step", "200",
    "--demo_every", "1000",
    "--save_every", "2000"
]

print(f"Running command: {' '.join(cmd)}")
sys.stdout.flush()

# 4. Execute
subprocess.run(cmd, env=env, check=True)
