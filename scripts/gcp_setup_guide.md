# Google Cloud Setup Guide for TPU Training

This guide will help you set up Google Cloud for training your Dia model on TPU.

## Prerequisites

1. **Google Cloud Account** with TRC (TPU Research Cloud) access
2. **gcloud CLI** installed and authenticated
3. **Project ID** - You'll need your GCP project ID

## Step 1: Set Up Your Environment

First, authenticate and set your project:

```bash
# Authenticate with Google Cloud
gcloud auth login

# Set your project ID (replace with your actual project ID)
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Set the zone for TPU
export ZONE="us-central2-b"
```

## Step 2: Create a GCS Bucket for Dataset Storage

GCS (Google Cloud Storage) is the cheapest option for storing your dataset that TPUs can access efficiently.

### Option A: Using the Setup Script

```bash
# Run the setup script
bash scripts/setup_gcs_bucket.sh
```

### Option B: Manual Setup

```bash
# Create a bucket (choose a unique name)
export BUCKET_NAME="your-dataset-bucket-name"
export REGION="us-central2"  # Same region as TPU for best performance

# Create the bucket
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_NAME

# Set lifecycle policy to use cheaper storage classes (optional but recommended)
gsutil lifecycle set scripts/gcs_lifecycle.json gs://$BUCKET_NAME
```

**Storage Classes:**
- **Standard**: Best for frequent access during training (~$0.020/GB/month)
- **Nearline**: Good for datasets accessed monthly (~$0.010/GB/month)
- **Coldline**: Cheapest for long-term storage (~$0.004/GB/month)

## Step 3: Upload Your Dataset

### For Pre-encoded Data (Recommended)

If you have pre-encoded `.pt` files:

```bash
# Upload pre-encoded directory
gsutil -m cp -r /path/to/preencoded_dir/* gs://$BUCKET_NAME/preencoded/

# Or if you have a structured dataset
gsutil -m cp -r /path/to/dataset/encoded_audio/* gs://$BUCKET_NAME/encoded_audio/
gsutil cp /path/to/dataset/metadata.json gs://$BUCKET_NAME/metadata.json
```

### For Raw Audio Files

If you need to upload raw audio:

```bash
# Upload audio folder
gsutil -m cp -r /path/to/audio_folder/* gs://$BUCKET_NAME/audio/

# Upload prompts
gsutil -m cp -r /path/to/audio_prompts/* gs://$BUCKET_NAME/audio_prompts/
```

**Note:** The `-m` flag enables parallel uploads for faster transfer.

## Step 4: Reserve a TPU Node

### Option A: Using the Setup Script

```bash
# Reserve a TPU node
bash scripts/reserve_tpu.sh
```

### Option B: Manual Reservation

```bash
# Reserve a TPU v4 node (on-demand)
gcloud compute tpus tpu-vm create dia-training-tpu \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-vm-v4-base \
    --network=default \
    --subnetwork=default

# For multiple chips, you can create multiple nodes or use a larger accelerator type
# Available types: v4-8 (single chip), v4-16, v4-32, v4-64, v4-128, v4-256, v4-512
```

**Important Notes:**
- TPU v4-8 = 1 chip, v4-16 = 2 chips, etc.
- You have access to 32 chips total, so you could reserve:
  - 4x v4-8 nodes (4 chips each = 16 chips)
  - 2x v4-16 nodes (8 chips each = 16 chips)
  - 1x v4-32 node (32 chips)
  - Or any combination that adds up to ≤32 chips

## Step 5: Connect to Your TPU

```bash
# SSH into the TPU VM
gcloud compute tpus tpu-vm ssh dia-training-tpu --zone=$ZONE
```

## Step 6: Set Up the Training Environment on TPU

Once connected to the TPU VM:

```bash
# Install Python and dependencies
sudo apt-get update
sudo apt-get install -y python3-pip git

# Clone your repository (or upload it)
git clone <your-repo-url>
cd dia-finetuning-stereo-main

# Install dependencies
pip install -r requirements.txt
pip install torch torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Set up environment variables
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export TPU_NAME="dia-training-tpu"
```

## Step 7: Mount GCS Bucket (Optional but Recommended)

For better performance, you can use GCS FUSE to mount your bucket as a filesystem:

```bash
# Install GCS FUSE
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse

# Mount your bucket
mkdir -p ~/dataset
gcsfuse --implicit-dirs $BUCKET_NAME ~/dataset
```

Now you can access your dataset at `~/dataset/` as if it were a local directory.

## Step 8: Run Training

### Using Pre-encoded Data from GCS

```bash
# If using GCS FUSE mount
accelerate launch dia/train_acc_tpu.py \
    --config configs/architecture/model.json \
    --preencoded_dir ~/dataset/preencoded \
    --output_dir ./checkpoints \
    --run_name dia_tpu_training \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --wandb_project dia-tpu

# Or directly from GCS (slower but works)
accelerate launch dia/train_acc_tpu.py \
    --config configs/architecture/model.json \
    --preencoded_dir gs://$BUCKET_NAME/preencoded \
    --output_dir ./checkpoints \
    --run_name dia_tpu_training \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --wandb_project dia-tpu
```

### Using Raw Audio from GCS

```bash
# If using GCS FUSE mount
accelerate launch dia/train_acc_tpu.py \
    --config configs/architecture/model.json \
    --audio_folder ~/dataset/audio \
    --output_dir ./checkpoints \
    --run_name dia_tpu_training \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --wandb_project dia-tpu
```

## Step 9: Monitor Costs

Keep track of your usage:

```bash
# Check TPU usage
gcloud compute tpus list --zone=$ZONE

# Check storage costs
gsutil du -sh gs://$BUCKET_NAME

# Monitor billing
# Visit: https://console.cloud.google.com/billing
```

## Step 10: Clean Up (When Done)

```bash
# Delete TPU node (IMPORTANT: TPUs are expensive when running!)
gcloud compute tpus tpu-vm delete dia-training-tpu --zone=$ZONE

# Optionally delete the bucket (if you don't need the data)
gsutil rm -r gs://$BUCKET_NAME
```

## Cost Optimization Tips

1. **Use Pre-encoded Data**: Pre-encode your audio to `.pt` files locally, then upload. This reduces storage size significantly.

2. **Choose Right Storage Class**: Use Nearline or Coldline for datasets you access infrequently.

3. **Delete TPU When Not Training**: TPUs cost money even when idle. Always delete them when not in use.

4. **Use Spot/Preemptible TPUs**: If available in your TRC program, these are much cheaper.

5. **Compress Data**: Compress your dataset before uploading to save storage costs.

## Troubleshooting

### TPU Not Found
```bash
# List available TPUs
gcloud compute tpus list --zone=$ZONE

# Check TPU status
gcloud compute tpus describe dia-training-tpu --zone=$ZONE
```

### Permission Issues
```bash
# Ensure you have the right permissions
gcloud projects get-iam-policy $PROJECT_ID
```

### Slow Data Loading
- Use GCS FUSE for better performance
- Pre-encode data locally before uploading
- Use a bucket in the same region as your TPU

## Additional Resources

- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [GCS Pricing](https://cloud.google.com/storage/pricing)
- [TPU Pricing](https://cloud.google.com/tpu/pricing)
- [TRC Program](https://sites.research.google/trc/)
