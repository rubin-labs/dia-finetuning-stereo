# GCP Quick Start Guide

## Quick Setup (5 minutes)

### 1. Authenticate and Set Project
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
export PROJECT_ID=$(gcloud config get-value project)
export ZONE="us-central2-b"
```

### 2. Create GCS Bucket
```bash
bash scripts/setup_gcs_bucket.sh
# Or manually:
export BUCKET_NAME="your-dataset-bucket"
gsutil mb -p $PROJECT_ID -c STANDARD -l us-central2 gs://$BUCKET_NAME
```

### 3. Upload Dataset
```bash
# For pre-encoded data
bash scripts/upload_dataset.sh $BUCKET_NAME /path/to/preencoded preencoded

# For raw audio
bash scripts/upload_dataset.sh $BUCKET_NAME /path/to/audio audio
```

### 4. Reserve TPU
```bash
# Reserve a single TPU v4-8 node (1 chip)
bash scripts/reserve_tpu.sh dia-training-tpu v4-8

# Or reserve multiple chips (you have 32 total)
bash scripts/reserve_tpu.sh dia-training-tpu v4-32  # Uses all 32 chips
```

### 5. Connect and Train
```bash
# SSH into TPU
gcloud compute tpus tpu-vm ssh dia-training-tpu --zone=$ZONE

# On the TPU, set up and run training
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
accelerate launch dia/train_acc_tpu.py \
    --preencoded_dir gs://$BUCKET_NAME/preencoded \
    --output_dir ./checkpoints \
    --run_name dia_tpu_training
```

## Common Commands

### Check TPU Status
```bash
gcloud compute tpus tpu-vm list --zone=$ZONE
gcloud compute tpus tpu-vm describe dia-training-tpu --zone=$ZONE
```

### Delete TPU (IMPORTANT: Do this when done!)
```bash
gcloud compute tpus tpu-vm delete dia-training-tpu --zone=$ZONE
```

### Check Storage Usage
```bash
gsutil du -sh gs://$BUCKET_NAME
gsutil ls -lh gs://$BUCKET_NAME/preencoded/
```

### Mount GCS Bucket (on TPU)
```bash
# Install GCS FUSE
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y gcsfuse

# Mount
mkdir -p ~/dataset
gcsfuse --implicit-dirs $BUCKET_NAME ~/dataset
```

## Cost Estimates

### Storage (GCS)
- **Standard**: ~$0.020/GB/month (frequent access)
- **Nearline**: ~$0.010/GB/month (monthly access)
- **Coldline**: ~$0.004/GB/month (rarely accessed)

### TPU v4 (On-Demand)
- **v4-8** (1 chip): ~$3.00/hour
- **v4-32** (4 chips): ~$12.00/hour
- **v4-128** (16 chips): ~$48.00/hour

**⚠️ Always delete TPUs when not training!**

## TPU Chip Allocation

You have access to **32 chips total**. Options:

- **Option 1**: 1x v4-32 node (all 32 chips in one node)
- **Option 2**: 4x v4-8 nodes (8 chips each = 32 chips total)
- **Option 3**: 2x v4-16 nodes (16 chips each = 32 chips total)
- **Option 4**: Any combination that adds up to ≤32 chips

## Troubleshooting

### TPU not found
```bash
# List all TPUs
gcloud compute tpus tpu-vm list --zone=$ZONE

# Check if TPU exists
gcloud compute tpus tpu-vm describe dia-training-tpu --zone=$ZONE
```

### Permission denied
```bash
# Check IAM permissions
gcloud projects get-iam-policy $PROJECT_ID
```

### Slow data loading
- Use GCS FUSE mount instead of direct gs:// paths
- Pre-encode data before uploading
- Ensure bucket is in same region as TPU

## Full Documentation

See `scripts/gcp_setup_guide.md` for detailed instructions.
