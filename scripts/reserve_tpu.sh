#!/bin/bash

# Script to reserve a TPU node for training
# Usage: ./reserve_tpu.sh [tpu-name] [accelerator-type] [zone] [--spot]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TPU_NAME=${1:-"dia-training-tpu"}
ACCELERATOR_TYPE=${2:-"v4-8"}
ZONE=${3:-"us-central2-b"}
SPOT_FLAG=${4:-""}

# Detect TPU version and set appropriate version string
# Default to PyTorch versions for ML training
if [[ "$ACCELERATOR_TYPE" == v6e-* ]]; then
    VERSION="tpu-vm-v6e-base"
    TPU_TYPE="v6e"
elif [[ "$ACCELERATOR_TYPE" == v5e-* ]]; then
    VERSION="tpu-vm-v5e-base"
    TPU_TYPE="v5e"
elif [[ "$ACCELERATOR_TYPE" == v4-* ]]; then
    VERSION="tpu-vm-v4-pt-2.0"  # PyTorch 2.0 for v4
    TPU_TYPE="v4"
else
    VERSION="tpu-vm-v4-pt-2.0"  # PyTorch 2.0 for v4
    TPU_TYPE="v4"
fi

# Check if spot/preemptible
IS_SPOT=false
if [[ "$SPOT_FLAG" == "--spot" ]] || [[ "$SPOT_FLAG" == "--preemptible" ]]; then
    IS_SPOT=true
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: No GCP project set. Run 'gcloud config set project YOUR_PROJECT_ID'${NC}"
    exit 1
fi

echo -e "${GREEN}Reserving TPU node for training...${NC}"
echo "Project ID: $PROJECT_ID"
echo "TPU Name: $TPU_NAME"
echo "Accelerator Type: $ACCELERATOR_TYPE"
echo "Zone: $ZONE"
if [ "$IS_SPOT" = true ]; then
    echo -e "${YELLOW}Type: SPOT/PREEMPTIBLE (can be interrupted)${NC}"
else
    echo "Type: On-Demand"
fi
echo ""

# Check if TPU already exists
if gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE &>/dev/null; then
    echo -e "${YELLOW}TPU node '$TPU_NAME' already exists in zone $ZONE.${NC}"
    echo "Current status:"
    gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE --format="value(state)"
    echo ""
    read -p "Do you want to delete and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deleting existing TPU node...${NC}"
        gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --quiet
        echo "Waiting for deletion to complete..."
        sleep 10
    else
        echo "Using existing TPU node."
        echo ""
        echo -e "${GREEN}To connect to the TPU:${NC}"
        echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
        exit 0
    fi
fi

# Calculate number of chips (handle v4, v5e, v6e formats)
if [[ "$ACCELERATOR_TYPE" =~ ^v[456]e- ]]; then
    CHIPS=$(echo $ACCELERATOR_TYPE | sed 's/v[456]e-//')
    echo -e "${BLUE}This will reserve $CHIPS TPU $TPU_TYPE chip(s)${NC}"
elif [[ "$ACCELERATOR_TYPE" =~ ^v4- ]]; then
    CHIPS=$(echo $ACCELERATOR_TYPE | sed 's/v4-//')
    echo -e "${BLUE}This will reserve $CHIPS TPU v4 chip(s)${NC}"
else
    echo -e "${YELLOW}Warning: Unrecognized accelerator type format, proceeding anyway...${NC}"
    CHIPS="?"
fi
echo ""

# Check available quota
echo -e "${YELLOW}Checking available quota...${NC}"
QUOTA=$(gcloud compute project-info describe --format="value(quotas[metric=TPU_V4_POD_QUOTA].limit)" 2>/dev/null || echo "unknown")
if [ "$QUOTA" != "unknown" ]; then
    echo "TPU v4 quota: $QUOTA chips"
fi

# Confirm before creating
read -p "Continue with TPU reservation? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create the TPU node
echo -e "${GREEN}Creating TPU node...${NC}"
if [ "$IS_SPOT" = true ]; then
    echo -e "${YELLOW}⚠️  Creating SPOT TPU (can be preempted, but much cheaper)${NC}"
fi
echo "This may take a few minutes..."

# Build the command
CREATE_CMD="gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$VERSION \
    --network=default \
    --subnetwork=default"

# Add preemptible flag if spot
if [ "$IS_SPOT" = true ]; then
    CREATE_CMD="$CREATE_CMD --preemptible"
fi

# Execute the command
eval $CREATE_CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ TPU node created successfully!${NC}"
else
    echo -e "${RED}✗ Failed to create TPU node${NC}"
    exit 1
fi

# Wait for TPU to be ready
echo -e "${YELLOW}Waiting for TPU to be ready...${NC}"
sleep 30

# Check TPU status
STATUS=$(gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE --format="value(state)" 2>/dev/null || echo "UNKNOWN")
echo "TPU Status: $STATUS"

# Display connection info
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TPU Node Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "TPU Name: $TPU_NAME"
echo "Zone: $ZONE"
echo "Accelerator: $ACCELERATOR_TYPE ($CHIPS chip(s))"
if [ "$IS_SPOT" = true ]; then
    echo -e "${YELLOW}Type: SPOT (can be preempted)${NC}"
fi
echo "Status: $STATUS"
echo ""
echo -e "${BLUE}To connect to the TPU:${NC}"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
echo ""
echo -e "${BLUE}To check TPU status:${NC}"
echo "  gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE"
echo ""
echo -e "${BLUE}To list all TPUs:${NC}"
echo "  gcloud compute tpus tpu-vm list --zone=$ZONE"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: TPUs are expensive! Delete when not in use:${NC}"
echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. SSH into the TPU:"
echo "   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE"
echo ""
echo "2. Set up environment variables:"
echo "   export XRT_TPU_CONFIG=\"localservice;0;localhost:51011\""
echo ""
echo "3. Install dependencies and run training"
echo ""
