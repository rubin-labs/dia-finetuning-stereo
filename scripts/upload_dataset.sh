#!/bin/bash

# Script to upload dataset to GCS bucket
# Usage: ./upload_dataset.sh [bucket-name] [local-path] [dataset-type]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get arguments
BUCKET_NAME=${1:-""}
LOCAL_PATH=${2:-""}
DATASET_TYPE=${3:-"preencoded"}  # preencoded or audio

if [ -z "$BUCKET_NAME" ]; then
    echo -e "${YELLOW}Enter GCS bucket name:${NC}"
    read BUCKET_NAME
fi

if [ -z "$LOCAL_PATH" ]; then
    echo -e "${YELLOW}Enter local dataset path:${NC}"
    read LOCAL_PATH
fi

if [ ! -d "$LOCAL_PATH" ]; then
    echo -e "${RED}Error: Local path '$LOCAL_PATH' does not exist${NC}"
    exit 1
fi

# Validate bucket exists
if ! gsutil ls -b gs://$BUCKET_NAME &>/dev/null; then
    echo -e "${RED}Error: Bucket 'gs://$BUCKET_NAME' does not exist${NC}"
    echo "Create it first with: bash scripts/setup_gcs_bucket.sh"
    exit 1
fi

echo -e "${GREEN}Uploading dataset to GCS...${NC}"
echo "Bucket: gs://$BUCKET_NAME"
echo "Local Path: $LOCAL_PATH"
echo "Dataset Type: $DATASET_TYPE"
echo ""

# Determine destination path
if [ "$DATASET_TYPE" == "preencoded" ]; then
    DEST_PATH="gs://$BUCKET_NAME/preencoded/"
    echo -e "${BLUE}Uploading pre-encoded dataset...${NC}"
    
    # Check if encoded_audio directory exists
    if [ -d "$LOCAL_PATH/encoded_audio" ]; then
        echo "Found encoded_audio directory"
        gsutil -m cp -r "$LOCAL_PATH/encoded_audio"/* gs://$BUCKET_NAME/preencoded/encoded_audio/
    elif [ -d "$LOCAL_PATH" ] && ls "$LOCAL_PATH"/*.pt &>/dev/null; then
        echo "Found .pt files in root directory"
        gsutil -m cp "$LOCAL_PATH"/*.pt gs://$BUCKET_NAME/preencoded/encoded_audio/
    else
        echo -e "${YELLOW}Warning: No encoded_audio directory or .pt files found${NC}"
    fi
    
    # Upload metadata if exists
    if [ -f "$LOCAL_PATH/metadata.json" ]; then
        echo "Uploading metadata.json"
        gsutil cp "$LOCAL_PATH/metadata.json" gs://$BUCKET_NAME/preencoded/metadata.json
    fi
    
    # Upload prompts if they exist
    if [ -d "$LOCAL_PATH/prompts" ]; then
        echo "Uploading prompts"
        gsutil -m cp -r "$LOCAL_PATH/prompts"/* gs://$BUCKET_NAME/preencoded/prompts/
    fi
    
elif [ "$DATASET_TYPE" == "audio" ]; then
    DEST_PATH="gs://$BUCKET_NAME/audio/"
    echo -e "${BLUE}Uploading raw audio dataset...${NC}"
    
    # Upload audio files
    if [ -d "$LOCAL_PATH" ]; then
        echo "Uploading audio files"
        gsutil -m cp -r "$LOCAL_PATH"/* gs://$BUCKET_NAME/audio/
    fi
    
    # Upload prompts if they exist in parent directory
    PROMPTS_DIR=$(dirname "$LOCAL_PATH")/audio_prompts
    if [ -d "$PROMPTS_DIR" ]; then
        echo "Uploading audio prompts"
        gsutil -m cp -r "$PROMPTS_DIR"/* gs://$BUCKET_NAME/audio_prompts/
    fi
else
    echo -e "${RED}Error: Invalid dataset type '$DATASET_TYPE'. Use 'preencoded' or 'audio'${NC}"
    exit 1
fi

# Calculate upload size
echo ""
echo -e "${GREEN}Upload complete!${NC}"
echo ""
echo "Dataset location: $DEST_PATH"
echo ""
echo -e "${BLUE}To verify upload:${NC}"
echo "  gsutil ls -lh gs://$BUCKET_NAME/preencoded/  # for preencoded"
echo "  gsutil ls -lh gs://$BUCKET_NAME/audio/       # for audio"
echo ""
echo -e "${BLUE}To use in training:${NC}"
if [ "$DATASET_TYPE" == "preencoded" ]; then
    echo "  --preencoded_dir gs://$BUCKET_NAME/preencoded"
else
    echo "  --audio_folder gs://$BUCKET_NAME/audio"
fi
echo ""
