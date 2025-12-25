#!/bin/bash
# 1. Set the credentials path explicitly
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/gcloud/application_default_credentials.json

# 2. Cleanup (Kill old processes and unmount force)
sudo pkill -9 -f gcsfuse
sudo fusermount -u -z $HOME/bucket_mount
sudo umount -l $HOME/bucket_mount
mkdir -p $HOME/bucket_mount

# 3. Mount (Log to file to avoid SSH hanging)
echo "Mounting..."
gcsfuse --implicit-dirs rubin-dia-dataset $HOME/bucket_mount > /tmp/mount.log 2>&1

# 4. Check success
if mount | grep -q "bucket_mount"; then
  echo "SUCCESS: Mounted on $(hostname)"
else
  echo "FAILURE: Could not mount on $(hostname)"
  echo "--- LOGS ---"
  cat /tmp/mount.log
fi
