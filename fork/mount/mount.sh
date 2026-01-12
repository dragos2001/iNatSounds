#!/bin/bash

# List of containers
#containers=("train" "test" "val")
containers=("train" "test" "val")
# BlobFuse temporary cache folder (must exist and be writable)
sudo mkdir -p /mnt/blobfuse_tmp
sudo chown -R $USER:$USER /mnt/blobfuse_tmp

# Iterate over containers and mount each
for c in "${containers[@]}"; do
  # Create local mount point
sudo mkdir -p /mnt/$c
sudo chown -R $USER /mnt/$c

  # Mount the container
  blobfuse2 mount /mnt/$c \
    --config-file=/home/azureuser/cloudfiles/code/Users/Bratfalean.Ra.Dragos/iNatSounds/mount/blob_fuse_$c.yaml \
    --tmp-path=/mnt/blobfuse_tmp
done
