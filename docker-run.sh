#!/bin/bash
# Run IrisPupilNet Docker container with GPU support

set -e

IMAGE_NAME="irispupilnet:latest"
DATA_PATH="${1:-/media/hdd2/datasets/eyes}"  # First argument or default to /media/hdd2/datasets/eyes

echo "Running Docker container: $IMAGE_NAME"
echo "Mounting data from: $DATA_PATH"

docker run --gpus all -it --rm \
  -v $(pwd):/app \
  -v $DATA_PATH:/data \
  --shm-size=8g \
  $IMAGE_NAME \
  /bin/bash

# Usage:
# ./docker-run.sh                    # Uses /data as data path
# ./docker-run.sh /path/to/your/data # Uses custom data path
