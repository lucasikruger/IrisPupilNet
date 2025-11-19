#!/bin/bash
# Build Docker image for IrisPupilNet

set -e

IMAGE_NAME="irispupilnet:latest"

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo ""
echo "Build complete! To run the container:"
echo "  docker run --gpus all -it --rm -v \$(pwd):/app -v /path/to/data:/data $IMAGE_NAME"
echo ""
echo "Or use the docker-run.sh script"
