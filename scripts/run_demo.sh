#!/bin/bash
# Simple wrapper to run the webcam demo
#
# Usage:
#   ./scripts/run_demo.sh model.onnx
#   ./scripts/run_demo.sh  # no model, just face/eye detection

set -e

if [ "$#" -eq 0 ]; then
    echo "Running demo without model (only face/eye detection)..."
    python demo/webcam_demo.py
elif [ "$#" -eq 1 ]; then
    MODEL="$1"
    if [ ! -f "$MODEL" ]; then
        echo "Error: Model file not found: $MODEL"
        exit 1
    fi
    echo "Running demo with model: $MODEL"
    echo "(Config will be auto-detected from model)"
    python demo/webcam_demo.py --model "$MODEL"
else
    echo "Usage: $0 [model.onnx]"
    echo ""
    echo "Examples:"
    echo "  $0                    # no model, just detection"
    echo "  $0 model.onnx         # with segmentation model"
    exit 1
fi
