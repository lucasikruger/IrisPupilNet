#!/bin/bash
# Simple wrapper to export a checkpoint to ONNX
#
# Usage:
#   ./scripts/export_model.sh runs/experiment/best.pt model.onnx
#
# The script automatically reads all configuration from the checkpoint.

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <checkpoint.pt> <output.onnx>"
    echo ""
    echo "Example:"
    echo "  $0 runs/experiment/best.pt model.onnx"
    exit 1
fi

CHECKPOINT="$1"
OUTPUT="$2"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

echo "Exporting checkpoint to ONNX..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT"
echo ""

python export/export_to_onnx.py \
    --checkpoint "$CHECKPOINT" \
    --out "$OUTPUT"

echo ""
echo "Done! You can now run the demo with:"
echo "  python demo/webcam_demo.py --model $OUTPUT"
