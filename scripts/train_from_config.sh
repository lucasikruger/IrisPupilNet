#!/bin/bash
# Simple wrapper to train from a config file
#
# Usage:
#   ./scripts/train_from_config.sh irispupilnet/configs/example_grayscale.yaml
#   ./scripts/train_from_config.sh myconfig.yaml

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [additional args...]"
    echo ""
    echo "Examples:"
    echo "  $0 irispupilnet/configs/example_grayscale.yaml"
    echo "  $0 myconfig.yaml --epochs 100"
    echo ""
    echo "Available example configs:"
    ls -1 irispupilnet/configs/*.yaml 2>/dev/null || true
    exit 1
fi

CONFIG="$1"
shift  # Remove first argument, keep rest for pass-through

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    echo ""
    echo "Available example configs:"
    ls -1 irispupilnet/configs/*.yaml 2>/dev/null || true
    exit 1
fi

echo "Training with config: $CONFIG"
if [ "$#" -gt 0 ]; then
    echo "Additional arguments: $@"
fi
echo ""

python -m irispupilnet.train --config "$CONFIG" "$@"
