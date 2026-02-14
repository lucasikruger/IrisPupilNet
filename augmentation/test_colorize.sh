#!/bin/bash
# Test script for IR eye colorization
# Creates a small test run with sample images

set -e  # Exit on error

echo "=== IR Eye Colorization Test ==="

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set"
    echo "Please set it with: export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

# Default test directory (override with first argument)
TEST_DIR="${1:-test_ir_images}"

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory not found: $TEST_DIR"
    echo "Usage: $0 [test_images_directory]"
    echo "Example: $0 data/sample_eyes"
    exit 1
fi

# Count images
NUM_IMAGES=$(find "$TEST_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)

if [ "$NUM_IMAGES" -eq 0 ]; then
    echo "Error: No images found in $TEST_DIR"
    exit 1
fi

echo "Found $NUM_IMAGES images in $TEST_DIR"

# Create output directory
OUTPUT_DIR="test_colorization_output"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Test 1: Basic colorization (auto color)
echo "Test 1: Basic colorization (auto color, 1 variant)..."
python augmentation/colorize_ir_eyes.py \
    --in_dir "$TEST_DIR" \
    --out_dir "$OUTPUT_DIR/auto" \
    --variants 1 \
    --temperature 0.7

echo "✓ Test 1 complete"
echo ""

# Test 2: Specific eye color
echo "Test 2: Brown eyes (1 variant)..."
python augmentation/colorize_ir_eyes.py \
    --in_dir "$TEST_DIR" \
    --out_dir "$OUTPUT_DIR/brown" \
    --variants 1 \
    --eye_color brown \
    --temperature 0.7

echo "✓ Test 2 complete"
echo ""

# Test 3: Multiple variants
echo "Test 3: Multiple variants (3 variants, auto color)..."
python augmentation/colorize_ir_eyes.py \
    --in_dir "$TEST_DIR" \
    --out_dir "$OUTPUT_DIR/multi_variant" \
    --variants 3 \
    --temperature 0.8

echo "✓ Test 3 complete"
echo ""

# Summary
echo "=== All tests complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated folders:"
ls -1 "$OUTPUT_DIR"
echo ""
echo "Total colorized images:"
find "$OUTPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l
