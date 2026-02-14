# Quick Start Guide - IR Eye Colorization

Get started with colorizing IR/grayscale eye images in 3 steps.

## Prerequisites

```bash
# Install dependencies
pip install google-genai==1.41.0 Pillow==11.3.0 tqdm==4.67.1

# Set API key (get it from https://aistudio.google.com/app/apikey)
export GEMINI_API_KEY="your-api-key-here"
```

## Basic Usage

### 1. Prepare Your Images

Organize your IR/grayscale eye images in a directory:

```
data/
└── ir_eyes/
    ├── image001.jpg
    ├── image002.jpg
    └── image003.jpg
```

### 2. Run Colorization

```bash
python augmentation/colorize_ir_eyes.py \
    --in_dir data/ir_eyes \
    --out_dir data/colorized
```

### 3. Check Results

```
data/colorized/
└── colorized_2025-12-16_14-30-45/
    ├── config.json                    # Settings used
    ├── run.log                        # Execution log
    ├── image001_colorized.jpg         # Colorized!
    ├── image002_colorized.jpg
    └── image003_colorized.jpg
```

## Common Options

### Generate Multiple Variants

```bash
python augmentation/colorize_ir_eyes.py \
    -i data/ir_eyes \
    -o data/colorized \
    --variants 3
```

### Specific Eye Color

```bash
# Brown eyes
python augmentation/colorize_ir_eyes.py -i data/ir_eyes -o data/brown --eye_color brown

# Blue eyes
python augmentation/colorize_ir_eyes.py -i data/ir_eyes -o data/blue --eye_color blue

# Green eyes
python augmentation/colorize_ir_eyes.py -i data/ir_eyes -o data/green --eye_color green
```

### Using a Config File

Create `my_config.json`:
```json
{
    "in_dir": "data/ir_eyes",
    "out_dir": "data/colorized",
    "variants": 2,
    "eye_color": "auto"
}
```

Run:
```bash
python augmentation/colorize_ir_eyes.py --config my_config.json
```

## Testing

Test with sample images:

```bash
# Copy some sample images to a test folder
mkdir test_images
cp data/sample/*.jpg test_images/

# Run test script
./augmentation/test_colorize.sh test_images

# Check results
ls test_colorization_output/
```

## Full Example Workflow

```bash
# 1. Set up environment
export GEMINI_API_KEY="your-key"

# 2. Test with 1 image
python augmentation/colorize_ir_eyes.py \
    -i test_single \
    -o test_output

# 3. If satisfied, process full dataset
python augmentation/colorize_ir_eyes.py \
    -i data/mobius/train/images \
    -o data/mobius_colorized/train/images \
    --variants 2 \
    --temperature 0.7

# 4. Process validation set
python augmentation/colorize_ir_eyes.py \
    -i data/mobius/val/images \
    -o data/mobius_colorized/val/images \
    --variants 2

# 5. Process test set
python augmentation/colorize_ir_eyes.py \
    -i data/mobius/test/images \
    -o data/mobius_colorized/test/images \
    --variants 1
```

## Need Help?

- **Full documentation:** See `augmentation/README.md`
- **All options:** Run `python augmentation/colorize_ir_eyes.py --help`
- **Troubleshooting:** Check the `run.log` file in output directory
- **API issues:** Verify `GEMINI_API_KEY` is set: `echo $GEMINI_API_KEY`

## Cost Estimate

Approximate costs at current Gemini pricing (~$0.05 per 1000 images):

| Images | Variants | Cost |
|--------|----------|------|
| 100    | 1        | $0.005 |
| 100    | 3        | $0.015 |
| 1,000  | 1        | $0.05 |
| 1,000  | 3        | $0.15 |
| 10,000 | 1        | $0.50 |
| 10,000 | 3        | $1.50 |

*Check current pricing at: https://ai.google.dev/pricing*
