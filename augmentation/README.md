# Data Augmentation Tools

This directory contains tools for augmenting eye image datasets using AI-powered transformations with Gemini Nano Banana.

## Available Tools

### 1. `colorize_ir_eyes.py` - IR Eye Colorization

Transform grayscale/infrared eye images into realistic colored eye images.

**Use case:** Augment IR/grayscale eye datasets by generating realistic colored versions, which can help improve model generalization and robustness.

**Key feature:** Automatic colorization without needing masks.

### 2. `generate_with_mask_guidance.py` - Mask-Guided Generation

Generate eye images using segmentation masks as visual guides. The masks are overlaid with colors (red=pupil, green=iris, gray=background) and explained to Gemini to help preserve anatomical structure.

**Use case:** Generate variations, colorizations, or transformations while maintaining exact anatomical boundaries. Useful for experiments testing if mask guidance improves structure preservation.

**Key feature:** Anatomically-aware generation with structure preservation.

---

## Installation

Install required dependencies:

```bash
pip install google-genai==1.41.0 Pillow==11.3.0 tqdm==4.67.1
```

**Get a Gemini API key:**
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Set it as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## Quick Start

### Basic Colorization

```bash
# Colorize all images in a directory
python augmentation/colorize_ir_eyes.py \
    --in_dir data/ir_eyes \
    --out_dir data/colorized
```

### With Specific Eye Color

```bash
# Generate brown-eyed versions
python augmentation/colorize_ir_eyes.py \
    --in_dir data/ir_eyes \
    --out_dir data/colorized_brown \
    --eye_color brown
```

### Multiple Variants

```bash
# Generate 3 different colorized versions of each image
python augmentation/colorize_ir_eyes.py \
    --in_dir data/ir_eyes \
    --out_dir data/colorized \
    --variants 3
```

### Using a Reference Image

```bash
# Use a reference image to guide the colorization style
python augmentation/colorize_ir_eyes.py \
    --in_dir data/ir_eyes \
    --out_dir data/colorized \
    --reference_image examples/reference_eye.jpg
```

### With Config File

```bash
# Use a JSON config file
python augmentation/colorize_ir_eyes.py --config colorize_config.json
```

---

## Configuration Options

### Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--in_dir` | `-i` | *required* | Input directory with IR/grayscale eye images |
| `--out_dir` | `-o` | `colorized_eyes` | Output directory for colorized images |
| `--variants` | `-v` | `1` | Number of colorized versions per image |
| `--temperature` | `-t` | `0.7` | Creativity level (0.0 = deterministic, 1.0 = creative) |
| `--eye_color` | `-e` | `auto` | Eye color: `auto`, `brown`, `blue`, `green`, `hazel`, `gray` |
| `--reference_image` | `-r` | None | Reference image for style guidance |
| `--config` | `-c` | None | JSON config file |
| `--api_key` | - | env var | Gemini API key (or set `GEMINI_API_KEY`) |
| `--model` | `-m` | `gemini-2.5-flash-image` | Gemini model to use |

### Eye Color Options

- **`auto`** (default): Let Gemini choose realistic, natural eye colors
- **`brown`**: Warm brown eyes with depth and detail
- **`blue`**: Vivid blue eyes with realistic iris patterns
- **`green`**: Striking green eyes with natural variations
- **`hazel`**: Hazel eyes with brown-green mixing
- **`gray`**: Cool gray eyes with subtle blue undertones

---

## Example Workflows

### 1. Augment Training Dataset

```bash
# Create colored versions of all IR training images
python augmentation/colorize_ir_eyes.py \
    --in_dir data/mobius/train/images \
    --out_dir data/mobius_colorized/train/images \
    --variants 2 \
    --eye_color auto
```

### 2. Generate Diverse Eye Colors

```bash
# Create different eye colors for variety
for color in brown blue green hazel gray; do
    python augmentation/colorize_ir_eyes.py \
        --in_dir data/ir_eyes \
        --out_dir "data/colorized_$color" \
        --eye_color $color \
        --variants 1
done
```

### 3. Style Transfer from Reference

```bash
# Match the style of a specific reference image
python augmentation/colorize_ir_eyes.py \
    --in_dir data/ir_eyes \
    --out_dir data/colorized_styled \
    --reference_image examples/target_style.jpg \
    --temperature 0.5
```

### 4. Batch Processing with Config

Create `colorize_config.json`:
```json
{
    "in_dir": "data/ir_eyes",
    "out_dir": "data/colorized",
    "variants": 3,
    "temperature": 0.7,
    "eye_color": "auto"
}
```

Then run:
```bash
python augmentation/colorize_ir_eyes.py --config colorize_config.json
```

---

## Output Structure

The script creates a timestamped output directory:

```
data/colorized/
└── colorized_2025-12-16_14-30-45/
    ├── config.json              # Saved configuration
    ├── run.log                  # Execution log
    ├── image001_colorized.jpg   # Colorized images
    ├── image002_colorized.jpg
    └── ...
```

With multiple variants:
```
colorized_2025-12-16_14-30-45/
├── image001_colorized_v00.jpg
├── image001_colorized_v01.jpg
├── image001_colorized_v02.jpg
├── image002_colorized_v00.jpg
└── ...
```

---

## Tips for Best Results

### Temperature Settings

- **0.3-0.5**: More consistent, conservative colorization
- **0.7** (default): Balanced creativity and consistency
- **0.8-1.0**: More creative, varied results

### Prompt Customization

You can modify the colorization prompt in the script to achieve specific effects. The prompt is defined in `build_colorization_prompt()`.

### Processing Large Datasets

For large datasets, run in batches to avoid API rate limits:

```bash
# Process in chunks
python augmentation/colorize_ir_eyes.py \
    --in_dir data/chunk_1 \
    --out_dir data/colorized/batch_1

python augmentation/colorize_ir_eyes.py \
    --in_dir data/chunk_2 \
    --out_dir data/colorized/batch_2
```

### Quality Control

Always review a sample of outputs before processing your entire dataset:

```bash
# Test on a small subset first
python augmentation/colorize_ir_eyes.py \
    --in_dir data/sample_10_images \
    --out_dir test_colorization \
    --variants 3
```

---

## API Costs

Gemini API pricing (as of 2025):
- **gemini-2.5-flash-image**: ~$0.05 per 1000 images

**Example costs:**
- 1000 images × 1 variant = ~$0.05
- 1000 images × 3 variants = ~$0.15

Check current pricing at: https://ai.google.dev/pricing

---

## Troubleshooting

### "Set GEMINI_API_KEY env var"

Set your API key:
```bash
export GEMINI_API_KEY="your-key-here"
```

Or pass it directly:
```bash
python augmentation/colorize_ir_eyes.py --api_key "your-key-here" ...
```

### "No images found"

Make sure your input directory contains supported image formats:
- `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`

### Rate Limit Errors

If you hit API rate limits, add delays between requests or process in smaller batches.

### Poor Colorization Quality

Try adjusting:
- **Temperature**: Lower (0.3-0.5) for more consistent results
- **Reference image**: Provide a high-quality reference image
- **Eye color**: Specify a specific color instead of "auto"

---

## Integration with Training Pipeline

### Option 1: Augment Dataset Offline

```bash
# 1. Colorize IR images
python augmentation/colorize_ir_eyes.py \
    --in_dir data/mobius/train/images \
    --out_dir data/mobius_colorized/train/images \
    --variants 2

# 2. Update CSV to include colorized images
# (manually or with a script)

# 3. Train on combined dataset
python -m irispupilnet.train --csv dataset/combined.csv
```

### Option 2: Create Separate Colorized Dataset

```bash
# Train on IR only
python -m irispupilnet.train --csv dataset/ir_only.csv --color false

# Train on colorized only
python -m irispupilnet.train --csv dataset/colorized_only.csv --color true

# Train on both
python -m irispupilnet.train --csv dataset/ir_and_colorized.csv --color true
```

---

## Advanced Usage

### Custom Prompts

Modify `build_colorization_prompt()` in the script to customize:
- Specific iris patterns (e.g., "central heterochromia")
- Lighting conditions (e.g., "soft natural daylight")
- Artistic styles (e.g., "portrait photography")

### Batch Processing Script

Create `process_all.sh`:
```bash
#!/bin/bash

for subset in train val test; do
    python augmentation/colorize_ir_eyes.py \
        --in_dir "data/mobius/$subset/images" \
        --out_dir "data/mobius_colorized/$subset/images" \
        --variants 2 \
        --temperature 0.7
done
```

---

## Limitations

1. **Anatomical preservation**: While Gemini tries to preserve iris/pupil structure, minor variations may occur
2. **Mask alignment**: Colorized images may have slight shifts; verify mask alignment if using for training
3. **API dependency**: Requires internet connection and valid API key
4. **Processing time**: ~1-2 seconds per image variant
5. **SynthID watermark**: Generated images include a watermark (invisible but detectable)

---

---

## Tool Comparison

| Feature | `colorize_ir_eyes.py` | `generate_with_mask_guidance.py` |
|---------|----------------------|----------------------------------|
| **Input** | Images only | Images + Masks |
| **Guidance** | Text prompt only | Text + Visual (colored masks) |
| **Structure Preservation** | Best effort | Explicit guidance |
| **Use Case** | General colorization | Anatomically-controlled generation |
| **Setup** | Simple (just images) | Requires masks |
| **Experiment** | Quick augmentation | Test mask guidance effectiveness |

**Recommendation:**
- Use `colorize_ir_eyes.py` for quick colorization without masks
- Use `generate_with_mask_guidance.py` when you need structure preservation or want to test if mask guidance helps

---

## Complete Workflow Examples

### Workflow 1: Simple Colorization

```bash
# Just colorize without masks
python augmentation/colorize_ir_eyes.py \
    -i data/ir_eyes \
    -o data/colorized \
    --variants 2
```

### Workflow 2: Mask-Guided Colorization (Experiment)

```bash
# Colorize with mask guidance
python augmentation/generate_with_mask_guidance.py \
    -i data/ir_eyes \
    -m data/ir_masks \
    -o data/colorized_guided \
    -p "colorize with realistic brown eyes" \
    --variants 2

# Compare: Does mask guidance preserve structure better?
```

### Workflow 3: Generate Diverse Eye Colors (Both Methods)

```bash
# Method 1: Without masks
for color in brown blue green; do
    python augmentation/colorize_ir_eyes.py \
        -i data/ir_eyes \
        -o "data/colorized_$color" \
        --eye_color $color
done

# Method 2: With mask guidance
for color in "brown" "blue" "green"; do
    python augmentation/generate_with_mask_guidance.py \
        -i data/ir_eyes \
        -m data/ir_masks \
        -o "data/guided_$color" \
        -p "colorize with realistic $color eyes"
done
```

---

## Research Experiments

### Experiment: Mask Guidance Effectiveness

**Question:** Does providing colored mask overlays improve anatomical preservation?

```bash
# Control group: No masks
python augmentation/colorize_ir_eyes.py \
    -i data/test_set \
    -o experiments/no_masks \
    -v 3

# Treatment group: With masks
python augmentation/generate_with_mask_guidance.py \
    -i data/test_set \
    -m data/test_masks \
    -o experiments/with_masks \
    -p "colorize realistically" \
    -v 3

# Analyze:
# 1. Segment both outputs with trained model
# 2. Compare IoU with original masks
# 3. Visual quality assessment
```

---

## Future Enhancements

Potential improvements:
- [ ] Automatic mask adjustment for generated images
- [ ] Bulk processing with progress checkpointing
- [ ] Quality assessment metrics (IoU preservation, etc.)
- [ ] Custom style presets (medical, artistic, etc.)
- [ ] Parallel processing for faster batch operations
- [ ] Automatic mask-image matching with fuzzy filenames
- [ ] A/B testing framework for comparing methods

---

## References

- [Gemini Image Generation Docs](https://ai.google.dev/gemini-api/docs/image-generation)
- [Imagen Configuration](https://ai.google.dev/gemini-api/docs/imagen#imagen-configuration)
- [Google AI Studio](https://aistudio.google.com/)

---

## Documentation

- **`README.md`** (this file): Overview of all augmentation tools
- **`QUICKSTART.md`**: Quick start guide for IR colorization
- **`MASK_GUIDANCE_GUIDE.md`**: Detailed guide for mask-guided generation
- **`colorize_config_example.json`**: Example config for colorization
- **`mask_guided_config_example.json`**: Example config for mask-guided generation
