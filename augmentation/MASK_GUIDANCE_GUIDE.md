# Mask-Guided Image Generation

This tool uses segmentation masks as **visual guides** to help Gemini generate anatomically accurate eye images.

## Concept

The script overlays colored masks on eye images to show Gemini exactly where each anatomical structure is located:

- 🔴 **RED** = Pupil (dark center)
- 🟢 **GREEN** = Iris (colored ring)
- ⚫ **GRAY** = Background/Sclera (white of eye)

This visual guidance helps the model understand and preserve the anatomical structure during generation.

## Use Cases

1. **Colorize IR eyes** while maintaining exact anatomical boundaries
2. **Generate variations** that preserve iris/pupil structure
3. **Test if mask guidance** improves anatomical accuracy
4. **Style transfer** with structure preservation
5. **Data augmentation** with controlled anatomical consistency

## Quick Start

### Basic Usage

```bash
python augmentation/generate_with_mask_guidance.py \
    --images_dir data/images \
    --masks_dir data/masks \
    --out_dir data/mask_guided_output
```

### With Transformation

```bash
# Colorize grayscale images
python augmentation/generate_with_mask_guidance.py \
    -i data/ir_images \
    -m data/ir_masks \
    -o data/colorized_guided \
    -p "colorize this grayscale eye image with realistic brown eyes"

# Make eyes more vivid
python augmentation/generate_with_mask_guidance.py \
    -i data/images \
    -m data/masks \
    -o data/vivid_eyes \
    -p "enhance the eye colors to be more vivid and striking"

# Add artistic style
python augmentation/generate_with_mask_guidance.py \
    -i data/images \
    -m data/masks \
    -o data/artistic \
    -p "transform to watercolor painting style"
```

## How It Works

### Step 1: Prepare Input

You need:
- **Images directory**: Eye images
- **Masks directory**: Corresponding segmentation masks

```
data/
├── images/
│   ├── eye001.jpg
│   ├── eye002.jpg
│   └── eye003.jpg
└── masks/
    ├── eye001.png
    ├── eye002.png
    └── eye003.png
```

### Step 2: Script Creates Mask Overlays

The script:
1. Loads each image + corresponding mask
2. Colors the mask (red=pupil, green=iris, gray=background)
3. Overlays it on the image with transparency
4. Saves the "guide image"

### Step 3: Gemini Generates

The script sends to Gemini:
- **The mask-guided image** (image + colored overlay)
- **A detailed prompt** explaining what each color means

Gemini then generates a new image that:
- Understands the anatomical structure from the colors
- Preserves the pupil/iris boundaries
- Applies the requested transformation

## Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--images_dir` | `-i` | *required* | Directory with eye images |
| `--masks_dir` | `-m` | *required* | Directory with segmentation masks |
| `--out_dir` | `-o` | `mask_guided_output` | Output directory |
| `--prompt` | `-p` | Default | Transformation prompt |
| `--variants` | `-v` | `1` | Number of variants per image |
| `--temperature` | `-t` | `0.7` | Creativity (0.0-1.0) |
| `--mask_opacity` | `-a` | `0.4` | Mask overlay opacity (0.0-1.0) |
| `--show_legend` | - | `False` | Add color legend to guides |
| `--config` | `-c` | None | JSON config file |

## Output Structure

```
mask_guided_output/
└── mask_guided_2025-12-16_14-30-45/
    ├── config.json
    ├── run.log
    ├── mask_guides/           # Images with mask overlays
    │   ├── eye001_guide.jpg
    │   ├── eye002_guide.jpg
    │   └── ...
    └── generated/            # Generated results
        ├── eye001_generated.jpg
        ├── eye002_generated.jpg
        └── ...
```

## Example Prompts

### Colorization

```bash
--prompt "colorize this grayscale infrared eye image. Use realistic brown eyes with natural skin tones"

--prompt "transform to colored image with vivid blue eyes and detailed iris patterns"

--prompt "add realistic colors: hazel eyes, natural eyelashes, healthy skin"
```

### Style Transfer

```bash
--prompt "make it look like a professional portrait photograph with studio lighting"

--prompt "transform to artistic watercolor painting while keeping anatomical structure"

--prompt "apply oil painting style with rich textures and colors"
```

### Enhancement

```bash
--prompt "enhance the image quality, add realistic details and texture"

--prompt "make the iris more detailed with visible patterns and depth"

--prompt "improve lighting and add natural reflections in the eye"
```

### Variations

```bash
--prompt "generate a natural variation of this eye with similar but different features"

--prompt "keep the structure but change the eye color to green"

--prompt "same anatomy but make it look like a different person"
```

## Mask Opacity Settings

The `--mask_opacity` parameter controls how visible the colored overlay is:

- **0.2**: Very subtle, barely visible guide
- **0.4** (default): Balanced, clear but not overwhelming
- **0.6**: Strong, very clear guidance
- **0.8**: Very strong, dominates the image

**Recommendation:** Start with 0.4, increase if you want stronger guidance.

## With Legend

Add `--show_legend` to include a color key on the guide images:

```bash
python augmentation/generate_with_mask_guidance.py \
    -i data/images \
    -m data/masks \
    -o data/output \
    --show_legend
```

This adds a legend at the bottom explaining what each color means, which can be helpful when reviewing the guides.

## Mask Format

The script expects masks in **MOBIUS format**:

### RGB Color-Coded Masks
- Red (255, 0, 0): Background/Sclera
- Green (0, 255, 0): Iris
- Blue (0, 0, 255): Pupil

### Or Class Index Masks
- 0: Background/Sclera
- 1: Iris
- 2: Pupil

The script automatically detects the format.

## Experimental Workflows

### Experiment 1: Does Mask Guidance Help Colorization?

```bash
# Without mask guidance (baseline)
python augmentation/colorize_ir_eyes.py \
    -i data/ir_eyes \
    -o data/baseline_colorized \
    -v 3

# With mask guidance
python augmentation/generate_with_mask_guidance.py \
    -i data/ir_eyes \
    -m data/ir_masks \
    -o data/guided_colorized \
    -p "colorize with realistic eye colors" \
    -v 3
```

**Compare:** Do the mask-guided results preserve anatomical structure better?

### Experiment 2: Structure Preservation

```bash
# Generate variations with mask guidance
python augmentation/generate_with_mask_guidance.py \
    -i data/images \
    -m data/masks \
    -o data/structure_test \
    -p "generate a natural variation" \
    -v 5
```

**Analyze:** Measure IoU between original masks and segmentations of generated images.

### Experiment 3: Opacity Impact

```bash
# Test different opacity levels
for opacity in 0.2 0.4 0.6 0.8; do
    python augmentation/generate_with_mask_guidance.py \
        -i data/sample \
        -m data/sample_masks \
        -o "data/opacity_$opacity" \
        -a $opacity
done
```

**Compare:** Which opacity gives best anatomical preservation?

## Integration with Training

### Workflow: Generate Training Data

```bash
# 1. Generate mask-guided variations
python augmentation/generate_with_mask_guidance.py \
    -i data/mobius/train/images \
    -m data/mobius/train/masks \
    -o data/mobius_guided/train \
    -v 2

# 2. Use original masks for generated images
# (Since structure is preserved, original masks should align well)

# 3. Update CSV to include generated images + original masks

# 4. Train on expanded dataset
python -m irispupilnet.train --csv dataset/with_guided_augmentation.csv
```

### Quality Control

Before using generated images for training:

1. **Visual inspection**: Check a sample of generated images
2. **Mask alignment**: Verify original masks still align with generated images
3. **Metrics**: Compute segmentation metrics on generated images

## Tips for Best Results

### Prompt Engineering

**Do:**
- Be specific about what you want
- Mention "preserve anatomical structure"
- Reference the colored regions if needed
- Use photographic terms (lighting, focus, etc.)

**Don't:**
- Ask for anatomical changes (moving pupil, changing iris size)
- Request removal of mask guidance in prompt (script does this automatically)
- Be too vague ("make it better")

### Temperature Settings

- **0.3-0.5**: Conservative, stays close to original
- **0.7** (default): Balanced creativity
- **0.8-1.0**: More creative, larger variations

### Mask Opacity

- **Lower (0.2-0.3)**: If masks are very accurate
- **Medium (0.4-0.5)**: Standard, good balance
- **Higher (0.6-0.8)**: If you want strong anatomical constraints

## Troubleshooting

### "No mask found for image"

Ensure mask filenames match image filenames:
- `eye001.jpg` → `eye001.png` (or `eye001_mask.png`)

### Generated images don't preserve structure

Try:
1. Increase `--mask_opacity` to 0.6 or 0.8
2. Lower `--temperature` to 0.5
3. Add "maintain exact structure" to your prompt
4. Use `--show_legend` to make guidance clearer

### Poor quality results

- Check that masks are accurate
- Try different prompts
- Adjust temperature
- Ensure images are good quality

## Advanced: Custom Prompts

You can modify the `build_mask_guidance_prompt()` function to customize:

- Explanation of colored regions
- Specific requirements
- Style guidelines

## Comparison: With vs Without Mask Guidance

| Aspect | Without Masks | With Masks |
|--------|---------------|------------|
| Structure Preservation | Variable | Better controlled |
| Anatomical Accuracy | Depends on prompt | Guided by visual cues |
| Pupil Position | May shift | Locked by red region |
| Iris Boundaries | May blur | Preserved by green region |
| Use Case | General transformations | Structure-critical augmentation |

## Example Config File

`mask_guided_config.json`:
```json
{
    "images_dir": "data/images",
    "masks_dir": "data/masks",
    "out_dir": "data/mask_guided_output",
    "prompt": "colorize with realistic eye colors",
    "variants": 2,
    "temperature": 0.7,
    "mask_opacity": 0.4,
    "show_legend": false
}
```

Usage:
```bash
python augmentation/generate_with_mask_guidance.py -c mask_guided_config.json
```

## Research Questions

This tool enables investigating:

1. **Does mask guidance improve anatomical preservation in generated images?**
2. **What opacity level gives optimal structure preservation vs. quality?**
3. **Can mask-guided generation create better training data than unguided?**
4. **How well do original masks align with generated images?**

Run experiments and measure:
- IoU between original and generated-then-segmented masks
- Visual quality scores
- Training performance on augmented datasets
