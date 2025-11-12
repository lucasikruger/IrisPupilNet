# Unity Eyes Dataset Preparation

## Overview

Unity Eyes is a synthetic eye dataset generated using Unity 3D engine. It provides:
- High-resolution face images (JPG)
- Detailed JSON annotations with:
  - Eye contour points (`interior_margin_2d`)
  - Iris boundary points (32 points in `iris_2d`)
  - Pupil size and iris size ratios
  - Lighting conditions
  - Head pose information
  - PCA shape coefficients

## Conversion to MOBIUS Format

The `unity_eyes_to_mobius.py` script converts Unity Eyes to MOBIUS-compatible format:

### What it does:

1. **Crops eyes** to just the eye region using `interior_margin_2d` bounding box
2. **Fits circles** to iris points using least-squares circle fitting
3. **Estimates pupil** size from `pupil_size` / `iris_size` ratio
4. **Clips circles** to eye region polygon (handles partially closed eyes)
5. **Creates RGB masks** in MOBIUS format:
   - Red (255,0,0): Background/sclera
   - Green (0,255,0): Iris
   - Blue (0,0,255): Pupil
6. **Outputs** to MOBIUS directory structure:
   - `images/`: Cropped eye images
   - `Masks/`: RGB segmentation masks
7. **Generates CSV** with metadata

### Usage

```bash
python tools/prepare/unity_eyes_to_mobius.py \
    --input_dir "/path/to/unity_eyes_dataset_1k" \
    --output_dir "data/unity_eyes_mobius" \
    --csv "dataset/unity_eyes_output/unity_eyes_dataset.csv" \
    --padding 30
```

### Arguments

- `--input_dir`: Unity Eyes dataset directory (with .jpg and .json files)
- `--output_dir`: Output directory for MOBIUS-format data
- `--csv`: Output CSV path
- `--padding`: Pixels to add around eye bounding box (default: 20)
- `--limit`: Process only first N images (for testing)

### Example Output

```
Found 1000 images to process
✓ Processed 1000 images successfully
✓ Images saved to: data/unity_eyes_mobius/images
✓ Masks saved to: data/unity_eyes_mobius/Masks
✓ CSV saved to: dataset/unity_eyes_output/unity_eyes_dataset.csv

Dataset statistics:
  Total samples: 1000
  Mean crop size: 204.0 x 129.7
  Mean iris radius: 355.4 px
  Mean pupil radius: 48.7 px
  Unique iris textures: 5
  Unique skin textures: 20
```

## CSV Format

The output CSV contains:

| Column | Description |
|--------|-------------|
| `id` | Image ID (from filename) |
| `rel_image_path` | Relative path to cropped eye image |
| `rel_mask_path` | Relative path to mask |
| `split` | Always 'train' for Unity Eyes |
| `dataset` | 'unity_eyes' |
| `dataset_format` | 'unity_eyes_3c' (MOBIUS-compatible) |
| `crop_width` | Width of cropped eye image |
| `crop_height` | Height of cropped eye image |
| `iris_radius` | Fitted iris radius (pixels) |
| `pupil_radius` | Estimated pupil radius (pixels) |
| `pupil_size` | Original pupil_size from JSON |
| `iris_size` | Original iris_size from JSON |
| `iris_texture` | Iris texture name |
| `lighting_skybox` | Lighting environment |
| `skin_texture` | Skin texture name |
| `head_pose` | Head pose angles |

## Training with Unity Eyes

Once processed, you can train with Unity Eyes using:

```bash
python irispupilnet/train.py \
  --dataset csv_seg \
  --data-root data/unity_eyes_mobius \
  --csv dataset/unity_eyes_output/unity_eyes_dataset.csv \
  --default-format unity_eyes_3c \
  --model unet_se_small \
  --img-size 160 \
  --num-classes 3 \
  --base 32 \
  --batch-size 32 \
  --epochs 20 \
  --out runs/unity_eyes_experiment
```

## Combining with MOBIUS Dataset

To train on both Unity Eyes and MOBIUS:

1. Create combined CSV:
```python
import pandas as pd

# Load both CSVs
unity = pd.read_csv('dataset/unity_eyes_output/unity_eyes_dataset.csv')
mobius = pd.read_csv('dataset/mobius_output/mobius_dataset_split.csv')

# Select common columns
common_cols = ['rel_image_path', 'rel_mask_path', 'split', 'dataset_format']
unity_sub = unity[common_cols].copy()
unity_sub['dataset_root'] = 'data/unity_eyes_mobius'

mobius_sub = mobius[common_cols].copy()
mobius_sub['dataset_root'] = '/media/agot-lkruger/X9 Pro/facu/facu/tesis/MOBIUS'

# Combine
combined = pd.concat([unity_sub, mobius_sub], ignore_index=True)
combined.to_csv('dataset/combined_dataset.csv', index=False)
```

2. Train with combined dataset (requires custom loader or multi-root support)

## Notes

### Handling Edge Cases

- **Negative pupil_size**: Script sets minimum pupil radius to prevent errors
- **Partially closed eyes**: Circles are clipped to `interior_margin` polygon
- **Invalid radii**: Minimum radius enforced (1.0 pixels)

### Mask Format Registration

The `unity_eyes_3c` format is registered in `irispupilnet/utils/mask_formats.py`:

```python
@register_mask_format("unity_eyes_3c")
def unity_eyes_3c(mask_bgr: np.ndarray) -> np.ndarray:
    # Identical to mobius_3c
    # Red=bg, Green=iris, Blue=pupil
    ...
```

### Dataset Characteristics

Unity Eyes provides:
- **Synthetic diversity**: Various lighting, skin tones, iris colors
- **Perfect annotations**: Exact iris/pupil boundaries
- **Controlled conditions**: No motion blur or occlusions
- **Augmentation baseline**: Good for pretraining before fine-tuning on real data

## Troubleshooting

### Import Errors

If you get module import errors, ensure you're running from repository root:
```bash
cd /home/agot-lkruger/tesis/IrisPupilNet
python tools/prepare/unity_eyes_to_mobius.py ...
```

### OpenCV Circle Errors

If you get "radius >= 0" assertion errors, update the script to handle negative pupil sizes (already fixed in current version).

### Memory Issues

If processing all 1000 images causes memory issues, use `--limit`:
```bash
python tools/prepare/unity_eyes_to_mobius.py \
  --input_dir ... \
  --limit 100  # Process only first 100
```

## Citation

If using Unity Eyes dataset, cite the original paper:
```
@article{wood2016learning,
  title={Learning an appearance-based gaze estimator from one million synthesised images},
  author={Wood, Erroll and Baltru{\v{s}}aitis, Tadas and Morency, Louis-Philippe and Robinson, Peter and Bulling, Andreas},
  journal={arXiv preprint arXiv:1610.04936},
  year={2016}
}
```
