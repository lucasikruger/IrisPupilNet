# Dataset Adapter Visualization Guide

This guide explains how to visualize samples from different datasets to verify that all adapters normalize to the same format.

---

## Quick Start

### Option 1: Command-Line Script (Easiest)

Visualize all datasets from your CSV:

```bash
# RGB visualization
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset

# Grayscale visualization (recommended - matches training)
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset \
    --grayscale

# Save to file
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset \
    --grayscale \
    --output visualizations/all_datasets.png
```

This will show one sample from each dataset type found in your CSV.

---

### Option 2: Python Functions (More Control)

Use the utility functions in your own scripts:

```python
from pathlib import Path
from irispupilnet.utils.visualize import (
    visualize_sample_rgb,
    visualize_sample_grayscale,
    visualize_multiple_samples
)

# Visualize a single sample (RGB)
visualize_sample_rgb(
    image_path=Path("dataset/mobius/train/img_001.png"),
    mask_path=Path("dataset/mobius/train/mask_001.png"),
    adapter_name="mobius"
)

# Visualize a single sample (Grayscale - recommended)
visualize_sample_grayscale(
    image_path=Path("dataset/mobius/train/img_001.png"),
    mask_path=Path("dataset/mobius/train/mask_001.png"),
    adapter_name="mobius",
    save_path=Path("mobius_visualization.png")
)

# Visualize multiple samples from different datasets
samples = [
    (Path("dataset/mobius/train/img_001.png"),
     Path("dataset/mobius/train/mask_001.png"),
     "mobius"),

    (Path("dataset/tayed/train/eye_020.png"),
     Path("dataset/tayed/train/eye_020_mask.png"),
     "tayed"),

    (Path("dataset/irispupileye/val/image_50.jpg"),
     Path("dataset/irispupileye/val/mask_50.png"),
     "irispupileye"),
]

visualize_multiple_samples(
    samples=samples,
    grayscale=True,
    save_path=Path("all_datasets_comparison.png")
)
```

---

## What Gets Visualized

Each visualization shows **3 panels**:

1. **Original Image** - The input image (RGB or grayscale)
2. **Mask (Normalized)** - The mask after adapter processing
3. **Overlay** - Image + semi-transparent mask

### Color Coding (Consistent Across All Datasets)

- **Black**: Background (Class 0)
- **Green**: Iris (Class 1)
- **Red**: Pupil (Class 2)

This demonstrates that **all adapters normalize different datasets to the same format**!

---

## Why This Matters

Different datasets use different mask formats:

| Dataset | Original Mask Format |
|---------|---------------------|
| MOBIUS | RGB color-coded (Red=bg, Green=iris, Blue=pupil) |
| TayedEyes | RGB color-coded (same as MOBIUS) |
| IrisPupilEye | RGB color-coded (same as MOBIUS) |
| UnityEyes | RGB color-coded (same as MOBIUS) |

**All adapters convert these to:**
- Class 0 = Background
- Class 1 = Iris
- Class 2 = Pupil

This allows:
✅ Training on mixed datasets
✅ Consistent evaluation
✅ Same augmentation pipeline
✅ Universal visualization

---

## Examples

### Example 1: Verify MOBIUS adapter works

```python
from pathlib import Path
from irispupilnet.utils.visualize import visualize_sample_grayscale

visualize_sample_grayscale(
    image_path=Path("dataset/mobius/train/img_001.png"),
    mask_path=Path("dataset/mobius/train/mask_001.png"),
    adapter_name="mobius"
)
```

### Example 2: Compare all datasets

```bash
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset \
    --grayscale \
    --output comparison.png
```

### Example 3: Check if new adapter works

After creating a new adapter, verify it:

```python
from pathlib import Path
from irispupilnet.utils.visualize import visualize_sample_grayscale

# Test your new adapter
visualize_sample_grayscale(
    image_path=Path("dataset/mynewdataset/img001.png"),
    mask_path=Path("dataset/mynewdataset/mask001.png"),
    adapter_name="mynewdataset"  # Your new adapter name
)

# Check the mask values
from irispupilnet.datasets.adapters import get_adapter
import numpy as np

adapter = get_adapter("mynewdataset")
_, mask = adapter.load_sample(img_path, mask_path)

print(f"Unique classes in mask: {np.unique(mask)}")
# Should print: [0 1 2] or subset thereof
```

---

## Integration with Dataset

These visualization functions use the **same adapters** as your training dataset:

```python
from irispupilnet.datasets.simple_dataset import SimpleGrayscaleDataset

# Create dataset
dataset = SimpleGrayscaleDataset(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)

# Get sample info
info = dataset.get_sample_info(0)
print(info)
# {'image_path': '...', 'mask_path': '...', 'adapter': 'MOBIUS', ...}

# Visualize this exact sample
from irispupilnet.utils.visualize import visualize_sample_grayscale

visualize_sample_grayscale(
    image_path=Path(info['image_path']),
    mask_path=Path(info['mask_path']),
    adapter_name=info['adapter'].lower()
)
```

---

## Troubleshooting

### "Failed to load image"

Check paths are correct:
```python
from pathlib import Path

img_path = Path("dataset/mobius/img001.png")
print(f"Exists: {img_path.exists()}")
print(f"Absolute: {img_path.resolve()}")
```

### "Unknown adapter"

Check adapter name matches registry:
```python
from irispupilnet.datasets.adapters import list_adapters

print("Available adapters:", list_adapters())
```

### Wrong colors in mask

Adapters expect specific formats. Check your mask format matches the adapter:

```python
import cv2
import numpy as np

# Load mask directly
mask = cv2.imread("dataset/mobius/mask001.png", cv2.IMREAD_COLOR)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# Check what colors are present
unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
print("Unique colors in mask:", unique_colors)

# Should see: [[0,0,0], [0,255,0], [255,0,0]] for RGB color-coded
# Or: [[0,0,255], [0,255,0], [255,0,0]] if BGR
```

---

## API Reference

### visualize_sample_rgb()

```python
def visualize_sample_rgb(
    image_path: Path,
    mask_path: Path,
    adapter_name: str,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
) -> plt.Figure
```

Visualize sample with RGB image.

**Parameters:**
- `image_path`: Path to image file
- `mask_path`: Path to mask file
- `adapter_name`: Name of adapter (e.g., 'mobius', 'tayed')
- `figsize`: Figure size (width, height)
- `save_path`: Optional path to save figure

**Returns:** Matplotlib figure

---

### visualize_sample_grayscale()

```python
def visualize_sample_grayscale(
    image_path: Path,
    mask_path: Path,
    adapter_name: str,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
) -> plt.Figure
```

Visualize sample with grayscale image (recommended - matches training).

**Parameters:** Same as `visualize_sample_rgb()`

**Returns:** Matplotlib figure

---

### visualize_multiple_samples()

```python
def visualize_multiple_samples(
    samples: list,
    grayscale: bool = True,
    figsize: Tuple[int, int] = (15, None),
    save_path: Optional[Path] = None
) -> plt.Figure
```

Visualize multiple samples in a grid.

**Parameters:**
- `samples`: List of tuples `(image_path, mask_path, adapter_name)`
- `grayscale`: If True, show grayscale; if False, show RGB
- `figsize`: Figure size (height auto-calculated if None)
- `save_path`: Optional path to save figure

**Returns:** Matplotlib figure

---

## Summary

You have **two ways** to visualize:

1. **Command-line script** (`examples/visualize_adapters.py`)
   - Quick visualization from CSV
   - Good for checking all datasets at once

2. **Python functions** (`irispupilnet.utils.visualize`)
   - More control
   - Good for debugging specific samples
   - Can be integrated into other scripts

Both use the **same adapters** as your training pipeline, ensuring consistency!

**Key functions:**
- `visualize_sample_rgb()` - Show RGB image
- `visualize_sample_grayscale()` - Show grayscale (recommended)
- `visualize_multiple_samples()` - Show multiple samples

All show the same normalized format: **Class 0=Background, 1=Iris, 2=Pupil** 🎯
