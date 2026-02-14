# Quick Start: Dataset Visualization

**Goal:** See how adapters normalize different datasets to the same format.

---

## 1. Test Setup (30 seconds)

```bash
python examples/test_visualization.py
```

Should print: `✓ All tests passed!`

---

## 2. Visualize All Datasets (1 minute)

```bash
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset \
    --grayscale
```

This shows one sample from each dataset type in your CSV.

**Save to file:**
```bash
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset \
    --grayscale \
    --output dataset_comparison.png
```

---

## 3. Visualize Specific Sample (30 seconds)

### Option A: Grayscale (Recommended - matches training)

```python
from pathlib import Path
from irispupilnet.utils.visualize import visualize_sample_grayscale

visualize_sample_grayscale(
    image_path=Path("dataset/mobius/train/img_001.png"),
    mask_path=Path("dataset/mobius/train/mask_001.png"),
    adapter_name="mobius"
)
```

### Option B: RGB

```python
from pathlib import Path
from irispupilnet.utils.visualize import visualize_sample_rgb

visualize_sample_rgb(
    image_path=Path("dataset/mobius/train/img_001.png"),
    mask_path=Path("dataset/mobius/train/mask_001.png"),
    adapter_name="mobius"
)
```

---

## 4. Compare Multiple Datasets

```python
from pathlib import Path
from irispupilnet.utils.visualize import visualize_multiple_samples

samples = [
    (Path("dataset/mobius/img.png"),
     Path("dataset/mobius/mask.png"),
     "mobius"),

    (Path("dataset/tayed/img.png"),
     Path("dataset/tayed/mask.png"),
     "tayed"),
]

visualize_multiple_samples(samples, grayscale=True)
```

---

## What You'll See

Each visualization shows **3 panels**:

```
┌─────────────┬─────────────┬─────────────┐
│   Image     │    Mask     │   Overlay   │
│ (BW or RGB) │ (Normalized)│             │
└─────────────┴─────────────┴─────────────┘
```

**Colors (same for ALL datasets):**
- 🖤 **Black** = Background (0)
- 💚 **Green** = Iris (1)
- ❤️ **Red** = Pupil (2)

---

## Common Commands

```bash
# Test
python examples/test_visualization.py

# Visualize all (grayscale)
python examples/visualize_adapters.py --csv CSV --data-root ROOT --grayscale

# Visualize all (RGB)
python examples/visualize_adapters.py --csv CSV --data-root ROOT

# Save to file
python examples/visualize_adapters.py --csv CSV --data-root ROOT --grayscale --output out.png
```

---

## Two Main Functions

As requested, here are the two functions:

### 1. RGB
```python
from irispupilnet.utils.visualize import visualize_sample_rgb

visualize_sample_rgb(img_path, mask_path, "mobius")
```

### 2. Grayscale (Recommended)
```python
from irispupilnet.utils.visualize import visualize_sample_grayscale

visualize_sample_grayscale(img_path, mask_path, "mobius")
```

**Both work with ALL adapters:** mobius, tayed, irispupileye, unity_eyes

---

## Available Adapters

```python
from irispupilnet.datasets.adapters import list_adapters

print(list_adapters())
# ['mobius', 'mobius_3c', 'tayed', 'tayed_3c',
#  'irispupileye', 'iris_pupil_eye_cls',
#  'unity_eyes', 'unity_eyes_3c']
```

---

## Full Documentation

- **Quick guide:** This file
- **Complete guide:** `examples/README_VISUALIZE.md`
- **Summary:** `VISUALIZATION_SUMMARY.md`
- **Code:** `irispupilnet/utils/visualize.py`

---

## That's It! 🎉

Two simple functions visualize any dataset using adapters:
- `visualize_sample_rgb()` - RGB images
- `visualize_sample_grayscale()` - Grayscale (recommended)

All datasets show the same normalized format! ✅
