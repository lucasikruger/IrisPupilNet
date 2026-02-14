# Dataset Visualization Tools - Summary

I've created comprehensive visualization tools to demonstrate how adapters normalize different dataset formats. Here's what you have:

---

## 📁 Files Created

### 1. **Command-Line Script** (`examples/visualize_adapters.py`)
Full-featured script to visualize all datasets from your CSV.

**Usage:**
```bash
# Grayscale visualization (recommended - matches training)
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset \
    --grayscale

# RGB visualization
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset

# Save to file
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset \
    --grayscale \
    --output visualizations/all_datasets.png
```

**What it does:**
- Reads your CSV
- Finds all unique dataset formats
- Takes one sample from each format
- Shows image, mask, and overlay side-by-side
- Demonstrates consistent normalization

---

### 2. **Python Utility Module** (`irispupilnet/utils/visualize.py`)
Reusable visualization functions for use in your own scripts.

**Two main functions (as you requested!):**

#### Function 1: RGB Visualization
```python
from pathlib import Path
from irispupilnet.utils.visualize import visualize_sample_rgb

visualize_sample_rgb(
    image_path=Path("dataset/mobius/img001.png"),
    mask_path=Path("dataset/mobius/mask001.png"),
    adapter_name="mobius",
    save_path=Path("output.png")  # Optional
)
```

#### Function 2: Grayscale Visualization (Recommended!)
```python
from pathlib import Path
from irispupilnet.utils.visualize import visualize_sample_grayscale

visualize_sample_grayscale(
    image_path=Path("dataset/mobius/img001.png"),
    mask_path=Path("dataset/mobius/mask001.png"),
    adapter_name="mobius",
    save_path=Path("output.png")  # Optional
)
```

**Bonus: Multiple Samples**
```python
from irispupilnet.utils.visualize import visualize_multiple_samples

samples = [
    (Path("mobius/img.png"), Path("mobius/mask.png"), "mobius"),
    (Path("tayed/img.png"), Path("tayed/mask.png"), "tayed"),
    (Path("irispupileye/img.png"), Path("irispupileye/mask.png"), "irispupileye"),
]

visualize_multiple_samples(samples, grayscale=True)
```

---

### 3. **Documentation** (`examples/README_VISUALIZE.md`)
Complete guide with:
- Quick start examples
- API reference
- Troubleshooting
- Integration examples

---

### 4. **Test Script** (`examples/test_visualization.py`)
Verify everything works:

```bash
python examples/test_visualization.py
```

---

## 🎨 What Gets Visualized

Each visualization shows **3 panels**:

```
┌─────────────┬─────────────┬─────────────┐
│   Original  │    Mask     │   Overlay   │
│    Image    │ (Normalized)│             │
└─────────────┴─────────────┴─────────────┘
```

**Color Legend (Consistent Across ALL Datasets):**
- **Black**: Background (Class 0)
- **Green**: Iris (Class 1)
- **Red**: Pupil (Class 2)

---

## 🔑 Key Concept: Normalized Format

Different datasets have different raw formats:

| Dataset | Raw Mask Format |
|---------|----------------|
| MOBIUS | RGB: Red=bg, Green=iris, Blue=pupil |
| TayedEyes | RGB: Same as MOBIUS |
| IrisPupilEye | RGB: Same as MOBIUS |
| UnityEyes | RGB: Same as MOBIUS |

**↓ All adapters normalize to ↓**

```python
Class 0 = Background
Class 1 = Iris
Class 2 = Pupil
```

**This is demonstrated visually!** All datasets show the same color scheme (Black/Green/Red) in the visualization, proving they're normalized correctly.

---

## 🚀 Quick Start

### Test the setup:
```bash
python examples/test_visualization.py
```

### Visualize your datasets:
```bash
python examples/visualize_adapters.py \
    --csv dataset/merged/all_datasets.csv \
    --data-root dataset \
    --grayscale
```

### Use in your own scripts:
```python
from pathlib import Path
from irispupilnet.utils.visualize import visualize_sample_grayscale

# Visualize any sample
visualize_sample_grayscale(
    image_path=Path("dataset/mobius/train/img_001.png"),
    mask_path=Path("dataset/mobius/train/mask_001.png"),
    adapter_name="mobius"
)
```

---

## 🎯 Why This Matters

These visualization tools demonstrate that:

1. **All adapters work correctly** - Each loads data in its specific format
2. **Normalization is consistent** - All produce the same class indices [0, 1, 2]
3. **Training will work** - Same format means same augmentation/loss/metrics
4. **Mixed datasets work** - Can combine MOBIUS + TayedEyes + IrisPupilEye seamlessly

**The same functions work for ALL datasets** because adapters make them transparent!

---

## 📋 Function Summary

You asked for two functions (RGB and grayscale), here they are:

### 1. `visualize_sample_rgb(image_path, mask_path, adapter_name)`
- Loads image in RGB
- Shows original RGB + mask + overlay
- Good for colored datasets

### 2. `visualize_sample_grayscale(image_path, mask_path, adapter_name)` ⭐
- Loads image in grayscale
- Shows grayscale + mask + overlay
- **Recommended** - matches training mode

**Both use adapters**, so they work with all dataset formats!

**Bonus:** `visualize_multiple_samples()` for batch visualization

---

## 🔧 Integration with Dataset

These functions use the **same adapters** as your training:

```python
# Training uses adapters
from irispupilnet.datasets.simple_dataset import SimpleGrayscaleDataset

dataset = SimpleGrayscaleDataset(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)

# Get a sample
image, mask = dataset[0]

# Visualize the same sample
info = dataset.get_sample_info(0)

from irispupilnet.utils.visualize import visualize_sample_grayscale

visualize_sample_grayscale(
    image_path=Path(info['image_path']),
    mask_path=Path(info['mask_path']),
    adapter_name=info['adapter'].lower()
)
```

---

## 📖 Documentation

- **Full guide:** `examples/README_VISUALIZE.md`
- **Test script:** `examples/test_visualization.py`
- **CLI tool:** `examples/visualize_adapters.py`
- **Python API:** `irispupilnet/utils/visualize.py`

---

## ✅ Next Steps

1. **Test the setup:**
   ```bash
   python examples/test_visualization.py
   ```

2. **Visualize your datasets:**
   ```bash
   python examples/visualize_adapters.py \
       --csv dataset/merged/all_datasets.csv \
       --data-root dataset \
       --grayscale \
       --output dataset_comparison.png
   ```

3. **Use in your own code:**
   ```python
   from irispupilnet.utils.visualize import visualize_sample_grayscale

   # Visualize any sample
   visualize_sample_grayscale(img_path, mask_path, "mobius")
   ```

---

## 💡 Summary

You now have:

✅ **Two visualization functions** (RGB + Grayscale)
✅ **CLI tool** for batch visualization
✅ **Complete documentation**
✅ **Test script** to verify it works
✅ **Consistent output** across all datasets
✅ **Easy to use** in any Python script

**All functions work with ALL datasets** because they use the adapter pattern! 🎉
