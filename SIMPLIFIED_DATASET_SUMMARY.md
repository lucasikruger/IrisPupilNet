# Simplified Dataset Architecture - Quick Summary

## What Was Created

I've simplified your dataset loading using the **Adapter Pattern** for clean, transparent, and polymorphic dataset handling. The new architecture is **grayscale-only** as you requested.

---

## Files Created

### 1. `irispupilnet/datasets/adapters.py` ✨
**The heart of the new system.**

- **DatasetAdapter (ABC)** - Base class defining the interface
- **Concrete Adapters:**
  - `MobiusAdapter` - For MOBIUS dataset
  - `TayedEyesAdapter` - For TayedEyes dataset
  - `IrisPupilEyeAdapter` - For IrisPupilEye dataset
  - `UnityEyesAdapter` - For Unity Eyes dataset

- **Registry:** `get_adapter(name)` function to get adapters by name

**Key interface:**
```python
class DatasetAdapter(ABC):
    def load_image(self, path) -> np.ndarray:
        """Load grayscale image (H, W) uint8"""

    def load_mask(self, path) -> np.ndarray:
        """Load mask as class indices (H, W) int64 [0,1,2]"""

    def load_sample(self, img_path, mask_path):
        """Load both image and mask"""
```

### 2. `irispupilnet/datasets/simple_dataset.py` ✨
**Simplified dataset class using adapters.**

- **SimpleGrayscaleDataset** - Main dataset class
- Always grayscale (no RGB complexity)
- Uses adapters transparently
- Registered as `"simple_grayscale"`

**Usage:**
```python
dataset = SimpleGrayscaleDataset(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)
```

### 3. `docs/SIMPLIFIED_DATASET_ARCHITECTURE.md` 📖
**Complete documentation** with:
- Architecture diagram
- Usage examples
- How to add new formats
- Migration guide
- FAQ

### 4. `examples/test_simple_dataset.py` 🧪
**Demo script** showing:
- How to use adapters
- How to create datasets
- How to add new formats
- Usage examples

---

## Key Benefits

### ✅ **Simplicity**
- Only grayscale (no RGB complexity)
- Clear separation: adapter loads, dataset orchestrates
- Easy to understand

### ✅ **Transparency**
Each adapter is self-contained and shows exactly how it loads data:
```python
class MobiusAdapter(DatasetAdapter):
    def load_image(self, path):
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def load_mask(self, path):
        # RGB color decoding logic here
        # Red → 0, Green → 1, Blue → 2
        ...
```

### ✅ **Polymorphism**
All adapters implement the same interface, so you can mix datasets transparently:
```csv
rel_image_path,rel_mask_path,split,dataset_format
mobius/img1.png,mobius/mask1.png,train,mobius
tayed/img2.png,tayed/mask2.png,train,tayed
irispupileye/img3.png,irispupileye/mask3.png,train,irispupileye
```

### ✅ **Easy to Extend**
Add a new dataset format in 3 steps:
1. Create adapter class
2. Register in `ADAPTER_REGISTRY`
3. Use in CSV

---

## Quick Start

### Using in Training

**Option 1: With registry (recommended for train.py)**
```python
from irispupilnet.datasets import DATASET_REGISTRY

# Get dataset class
DatasetClass = DATASET_REGISTRY["simple_grayscale"]

# Create dataset
train_ds = DatasetClass(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)
```

**Option 2: Direct import**
```python
from irispupilnet.datasets.simple_dataset import SimpleGrayscaleDataset

train_ds = SimpleGrayscaleDataset(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)
```

### Using Adapters Directly

```python
from irispupilnet.datasets.adapters import get_adapter

# Get adapter
adapter = get_adapter("mobius")

# Load sample
image, mask = adapter.load_sample(img_path, mask_path)
# image: (H, W) uint8 grayscale
# mask: (H, W) int64 class indices [0, 1, 2]
```

---

## CSV Format

Your CSV needs these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `rel_image_path` | Image path relative to `data_root` | `mobius/train/img_001.png` |
| `rel_mask_path` | Mask path relative to `data_root` | `mobius/train/mask_001.png` |
| `split` | Dataset split | `train`, `val`, `test` |
| `dataset_format` | Adapter name | `mobius`, `tayed`, `irispupileye` |

**Example:**
```csv
rel_image_path,rel_mask_path,split,dataset_format
mobius/train/img_001.png,mobius/train/mask_001.png,train,mobius
tayed/train/eye_020.png,tayed/train/eye_020_mask.png,train,tayed
irispupileye/val/image_50.jpg,irispupileye/val/mask_50.png,val,irispupileye
```

---

## Adding a New Dataset Format

Let's say you want to add support for "OpenEDS" dataset:

### Step 1: Create Adapter

```python
# In irispupilnet/datasets/adapters.py

class OpenEDSAdapter(DatasetAdapter):
    @property
    def name(self) -> str:
        return "OpenEDS"

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image as grayscale."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load: {image_path}")
        return img

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Load mask and convert to class indices.

        OpenEDS uses different encoding - implement conversion here.
        Must return (H, W) int64 with values [0, 1, 2].
        """
        # Example: If OpenEDS uses grayscale where pixel value = class
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load: {mask_path}")

        # Ensure it's int64
        return mask.astype(np.int64)
```

### Step 2: Register

```python
# Add to ADAPTER_REGISTRY
ADAPTER_REGISTRY = {
    "mobius": MobiusAdapter,
    "tayed": TayedEyesAdapter,
    "irispupileye": IrisPupilEyeAdapter,
    "unity_eyes": UnityEyesAdapter,
    "openeds": OpenEDSAdapter,  # ← Add here
}
```

### Step 3: Use in CSV

```csv
rel_image_path,rel_mask_path,split,dataset_format
openeds/img001.png,openeds/mask001.png,train,openeds
```

**Done!** The adapter is automatically used for OpenEDS samples.

---

## Comparison: Old vs New

### Old Way (csv_seg.py)

```python
dataset = DATASET_REGISTRY["csv_seg"](
    dataset_base_dir="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160,
    default_format="mobius_3c",      # ← Format as parameter
    convert_to_grayscale=True        # ← Manual flag
)
```

**Issues:**
- Mixed RGB/grayscale handling
- Mask format separate from loading logic
- Less clear where conversion happens

### New Way (simple_dataset.py)

```python
dataset = DATASET_REGISTRY["simple_grayscale"](
    data_root="dataset",             # ← Clearer name
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
    # Format is in CSV, grayscale is automatic!
)
```

**Benefits:**
- Always grayscale (simpler)
- Format per sample (flexible)
- Adapter handles everything (transparent)

---

## Architecture Diagram

```
SimpleGrayscaleDataset
         │
         │ uses
         ▼
   DatasetAdapter (ABC)
         │
         │ implements
         ▼
    ┌────┴────┬────────┬───────────┐
    │         │        │           │
MobiusAdapter │  IrisPupilEye  UnityEyes
         TayedEyes  Adapter     Adapter
         Adapter
```

**Flow:**
1. Dataset reads CSV
2. For each sample, gets adapter from `dataset_format` column
3. Adapter loads image (grayscale) and mask (class indices)
4. Dataset applies augmentations
5. Returns (image, mask) tensors

---

## Testing

Run the demo:
```bash
python examples/test_simple_dataset.py
```

Test with your data:
```python
from irispupilnet.datasets.simple_dataset import SimpleGrayscaleDataset

# Create dataset
ds = SimpleGrayscaleDataset(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)

print(f"Dataset size: {len(ds)}")

# Get first sample
image, mask = ds[0]
print(f"Image: {image.shape} {image.dtype}")  # (1, 160, 160) float32
print(f"Mask: {mask.shape} {mask.dtype}")     # (160, 160) int64

# Get sample info
info = ds.get_sample_info(0)
print(info)
```

---

## Next Steps

### 1. Update Your CSV (if needed)

Make sure your CSV has `dataset_format` column:
```python
import pandas as pd

df = pd.read_csv("dataset/merged/all_datasets.csv")

# Check if column exists
if "dataset_format" not in df.columns:
    # Add it (adjust format name as needed)
    df["dataset_format"] = "mobius"
    df.to_csv("dataset/merged/all_datasets.csv", index=False)
    print("✓ Added dataset_format column")
else:
    print("✓ dataset_format column already exists")
```

### 2. Update train.py (optional)

You can keep using `csv_seg` or switch to `simple_grayscale`:

```python
# In train.py, change dataset name:
# OLD:
# train_ds = DATASET_REGISTRY["csv_seg"](...)

# NEW:
train_ds = DATASET_REGISTRY["simple_grayscale"](
    data_root=args.data_root,
    csv_path=args.csv,
    split="train",
    img_size=args.img_size
)
```

### 3. Remove RGB Complexity (optional)

If you want to fully simplify:
1. Remove `--color` flag from train.py
2. Remove RGB handling from models
3. Always set `in_channels=1`

---

## Documentation

- **Full guide:** `docs/SIMPLIFIED_DATASET_ARCHITECTURE.md`
- **Demo script:** `examples/test_simple_dataset.py`
- **Code:** `irispupilnet/datasets/adapters.py` and `simple_dataset.py`

---

## Summary

You now have:

✅ **Clean adapter pattern** for dataset loading
✅ **Grayscale-only** (simpler, faster)
✅ **Transparent** (each adapter shows exactly what it does)
✅ **Polymorphic** (mix datasets seamlessly)
✅ **Easy to extend** (add new formats in 3 steps)
✅ **Well documented** (guides + examples)

**The old `csv_seg` still works** if you need backward compatibility, but new code should use `simple_grayscale` with adapters!
