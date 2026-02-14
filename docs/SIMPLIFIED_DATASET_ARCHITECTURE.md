# Simplified Dataset Architecture (Grayscale-Only)

## Overview

The simplified dataset architecture uses the **Adapter Pattern** for transparent, polymorphic loading of different dataset formats. This makes the codebase easier to understand, maintain, and extend.

**Key principle:** Work only with grayscale images (no RGB complexity).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimpleGrayscaleDataset                       │
│                                                                 │
│  Responsibilities:                                              │
│  - Load CSV and filter by split                                 │
│  - Validate paths                                               │
│  - Apply augmentations                                          │
│  - Return (image, mask) tensors                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DatasetAdapter (ABC)                       │
│                                                                 │
│  Interface:                                                     │
│  - load_image(path) -> grayscale (H, W) uint8                  │
│  - load_mask(path) -> class indices (H, W) int64                │
│  - load_sample(img_path, mask_path) -> (image, mask)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ implements
                              ▼
        ┌─────────────────────┬─────────────────────┬──────────────┐
        │                     │                     │              │
        ▼                     ▼                     ▼              ▼
┌───────────────┐  ┌───────────────┐  ┌──────────────┐  ┌─────────────┐
│MobiusAdapter  │  │TayedEyesAdapter│  │IrisPupilEye │  │UnityEyes    │
│               │  │                │  │Adapter       │  │Adapter      │
│- load_image() │  │- load_image()  │  │- load_image()│  │- load_image │
│- load_mask()  │  │- load_mask()   │  │- load_mask() │  │- load_mask()│
└───────────────┘  └────────────────┘  └──────────────┘  └─────────────┘
```

---

## Key Components

### 1. DatasetAdapter (Base Class)

**Location:** `irispupilnet/datasets/adapters.py`

Abstract base class defining the interface for loading data:

```python
class DatasetAdapter(ABC):
    @abstractmethod
    def load_image(self, image_path: Path) -> np.ndarray:
        """Load grayscale image (H, W) uint8."""
        pass

    @abstractmethod
    def load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask as class indices (H, W) int64."""
        pass

    def load_sample(self, image_path, mask_path):
        """Load both image and mask."""
        return self.load_image(image_path), self.load_mask(mask_path)
```

### 2. Concrete Adapters

Each dataset format has its own adapter:

| Adapter | Dataset | Image Format | Mask Format |
|---------|---------|--------------|-------------|
| `MobiusAdapter` | MOBIUS | Grayscale | RGB color-coded |
| `TayedEyesAdapter` | TayedEyes | Grayscale | RGB color-coded |
| `IrisPupilEyeAdapter` | IrisPupilEye | Grayscale | RGB color-coded |
| `UnityEyesAdapter` | Unity Eyes | Grayscale | RGB color-coded |

**All masks use same color coding:**
- Red (255, 0, 0): Background → Class 0
- Green (0, 255, 0): Iris → Class 1
- Blue (0, 0, 255): Pupil → Class 2

### 3. SimpleGrayscaleDataset

**Location:** `irispupilnet/datasets/simple_dataset.py`

Main dataset class that:
1. Reads CSV with paths and dataset formats
2. Uses appropriate adapter for each sample
3. Applies augmentations
4. Returns PyTorch tensors

**Registered as:** `"simple_grayscale"`

---

## Usage Examples

### Basic Usage

```python
from irispupilnet.datasets.simple_dataset import SimpleGrayscaleDataset

# Create dataset
train_ds = SimpleGrayscaleDataset(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)

# Use with DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate
for images, masks in train_loader:
    # images: (B, 1, 160, 160) float32
    # masks: (B, 160, 160) int64
    pass
```

### Using with Training Script

```python
from irispupilnet.datasets import DATASET_REGISTRY

# Get dataset class
dataset_class = DATASET_REGISTRY["simple_grayscale"]

# Create train/val datasets
train_ds = dataset_class(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)

val_ds = dataset_class(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="val",
    img_size=160
)
```

### Using Adapters Directly

```python
from pathlib import Path
from irispupilnet.datasets.adapters import get_adapter

# Get adapter
adapter = get_adapter("mobius")

# Load sample
image, mask = adapter.load_sample(
    Path("dataset/mobius/image_001.png"),
    Path("dataset/mobius/mask_001.png")
)

# image: (H, W) uint8 grayscale
# mask: (H, W) int64 class indices [0, 1, 2]
```

---

## CSV Format

The CSV must have these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `rel_image_path` | str | Image path relative to `data_root` | `mobius/train/img_001.png` |
| `rel_mask_path` | str | Mask path relative to `data_root` | `mobius/train/mask_001.png` |
| `split` | str | Dataset split | `train`, `val`, `test` |
| `dataset_format` | str | Adapter name | `mobius`, `tayed`, `irispupileye` |

**Example CSV:**

```csv
rel_image_path,rel_mask_path,split,dataset_format
mobius/train/img_001.png,mobius/train/mask_001.png,train,mobius
tayed/train/eye_020.png,tayed/train/eye_020_mask.png,train,tayed
irispupileye/val/image_50.jpg,irispupileye/val/mask_50.png,val,irispupileye
```

---

## Adding a New Dataset Format

To add support for a new dataset, create a new adapter:

### Step 1: Create Adapter Class

```python
# In irispupilnet/datasets/adapters.py

class MyNewDatasetAdapter(DatasetAdapter):
    @property
    def name(self) -> str:
        return "MyNewDataset"

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image as grayscale."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load: {image_path}")
        return img

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Load mask and convert to class indices.

        Implement your custom mask conversion logic here.
        Must return (H, W) int64 array with values [0, 1, 2].
        """
        # Example: Load grayscale mask where pixel values are class IDs
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load: {mask_path}")

        # Convert to int64
        return mask.astype(np.int64)
```

### Step 2: Register in ADAPTER_REGISTRY

```python
# Add to ADAPTER_REGISTRY dict
ADAPTER_REGISTRY = {
    "mobius": MobiusAdapter,
    "tayed": TayedEyesAdapter,
    "irispupileye": IrisPupilEyeAdapter,
    "unity_eyes": UnityEyesAdapter,
    "mynewdataset": MyNewDatasetAdapter,  # ← Add here
}
```

### Step 3: Use in CSV

```csv
rel_image_path,rel_mask_path,split,dataset_format
mynewdataset/train/img001.png,mynewdataset/train/mask001.png,train,mynewdataset
```

That's it! The adapter will be automatically used for samples with `dataset_format=mynewdataset`.

---

## Benefits of This Architecture

### 1. **Simplicity**
- Only grayscale (no RGB complexity)
- Clear separation of concerns
- Easy to understand

### 2. **Transparency**
- Each adapter is self-contained
- Easy to see what each dataset format does
- No hidden magic

### 3. **Polymorphism**
- All adapters implement same interface
- Dataset class doesn't need to know details
- Different formats work seamlessly together

### 4. **Extensibility**
- Add new dataset formats easily
- No need to modify existing code
- Just create new adapter + register

### 5. **Maintainability**
- Easy to debug (one adapter at a time)
- Changes to one format don't affect others
- Clear code structure

### 6. **Testability**
- Each adapter can be tested independently
- Mock adapters for unit testing
- Easy to verify behavior

---

## Comparison with Old Architecture

| Aspect | Old (csv_seg.py) | New (simple_dataset.py) |
|--------|-----------------|------------------------|
| **Image Loading** | RGB then optional grayscale | Always grayscale |
| **Mask Conversion** | Separate registry + converter functions | Integrated in adapter |
| **Adding Format** | Create converter, register, import | Create adapter, register |
| **Complexity** | Medium (RGB handling, separate converters) | Low (single responsibility) |
| **Code Lines** | ~90 lines | ~150 lines (but clearer) |
| **Dependencies** | Tight coupling with mask_formats.py | Self-contained adapters |
| **Testability** | Need to mock converters | Test adapters directly |

---

## Migration Guide

### For Training Scripts

**Old way:**
```python
from irispupilnet.datasets import DATASET_REGISTRY

dataset = DATASET_REGISTRY["csv_seg"](
    dataset_base_dir="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160,
    default_format="mobius_3c",
    convert_to_grayscale=True  # ← No longer needed!
)
```

**New way:**
```python
from irispupilnet.datasets import DATASET_REGISTRY

dataset = DATASET_REGISTRY["simple_grayscale"](
    data_root="dataset",  # ← Renamed from dataset_base_dir
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
    # No default_format, no convert_to_grayscale!
    # Format is in CSV, grayscale is always on
)
```

### For CSV Files

**Required change:** Add `dataset_format` column if not present

```bash
# Check if column exists
head -1 dataset/merged/all_datasets.csv

# If missing, add it (example with pandas)
python -c "
import pandas as pd
df = pd.read_csv('dataset/merged/all_datasets.csv')
df['dataset_format'] = 'mobius'  # or appropriate format
df.to_csv('dataset/merged/all_datasets.csv', index=False)
"
```

---

## Testing

### Test an Adapter

```python
from pathlib import Path
from irispupilnet.datasets.adapters import MobiusAdapter

adapter = MobiusAdapter()

# Test image loading
img = adapter.load_image(Path("test_image.png"))
assert img.ndim == 2  # Grayscale (H, W)
assert img.dtype == np.uint8

# Test mask loading
mask = adapter.load_mask(Path("test_mask.png"))
assert mask.ndim == 2  # (H, W)
assert mask.dtype == np.int64
assert set(np.unique(mask)).issubset({0, 1, 2})  # Only valid classes

print(f"✓ {adapter.name} adapter works!")
```

### Test Dataset

```python
from irispupilnet.datasets.simple_dataset import SimpleGrayscaleDataset

dataset = SimpleGrayscaleDataset(
    data_root="dataset",
    csv_path="dataset/merged/all_datasets.csv",
    split="train",
    img_size=160
)

# Test sample loading
image, mask = dataset[0]

assert image.shape == (1, 160, 160)  # (C, H, W) grayscale
assert mask.shape == (160, 160)      # (H, W)
assert image.dtype == torch.float32
assert mask.dtype == torch.int64

print(f"✓ Dataset works! {len(dataset)} samples")
```

---

## Performance Considerations

### Grayscale Loading
- **3x faster** than RGB (only 1 channel to load/process)
- **3x less memory** during training
- Simpler augmentation pipeline

### Caching
If you need to speed up loading further, consider caching:

```python
from functools import lru_cache

class CachedAdapter(MobiusAdapter):
    @lru_cache(maxsize=1000)
    def load_image(self, image_path: Path) -> np.ndarray:
        return super().load_image(image_path)
```

---

## FAQ

**Q: Why not support RGB?**
A: Simplicity. Grayscale is 3x faster, uses 3x less memory, and is sufficient for iris/pupil segmentation. RGB adds complexity without clear benefit for this task.

**Q: Can I still use the old csv_seg dataset?**
A: Yes, it's not removed. But new code should use `simple_grayscale`.

**Q: Do I need to change my CSV?**
A: Yes, if you don't have a `dataset_format` column. Old CSVs used `default_format` parameter; new dataset requires format per row.

**Q: How do I handle mixed datasets?**
A: That's the beauty of adapters! Put all datasets in one CSV with different `dataset_format` values:

```csv
rel_image_path,rel_mask_path,split,dataset_format
mobius/img1.png,mobius/mask1.png,train,mobius
tayed/img2.png,tayed/mask2.png,train,tayed
irispupileye/img3.png,irispupileye/mask3.png,train,irispupileye
```

**Q: What if my mask format is different?**
A: Create a custom adapter! See "Adding a New Dataset Format" section above.

---

## Summary

The simplified architecture provides:

✅ **Grayscale-only** (no RGB complexity)
✅ **Adapter pattern** (transparent, polymorphic loading)
✅ **Easy to extend** (add new adapters easily)
✅ **Self-contained** (each adapter independent)
✅ **Type-safe** (clear interfaces with type hints)
✅ **Testable** (test adapters independently)

**Result:** Cleaner, simpler, more maintainable dataset loading!
