"""
Test script for simplified grayscale dataset with adapters.

This demonstrates how easy it is to:
1. Use different dataset formats transparently
2. Load data with adapters
3. Create PyTorch datasets
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from irispupilnet.datasets.adapters import (
    get_adapter,
    list_adapters,
    MobiusAdapter,
    TayedEyesAdapter,
)
from irispupilnet.datasets.simple_dataset import SimpleGrayscaleDataset


def test_adapters():
    """Test adapter pattern."""
    print("=" * 60)
    print("Testing Adapters")
    print("=" * 60)

    # List available adapters
    print(f"\nAvailable adapters: {list_adapters()}")

    # Get adapter by name
    print("\n1. Getting adapter by name:")
    mobius = get_adapter("mobius")
    print(f"   - Got adapter: {mobius.name}")

    tayed = get_adapter("tayed_3c")  # Using alias
    print(f"   - Got adapter: {tayed.name}")

    # Test direct instantiation
    print("\n2. Direct instantiation:")
    adapter = MobiusAdapter()
    print(f"   - Created: {adapter.name}")

    print("\n✓ Adapters work!")


def test_adapter_loading():
    """Test loading with a specific adapter."""
    print("\n" + "=" * 60)
    print("Testing Adapter Loading")
    print("=" * 60)

    # You need actual test files for this to work
    # This is just a demonstration
    print("\nTo test loading:")
    print("1. Create test image: test_img.png (grayscale)")
    print("2. Create test mask: test_mask.png (RGB color-coded)")
    print("3. Run:")
    print("""
    adapter = get_adapter("mobius")
    image, mask = adapter.load_sample(
        Path("test_img.png"),
        Path("test_mask.png")
    )
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask classes: {np.unique(mask)}")
    """)


def test_dataset():
    """Test dataset creation."""
    print("\n" + "=" * 60)
    print("Testing SimpleGrayscaleDataset")
    print("=" * 60)

    print("\nTo create a dataset:")
    print("""
    dataset = SimpleGrayscaleDataset(
        data_root="dataset",
        csv_path="dataset/merged/all_datasets.csv",
        split="train",
        img_size=160
    )

    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    image, mask = dataset[0]
    print(f"Image: {image.shape}, {image.dtype}")
    print(f"Mask: {mask.shape}, {mask.dtype}")

    # Get sample info
    info = dataset.get_sample_info(0)
    print(f"Sample info: {info}")
    """)

    print("\n✓ See usage in train.py for real example!")


def show_adding_new_format():
    """Show how to add a new dataset format."""
    print("\n" + "=" * 60)
    print("Adding a New Dataset Format")
    print("=" * 60)

    print("""
To add a new dataset format:

1. Create adapter class in irispupilnet/datasets/adapters.py:

   class MyDatasetAdapter(DatasetAdapter):
       @property
       def name(self) -> str:
           return "MyDataset"

       def load_image(self, image_path: Path) -> np.ndarray:
           # Load grayscale image
           img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
           return img

       def load_mask(self, mask_path: Path) -> np.ndarray:
           # Load and convert mask to class indices [0, 1, 2]
           mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
           # ... convert to class indices ...
           return mask.astype(np.int64)

2. Register in ADAPTER_REGISTRY:

   ADAPTER_REGISTRY = {
       "mobius": MobiusAdapter,
       "tayed": TayedEyesAdapter,
       "mydataset": MyDatasetAdapter,  # ← Add here
   }

3. Use in CSV:

   rel_image_path,rel_mask_path,split,dataset_format
   mydataset/img1.png,mydataset/mask1.png,train,mydataset

Done! The adapter will be automatically used.
    """)


def show_usage_examples():
    """Show usage examples."""
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)

    print("\n1. Using with DataLoader:")
    print("""
    from torch.utils.data import DataLoader

    train_ds = SimpleGrayscaleDataset(
        data_root="dataset",
        csv_path="dataset/merged/all_datasets.csv",
        split="train",
        img_size=160
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    for images, masks in train_loader:
        # images: (B, 1, 160, 160) float32
        # masks: (B, 160, 160) int64
        model(images)
    """)

    print("\n2. Using adapter directly:")
    print("""
    from irispupilnet.datasets.adapters import get_adapter

    adapter = get_adapter("mobius")
    image, mask = adapter.load_sample(img_path, mask_path)
    # image: (H, W) uint8 grayscale
    # mask: (H, W) int64 class indices
    """)

    print("\n3. Mixed datasets:")
    print("""
    CSV with multiple formats:

    rel_image_path,rel_mask_path,split,dataset_format
    mobius/img1.png,mobius/mask1.png,train,mobius
    tayed/img2.png,tayed/mask2.png,train,tayed
    irispupileye/img3.png,irispupileye/mask3.png,train,irispupileye

    All work transparently in the same dataset!
    """)


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Simplified Dataset Architecture Demo" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")

    try:
        test_adapters()
        test_adapter_loading()
        test_dataset()
        show_adding_new_format()
        show_usage_examples()

        print("\n" + "=" * 60)
        print("✓ All demonstrations completed!")
        print("=" * 60)
        print("\nFor full documentation, see:")
        print("  docs/SIMPLIFIED_DATASET_ARCHITECTURE.md")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
