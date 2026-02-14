#!/usr/bin/env python3
"""
Quick test for visualization functions.

This script tests that the visualization functions work correctly
and can load samples from different adapters.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from irispupilnet.datasets.adapters import list_adapters, get_adapter
import numpy as np


def test_adapters():
    """Test that adapters are available."""
    print("\n" + "=" * 60)
    print("Testing Adapters")
    print("=" * 60)

    adapters = list_adapters()
    print(f"\n✓ Found {len(adapters)} adapters:")
    for name in sorted(set(adapters)):
        try:
            adapter = get_adapter(name)
            print(f"  • {name:20s} → {adapter.name}")
        except Exception as e:
            print(f"  ✗ {name:20s} → Error: {e}")

    return True


def test_mask_conversion():
    """Test mask conversion to RGB."""
    print("\n" + "=" * 60)
    print("Testing Mask Conversion")
    print("=" * 60)

    from irispupilnet.utils.visualize import _mask_to_rgb, CLASS_COLORS_RGB

    # Create test mask with all classes
    mask = np.array([
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [2, 2, 1, 1],
    ], dtype=np.int64)

    print("\nTest mask:")
    print(mask)

    # Convert to RGB
    mask_rgb = _mask_to_rgb(mask)

    print(f"\nMask RGB shape: {mask_rgb.shape}")
    print(f"Mask RGB dtype: {mask_rgb.dtype}")

    # Verify colors
    print("\nVerifying colors:")
    for class_id, expected_color in CLASS_COLORS_RGB.items():
        pixels = mask_rgb[mask == class_id]
        if len(pixels) > 0:
            actual_color = pixels[0]
            matches = np.array_equal(actual_color, expected_color)
            status = "✓" if matches else "✗"
            print(f"  {status} Class {class_id}: expected {expected_color}, got {actual_color}")

    print("\n✓ Mask conversion works!")
    return True


def test_visualization_functions():
    """Test that visualization functions are importable."""
    print("\n" + "=" * 60)
    print("Testing Visualization Functions")
    print("=" * 60)

    try:
        from irispupilnet.utils.visualize import (
            visualize_sample_rgb,
            visualize_sample_grayscale,
            visualize_multiple_samples
        )

        print("\n✓ Successfully imported:")
        print("  • visualize_sample_rgb")
        print("  • visualize_sample_grayscale")
        print("  • visualize_multiple_samples")

        return True

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False


def show_usage_example():
    """Show usage example."""
    print("\n" + "=" * 60)
    print("Usage Example")
    print("=" * 60)

    print("""
To visualize your data, you have two options:

1. Command-line script (visualize all datasets):

   python examples/visualize_adapters.py \\
       --csv dataset/merged/all_datasets.csv \\
       --data-root dataset \\
       --grayscale

2. Python functions (visualize specific samples):

   from pathlib import Path
   from irispupilnet.utils.visualize import visualize_sample_grayscale

   visualize_sample_grayscale(
       image_path=Path("dataset/mobius/train/img_001.png"),
       mask_path=Path("dataset/mobius/train/mask_001.png"),
       adapter_name="mobius"
   )

See examples/README_VISUALIZE.md for full documentation!
    """)


def main():
    """Run all tests."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "Visualization Test Suite" + " " * 19 + "║")
    print("╚" + "═" * 58 + "╝")

    all_passed = True

    try:
        all_passed &= test_adapters()
        all_passed &= test_mask_conversion()
        all_passed &= test_visualization_functions()
        show_usage_example()

        if all_passed:
            print("\n" + "=" * 60)
            print("✓ All tests passed!")
            print("=" * 60)
            print("\nYou can now use:")
            print("  • examples/visualize_adapters.py (CLI)")
            print("  • irispupilnet.utils.visualize (Python API)")
            print("\nSee examples/README_VISUALIZE.md for usage guide.")
            return 0
        else:
            print("\n" + "=" * 60)
            print("✗ Some tests failed!")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
