"""
Visualize Dataset Adapters

This script demonstrates how different dataset adapters normalize
various dataset formats into a consistent representation.

Shows:
1. Original images (RGB or grayscale)
2. Normalized masks (class indices with consistent colors)
3. That all adapters produce the same output format

Usage:
    python examples/visualize_adapters.py --csv dataset/merged/all_datasets.csv --data-root dataset
    python examples/visualize_adapters.py --csv dataset/merged/all_datasets.csv --data-root dataset --grayscale
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from irispupilnet.datasets.adapters import get_adapter, list_adapters


# ============================================================================
# Color mapping for consistent visualization
# ============================================================================

# Standard colors for each class (for visualization)
CLASS_COLORS = {
    0: [0, 0, 0],        # Background: Black
    1: [0, 255, 0],      # Iris: Green
    2: [255, 0, 0],      # Pupil: Red
}

CLASS_NAMES = {
    0: "Background",
    1: "Iris",
    2: "Pupil",
}


# ============================================================================
# Loading functions
# ============================================================================

def load_image_rgb(image_path: Path) -> np.ndarray:
    """
    Load image in RGB format.

    Args:
        image_path: Path to image file

    Returns:
        RGB image (H, W, 3) as uint8
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_image_grayscale(image_path: Path) -> np.ndarray:
    """
    Load image in grayscale format.

    Args:
        image_path: Path to image file

    Returns:
        Grayscale image (H, W) as uint8
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return img


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert class index mask to RGB visualization.

    Args:
        mask: Class indices (H, W) with values [0, 1, 2]

    Returns:
        RGB image (H, W, 3) with consistent colors for each class
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color

    return rgb


# ============================================================================
# Visualization functions
# ============================================================================

def visualize_sample_rgb(
    image_path: Path,
    mask_path: Path,
    adapter_name: str,
    ax_img,
    ax_mask,
    ax_overlay
):
    """
    Visualize a sample with RGB image.

    Args:
        image_path: Path to image
        mask_path: Path to mask
        adapter_name: Name of adapter to use
        ax_img: Matplotlib axis for image
        ax_mask: Matplotlib axis for mask
        ax_overlay: Matplotlib axis for overlay
    """
    # Load with adapter
    adapter = get_adapter(adapter_name)
    img_gray = adapter.load_image(image_path)  # Adapter loads as grayscale
    mask = adapter.load_mask(mask_path)

    # Load original image in RGB for visualization
    img_rgb = load_image_rgb(image_path)

    # Convert mask to RGB for visualization
    mask_rgb = mask_to_rgb(mask)

    # Create overlay (image + semi-transparent mask)
    overlay = img_rgb.copy()
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)

    # Plot
    ax_img.imshow(img_rgb)
    ax_img.set_title(f"Original Image (RGB)\n{adapter.name}", fontsize=10, fontweight='bold')
    ax_img.axis('off')

    ax_mask.imshow(mask_rgb)
    ax_mask.set_title(f"Mask (Normalized)\nClasses: {np.unique(mask)}", fontsize=10, fontweight='bold')
    ax_mask.axis('off')

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Overlay", fontsize=10, fontweight='bold')
    ax_overlay.axis('off')


def visualize_sample_grayscale(
    image_path: Path,
    mask_path: Path,
    adapter_name: str,
    ax_img,
    ax_mask,
    ax_overlay
):
    """
    Visualize a sample with grayscale image.

    Args:
        image_path: Path to image
        mask_path: Path to mask
        adapter_name: Name of adapter to use
        ax_img: Matplotlib axis for image
        ax_mask: Matplotlib axis for mask
        ax_overlay: Matplotlib axis for overlay
    """
    # Load with adapter
    adapter = get_adapter(adapter_name)
    img_gray = adapter.load_image(image_path)
    mask = adapter.load_mask(mask_path)

    # Convert mask to RGB for visualization
    mask_rgb = mask_to_rgb(mask)

    # Create overlay (grayscale image + colored mask)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)

    # Plot
    ax_img.imshow(img_gray, cmap='gray')
    ax_img.set_title(f"Original Image (Grayscale)\n{adapter.name}", fontsize=10, fontweight='bold')
    ax_img.axis('off')

    ax_mask.imshow(mask_rgb)
    ax_mask.set_title(f"Mask (Normalized)\nClasses: {np.unique(mask)}", fontsize=10, fontweight='bold')
    ax_mask.axis('off')

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Overlay", fontsize=10, fontweight='bold')
    ax_overlay.axis('off')


# ============================================================================
# Main visualization function
# ============================================================================

def visualize_all_datasets(
    csv_path: Path,
    data_root: Path,
    grayscale: bool = False,
    samples_per_dataset: int = 1,
    output_path: Path = None
):
    """
    Visualize samples from all dataset types.

    Args:
        csv_path: Path to CSV file
        data_root: Base directory for dataset
        grayscale: If True, show grayscale images; if False, show RGB
        samples_per_dataset: Number of samples to show per dataset type
        output_path: Optional path to save figure
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    if 'dataset_format' not in df.columns:
        raise ValueError("CSV must have 'dataset_format' column")

    # Get unique dataset formats
    dataset_formats = df['dataset_format'].dropna().unique()
    print(f"\nFound {len(dataset_formats)} dataset formats: {list(dataset_formats)}")

    # Collect samples for each format
    samples_by_format = {}

    for fmt in dataset_formats:
        fmt_df = df[df['dataset_format'] == fmt].head(samples_per_dataset)
        samples = []

        for _, row in fmt_df.iterrows():
            img_path = (data_root / row['rel_image_path']).resolve()
            mask_path = (data_root / row['rel_mask_path']).resolve()

            if img_path.exists() and mask_path.exists():
                samples.append((img_path, mask_path, fmt))
                if len(samples) >= samples_per_dataset:
                    break

        if samples:
            samples_by_format[fmt] = samples

    if not samples_by_format:
        raise RuntimeError("No valid samples found!")

    print(f"\nLoaded samples from {len(samples_by_format)} formats")

    # Calculate grid size
    n_formats = len(samples_by_format)
    n_cols = 3  # Image, Mask, Overlay
    n_rows = n_formats

    # Create figure
    fig = plt.figure(figsize=(15, 5 * n_formats))
    fig.suptitle(
        f"Dataset Adapter Visualization ({'Grayscale' if grayscale else 'RGB'})\n"
        "All adapters normalize to consistent format: Class 0=Background, 1=Iris, 2=Pupil",
        fontsize=14,
        fontweight='bold'
    )

    # Plot each format
    row_idx = 0
    for fmt, samples in sorted(samples_by_format.items()):
        img_path, mask_path, adapter_name = samples[0]  # Take first sample

        # Create axes for this row
        ax_img = plt.subplot(n_rows, n_cols, row_idx * n_cols + 1)
        ax_mask = plt.subplot(n_rows, n_cols, row_idx * n_cols + 2)
        ax_overlay = plt.subplot(n_rows, n_cols, row_idx * n_cols + 3)

        try:
            if grayscale:
                visualize_sample_grayscale(
                    img_path, mask_path, adapter_name,
                    ax_img, ax_mask, ax_overlay
                )
            else:
                visualize_sample_rgb(
                    img_path, mask_path, adapter_name,
                    ax_img, ax_mask, ax_overlay
                )

            print(f"  ✓ Visualized: {fmt}")

        except Exception as e:
            print(f"  ✗ Failed to visualize {fmt}: {e}")
            ax_img.text(0.5, 0.5, f"Error loading\n{fmt}",
                       ha='center', va='center', transform=ax_img.transAxes)
            ax_img.axis('off')
            ax_mask.axis('off')
            ax_overlay.axis('off')

        row_idx += 1

    # Add legend
    legend_elements = [
        Patch(facecolor=np.array(CLASS_COLORS[0])/255, label=CLASS_NAMES[0]),
        Patch(facecolor=np.array(CLASS_COLORS[1])/255, label=CLASS_NAMES[1]),
        Patch(facecolor=np.array(CLASS_COLORS[2])/255, label=CLASS_NAMES[2]),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=12,
        frameon=True,
        fancybox=True
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved figure to: {output_path}")
    else:
        plt.show()

    return fig


# ============================================================================
# Demo with specific samples
# ============================================================================

def demo_adapter_normalization():
    """
    Demo showing how adapters normalize different formats.

    This creates a side-by-side comparison showing:
    1. Different datasets have different raw mask formats
    2. All adapters normalize to the same class indices [0, 1, 2]
    3. Final visualization is consistent across all datasets
    """
    print("\n" + "=" * 60)
    print("Dataset Adapter Normalization Demo")
    print("=" * 60)

    print("\nKey Concept:")
    print("  - Different datasets use different mask encodings")
    print("  - RGB color-coded: Red=bg, Green=iris, Blue=pupil")
    print("  - Grayscale indexed: Pixel value = class ID")
    print("  - Binary masks: Only one class")
    print()
    print("  → Adapters normalize ALL formats to:")
    print("    Class 0 = Background")
    print("    Class 1 = Iris")
    print("    Class 2 = Pupil")
    print()
    print("  → This allows:")
    print("    ✓ Consistent training across datasets")
    print("    ✓ Mixed datasets in same batch")
    print("    ✓ Easy visualization with same function")
    print("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize dataset adapters and their normalization"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with dataset paths"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Base directory for dataset"
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Show grayscale images instead of RGB"
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=1,
        help="Number of samples to visualize per dataset type"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path to save figure (if not provided, displays interactively)"
    )

    args = parser.parse_args()

    # Show demo
    demo_adapter_normalization()

    # Visualize
    print("\nGenerating visualization...")

    csv_path = Path(args.csv)
    data_root = Path(args.data_root)
    output_path = Path(args.output) if args.output else None

    if not csv_path.exists():
        print(f"✗ CSV not found: {csv_path}")
        return 1

    if not data_root.exists():
        print(f"✗ Data root not found: {data_root}")
        return 1

    try:
        visualize_all_datasets(
            csv_path=csv_path,
            data_root=data_root,
            grayscale=args.grayscale,
            samples_per_dataset=args.samples_per_dataset,
            output_path=output_path
        )

        print("\n✓ Visualization complete!")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
