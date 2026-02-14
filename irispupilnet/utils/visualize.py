"""
Visualization utilities for IrisPupilNet

Simple functions to visualize images and masks using adapters.
All datasets are normalized to the same format, so the same
visualization function works for all.
"""

from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Import adapters
try:
    from ..datasets.adapters import get_adapter, DatasetAdapter
except ImportError:
    # For standalone usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from irispupilnet.datasets.adapters import get_adapter, DatasetAdapter


# ============================================================================
# Constants
# ============================================================================

# Standard colors for visualization
CLASS_COLORS_RGB = {
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
# Core visualization functions
# ============================================================================

def visualize_sample_rgb(
    image_path: Path,
    mask_path: Path,
    adapter_name: str,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize a sample with RGB image.

    Shows original RGB image, normalized mask, and overlay side-by-side.

    Args:
        image_path: Path to image file
        mask_path: Path to mask file
        adapter_name: Name of adapter (e.g., 'mobius', 'tayed')
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> visualize_sample_rgb(
        ...     Path("dataset/mobius/img001.png"),
        ...     Path("dataset/mobius/mask001.png"),
        ...     "mobius"
        ... )
    """
    # Load with adapter
    adapter = get_adapter(adapter_name)
    img_gray = adapter.load_image(image_path)  # Loads as grayscale
    mask = adapter.load_mask(mask_path)        # Normalized class indices

    # Load original image in RGB for visualization
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert mask to RGB visualization
    mask_rgb = _mask_to_rgb(mask)

    # Create overlay
    overlay = img_rgb.copy()
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot image
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original Image (RGB)\n{adapter.name}", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Plot mask
    axes[1].imshow(mask_rgb)
    axes[1].set_title(f"Mask (Normalized)\nClasses: {sorted(np.unique(mask))}", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Plot overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Add legend
    _add_legend(fig)

    # Add info text
    fig.suptitle(
        f"Dataset: {adapter.name} | Format: {adapter_name} | All datasets normalized to [0=bg, 1=iris, 2=pupil]",
        fontsize=10,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    else:
        plt.show()

    return fig


def visualize_sample_grayscale(
    image_path: Path,
    mask_path: Path,
    adapter_name: str,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize a sample with grayscale image.

    Shows original grayscale image, normalized mask, and overlay side-by-side.

    Args:
        image_path: Path to image file
        mask_path: Path to mask file
        adapter_name: Name of adapter (e.g., 'mobius', 'tayed')
        figsize: Figure size (width, height)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> visualize_sample_grayscale(
        ...     Path("dataset/mobius/img001.png"),
        ...     Path("dataset/mobius/mask001.png"),
        ...     "mobius"
        ... )
    """
    # Load with adapter
    adapter = get_adapter(adapter_name)
    img_gray = adapter.load_image(image_path)  # Loads as grayscale
    mask = adapter.load_mask(mask_path)        # Normalized class indices

    # Convert mask to RGB visualization
    mask_rgb = _mask_to_rgb(mask)

    # Create overlay (grayscale converted to RGB + colored mask)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot image
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title(f"Original Image (Grayscale)\n{adapter.name}", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Plot mask
    axes[1].imshow(mask_rgb)
    axes[1].set_title(f"Mask (Normalized)\nClasses: {sorted(np.unique(mask))}", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Plot overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Add legend
    _add_legend(fig)

    # Add info text
    fig.suptitle(
        f"Dataset: {adapter.name} | Format: {adapter_name} | All datasets normalized to [0=bg, 1=iris, 2=pupil]",
        fontsize=10,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    else:
        plt.show()

    return fig


# ============================================================================
# Helper functions
# ============================================================================

def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert class index mask to RGB visualization.

    Args:
        mask: Class indices (H, W) with values [0, 1, 2]

    Returns:
        RGB image (H, W, 3) with consistent colors
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS_RGB.items():
        rgb[mask == class_id] = color

    return rgb


def _add_legend(fig: plt.Figure):
    """Add class legend to figure."""
    legend_elements = [
        Patch(facecolor=np.array(CLASS_COLORS_RGB[i])/255, label=CLASS_NAMES[i])
        for i in sorted(CLASS_NAMES.keys())
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=3,
        fontsize=11,
        frameon=True,
        fancybox=True
    )


# ============================================================================
# Batch visualization
# ============================================================================

def visualize_multiple_samples(
    samples: list,
    grayscale: bool = True,
    figsize: Tuple[int, int] = (15, None),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize multiple samples in a grid.

    Args:
        samples: List of tuples (image_path, mask_path, adapter_name)
        grayscale: If True, show grayscale; if False, show RGB
        figsize: Figure size (width, height auto-calculated if None)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure

    Example:
        >>> samples = [
        ...     (Path("mobius/img1.png"), Path("mobius/mask1.png"), "mobius"),
        ...     (Path("tayed/img2.png"), Path("tayed/mask2.png"), "tayed"),
        ... ]
        >>> visualize_multiple_samples(samples, grayscale=True)
    """
    n_samples = len(samples)
    n_cols = 3  # Image, Mask, Overlay

    # Calculate figure size
    if figsize[1] is None:
        figsize = (figsize[0], 5 * n_samples)

    # Create figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Multiple Dataset Samples ({'Grayscale' if grayscale else 'RGB'})\n"
        "All adapters normalize to consistent format",
        fontsize=14,
        fontweight='bold'
    )

    # Plot each sample
    for idx, (img_path, mask_path, adapter_name) in enumerate(samples):
        # Load with adapter
        adapter = get_adapter(adapter_name)
        img_gray = adapter.load_image(img_path)
        mask = adapter.load_mask(mask_path)

        # Convert mask to RGB
        mask_rgb = _mask_to_rgb(mask)

        # Create axes
        ax_img = plt.subplot(n_samples, n_cols, idx * n_cols + 1)
        ax_mask = plt.subplot(n_samples, n_cols, idx * n_cols + 2)
        ax_overlay = plt.subplot(n_samples, n_cols, idx * n_cols + 3)

        if grayscale:
            # Grayscale visualization
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            overlay = img_rgb.copy()
            alpha = 0.5
            overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)

            ax_img.imshow(img_gray, cmap='gray')
            ax_img.set_title(f"Image (Grayscale)\n{adapter.name}", fontsize=10)

        else:
            # RGB visualization
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            overlay = img_rgb.copy()
            alpha = 0.5
            overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)

            ax_img.imshow(img_rgb)
            ax_img.set_title(f"Image (RGB)\n{adapter.name}", fontsize=10)

        ax_img.axis('off')

        ax_mask.imshow(mask_rgb)
        ax_mask.set_title(f"Mask\nClasses: {sorted(np.unique(mask))}", fontsize=10)
        ax_mask.axis('off')

        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Overlay", fontsize=10)
        ax_overlay.axis('off')

    # Add legend
    _add_legend(fig)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    else:
        plt.show()

    return fig


# ============================================================================
# Quick test function
# ============================================================================

def quick_test():
    """Quick test of visualization functions."""
    print("=" * 60)
    print("Visualization Module Test")
    print("=" * 60)

    print("\nAvailable functions:")
    print("  1. visualize_sample_rgb(img_path, mask_path, adapter_name)")
    print("  2. visualize_sample_grayscale(img_path, mask_path, adapter_name)")
    print("  3. visualize_multiple_samples([(img, mask, adapter), ...])")

    print("\nExample usage:")
    print("""
    from pathlib import Path
    from irispupilnet.utils.visualize import (
        visualize_sample_rgb,
        visualize_sample_grayscale
    )

    # RGB visualization
    visualize_sample_rgb(
        Path("dataset/mobius/img001.png"),
        Path("dataset/mobius/mask001.png"),
        "mobius"
    )

    # Grayscale visualization (recommended for training)
    visualize_sample_grayscale(
        Path("dataset/mobius/img001.png"),
        Path("dataset/mobius/mask001.png"),
        "mobius",
        save_path=Path("visualization.png")
    )

    # Multiple samples
    samples = [
        (Path("mobius/img.png"), Path("mobius/mask.png"), "mobius"),
        (Path("tayed/img.png"), Path("tayed/mask.png"), "tayed"),
    ]
    visualize_multiple_samples(samples, grayscale=True)
    """)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    quick_test()
