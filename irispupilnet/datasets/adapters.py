"""
Dataset Adapters for IrisPupilNet

Simplified adapter pattern for loading images from different datasets.
All adapters work with GRAYSCALE images only for simplicity.

Architecture:
    DatasetAdapter (abstract base class)
        ├── MobiusAdapter
        ├── TayedEyesAdapter
        ├── IrisPupilEyeAdapter
        └── UnityEyesAdapter

Usage:
    adapter = MobiusAdapter()
    image, mask = adapter.load_sample(image_path, mask_path)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.

    Each adapter knows how to:
    1. Load an image from a specific dataset (as grayscale)
    2. Load and convert a mask to class indices [0, 1, 2]
    """

    @abstractmethod
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image as grayscale.

        Args:
            image_path: Path to image file

        Returns:
            Grayscale image (H, W) as uint8
        """
        pass

    @abstractmethod
    def load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Load mask and convert to class indices.

        Args:
            mask_path: Path to mask file

        Returns:
            Class indices (H, W) as int64 with values [0, 1, 2]
            - 0: Background
            - 1: Iris
            - 2: Pupil
        """
        pass

    def load_sample(self, image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load both image and mask.

        Args:
            image_path: Path to image
            mask_path: Path to mask

        Returns:
            Tuple of (image, mask)
            - image: (H, W) grayscale uint8
            - mask: (H, W) int64 class indices
        """
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        return image, mask

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this adapter."""
        pass


class MobiusAdapter(DatasetAdapter):
    """
    Adapter for MOBIUS dataset.

    Image format: Grayscale or RGB (converted to grayscale)
    Mask format: RGB color-coded PNG
        - Red (255, 0, 0): Background → 0
        - Green (0, 255, 0): Iris → 1
        - Blue (0, 0, 255): Pupil → 2
    """

    @property
    def name(self) -> str:
        return "MOBIUS"

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image as grayscale."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Load RGB mask and convert to class indices.

        MOBIUS format:
        - Red: Background (0)
        - Green: Iris (1)
        - Blue: Pupil (2)
        """
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # Convert BGR to RGB
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        # Identify pixels by color
        red = (mask_rgb[..., 0] == 255) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 0)
        green = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 255) & (mask_rgb[..., 2] == 0)
        blue = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 255)

        # Create class mask
        out = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        out[green] = 1  # Iris
        out[blue] = 2   # Pupil
        # Red pixels stay 0 (background)

        return out


class TayedEyesAdapter(DatasetAdapter):
    """
    Adapter for TayedEyes dataset.

    Image format: Grayscale or RGB synthetic images
    Mask format: Same as MOBIUS (RGB color-coded)
    """

    @property
    def name(self) -> str:
        return "TayedEyes"

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image as grayscale."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask (same format as MOBIUS)."""
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        red = (mask_rgb[..., 0] == 255) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 0)
        green = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 255) & (mask_rgb[..., 2] == 0)
        blue = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 255)

        out = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        out[green] = 1
        out[blue] = 2

        return out


class IrisPupilEyeAdapter(DatasetAdapter):
    """
    Adapter for IrisPupilEye dataset.

    Image format: Grayscale or RGB
    Mask format: Same as MOBIUS (RGB color-coded)
    """

    @property
    def name(self) -> str:
        return "IrisPupilEye"

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image as grayscale."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask (same format as MOBIUS)."""
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        red = (mask_rgb[..., 0] == 255) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 0)
        green = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 255) & (mask_rgb[..., 2] == 0)
        blue = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 255)

        out = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        out[green] = 1
        out[blue] = 2

        return out


class UnityEyesAdapter(DatasetAdapter):
    """
    Adapter for Unity Eyes synthetic dataset.

    Image format: Grayscale synthetic images
    Mask format: Same as MOBIUS (RGB color-coded)
    """

    @property
    def name(self) -> str:
        return "UnityEyes"

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load image as grayscale."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img

    def load_mask(self, mask_path: Path) -> np.ndarray:
        """Load mask (same format as MOBIUS)."""
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        red = (mask_rgb[..., 0] == 255) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 0)
        green = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 255) & (mask_rgb[..., 2] == 0)
        blue = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 255)

        out = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        out[green] = 1
        out[blue] = 2

        return out


# ============================================================================
# Adapter Registry
# ============================================================================

ADAPTER_REGISTRY = {
    "mobius": MobiusAdapter,
    "mobius_3c": MobiusAdapter,  # Alias
    "tayed": TayedEyesAdapter,
    "tayed_3c": TayedEyesAdapter,  # Alias
    "irispupileye": IrisPupilEyeAdapter,
    "iris_pupil_eye_cls": IrisPupilEyeAdapter,  # Alias
    "unity_eyes": UnityEyesAdapter,
    "unity_eyes_3c": UnityEyesAdapter,  # Alias
}


def get_adapter(adapter_name: str) -> DatasetAdapter:
    """
    Get dataset adapter by name.

    Args:
        adapter_name: Name of the adapter (case-insensitive)

    Returns:
        DatasetAdapter instance

    Raises:
        ValueError: If adapter not found

    Examples:
        >>> adapter = get_adapter("mobius")
        >>> image, mask = adapter.load_sample(img_path, mask_path)
    """
    key = adapter_name.lower().strip()

    if key not in ADAPTER_REGISTRY:
        available = sorted(set(ADAPTER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown adapter '{adapter_name}'. "
            f"Available adapters: {available}"
        )

    adapter_class = ADAPTER_REGISTRY[key]
    return adapter_class()


def list_adapters() -> list[str]:
    """Get list of available adapter names."""
    return sorted(set(ADAPTER_REGISTRY.keys()))
