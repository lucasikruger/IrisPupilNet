"""
Simplified Dataset for IrisPupilNet

Grayscale-only dataset using adapter pattern for transparent loading
from different dataset formats.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import register_dataset
from .adapters import get_adapter, DatasetAdapter
from ..utils.augment import get_transforms


@register_dataset("simple_grayscale")
class SimpleGrayscaleDataset(Dataset):
    """
    Simplified grayscale-only dataset using adapter pattern.

    CSV columns (required):
      - rel_image_path: Path relative to data_root
      - rel_mask_path: Path relative to data_root
      - split: 'train', 'val', or 'test'
      - dataset_format: Adapter name (e.g., 'mobius', 'tayed', 'irispupileye')

    Features:
      - Grayscale only (simplified)
      - Transparent loading via adapters
      - Easy to add new dataset formats
      - Clean separation of concerns

    Example:
        dataset = SimpleGrayscaleDataset(
            data_root="dataset",
            csv_path="dataset/merged/all_datasets.csv",
            split="train",
            img_size=160
        )
    """

    REQUIRED_COLUMNS = {"rel_image_path", "rel_mask_path", "split", "dataset_format"}
    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        split: str,
        img_size: int,
        extra_filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_root: Base directory for dataset paths
            csv_path: Path to CSV file with image/mask paths
            split: One of 'train', 'val', 'test'
            img_size: Target image size (square)
            extra_filters: Optional dict for filtering rows by column values
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.csv_path = Path(csv_path)
        self.split = split.lower().strip()
        self.img_size = int(img_size)

        # Validate inputs
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        if self.split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of: {self.VALID_SPLITS}"
            )

        # Load and validate CSV
        df = pd.read_csv(self.csv_path)
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )

        # Filter by split
        df = df[df["split"].str.lower() == self.split].copy()

        # Apply extra filters
        if extra_filters:
            for column, value in extra_filters.items():
                if column in df.columns:
                    df = df[df[column] == value]

        # Build sample list: (image_path, mask_path, adapter)
        self.samples: List[Tuple[Path, Path, DatasetAdapter]] = []

        for _, row in df.iterrows():
            # Resolve paths
            img_path = (self.data_root / row["rel_image_path"]).resolve()
            mask_path = (self.data_root / row["rel_mask_path"]).resolve()

            # Check existence
            if not img_path.exists():
                print(f"Warning: Image not found, skipping: {img_path}")
                continue
            if not mask_path.exists():
                print(f"Warning: Mask not found, skipping: {mask_path}")
                continue

            # Get adapter
            adapter_name = str(row["dataset_format"]).strip()
            try:
                adapter = get_adapter(adapter_name)
            except ValueError as e:
                print(f"Warning: {e}, skipping sample")
                continue

            self.samples.append((img_path, mask_path, adapter))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found for split='{self.split}' in {self.csv_path}"
            )

        # Setup transforms (grayscale: 1 channel)
        self.transforms = get_transforms(
            self.img_size,
            mode="train" if self.split == "train" else "val",
            num_channels=1  # Always grayscale
        )

        print(f"Loaded {len(self.samples)} samples for split='{self.split}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            Tuple of (image, mask)
            - image: Tensor (1, H, W) float32, normalized
            - mask: Tensor (H, W) int64, class indices [0, 1, 2]
        """
        img_path, mask_path, adapter = self.samples[idx]

        # Load using adapter
        image = adapter.load_image(img_path)  # (H, W) uint8 grayscale
        mask = adapter.load_mask(mask_path)   # (H, W) int64 class indices

        # Add channel dimension for augmentation: (H, W) -> (H, W, 1)
        image = image[..., None]

        # Apply transforms
        augmented = self.transforms(image=image, mask=mask)
        x = augmented["image"]    # Tensor (1, H, W) float32
        y = augmented["mask"]     # Tensor (H, W) int64

        return x, y.long()

    def get_sample_info(self, idx: int) -> dict:
        """
        Get metadata about a sample.

        Args:
            idx: Sample index

        Returns:
            Dict with sample information
        """
        img_path, mask_path, adapter = self.samples[idx]
        return {
            "index": idx,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "adapter": adapter.name,
            "split": self.split,
        }


# ============================================================================
# Convenience function
# ============================================================================

def create_grayscale_dataset(
    data_root: str,
    csv_path: str,
    split: str,
    img_size: int,
    **kwargs
) -> SimpleGrayscaleDataset:
    """
    Convenience function to create a grayscale dataset.

    Args:
        data_root: Base directory for dataset
        csv_path: Path to CSV file
        split: 'train', 'val', or 'test'
        img_size: Target image size
        **kwargs: Additional arguments for dataset

    Returns:
        SimpleGrayscaleDataset instance

    Example:
        >>> train_ds = create_grayscale_dataset(
        ...     data_root="dataset",
        ...     csv_path="dataset/merged/all_datasets.csv",
        ...     split="train",
        ...     img_size=160
        ... )
    """
    return SimpleGrayscaleDataset(
        data_root=data_root,
        csv_path=csv_path,
        split=split,
        img_size=img_size,
        **kwargs
    )
