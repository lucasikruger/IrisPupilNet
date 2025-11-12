from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import register_dataset
from ..utils.augment import get_transforms
from ..utils.mask_formats import get_mask_converter

@register_dataset("csv_seg")
class CSVIrisPupilSeg(Dataset):
    """
    CSV columns (required):
      - rel_image_path
      - rel_mask_path
      - split               ('train'|'val'|'test')
    Optional:
      - dataset_format      (e.g., 'mobius_3c', 'pascal_indexed', etc.)
    """

    REQUIRED = {"rel_image_path", "rel_mask_path", "split"}
    SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        dataset_base_dir: str,
        csv_path: str,
        split: str,
        img_size: int,
        default_format: str = "mobius_3c",
        extra_filters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.base = Path(dataset_base_dir)
        self.csv_path = Path(csv_path)
        self.split = split.lower().strip()
        self.default_format = default_format
        assert self.csv_path.exists(), f"CSV not found: {self.csv_path}"
        assert self.split in self.SPLITS, f"Invalid split={split}. Expected one of {self.SPLITS}"

        df = pd.read_csv(self.csv_path)
        missing = self.REQUIRED - set(df.columns)
        assert not missing, f"CSV missing required columns: {missing}"

        df = df[df["split"].str.lower() == self.split].copy()

        # Optional per-row filtering by extra columns
        if extra_filters:
            for k, v in extra_filters.items():
                if k in df.columns:
                    df = df[df[k] == v]

        # Build items: (img_abs, msk_abs, fmt)
        self.items: List[Tuple[Path, Path, str]] = []
        fmt_col_present = "dataset_format" in df.columns

        for _, row in df.iterrows():
            img_abs = (self.base / row["rel_image_path"]).resolve()
            msk_abs = (self.base / row["rel_mask_path"]).resolve()
            if not img_abs.exists() or not msk_abs.exists():
                continue
            fmt = str(row["dataset_format"]).strip() if fmt_col_present and pd.notna(row["dataset_format"]) else self.default_format
            self.items.append((img_abs, msk_abs, fmt))

        assert len(self.items) > 0, f"No samples found for split={self.split} in {self.csv_path}"

        self.img_size = int(img_size)
        self.tf = get_transforms(self.img_size, "train" if self.split == "train" else "val")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, msk_path, fmt = self.items[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        raw_mask_bgr = cv2.imread(str(msk_path), cv2.IMREAD_COLOR)  # keep as BGR for converters that expect it
        mask_converter = get_mask_converter(fmt)
        msk = mask_converter(raw_mask_bgr)  # -> int64 class mask

        aug = self.tf(image=img, mask=msk)
        x = aug["image"]       # float tensor [3,H,W]
        y = aug["mask"].long() # long tensor [H,W]
        return x, y
