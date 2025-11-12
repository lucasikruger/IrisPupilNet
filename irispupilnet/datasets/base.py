from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Any

class SegmentationDataset(Dataset):
    """Abstract base for segmentation datasets."""
    def __init__(self, root: Path, split: str, img_size: int):
        self.root = Path(root)
        self.split = split
        self.img_size = int(img_size)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Return (image_tensor, mask_tensor_long)."""
        raise NotImplementedError
