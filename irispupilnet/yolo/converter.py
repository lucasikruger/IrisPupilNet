"""
MOBIUS to YOLO Format Converter

Converts MOBIUS semantic segmentation masks to YOLO polygon format.

Class Mapping:
    MOBIUS (semantic):          YOLO (instance):
    - 0 = Background (skip)     - 0 = Iris
    - 1 = Iris (green)    →     - 1 = Pupil
    - 2 = Pupil (blue)    →

Directory Structure:
    yolo_dataset/
    ├── data.yaml          # YOLO config
    ├── train.txt          # List of training image paths
    ├── val.txt            # List of validation image paths
    ├── images/
    │   ├── img_0001.jpg
    │   ├── img_0002.jpg
    │   └── ...
    └── labels/
        ├── img_0001.txt   # YOLO polygon labels
        ├── img_0002.txt
        └── ...
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import yaml
import hashlib
from tqdm import tqdm


def mask_to_yolo_labels(mask_rgb: np.ndarray) -> str:
    """
    Convert MOBIUS RGB mask to YOLO label format.

    MOBIUS format:
        Red (255,0,0) = Background (ignored)
        Green (0,255,0) = Iris → YOLO class 0
        Blue (0,0,255) = Pupil → YOLO class 1

    Args:
        mask_rgb: RGB mask (H, W, 3)

    Returns:
        YOLO label string (one polygon per line):
        "class_id x1 y1 x2 y2 ... xn yn" (normalized coords)
    """
    h, w = mask_rgb.shape[:2]
    lines = []

    # Process iris (YOLO class 0) and pupil (YOLO class 1)
    class_masks = [
        ((mask_rgb[..., 1] == 255) & (mask_rgb[..., 0] == 0) & (mask_rgb[..., 2] == 0), 0),  # Green → Iris
        ((mask_rgb[..., 2] == 255) & (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 0), 1),  # Blue → Pupil
    ]

    for binary_mask, class_id in class_masks:
        if not binary_mask.any():
            continue

        mask_uint8 = binary_mask.astype(np.uint8) * 255

        # Fill holes in mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue

            # Simplify contour
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            # Normalize points
            points = approx.reshape(-1, 2).astype(np.float32)
            points[:, 0] /= w
            points[:, 1] /= h

            # Format: class_id x1 y1 x2 y2 ...
            coords = ' '.join(f'{p[0]:.6f} {p[1]:.6f}' for p in points)
            lines.append(f'{class_id} {coords}')

    return '\n'.join(lines)


def _compute_cache_hash(csv_path: Path, data_root: Path, img_size: int) -> str:
    """Compute hash for cache validation."""
    df = pd.read_csv(csv_path)
    hash_input = f"{csv_path}:{data_root}:{img_size}:{len(df)}:{df['rel_image_path'].iloc[0]}:{df['rel_image_path'].iloc[-1]}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def _is_cache_valid(output_dir: Path, expected_hash: str) -> bool:
    """Check if cached dataset is valid."""
    cache_file = output_dir / '.cache_hash'
    if not cache_file.exists():
        return False

    stored_hash = cache_file.read_text().strip()
    return stored_hash == expected_hash


def _save_cache_hash(output_dir: Path, hash_value: str):
    """Save cache hash for future validation."""
    cache_file = output_dir / '.cache_hash'
    cache_file.write_text(hash_value)


def prepare_yolo_dataset(
    csv_path: str,
    data_root: str,
    output_dir: str = 'yolo_dataset',
    img_size: int = 160,
    grayscale: bool = True,
    force: bool = False,
) -> str:
    """
    Prepare YOLO dataset from MOBIUS CSV.

    Converts images and masks to YOLO format. Results are cached -
    subsequent calls skip conversion if data hasn't changed.

    Args:
        csv_path: Path to CSV with rel_image_path, rel_mask_path, split columns
        data_root: Base directory for images/masks
        output_dir: Output directory for YOLO dataset
        img_size: Target image size (square)
        grayscale: Convert images to grayscale (recommended for eye images)
        force: Force reconversion even if cache is valid

    Returns:
        Path to data.yaml file for YOLO training

    Example:
        >>> yaml_path = prepare_yolo_dataset(
        ...     csv_path='dataset/merged/all_datasets.csv',
        ...     data_root='data',
        ...     output_dir='yolo_dataset',
        ...     img_size=160
        ... )
        >>> model = YOLO('yolo11n-seg.pt')
        >>> model.train(data=yaml_path, epochs=50)
    """
    csv_path = Path(csv_path)
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    # Check cache
    cache_hash = _compute_cache_hash(csv_path, data_root, img_size)
    yaml_path = output_dir / 'data.yaml'

    if not force and _is_cache_valid(output_dir, cache_hash):
        print(f"✓ Using cached YOLO dataset: {output_dir}")
        return str(yaml_path)

    print(f"Converting MOBIUS dataset to YOLO format...")
    print(f"  CSV: {csv_path}")
    print(f"  Data root: {data_root}")
    print(f"  Output: {output_dir}")
    print(f"  Image size: {img_size}x{img_size}")

    # Create directories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    df = df[df['split'].isin(['train', 'val'])].reset_index(drop=True)

    train_paths = []
    val_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        split = row['split']
        img_src = data_root / row['rel_image_path']
        mask_src = data_root / row['rel_mask_path']

        if not img_src.exists() or not mask_src.exists():
            continue

        # Generate unique filename
        img_name = f"img_{idx:06d}.jpg"
        label_name = f"img_{idx:06d}.txt"

        img_dst = images_dir / img_name
        label_dst = labels_dir / label_name

        # Process image
        img = cv2.imread(str(img_src))
        if img is None:
            continue

        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(str(img_dst), img)

        # Process mask
        mask = cv2.imread(str(mask_src))
        if mask is None:
            continue

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.resize(mask_rgb, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        # Convert to YOLO labels
        labels = mask_to_yolo_labels(mask_rgb)
        label_dst.write_text(labels)

        # Add to split list (absolute path)
        abs_img_path = str(img_dst.absolute())
        if split == 'train':
            train_paths.append(abs_img_path)
        else:
            val_paths.append(abs_img_path)

    # Write train.txt and val.txt
    train_txt = output_dir / 'train.txt'
    val_txt = output_dir / 'val.txt'

    train_txt.write_text('\n'.join(train_paths))
    val_txt.write_text('\n'.join(val_paths))

    # Write data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': str(train_txt.absolute()),
        'val': str(val_txt.absolute()),
        'names': {
            0: 'iris',
            1: 'pupil',
        },
        'nc': 2,
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Save cache hash
    _save_cache_hash(output_dir, cache_hash)

    print(f"\n✓ YOLO dataset ready: {output_dir}")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Config: {yaml_path}")

    return str(yaml_path)
