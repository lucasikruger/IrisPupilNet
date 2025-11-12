#!/usr/bin/env python3
"""
Convert the IRIS + PUPIL + EYE dataset into the internal MOBIUS-style layout.

Dataset assumptions
-------------------
- Directory layout:
      <dataset_root>/
          train/
              image/*.png
              segmentation/*.png
          val/
              image/*.png
              segmentation/*.png
  (additional splits can be passed via --splits)
- Segmentation PNGs contain class ids:
      0 = background
      1 = pupil
      2 = iris
      3 = sclera / visible eye region
  When files are palette/RGBA encoded the script auto-selects the most
  informative channel.

Output
------
images/ : eye crops saved as JPG
Masks/  : MOBIUS RGB masks (Blue=pupil, Green=iris, Red=sclera)
CSV     : metadata with relative paths and source information

Usage example
-------------
    python iris_pupil_eye_to_mobius.py \
        --input_dir "/media/.../IrisPupilEye" \
        --output_dir "data/irispupileye_mobius" \
        --csv "dataset/irispupileye_output/irispupileye.csv"
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IRIS + PUPIL + EYE dataset to MOBIUS format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing the dataset splits (train/val/...)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/irispupileye_mobius",
        help="Directory where MOBIUS-style images and masks will be saved",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="dataset/irispupileye_output/irispupileye_dataset.csv",
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated list of dataset splits to process",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="Extension for exported images (default: .jpg)",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    if mask.ndim == 3:
        variances = [mask[..., c].std() for c in range(mask.shape[2])]
        channel = int(np.argmax(variances))
        mask = mask[..., channel]

    return mask.astype(np.uint8)


def convert_mask_to_mobius(mask_2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Convert single-channel class id mask to MOBIUS RGB encoding.
    Blue channel -> pupil, Green -> iris, Red -> sclera.
    """
    target_h, target_w = target_hw
    if mask_2d.shape[:2] != (target_h, target_w):
        mask_2d = cv2.resize(
            mask_2d,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mobius_mask = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    uniq = np.unique(mask_2d)

    if uniq.size and (uniq <= 5).all() and uniq.max() >= 2:
        pupil_mask = (mask_2d == 1).astype(np.uint8) * 255
        iris_mask = (mask_2d == 2).astype(np.uint8) * 255
        sclera_mask = np.isin(mask_2d, [3, 4, 5]).astype(np.uint8) * 255
    else:
        _, binary = cv2.threshold(
            mask_2d.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        pupil_mask = np.zeros_like(binary)
        iris_mask = np.zeros_like(binary)
        sclera_mask = binary

    mobius_mask[..., 0] = pupil_mask
    mobius_mask[..., 1] = iris_mask
    mobius_mask[..., 2] = sclera_mask
    return mobius_mask


def process_example(
    img_path: Path,
    mask_path: Path,
    split: str,
    img_output_dir: Path,
    mask_output_dir: Path,
    image_ext: str,
) -> Dict[str, Any]:
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")

    mask_raw = load_mask(mask_path)
    mobius_mask = convert_mask_to_mobius(mask_raw, image.shape[:2])

    base_id = f"{split}_{img_path.stem}"
    img_filename = f"{base_id}{image_ext}"
    mask_filename = f"{base_id}.png"

    ensure_dir(img_output_dir)
    ensure_dir(mask_output_dir)

    dataset_folder = img_output_dir.parent.name or img_output_dir.parent.as_posix()
    rel_image_path = (Path(dataset_folder) / 'images' / img_filename).as_posix()
    rel_mask_path = (Path(dataset_folder) / 'Masks' / mask_filename).as_posix()

    img_save_path = img_output_dir / img_filename
    mask_save_path = mask_output_dir / mask_filename

    cv2.imwrite(str(img_save_path), image)
    cv2.imwrite(str(mask_save_path), mobius_mask)

    metadata = {
        "id": base_id,
        "split": split,
        "original_filename": img_path.name,
        "rel_image_path": rel_image_path,
        "rel_mask_path": rel_mask_path,
        "dataset": "irispupileye",
        "dataset_format": "iris_pupil_eye_cls",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }
    return metadata


def gather_pairs(split_dir: Path) -> List[Tuple[Path, Path]]:
    img_dir = split_dir / "image"
    mask_dir = split_dir / "segmentation"

    if not img_dir.exists() or not mask_dir.exists():
        return []

    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / img_path.name
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv)

    img_output_dir = output_dir / "images"
    mask_output_dir = output_dir / "Masks"
    ensure_dir(img_output_dir)
    ensure_dir(mask_output_dir)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise ValueError("No dataset splits provided. Use --splits to list them.")

    all_metadata: List[Dict[str, Any]] = []

    for split in splits:
        split_dir = input_dir / split
        if not split_dir.exists():
            print(f"Warning: split '{split}' not found at {split_dir}. Skipping.")
            continue

        pairs = gather_pairs(split_dir)
        if not pairs:
            print(f"Warning: No image/mask pairs found for split '{split}'.")
            continue

        for img_path, mask_path in tqdm(
            pairs, desc=f"Processing {split}", unit="pair"
        ):
            metadata = process_example(
                img_path=img_path,
                mask_path=mask_path,
                split=split,
                img_output_dir=img_output_dir,
                mask_output_dir=mask_output_dir,
                image_ext=args.image_ext,
            )
            all_metadata.append(metadata)

    if not all_metadata:
        print("No samples processed. Nothing to save.")
        return

    df = pd.DataFrame(all_metadata)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"\n✓ Processed {len(df)} samples successfully")
    print(f"✓ Images saved to: {img_output_dir}")
    print(f"✓ Masks saved to: {mask_output_dir}")
    print(f"✓ CSV saved to: {csv_path}")
    print("\nDataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Splits covered: {df['split'].nunique()}")
    print(f"  Mean resolution: {df['width'].mean():.1f} x {df['height'].mean():.1f}")


if __name__ == "__main__":
    main()
