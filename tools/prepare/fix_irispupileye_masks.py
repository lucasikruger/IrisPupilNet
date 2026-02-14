#!/usr/bin/env python3
"""
Fix IrisPupilEye masks to use correct MOBIUS color scheme.

Current (wrong) colors:
    Blue  -> Sclera (should be background)
    Red   -> Iris
    Green -> Pupil
    Black -> Background

MOBIUS expected colors:
    Red   -> Background (class 0)
    Green -> Iris (class 1)
    Blue  -> Pupil (class 2)

This script converts the masks to use MOBIUS colors.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def fix_mask(mask_path: Path, output_path: Path) -> None:
    """
    Convert mask from wrong colors to MOBIUS colors.

    Input mapping (wrong):
        Blue [0,0,255]  -> Sclera
        Red [255,0,0]   -> Iris
        Green [0,255,0] -> Pupil
        Black [0,0,0]   -> Background

    Output mapping (MOBIUS):
        Red [255,0,0]   -> Background (sclera + outside)
        Green [0,255,0] -> Iris
        Blue [0,0,255]  -> Pupil
    """
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask_bgr is None:
        raise ValueError(f"Failed to load mask: {mask_path}")

    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    # Identify current colors
    is_blue = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 255)
    is_red = (mask_rgb[..., 0] == 255) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 0)
    is_green = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 255) & (mask_rgb[..., 2] == 0)
    is_black = (mask_rgb[..., 0] == 0) & (mask_rgb[..., 1] == 0) & (mask_rgb[..., 2] == 0)

    # Create new mask with MOBIUS colors
    new_mask = np.zeros_like(mask_rgb)

    # Blue (sclera) + Black (outside) -> Red (background)
    new_mask[is_blue] = [255, 0, 0]
    new_mask[is_black] = [255, 0, 0]

    # Red (iris) -> Green (iris)
    new_mask[is_red] = [0, 255, 0]

    # Green (pupil) -> Blue (pupil)
    new_mask[is_green] = [0, 0, 255]

    # Convert back to BGR for saving
    new_mask_bgr = cv2.cvtColor(new_mask, cv2.COLOR_RGB2BGR)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), new_mask_bgr)


def main():
    parser = argparse.ArgumentParser(
        description="Fix IrisPupilEye masks to use MOBIUS color scheme"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/irispupileye_mobius/Masks",
        help="Input directory with wrong masks"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: overwrite input)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Find all mask files
    mask_files = list(input_dir.glob("*.png"))

    if not mask_files:
        print(f"No PNG files found in {input_dir}")
        return 1

    print(f"Found {len(mask_files)} mask files")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    if args.dry_run:
        print("\n[DRY RUN] Would convert:")
        for f in mask_files[:5]:
            print(f"  {f.name}")
        if len(mask_files) > 5:
            print(f"  ... and {len(mask_files) - 5} more")
        return 0

    print("\nConverting masks...")

    for mask_path in tqdm(mask_files, desc="Fixing masks"):
        output_path = output_dir / mask_path.name
        try:
            fix_mask(mask_path, output_path)
        except Exception as e:
            print(f"\nError processing {mask_path}: {e}")

    print(f"\n✓ Done! Fixed {len(mask_files)} masks")
    return 0


if __name__ == "__main__":
    exit(main())
