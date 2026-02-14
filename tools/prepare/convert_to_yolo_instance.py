#!/usr/bin/env python3
"""
Convert IrisPupilNet semantic segmentation dataset to YOLO instance segmentation format.

This script converts semantic segmentation masks (pixel-wise class labels) to
YOLO instance segmentation format (bounding boxes + polygon masks per instance).

For each eye image:
- Semantic: (H, W) with values [0=background, 1=iris, 2=pupil]
- Instance: 2 instances (iris + pupil) with bboxes and polygon masks

Output YOLO format:
- images/ directory with copied images
- labels/ directory with .txt files (one per image)
- Each line: <class_id> <polygon_points_normalized>

Usage:
    python tools/prepare/convert_to_yolo_instance.py \
        --csv dataset/merged/all_datasets.csv \
        --data-root dataset \
        --output yolo_dataset \
        --img-size 160
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from irispupilnet.utils.mask_formats import get_mask_converter


def mask_to_polygon(binary_mask, simplify_epsilon=2.0):
    """
    Convert binary mask to polygon points.

    Args:
        binary_mask: (H, W) binary mask
        simplify_epsilon: Epsilon for polygon simplification (Douglas-Peucker)

    Returns:
        List of (x, y) points, or None if no contour found
    """
    # Find contours
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Get largest contour
    contour = max(contours, key=cv2.contourArea)

    # Simplify polygon
    epsilon = simplify_epsilon
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Convert to list of points
    points = approx.reshape(-1, 2)

    return points


def get_bbox_from_mask(binary_mask):
    """
    Get bounding box from binary mask.

    Args:
        binary_mask: (H, W) binary mask

    Returns:
        (x, y, w, h) bounding box, or None if mask is empty
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def convert_semantic_to_yolo_instance(mask, class_id_map=None):
    """
    Convert semantic segmentation mask to YOLO instance format.

    Args:
        mask: (H, W) semantic mask with class indices [0, 1, 2]
        class_id_map: Dict mapping semantic class to YOLO class id
                      Default: {1: 0, 2: 1} (iris=0, pupil=1)

    Returns:
        List of instances, each: {
            'class_id': int,
            'bbox': (x, y, w, h),
            'polygon': [(x1, y1), (x2, y2), ...]
        }
    """
    if class_id_map is None:
        class_id_map = {1: 0, 2: 1}  # iris=0, pupil=1 in YOLO

    instances = []
    height, width = mask.shape

    for semantic_class, yolo_class in class_id_map.items():
        # Extract binary mask for this class
        binary_mask = (mask == semantic_class).astype(np.uint8)

        if binary_mask.sum() == 0:
            continue  # Skip if no pixels

        # Get bounding box
        bbox = get_bbox_from_mask(binary_mask)
        if bbox is None:
            continue

        # Get polygon
        polygon = mask_to_polygon(binary_mask)
        if polygon is None or len(polygon) < 3:
            continue  # Need at least 3 points for polygon

        # Normalize polygon points to [0, 1]
        polygon_normalized = polygon.astype(float)
        polygon_normalized[:, 0] /= width
        polygon_normalized[:, 1] /= height

        instances.append({
            'class_id': yolo_class,
            'bbox': bbox,
            'polygon': polygon_normalized
        })

    return instances


def write_yolo_label(instances, output_path):
    """
    Write YOLO instance segmentation label file.

    Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    All coordinates normalized to [0, 1]

    Args:
        instances: List of instance dicts
        output_path: Path to output .txt file
    """
    with open(output_path, 'w') as f:
        for inst in instances:
            class_id = inst['class_id']
            polygon = inst['polygon']

            # Format: class_id x1 y1 x2 y2 ...
            coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in polygon)
            f.write(f'{class_id} {coords}\n')


def main():
    parser = argparse.ArgumentParser(description='Convert semantic segmentation to YOLO instance format')
    parser.add_argument('--csv', required=True, help='Input CSV file with dataset')
    parser.add_argument('--data-root', required=True, help='Root directory for data')
    parser.add_argument('--output', required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--img-size', type=int, default=160, help='Resize images to this size')
    parser.add_argument('--default-format', default='mobius_3c', help='Default mask format')
    parser.add_argument('--copy-images', action='store_true', help='Copy images (default: symlink)')

    args = parser.parse_args()

    # Read CSV
    print(f"Reading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"  Found {len(df)} samples")

    # Create output directories
    output_dir = Path(args.output)
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Process each sample
    data_root = Path(args.data_root)
    stats = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        split = row['split']
        if split not in ['train', 'val', 'test']:
            stats['skipped'] += 1
            continue

        # Load image
        img_path = data_root / row['rel_image_path']
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            stats['skipped'] += 1
            continue

        # Load mask
        mask_path = data_root / row['rel_mask_path']
        if not mask_path.exists():
            print(f"Warning: Mask not found: {mask_path}")
            stats['skipped'] += 1
            continue

        # Read and convert mask
        mask_bgr = cv2.imread(str(mask_path))
        if mask_bgr is None:
            print(f"Warning: Could not read mask: {mask_path}")
            stats['skipped'] += 1
            continue

        # Get mask format
        mask_format = row.get('dataset_format', args.default_format)
        try:
            converter = get_mask_converter(mask_format)
        except KeyError as e:
            print(f"Warning: {e}")
            stats['skipped'] += 1
            continue

        # Convert to semantic mask
        semantic_mask = converter(mask_bgr)  # (H, W) with [0, 1, 2]

        # Resize if needed
        if args.img_size:
            semantic_mask = cv2.resize(
                semantic_mask,
                (args.img_size, args.img_size),
                interpolation=cv2.INTER_NEAREST
            )

        # Convert to YOLO instance format
        instances = convert_semantic_to_yolo_instance(semantic_mask)

        if not instances:
            print(f"Warning: No instances found in {img_path}")
            stats['skipped'] += 1
            continue

        # Generate output filename (use stem to avoid conflicts)
        output_name = f"{img_path.stem}_{idx}"

        # Copy/resize image
        img = cv2.imread(str(img_path))
        if args.img_size:
            img = cv2.resize(img, (args.img_size, args.img_size))

        img_out_path = output_dir / 'images' / split / f'{output_name}.jpg'
        cv2.imwrite(str(img_out_path), img)

        # Write label file
        label_out_path = output_dir / 'labels' / split / f'{output_name}.txt'
        write_yolo_label(instances, label_out_path)

        stats[split] += 1

    # Generate dataset YAML
    yaml_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'iris',
            1: 'pupil'
        },
        'nc': 2  # Number of classes
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"\nStatistics:")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  Test:  {stats['test']} images")
    print(f"  Skipped: {stats['skipped']} images")
    print(f"\nYou can now train with:")
    print(f"  python tools/yolo/train_yolo_native.py --data {yaml_path} --epochs 50")


if __name__ == '__main__':
    main()
