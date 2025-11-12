#!/usr/bin/env python3
"""
Convert Unity Eyes synthetic dataset to MOBIUS format.

Unity Eyes provides face images with JSON annotations containing:
- interior_margin_2d: Eye contour points
- iris_2d: Iris boundary points (32 points)
- Eye details: pupil_size, iris_size, etc.

This script:
1. Crops images to eye region using interior_margin bounding box
2. Creates RGB masks in MOBIUS format (red=bg, green=iris, blue=pupil)
3. Clips iris/pupil circles if eye is partially closed
4. Outputs to images/ and Masks/ directories
5. Generates CSV with metadata

Usage:
    python unity_eyes_to_mobius.py \
        --input_dir "/path/to/unity_eyes_dataset_1k" \
        --output_dir "data/unity_eyes_mobius" \
        --csv "dataset/unity_eyes_output/unity_eyes_dataset.csv" \
        --padding 20
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
from tqdm import tqdm


def parse_point_string(point_str: str, img_height: int = None) -> Tuple[float, float]:
    """
    Parse a point string like "(326.25, 271.44, 9.42)" to (x, y).
    Ignores the z-coordinate.

    Unity uses Y-up coordinate system, but OpenCV uses Y-down.
    If img_height is provided, Y coordinate will be flipped.
    """
    point_str = point_str.strip().strip('()').strip('"')
    parts = [float(x.strip()) for x in point_str.split(',')]
    x, y = parts[0], parts[1]

    # Flip Y coordinate if image height is provided (Unity Y-up -> OpenCV Y-down)
    if img_height is not None:
        y = img_height - y

    return x, y


def parse_points_list(points_list: List[str], img_height: int = None) -> np.ndarray:
    """
    Parse a list of point strings to numpy array of shape [N, 2].

    Args:
        points_list: List of point strings
        img_height: Image height for Y-coordinate flipping (Unity Y-up -> OpenCV Y-down)
    """
    points = [parse_point_string(p, img_height) for p in points_list]
    return np.array(points, dtype=np.float32)


def fit_circle(points: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """
    Fit a circle to a set of 2D points using least squares.

    Returns:
        center: (cx, cy)
        radius: r
    """
    x = points[:, 0]
    y = points[:, 1]

    # Set up least squares problem: (x - cx)^2 + (y - cy)^2 = r^2
    # Expand: x^2 + y^2 - 2*cx*x - 2*cy*y + cx^2 + cy^2 - r^2 = 0
    # Linear system: A @ [cx, cy, c] = b where c = cx^2 + cy^2 - r^2

    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2

    # Solve: 2*cx*x + 2*cy*y - c = x^2 + y^2
    params, _, _, _ = np.linalg.lstsq(2 * A, b, rcond=None)

    cx, cy, c = params
    radius = np.sqrt(c + cx**2 + cy**2)

    return (cx, cy), radius


def create_eye_mask(
    img_shape: Tuple[int, int],
    interior_margin: np.ndarray,
    iris_points: np.ndarray,
    pupil_center: Tuple[float, float],
    pupil_radius: float,
    iris_center: Tuple[float, float],
    iris_radius: float
) -> np.ndarray:
    """
    Create MOBIUS-format mask with 4 regions:
    - Black (0,0,0): Background/exterior (outside eye region)
    - Red (0,0,255) in BGR: Sclera (white of eye)
    - Green (0,255,0) in BGR: Iris (colored ring)
    - Blue (255,0,0) in BGR: Pupil (black center)

    Clips iris and pupil circles to the eye region (interior_margin polygon).

    Args:
        img_shape: (height, width) of the image
        interior_margin: [N, 2] eye contour points
        iris_points: [M, 2] iris boundary points
        pupil_center: (cx, cy) pupil center
        pupil_radius: pupil radius
        iris_center: (cx, cy) iris center
        iris_radius: iris radius

    Returns:
        mask: [H, W, 3] uint8 BGR image
    """
    h, w = img_shape
    # Start with all black (background/exterior)
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Eye region polygon (sclera area)
    eye_poly = interior_margin.astype(np.int32)
    eye_region_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(eye_region_mask, [eye_poly], 255)

    # Draw full iris circle
    iris_full_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(iris_full_mask, (int(iris_center[0]), int(iris_center[1])),
               int(iris_radius), 255, -1)

    # Clip iris to eye region
    iris_full_mask = cv2.bitwise_and(iris_full_mask, eye_region_mask)

    # Draw pupil circle
    pupil_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(pupil_mask, (int(pupil_center[0]), int(pupil_center[1])),
               int(pupil_radius), 255, -1)

    # Clip pupil to iris and eye region
    pupil_mask = cv2.bitwise_and(pupil_mask, iris_full_mask)

    # Iris is the ring: full iris circle MINUS pupil circle
    iris_ring_mask = cv2.bitwise_and(iris_full_mask, cv2.bitwise_not(pupil_mask))

    # Sclera is eye region MINUS iris full circle
    sclera_mask = cv2.bitwise_and(eye_region_mask, cv2.bitwise_not(iris_full_mask))

    # Assign to color channels (BGR format):
    # Blue channel (index 0) = Pupil
    # Green channel (index 1) = Iris ring
    # Red channel (index 2) = Sclera
    # Black (0,0,0) = Background (already set)
    mask[:, :, 0] = pupil_mask  # Blue = Pupil
    mask[:, :, 1] = iris_ring_mask  # Green = Iris
    mask[:, :, 2] = sclera_mask  # Red = Sclera

    return mask


def process_unity_eyes_image(
    json_path: Path,
    img_path: Path,
    output_img_path: Path,
    output_mask_path: Path,
    padding: int = 20
) -> Dict[str, Any]:
    """
    Process a single Unity Eyes image and create cropped eye image + mask.

    Args:
        json_path: Path to JSON annotation
        img_path: Path to original image
        output_img_path: Where to save cropped eye image
        output_mask_path: Where to save mask
        padding: Pixels to add around eye bounding box

    Returns:
        metadata: Dict with processing info
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    h, w = img.shape[:2]

    # Parse annotations (flip Y coordinates: Unity Y-up -> OpenCV Y-down)
    interior_margin = parse_points_list(data['interior_margin_2d'], img_height=h)
    iris_points = parse_points_list(data['iris_2d'], img_height=h)

    # Filter iris points to only those visible within eye region
    # (Unity provides full 3D iris projection, some points may be behind eyelids)
    eye_poly = interior_margin.astype(np.int32)
    visible_iris_points = []
    for pt in iris_points:
        result = cv2.pointPolygonTest(eye_poly, (float(pt[0]), float(pt[1])), False)
        if result >= 0:  # Point is inside or on the boundary
            visible_iris_points.append(pt)

    visible_iris_points = np.array(visible_iris_points, dtype=np.float32)

    # Estimate pupil from iris size and pupil_size ratio
    pupil_size = float(data['eye_details']['pupil_size'])
    iris_size = float(data['eye_details']['iris_size'])

    # Estimate iris center and radius from iris boundary points
    # Unity provides iris boundary points in 2D (projected from 3D)
    # Use mean center and mean distance as radius (better than least-squares circle fitting)
    iris_center = iris_points.mean(axis=0)

    # Calculate mean distance from center to all iris boundary points
    distances = np.sqrt(np.sum((iris_points - iris_center)**2, axis=1))
    iris_radius = distances.mean()

    # Calculate pupil radius from pupil_size
    # pupil_size appears to be a normalized value (0-1 range, can be negative)
    # Scale it proportionally to iris radius for realistic pupil sizes
    if pupil_size > 0:
        # Scale pupil_size by iris_radius with a multiplier for realistic sizes
        # Typical pupil is 30-50% of iris radius in normal lighting
        pupil_radius = pupil_size * iris_radius * 10.0
        # Cap at maximum 50% of iris radius (large pupil in dim light)
        pupil_radius = min(pupil_radius, iris_radius * 0.5)
    else:
        # Negative or zero pupil_size: use minimum pupil (bright light/constricted)
        pupil_radius = max(1.0, iris_radius * 0.15)

    # Ensure valid radius
    pupil_radius = max(1.0, pupil_radius)
    iris_radius = max(1.0, iris_radius)

    # Pupil center is at iris center
    # In Unity Eyes, look_vec indicates the gaze direction of the whole eye,
    # not a displacement of the pupil within the iris
    pupil_center = iris_center

    # Create mask FIRST in original image space (full size)
    mask_full = create_eye_mask(
        (h, w),
        interior_margin,
        iris_points,
        pupil_center,
        pupil_radius,
        iris_center,
        iris_radius
    )

    # Calculate bounding box from interior_margin with padding
    x_min, y_min = interior_margin.min(axis=0)
    x_max, y_max = interior_margin.max(axis=0)

    # Add padding to ensure eye is not at the edge
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(w, int(x_max) + padding)
    y_max = min(h, int(y_max) + padding)

    # Crop BOTH image and mask with same bounding box
    crop_img = img[y_min:y_max, x_min:x_max].copy()
    mask = mask_full[y_min:y_max, x_min:x_max].copy()

    # Save outputs
    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_img_path), crop_img)
    cv2.imwrite(str(output_mask_path), mask)

    # Metadata (coordinates in cropped space)
    iris_center_crop = (iris_center[0] - x_min, iris_center[1] - y_min)
    pupil_center_crop = (pupil_center[0] - x_min, pupil_center[1] - y_min)

    metadata = {
        'original_size': (w, h),
        'crop_bbox': (x_min, y_min, x_max, y_max),
        'crop_size': crop_img.shape[:2],
        'iris_center': iris_center_crop,
        'iris_radius': iris_radius,
        'pupil_center': pupil_center_crop,
        'pupil_radius': pupil_radius,
        'pupil_size': pupil_size,
        'iris_size': iris_size,
        'iris_texture': data['eye_details']['iris_texture'],
        'lighting_skybox': data['lighting_details']['skybox_texture'],
        'skin_texture': data['eye_region_details']['primary_skin_texture'],
        'head_pose': data['head_pose']
    }

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert Unity Eyes dataset to MOBIUS format"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to Unity Eyes dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/unity_eyes_mobius',
        help='Output directory for MOBIUS-format data'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='dataset/unity_eyes_output/unity_eyes_dataset.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--padding',
        type=int,
        default=20,
        help='Padding around eye bounding box (pixels)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only first N images (for testing)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv)

    # Create output directories
    img_output_dir = output_dir / 'images'
    mask_output_dir = output_dir / 'Masks'
    img_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(input_dir.glob('*.json'))

    if args.limit:
        json_files = json_files[:args.limit]

    print(f"Found {len(json_files)} images to process")

    # Process each image
    csv_rows = []

    for json_path in tqdm(json_files, desc="Processing Unity Eyes"):
        img_id = json_path.stem
        img_path = json_path.with_suffix('.jpg')

        if not img_path.exists():
            print(f"Warning: Image not found for {json_path}")
            continue

        # Output paths
        rel_img_path = f'images/{img_id}.jpg'
        rel_mask_path = f'Masks/{img_id}.png'

        output_img_path = output_dir / rel_img_path
        output_mask_path = output_dir / rel_mask_path

        try:
            metadata = process_unity_eyes_image(
                json_path,
                img_path,
                output_img_path,
                output_mask_path,
                padding=args.padding
            )

            # Create CSV row
            row = {
                'id': img_id,
                'rel_image_path': rel_img_path,
                'rel_mask_path': rel_mask_path,
                'split': 'train',  # All Unity Eyes in train split
                'dataset': 'unity_eyes',
                'dataset_format': 'unity_eyes_3c',  # Custom format identifier
                'crop_width': metadata['crop_size'][1],
                'crop_height': metadata['crop_size'][0],
                'iris_radius': float(metadata['iris_radius']),
                'pupil_radius': float(metadata['pupil_radius']),
                'pupil_size': float(metadata['pupil_size']),
                'iris_size': float(metadata['iris_size']),
                'iris_texture': metadata['iris_texture'],
                'lighting_skybox': metadata['lighting_skybox'],
                'skin_texture': metadata['skin_texture'],
                'head_pose': metadata['head_pose']
            }

            csv_rows.append(row)

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            continue

    # Create CSV
    df = pd.DataFrame(csv_rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"\n✓ Processed {len(csv_rows)} images successfully")
    print(f"✓ Images saved to: {img_output_dir}")
    print(f"✓ Masks saved to: {mask_output_dir}")
    print(f"✓ CSV saved to: {csv_path}")
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Mean crop size: {df['crop_width'].mean():.1f} x {df['crop_height'].mean():.1f}")
    print(f"  Mean iris radius: {df['iris_radius'].mean():.1f} px")
    print(f"  Mean pupil radius: {df['pupil_radius'].mean():.1f} px")
    print(f"  Unique iris textures: {df['iris_texture'].nunique()}")
    print(f"  Unique skin textures: {df['skin_texture'].nunique()}")


if __name__ == '__main__':
    main()
