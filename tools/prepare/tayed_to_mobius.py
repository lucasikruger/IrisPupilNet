#!/usr/bin/env python3
"""
Convert TayedEyes reduced dataset to MOBIUS format.

TayedEyes provides video frames with segmentation masks:
- pupil_seg_2D: Pupil mask
- iris_seg_2D: Iris mask
- lid_seg_2D: Lid/sclera mask (eyelid region)
- video: Grayscale video frames

This script:
1. Extracts frames and masks from .npz files
2. Creates RGB masks in MOBIUS format (red=sclera, green=iris, blue=pupil)
3. Clips iris/pupil to lid region (visible eye area)
4. Outputs to images/ and Masks/ directories
5. Generates CSV with metadata (all marked as 'train' split)

Usage:
    python tayed_to_mobius.py \
        --input_dir "/path/to/tayed_reduced" \
        --output_dir "data/tayed_mobius" \
        --csv "dataset/tayed_output/tayed_dataset.csv"
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_mobius_mask(
    pupil_mask: np.ndarray,
    iris_mask: np.ndarray,
    lid_mask: np.ndarray
) -> np.ndarray:
    """
    Create MOBIUS-format mask from TayedEyes segmentation masks.

    TayedEyes masks are binary (0/255), need to combine them into RGB:
    - Black (0,0,0): Background (exterior)
    - Red (0,0,255) in BGR: Sclera/lid (visible eye region)
    - Green (0,255,0) in BGR: Iris (ring)
    - Blue (255,0,0) in BGR: Pupil (center)

    Args:
        pupil_mask: [H, W] binary mask (0 or 255)
        iris_mask: [H, W] binary mask (0 or 255)
        lid_mask: [H, W] binary mask (0 or 255) - visible eye region

    Returns:
        mask: [H, W, 3] uint8 BGR image
    """
    h, w = pupil_mask.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert to binary (0/1)
    pupil_binary = (pupil_mask > 127).astype(np.uint8)
    iris_binary = (iris_mask > 127).astype(np.uint8)
    lid_binary = (lid_mask > 127).astype(np.uint8)

    # Clip pupil to iris region (pupil shouldn't extend beyond iris)
    pupil_clipped = cv2.bitwise_and(pupil_binary, iris_binary)

    # Clip both to lid region (nothing should extend beyond visible eye)
    pupil_clipped = cv2.bitwise_and(pupil_clipped, lid_binary)
    iris_clipped = cv2.bitwise_and(iris_binary, lid_binary)

    # Iris ring: iris area MINUS pupil
    iris_ring = cv2.bitwise_and(iris_clipped, cv2.bitwise_not(pupil_clipped))

    # Sclera: lid/eye region MINUS iris (includes both iris and pupil)
    sclera = cv2.bitwise_and(lid_binary, cv2.bitwise_not(iris_clipped))

    # Assign to BGR channels
    mask[:, :, 0] = pupil_clipped * 255  # Blue = Pupil
    mask[:, :, 1] = iris_ring * 255      # Green = Iris ring
    mask[:, :, 2] = sclera * 255         # Red = Sclera

    return mask


def select_evenly_spaced_indices(total_frames: int, desired_count: int) -> List[int]:
    """
    Return indices that are evenly distributed across the sequence length.
    """
    if desired_count <= 0 or total_frames == 0:
        return []
    if desired_count >= total_frames:
        return list(range(total_frames))
    if desired_count == 1:
        return [total_frames // 2]

    step = (total_frames - 1) / (desired_count - 1)
    indices: List[int] = []
    for i in range(desired_count):
        idx = int(round(i * step))
        idx = min(idx, total_frames - 1)
        if indices and idx <= indices[-1]:
            idx = min(indices[-1] + 1, total_frames - 1)
        indices.append(idx)
    return indices


def process_tayed_npz(
    npz_path: Path,
    output_img_dir: Path,
    output_mask_dir: Path,
    frames_to_extract: int
) -> list:
    """
    Process a single TayedEyes .npz file and extract evenly-spaced frames.

    Args:
        npz_path: Path to .npz file
        output_img_dir: Directory to save images
        output_mask_dir: Directory to save masks
        frames_to_extract: Number of frames to extract for this video (evenly spaced)

    Returns:
        List of metadata dicts for each frame
    """
    video_name = npz_path.stem
    data = dict(np.load(npz_path))

    # Extract arrays
    frames = data['video']  # Grayscale frames [N, H, W]
    pupil_masks = data['pupil_seg_2D.mp4_path']  # [N, H, W]
    iris_masks = data['iris_seg_2D.mp4_path']    # [N, H, W]
    lid_masks = data['lid_seg_2D.mp4_path']      # [N, H, W]
    frame_indices = data['frames']  # Frame numbers

    # Normalize masks to 0-255
    pupil_masks = (pupil_masks * 255).astype(np.uint8)
    iris_masks = (iris_masks * 255).astype(np.uint8)
    lid_masks = (lid_masks * 255).astype(np.uint8)

    # Convert grayscale frames to uint8 if needed
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)

    # Select evenly-spaced frame indices
    total_frames = len(frame_indices)
    selected_indices = select_evenly_spaced_indices(total_frames, frames_to_extract)

    metadata_list = []

    dataset_folder = output_img_dir.parent.name or output_img_dir.parent.as_posix()

    for idx in selected_indices:
        frame_num = frame_indices[idx]
        # Get frame and masks
        frame = frames[idx]
        pupil_mask = pupil_masks[idx]
        iris_mask = iris_masks[idx]
        lid_mask = lid_masks[idx]

        # Convert grayscale to RGB for consistency
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_rgb = frame

        # Create MOBIUS mask
        mobius_mask = create_mobius_mask(pupil_mask, iris_mask, lid_mask)

        # Create filename
        frame_id = f"{video_name}_frame_{frame_num:05d}"
        img_filename = f"{frame_id}.jpg"
        mask_filename = f"{frame_id}.png"

        # Save paths
        img_path = output_img_dir / img_filename
        mask_path = output_mask_dir / mask_filename

        # Save files
        cv2.imwrite(str(img_path), frame_rgb)
        cv2.imwrite(str(mask_path), mobius_mask)

        # Metadata
        rel_image_path = (Path(dataset_folder) / 'images' / img_filename).as_posix()
        rel_mask_path = (Path(dataset_folder) / 'Masks' / mask_filename).as_posix()

        metadata = {
            'id': frame_id,
            'video_name': video_name,
            'frame_number': int(frame_num),
            'rel_image_path': rel_image_path,
            'rel_mask_path': rel_mask_path,
            'split': 'train',  # All TayedEyes in train split
            'dataset': 'tayed_eyes',
            'dataset_format': 'tayed_3c',
            'width': frame_rgb.shape[1],
            'height': frame_rgb.shape[0]
        }

        metadata_list.append(metadata)

    return metadata_list


def compute_frame_allocations(
    npz_files: List[Path],
    target_total_frames: int
) -> Dict[Path, int]:
    """
    Allocate how many frames to sample from each video so the sum
    approximates the requested total while respecting per-video limits.
    """
    video_lengths: List[Tuple[Path, int]] = []
    total_available = 0

    for npz_path in tqdm(npz_files, desc="Scanning videos for lengths"):
        with np.load(npz_path) as data:
            frame_count = len(data['frames'])
        video_lengths.append((npz_path, frame_count))
        total_available += frame_count

    if target_total_frames >= total_available:
        print(
            f"Requested {target_total_frames} frames but only "
            f"{total_available} exist. Using all available frames."
        )
        return {path: frame_count for path, frame_count in video_lengths}

    num_videos = len(video_lengths)
    allocations: List[int] = [0] * num_videos

    base = target_total_frames // num_videos
    remainder = target_total_frames % num_videos

    for idx, (_, frame_count) in enumerate(video_lengths):
        desired = base + (1 if idx < remainder else 0)
        allocations[idx] = min(frame_count, desired)

    assigned = sum(allocations)
    leftover = target_total_frames - assigned

    while leftover > 0:
        progress = False
        for idx, (_, frame_count) in enumerate(video_lengths):
            if leftover == 0:
                break
            if allocations[idx] < frame_count:
                allocations[idx] += 1
                leftover -= 1
                progress = True
        if not progress:
            # No remaining capacity; exit early
            break

    return {
        path: allocations[idx]
        for idx, (path, _) in enumerate(video_lengths)
        if allocations[idx] > 0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert TayedEyes dataset to MOBIUS format"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to TayedEyes dataset directory (contains .npz files)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tayed_mobius',
        help='Output directory for MOBIUS-format data'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='dataset/tayed_output/tayed_dataset.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--frames_per_video',
        type=int,
        default=4,
        help='Frames per video when --target_frames is not provided'
    )
    parser.add_argument(
        '--target_frames',
        type=int,
        default=None,
        help='Approximate total number of frames to extract across all videos'
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv)

    img_output_dir = output_dir / 'images'
    mask_output_dir = output_dir / 'Masks'
    img_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NPZ files
    npz_files = sorted(input_dir.glob('*.npz'))

    if not npz_files:
        print(f"Error: No .npz files found in {input_dir}")
        return

    print(f"Found {len(npz_files)} NPZ files to process")

    frame_plan: Dict[Path, int]
    if args.target_frames is not None:
        target = max(0, args.target_frames)
        if target == 0:
            print("Target frames set to 0; nothing to do.")
            return
        frame_plan = compute_frame_allocations(npz_files, target)
    else:
        per_video = max(1, args.frames_per_video)
        frame_plan = {path: per_video for path in npz_files}

    # Process each NPZ file
    all_metadata = []

    for npz_path in tqdm(npz_files, desc="Processing TayedEyes videos"):
        frames_to_extract = frame_plan.get(npz_path, 0)
        if frames_to_extract <= 0:
            continue
        try:
            metadata_list = process_tayed_npz(
                npz_path,
                img_output_dir,
                mask_output_dir,
                frames_to_extract
            )
            all_metadata.extend(metadata_list)
        except Exception as e:
            print(f"\nError processing {npz_path.name}: {e}")
            continue

    # Create CSV
    df = pd.DataFrame(all_metadata)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"\n✓ Processed {len(all_metadata)} frames successfully")
    print(f"✓ Images saved to: {img_output_dir}")
    print(f"✓ Masks saved to: {mask_output_dir}")
    print(f"✓ CSV saved to: {csv_path}")
    print(f"\nDataset statistics:")
    print(f"  Total frames: {len(df)}")
    print(f"  Total videos: {df['video_name'].nunique()}")
    print(f"  Mean resolution: {df['width'].mean():.1f} x {df['height'].mean():.1f}")


if __name__ == '__main__':
    main()
