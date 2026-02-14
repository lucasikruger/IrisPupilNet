#!/usr/bin/env python3
"""
Run YOLO11 instance segmentation and convert output to semantic segmentation masks.

This script:
1. Loads a trained YOLO11 instance segmentation model
2. Runs inference on images
3. Converts instance masks (separate iris/pupil instances) to semantic masks
4. Saves semantic masks compatible with IrisPupilNet format

Usage:
    # Predict on single image
    python tools/yolo/predict_yolo_to_semantic.py \
        --weights runs/yolo_native/train/weights/best.pt \
        --source image.jpg \
        --output predictions

    # Predict on directory
    python tools/yolo/predict_yolo_to_semantic.py \
        --weights runs/yolo_native/train/weights/best.pt \
        --source images/ \
        --output predictions \
        --save-vis

    # Export to ONNX
    python tools/yolo/predict_yolo_to_semantic.py \
        --weights runs/yolo_native/train/weights/best.pt \
        --export-onnx model.onnx \
        --imgsz 160
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


def instances_to_semantic_mask(masks, boxes, classes, img_shape, class_map=None):
    """
    Convert YOLO instance segmentation output to semantic segmentation mask.

    Args:
        masks: Instance masks from YOLO (list of binary masks or tensor)
        boxes: Bounding boxes (N, 4) [x1, y1, x2, y2]
        classes: Class IDs (N,) [0=iris, 1=pupil]
        img_shape: (H, W) output shape
        class_map: Dict mapping YOLO class to semantic class
                   Default: {0: 1, 1: 2} (iris=1, pupil=2)

    Returns:
        semantic_mask: (H, W) with values [0=background, 1=iris, 2=pupil]
    """
    if class_map is None:
        class_map = {0: 1, 1: 2}  # YOLO iris=0 → semantic iris=1, etc.

    height, width = img_shape
    semantic_mask = np.zeros((height, width), dtype=np.uint8)

    if masks is None or len(masks) == 0:
        return semantic_mask

    # Convert masks to numpy if needed
    if hasattr(masks, 'data'):
        masks = masks.data.cpu().numpy()

    # Process each instance
    for i, (mask, cls) in enumerate(zip(masks, classes)):
        semantic_class = class_map.get(int(cls), 0)
        if semantic_class == 0:
            continue  # Skip background

        # Resize mask to image size if needed
        if mask.shape != (height, width):
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask.astype(np.uint8)

        # Write to semantic mask (later instances overwrite earlier ones)
        semantic_mask[mask_resized > 0] = semantic_class

    return semantic_mask


def visualize_semantic_mask(image, semantic_mask, alpha=0.5):
    """
    Create visualization of semantic segmentation.

    Args:
        image: (H, W, 3) BGR image
        semantic_mask: (H, W) with values [0, 1, 2]
        alpha: Overlay transparency

    Returns:
        vis: (H, W, 3) BGR visualization
    """
    # Color map: background=black, iris=green, pupil=red
    color_map = {
        0: (0, 0, 0),       # background - black
        1: (0, 255, 0),     # iris - green
        2: (0, 0, 255),     # pupil - red
    }

    # Create colored mask
    colored_mask = np.zeros_like(image)
    for class_id, color in color_map.items():
        colored_mask[semantic_mask == class_id] = color

    # Blend with original image
    vis = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    # Draw contours
    for class_id in [1, 2]:
        mask = (semantic_mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = color_map[class_id]
        cv2.drawContours(vis, contours, -1, color, 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description='YOLO instance → semantic segmentation')

    # Model
    parser.add_argument('--weights', required=True, help='Path to trained YOLO model')
    parser.add_argument('--imgsz', type=int, default=160, help='Image size for inference')

    # Input/Output
    parser.add_argument('--source', required=True, help='Image file or directory')
    parser.add_argument('--output', default='predictions', help='Output directory')
    parser.add_argument('--save-vis', action='store_true', help='Save visualizations')
    parser.add_argument('--save-masks', action='store_true', default=True,
                       help='Save semantic masks')

    # Inference settings
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='Device (e.g., "0" or "cpu")')

    # Export
    parser.add_argument('--export-onnx', help='Export model to ONNX format')

    args = parser.parse_args()

    # Load model
    print(f"Loading YOLO model: {args.weights}")
    model = YOLO(args.weights)

    # Export to ONNX if requested
    if args.export_onnx:
        print(f"\nExporting to ONNX: {args.export_onnx}")
        model.export(
            format='onnx',
            imgsz=args.imgsz,
            simplify=True,
            dynamic=False,
        )
        # Move exported file
        exported_path = Path(args.weights).parent / f"{Path(args.weights).stem}.onnx"
        if exported_path.exists():
            import shutil
            shutil.move(str(exported_path), args.export_onnx)
            print(f"✓ Exported to {args.export_onnx}")
        return

    # Setup output directories
    output_dir = Path(args.output)
    if args.save_masks:
        (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        (output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)

    # Get image paths
    source = Path(args.source)
    if source.is_file():
        image_paths = [source]
    elif source.is_dir():
        image_paths = list(source.glob('*.jpg')) + list(source.glob('*.png'))
    else:
        raise ValueError(f"Invalid source: {source}")

    print(f"\nProcessing {len(image_paths)} images...")

    # Process images
    for img_path in tqdm(image_paths, desc="Inference"):
        # Run YOLO inference
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        # Load original image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue

        # Convert instance masks to semantic mask
        if results.masks is not None:
            semantic_mask = instances_to_semantic_mask(
                masks=results.masks,
                boxes=results.boxes.xyxy,
                classes=results.boxes.cls,
                img_shape=image.shape[:2]
            )
        else:
            # No detections - all background
            semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Save semantic mask
        if args.save_masks:
            mask_path = output_dir / 'masks' / f'{img_path.stem}_mask.png'
            # Convert to color-coded format (mobius_3c style)
            color_mask = np.zeros((*semantic_mask.shape, 3), dtype=np.uint8)
            color_mask[semantic_mask == 0] = [255, 0, 0]    # background - red
            color_mask[semantic_mask == 1] = [0, 255, 0]    # iris - green
            color_mask[semantic_mask == 2] = [0, 0, 255]    # pupil - blue
            cv2.imwrite(str(mask_path), color_mask)

        # Save visualization
        if args.save_vis:
            vis = visualize_semantic_mask(image, semantic_mask)
            vis_path = output_dir / 'visualizations' / f'{img_path.stem}_vis.png'
            cv2.imwrite(str(vis_path), vis)

    print(f"\n{'='*60}")
    print("Prediction complete!")
    print(f"{'='*60}")
    if args.save_masks:
        print(f"Semantic masks: {output_dir / 'masks'}")
    if args.save_vis:
        print(f"Visualizations: {output_dir / 'visualizations'}")


if __name__ == '__main__':
    main()
