#!/usr/bin/env python3
"""
Train YOLO11 using native ultralytics training API for iris/pupil instance segmentation.

This script uses YOLO's built-in training pipeline, which includes:
- Optimized data augmentation
- Learning rate scheduling
- Automatic mixed precision training
- Built-in logging and checkpointing

Usage:
    # Train YOLO11 nano
    python tools/yolo/train_yolo_native.py \
        --data yolo_dataset/dataset.yaml \
        --model yolo11n-seg \
        --epochs 50 \
        --imgsz 160 \
        --batch 16

    # Train YOLO11 small with custom settings
    python tools/yolo/train_yolo_native.py \
        --data yolo_dataset/dataset.yaml \
        --model yolo11s-seg \
        --epochs 100 \
        --imgsz 192 \
        --batch 8 \
        --lr0 0.001 \
        --project runs/yolo_native

    # Resume from checkpoint
    python tools/yolo/train_yolo_native.py \
        --data yolo_dataset/dataset.yaml \
        --resume runs/yolo_native/train/weights/last.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 for iris/pupil segmentation')

    # Dataset
    parser.add_argument('--data', required=True, help='Path to dataset YAML file')

    # Model
    parser.add_argument('--model', default='yolo11n-seg',
                       choices=['yolo11n-seg', 'yolo11s-seg', 'yolo11m-seg', 'yolo11l-seg'],
                       help='YOLO model variant')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=160, help='Image size (must be multiple of 32)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')
    parser.add_argument('--optimizer', default='AdamW', choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')

    # Output
    parser.add_argument('--project', default='runs/yolo_native', help='Project directory')
    parser.add_argument('--name', default='train', help='Experiment name')
    parser.add_argument('--resume', help='Resume from checkpoint')

    # Other
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--device', default='', help='Device (e.g., "0" or "0,1" or "cpu")')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='Save checkpoint every N epochs (-1 = disabled)')
    parser.add_argument('--val', action='store_true', default=True,
                       help='Validate during training')

    args = parser.parse_args()

    # Validate image size
    if args.imgsz % 32 != 0:
        raise ValueError(f"Image size must be multiple of 32, got {args.imgsz}")

    # Load model
    print(f"\n{'='*60}")
    print(f"Training YOLO11 for Iris/Pupil Instance Segmentation")
    print(f"{'='*60}")

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        model_name = f"{args.model}.pt" if args.pretrained else f"{args.model}.yaml"
        print(f"\nLoading model: {model_name}")
        print(f"  Pretrained: {args.pretrained}")
        model = YOLO(model_name)

    # Train
    print(f"\nStarting training...")
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Learning rate: {args.lr0}")
    print(f"  Device: {args.device if args.device else 'auto'}")
    print(f"  Output: {args.project}/{args.name}\n")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        project=args.project,
        name=args.name,
        workers=args.workers,
        device=args.device,
        pretrained=args.pretrained,
        patience=args.patience,
        save_period=args.save_period,
        val=args.val,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Best weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print(f"Last weights: {Path(args.project) / args.name / 'weights' / 'last.pt'}")
    print(f"Results: {Path(args.project) / args.name}")

    # Validation
    if args.val:
        print(f"\nRunning final validation...")
        metrics = model.val()
        print(f"\nFinal Metrics:")
        print(f"  Box mAP50: {metrics.box.map50:.4f}")
        print(f"  Box mAP50-95: {metrics.box.map:.4f}")
        print(f"  Mask mAP50: {metrics.seg.map50:.4f}")
        print(f"  Mask mAP50-95: {metrics.seg.map:.4f}")

    print(f"\nTo use this model with IrisPupilNet:")
    print(f"  python tools/yolo/predict_yolo_to_semantic.py \\")
    print(f"    --weights {Path(args.project) / args.name / 'weights' / 'best.pt'} \\")
    print(f"    --source path/to/images \\")
    print(f"    --output predictions")


if __name__ == '__main__':
    main()
