"""
YOLO Integration for IrisPupilNet

Converts MOBIUS semantic masks to YOLO polygon format for training
with ultralytics YOLO.train().

Usage:
    from irispupilnet.yolo import prepare_yolo_dataset
    from ultralytics import YOLO

    # Prepare dataset (cached - only converts once)
    yaml_path = prepare_yolo_dataset(
        csv_path='dataset/merged/all_datasets.csv',
        data_root='data',
        output_dir='yolo_dataset',
        img_size=160
    )

    # Train with YOLO
    model = YOLO('yolo11n-seg.pt')
    model.train(data=yaml_path, epochs=50, imgsz=160)
"""

from .converter import prepare_yolo_dataset, mask_to_yolo_labels

__all__ = [
    'prepare_yolo_dataset',
    'mask_to_yolo_labels',
]
