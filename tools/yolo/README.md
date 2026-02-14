# YOLO11 Native Training Approach

This directory contains tools for training YOLO11 using its native instance segmentation API.

## Why This Approach?

YOLO11 was designed for **instance segmentation** (detecting and segmenting individual object instances), while IrisPupilNet needs **semantic segmentation** (per-pixel classification). This approach bridges the gap by:

1. Converting semantic masks → instance format (1 iris + 1 pupil per image)
2. Training with YOLO's full model and optimized pipeline
3. Converting predictions back to semantic format

**Benefits:**
- Leverages YOLO's full pretrained segmentation head
- Uses YOLO's optimized training, augmentation, and learning rate scheduling
- May achieve better performance than backbone-only approach

**Trade-offs:**
- Requires dataset conversion step
- Separate training workflow (not integrated with IrisPupilNet's train.py)
- Need to convert instance predictions back to semantic masks

## Complete Workflow

### Step 1: Convert Dataset to YOLO Format

```bash
# Convert your semantic segmentation dataset to YOLO instance format
python tools/prepare/convert_to_yolo_instance.py \
  --csv dataset/merged/all_datasets.csv \
  --data-root dataset \
  --output yolo_dataset \
  --img-size 160
```

**What this does:**
- Reads images and semantic masks from your CSV dataset
- Extracts iris and pupil as separate instances
- Creates bounding boxes and polygon masks for each
- Generates YOLO-format labels (`.txt` files)
- Creates `dataset.yaml` configuration file

**Output structure:**
```
yolo_dataset/
├── dataset.yaml          # YOLO dataset config
├── images/
│   ├── train/            # Training images
│   ├── val/              # Validation images
│   └── test/             # Test images
└── labels/
    ├── train/            # Training labels (.txt)
    ├── val/              # Validation labels
    └── test/             # Test labels
```

### Step 2: Train with YOLO's Native API

```bash
# Train YOLO11 nano (fastest)
python tools/yolo/train_yolo_native.py \
  --data yolo_dataset/dataset.yaml \
  --model yolo11n-seg \
  --epochs 50 \
  --imgsz 160 \
  --batch 16 \
  --project runs/yolo_native

# Train YOLO11 small (balanced)
python tools/yolo/train_yolo_native.py \
  --data yolo_dataset/dataset.yaml \
  --model yolo11s-seg \
  --epochs 100 \
  --imgsz 192 \
  --batch 8 \
  --lr0 0.001 \
  --project runs/yolo_native

# Train YOLO11 medium (high accuracy)
python tools/yolo/train_yolo_native.py \
  --data yolo_dataset/dataset.yaml \
  --model yolo11m-seg \
  --epochs 100 \
  --imgsz 224 \
  --batch 4 \
  --project runs/yolo_native

# Resume from checkpoint
python tools/yolo/train_yolo_native.py \
  --data yolo_dataset/dataset.yaml \
  --resume runs/yolo_native/train/weights/last.pt
```

**Training outputs:**
```
runs/yolo_native/train/
├── weights/
│   ├── best.pt           # Best checkpoint (highest mAP)
│   └── last.pt           # Latest checkpoint
├── results.png           # Training curves
├── confusion_matrix.png  # Confusion matrix
├── val_batch0_*.jpg      # Validation visualizations
└── args.yaml             # Training arguments
```

### Step 3: Run Inference (Instance → Semantic)

```bash
# Predict on images and convert to semantic masks
python tools/yolo/predict_yolo_to_semantic.py \
  --weights runs/yolo_native/train/weights/best.pt \
  --source path/to/images/ \
  --output predictions \
  --save-vis \
  --imgsz 160

# Predict on single image
python tools/yolo/predict_yolo_to_semantic.py \
  --weights runs/yolo_native/train/weights/best.pt \
  --source image.jpg \
  --output predictions \
  --save-vis
```

**Prediction outputs:**
```
predictions/
├── masks/                   # Semantic masks (color-coded PNG)
│   ├── image1_mask.png
│   └── image2_mask.png
└── visualizations/          # Overlay visualizations
    ├── image1_vis.png
    └── image2_vis.png
```

**Mask format:** Color-coded PNG compatible with IrisPupilNet:
- Red (255, 0, 0): Background
- Green (0, 255, 0): Iris
- Blue (0, 0, 255): Pupil

### Step 4: Export to ONNX (Optional)

```bash
# Export trained model to ONNX
python tools/yolo/predict_yolo_to_semantic.py \
  --weights runs/yolo_native/train/weights/best.pt \
  --export-onnx model.onnx \
  --imgsz 160
```

**Note:** The exported ONNX model contains YOLO's instance segmentation. You'll need to use the prediction script to convert outputs to semantic masks.

## Hyperparameter Tuning

### Learning Rate

```bash
# Lower LR for fine-tuning
--lr0 0.0005 --lrf 0.001

# Higher LR for training from scratch
--lr0 0.01 --lrf 0.01
```

### Batch Size

Adjust based on GPU memory:
- **YOLO11n**: batch=16-32 (RTX 3090)
- **YOLO11s**: batch=8-16
- **YOLO11m**: batch=4-8
- **YOLO11l**: batch=2-4

### Image Size

Must be multiple of 32:
- **Fast training**: 128 or 160
- **Balanced**: 192 or 224
- **High quality**: 256 or 320

### Early Stopping

```bash
--patience 50  # Stop if no improvement for 50 epochs
```

## Metrics

YOLO reports instance segmentation metrics:
- **Box mAP50**: Mean Average Precision at IoU=0.5 for bounding boxes
- **Box mAP50-95**: mAP across IoU thresholds 0.5-0.95
- **Mask mAP50**: mAP for segmentation masks at IoU=0.5
- **Mask mAP50-95**: mAP for masks across IoU thresholds

**Focus on:** Mask mAP50-95 for segmentation quality

## Comparison with Approach 1 (Backbone)

| Aspect | Approach 1 (Backbone) | Approach 2 (Native) |
|--------|----------------------|---------------------|
| **Setup** | Single command | 2-step conversion |
| **Training** | `irispupilnet.train` | Custom YOLO script |
| **Metrics** | Dice, IoU, HD95 | mAP (box + mask) |
| **Inference** | Direct semantic output | Instance → semantic |
| **Integration** | Seamless with pipeline | Standalone workflow |
| **YOLO Model** | Backbone only (~60%) | Full model (100%) |
| **Performance** | Good | Potentially better |
| **Use Case** | Quick experiments, research | Production, maximum accuracy |

## Tips for Best Results

1. **Start with nano variant** for quick experiments
2. **Use pretrained weights** (enabled by default)
3. **Monitor validation metrics** during training
4. **Try different image sizes** (160, 192, 224)
5. **Adjust confidence threshold** (`--conf`) if predictions are too strict/lenient
6. **Use data augmentation** (built into YOLO by default)

## Troubleshooting

**Out of memory:**
- Reduce `--batch`
- Reduce `--imgsz`
- Use smaller variant (nano instead of small)

**Poor segmentation quality:**
- Increase `--epochs`
- Try larger `--imgsz`
- Adjust `--conf` threshold
- Check dataset quality

**Training too slow:**
- Use smaller `--imgsz`
- Reduce `--batch` (may need to adjust LR)
- Use nano variant

## Example: Complete Pipeline

```bash
# 1. Convert dataset
python tools/prepare/convert_to_yolo_instance.py \
  --csv dataset/mobius.csv \
  --data-root dataset \
  --output yolo_mobius \
  --img-size 192

# 2. Train YOLO11 small
python tools/yolo/train_yolo_native.py \
  --data yolo_mobius/dataset.yaml \
  --model yolo11s-seg \
  --epochs 100 \
  --imgsz 192 \
  --batch 8 \
  --project runs/mobius_yolo

# 3. Predict on test images
python tools/yolo/predict_yolo_to_semantic.py \
  --weights runs/mobius_yolo/train/weights/best.pt \
  --source dataset/test_images/ \
  --output predictions/mobius \
  --save-vis \
  --imgsz 192

# 4. Export to ONNX
python tools/yolo/predict_yolo_to_semantic.py \
  --weights runs/mobius_yolo/train/weights/best.pt \
  --export-onnx models/yolo11s_mobius.onnx \
  --imgsz 192
```

## Further Reading

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [YOLO Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [Hyperparameter Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
