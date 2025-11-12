# IrisPupilNet

**A PyTorch-based iris and pupil segmentation system with ONNX export for real-time deployment.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

IrisPupilNet is a complete pipeline for training, exporting, and deploying deep learning models for iris and pupil segmentation. It features:

- **UNet with Squeeze-and-Excitation blocks** for improved feature learning
- **Multi-source dataset support** via CSV-driven loader with format converters
- **Registry pattern** for easy addition of models, datasets, and mask formats
- **ONNX export** with NHWC layout for browser/mobile deployment
- **Real-time webcam demo** using MediaPipe face detection

## Features

- ✅ **Flexible Training Pipeline**
  - Registry-based model and dataset management
  - Support for multiple mask formats (RGB-coded, paletted PNG, etc.)
  - Albumentations-based augmentation
  - Automatic best model checkpoint saving

- ✅ **Production-Ready Export**
  - ONNX export with dynamic batch size
  - NHWC layout for web/mobile inference
  - Compatible with ONNX Runtime and browser deployment

- ✅ **Real-Time Demo**
  - WebCam-based eye detection using MediaPipe
  - Live segmentation overlay
  - 4-panel visualization (left/right eyes, raw + segmentation)

- ✅ **Multi-Dataset Support**
  - MOBIUS dataset
  - Kaggle iris datasets
  - Tayed dataset
  - Easy to add custom datasets via CSV format

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/IrisPupilNet.git
   cd IrisPupilNet
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Training a Model

1. **Prepare your dataset** (see [Dataset Preparation](#dataset-preparation))

2. **Run training**
   ```bash
   python irispupilnet/train.py \
     --dataset csv_seg \
     --data-root /path/to/dataset \
     --csv /path/to/dataset.csv \
     --default-format mobius_3c \
     --model unet_se_small \
     --img-size 160 \
     --num-classes 3 \
     --base 32 \
     --batch-size 32 \
     --epochs 20 \
     --lr 1e-3 \
     --workers 4 \
     --out runs/my_experiment
   ```

3. **Monitor training**
   ```
   epoch 01 | train 0.4521 | val 0.3892 | IoU(iris+pupil) 0.782
   epoch 02 | train 0.3124 | val 0.2845 | IoU(iris+pupil) 0.834
   ...
   ↳ saved runs/my_experiment/best.pt
   ```

### Exporting to ONNX

```bash
python export/export_to_onnx.py \
  --checkpoint runs/my_experiment/best.pt \
  --out checkpoints/model.onnx \
  --size 160 \
  --classes 3 \
  --base 32
```

### Running the Demo

```bash
cd demo
python webcam_demo.py --model ../checkpoints/model.onnx
```

Press 'q' to quit, 'm' to toggle mirror mode.

---

## Architecture

### Model: UNet with Squeeze-and-Excitation

The default model (`unet_se_small`) is a UNet architecture enhanced with Squeeze-and-Excitation blocks:

```
Input (3×H×W)
  ↓
[Encoder]
  inc: DoubleConvSE(3 → 32)           # Initial convolution
  d1:  MaxPool + DoubleConvSE(32 → 64)
  d2:  MaxPool + DoubleConvSE(64 → 128)
  d3:  MaxPool + DoubleConvSE(128 → 256)
  ↓
[Decoder]
  u1:  Upsample + Concat(256+128) + DoubleConvSE(384 → 128)
  u2:  Upsample + Concat(128+64)  + DoubleConvSE(192 → 64)
  u3:  Upsample + Concat(64+32)   + DoubleConvSE(96 → 32)
  ↓
Output: Conv1x1(32 → 3) → [B, 3, H, W] logits
```

**DoubleConvSE Block:**
- Conv3x3 → BatchNorm → ReLU
- Conv3x3 → BatchNorm → ReLU
- SEBlock (channel attention with reduction=16)

### Dataset Format

Training uses a CSV-driven loader that supports multiple mask formats:

**Required CSV columns:**
- `rel_image_path`: Relative path to image (from data_root)
- `rel_mask_path`: Relative path to mask
- `split`: 'train', 'val', or 'test'

**Optional CSV column:**
- `dataset_format`: Mask format ('mobius_3c', 'pascal_indexed', etc.)

**Supported Mask Formats:**
- **mobius_3c**: RGB-coded (red=bg, green=iris, blue=pupil)
- **mobius_2c_pupil_only**: 2-class (bg, pupil only)
- **pascal_indexed**: Paletted PNG with class indices

See [irispupilnet/utils/mask_formats.py](irispupilnet/utils/mask_formats.py) for all formats.

---

## Dataset Preparation

### MOBIUS Dataset

```bash
python tools/prepare/create_mobius_csv_from_dir.py \
  --input_dir /path/to/MOBIUS \
  --output_csv dataset/mobius_dataset.csv
```

Then add train/val/test splits:

```bash
python tools/prepare/add_split_column.py \
  --csv dataset/mobius_dataset.csv \
  --train-ratio 0.7 \
  --val-ratio 0.15
```

### Custom Dataset

Create a CSV file with the required columns:

```csv
rel_image_path,rel_mask_path,split,dataset_format
images/001.jpg,masks/001.png,train,mobius_3c
images/002.jpg,masks/002.png,train,mobius_3c
images/003.jpg,masks/003.png,val,mobius_3c
...
```

---

## Training Guide

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `csv_seg` | Dataset type (registered name) |
| `--data-root` | *required* | Base directory for relative paths |
| `--csv` | *required* | Path to CSV file |
| `--default-format` | `mobius_3c` | Fallback mask format |
| `--model` | `unet_se_small` | Model architecture |
| `--img-size` | `160` | Input image size (H=W) |
| `--num-classes` | `3` | Number of segmentation classes |
| `--base` | `32` | Base number of channels |
| `--batch-size` | `32` | Training batch size |
| `--epochs` | `20` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--workers` | `2` | DataLoader workers |
| `--out` | `runs/...` | Output directory |

### Training Output

```
runs/my_experiment/
├── best.pt           # Best checkpoint (highest val IoU)
└── (tensorboard logs, if enabled)
```

The `best.pt` file contains:
```python
{
  'model': model.state_dict(),
  'args': training_args_dict
}
```

### Metrics

- **Loss**: CrossEntropyLoss on class logits
- **Metric**: Mean IoU (Intersection over Union)
  - Computed on foreground classes only (iris, pupil)
  - Background (class 0) is ignored

---

## Export and Deployment

### ONNX Export

The export script converts PyTorch models to ONNX with NHWC layout:

```bash
python export/export_to_onnx.py \
  --checkpoint runs/experiment/best.pt \
  --out checkpoints/model.onnx \
  --size 160 \
  --classes 3 \
  --base 32
```

**Output format:**
- Input: `[batch, height, width, 3]` (NHWC)
- Output: `[batch, height, width, num_classes]` (logits, NHWC)
- Dynamic batch axis for flexible inference

### Using Exported Model

**Python (ONNX Runtime):**
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("checkpoints/model.onnx")
input_data = np.random.rand(1, 160, 160, 3).astype(np.float32)
logits = session.run(None, {"input": input_data})[0]
predictions = np.argmax(logits, axis=-1)  # [1, 160, 160]
```

**Browser (ONNX.js / Transformers.js):**
```javascript
const session = await ort.InferenceSession.create("model.onnx");
const tensor = new ort.Tensor('float32', imageData, [1, 160, 160, 3]);
const outputs = await session.run({ input: tensor });
const logits = outputs.logits.data;
```

---

## Demo Application

### Requirements

```bash
pip install opencv-python mediapipe onnxruntime
```

### Usage

```bash
cd demo
python webcam_demo.py [OPTIONS]
```

**Options:**
- `--model PATH`: Path to ONNX model (optional, shows detection only without model)
- `--no-mirror`: Disable mirrored camera feed

**Controls:**
- `q`: Quit
- `m`: Toggle mirror mode

### Demo Output

4-panel display:
```
┌─────────────┬─────────────┐
│  Left Eye   │  Right Eye  │  ← Raw cropped eyes
├─────────────┼─────────────┤
│ Left + Seg  │ Right + Seg │  ← With segmentation overlay
└─────────────┴─────────────┘
```

Colors:
- Green: Iris
- Blue: Pupil

---

## Project Structure

```
IrisPupilNet/
├── README.md                    # This file
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Development dependencies
├── setup.py                     # Package installation (coming soon)
│
├── irispupilnet/                # Main training package
│   ├── train.py                # Training script
│   ├── models/                 # Model architectures
│   │   ├── __init__.py         # Model registry
│   │   ├── unet_se.py          # UNet-SE (primary)
│   │   └── baseline.py         # Simple baseline
│   ├── datasets/               # Dataset loaders
│   │   ├── __init__.py         # Dataset registry
│   │   ├── base.py             # Base dataset class
│   │   └── csv_seg.py          # CSV-driven loader
│   ├── utils/                  # Utilities
│   │   ├── augment.py          # Augmentation pipeline
│   │   ├── mask_formats.py     # Mask format converters
│   │   └── metrics.py          # IoU metric
│   └── configs/                # Configuration files
│       └── default.yaml        # Default config
│
├── tools/                       # Dataset preparation tools
│   ├── prepare/                # Dataset conversion scripts
│   │   ├── create_mobius_csv.py
│   │   ├── create_kaggle_csv.py
│   │   └── add_split_column.py
│   ├── analyze/                # Analysis and visualization
│   │   └── plot_dataset_summary.py
│   └── resources/              # Reference files
│       ├── kaggle_datasets.txt
│       └── research_papers.txt
│
├── export/                      # Model export utilities
│   ├── export_to_onnx.py       # ONNX export script
│   └── README.md               # Export documentation
│
├── demo/                        # Real-time demo application
│   ├── webcam_demo.py          # Webcam demo
│   ├── requirements.txt        # Demo-specific deps
│   └── README.md               # Demo usage
│
├── checkpoints/                 # Trained model weights
│   ├── .gitkeep                # (large files not tracked)
│   └── README.md               # Model documentation
│
├── scripts/                     # Helper scripts
│   └── train.sh                # Example training script
│
├── runs/                        # Training experiment outputs
├── data/                        # Dataset storage (gitignored)
├── tests/                       # Unit tests
└── docs/                        # Additional documentation
```

---

## Adding Custom Components

### Adding a New Model

1. Create model file in `irispupilnet/models/`:

```python
from . import register_model
import torch.nn as nn

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, base: int):
        super().__init__()
        # Model definition

    def forward(self, x):
        # Forward pass
        return logits  # [B, n_classes, H, W]
```

2. Use with `--model my_model`

### Adding a New Mask Format

1. Add converter to `irispupilnet/utils/mask_formats.py`:

```python
@register_mask_format("my_format")
def convert_my_format(mask_bgr: np.ndarray) -> np.ndarray:
    """Convert custom mask format to class indices"""
    # mask_bgr: [H, W, 3] BGR uint8
    # return: [H, W] int64 with class indices (0=bg, 1=iris, 2=pupil)
    ...
    return class_mask
```

2. Use in CSV with `dataset_format=my_format`

---

## Troubleshooting

### Architecture Mismatch Error

If you get a state_dict mismatch when exporting:
- Ensure export script uses same architecture as training (now fixed with SE blocks)
- Check `base` parameter matches training

### Out of Memory

- Reduce `--batch-size`
- Reduce `--img-size`
- Reduce `--base` (fewer channels)

### Low IoU

- Increase `--epochs`
- Tune `--lr` (try 5e-4 or 2e-3)
- Check mask format conversion is correct
- Verify dataset quality

### Dataset CSV Not Found

- Use absolute paths for `--csv`
- Ensure `rel_image_path` and `rel_mask_path` are relative to `--data-root`

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{irispupilnet2025,
  author = {Your Name},
  title = {IrisPupilNet: Iris and Pupil Segmentation with UNet-SE},
  year = {2025},
  url = {https://github.com/yourusername/IrisPupilNet}
}
```

### Dataset Citations

**MOBIUS:**
- Dataset URL and paper citation here

**Other datasets:**
- Add citations as applicable

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- UNet architecture based on [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Squeeze-and-Excitation blocks from [Hu et al., 2018](https://arxiv.org/abs/1709.01507)
- MediaPipe for face/eye detection
- MOBIUS, Kaggle, and Tayed dataset contributors

---

## Contact

For questions or issues, please open an issue on GitHub or contact: your.email@example.com
