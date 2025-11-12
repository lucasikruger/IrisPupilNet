# Model Checkpoints

This directory stores trained model weights.

## Available Models

| Model | Input Size | Classes | Base Channels | Dataset | Val IoU | File |
|-------|-----------|---------|---------------|---------|---------|------|
| UNet-SE Small | 160×160 | 3 | 32 | MOBIUS | ~0.85+ | `unet_mobius_best.pt` |
| UNet-SE Small (ONNX) | 160×160 | 3 | 32 | MOBIUS | ~0.85+ | `unet_mobius_best.onnx` |

*Note: Large checkpoint files (.pt, .onnx) are not tracked in git. Download from releases or train your own.*

## Checkpoint Format

PyTorch checkpoints (`.pt`) contain:

```python
{
    'model': OrderedDict(...),  # state_dict
    'args': {                    # Training arguments
        'model': 'unet_se_small',
        'img_size': 160,
        'num_classes': 3,
        'base': 32,
        ...
    }
}
```

## Loading Checkpoints

### For Inference (PyTorch)

```python
import torch
from irispupilnet.models.unet_se import UNetSESmall

checkpoint = torch.load('checkpoints/unet_mobius_best.pt', map_location='cpu')
model = UNetSESmall(in_channels=3, n_classes=3, base=32)
model.load_state_dict(checkpoint['model'])
model.eval()
```

### For ONNX Export

```bash
python export/export_to_onnx.py \
  --checkpoint checkpoints/unet_mobius_best.pt \
  --out checkpoints/unet_mobius_best.onnx \
  --size 160 --classes 3 --base 32
```

## File Sizes

- PyTorch checkpoints: ~7-10 MB (depending on base channels)
- ONNX models: ~7-10 MB

## Git LFS (Optional)

For teams sharing large model files, consider using Git LFS:

```bash
git lfs track "*.pt"
git lfs track "*.onnx"
git add .gitattributes
```

## Model Zoo (Coming Soon)

Pre-trained models on different datasets:
- MOBIUS (indoor/natural/poor lighting)
- Kaggle iris datasets
- Tayed dataset
- Combined multi-source model

Check releases page for downloads.
