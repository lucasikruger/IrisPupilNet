# Quick Start Scripts

This directory contains simple wrapper scripts for common workflows.

## Available Scripts

### 1. `train_from_config.sh` - Train a model

Train with a config file:

```bash
./scripts/train_from_config.sh irispupilnet/configs/example_grayscale.yaml
```

Override config values from command line:

```bash
./scripts/train_from_config.sh myconfig.yaml --epochs 100 --lr 5e-4
```

### 2. `export_model.sh` - Export checkpoint to ONNX

Automatically reads configuration from checkpoint:

```bash
./scripts/export_model.sh runs/experiment/best.pt model.onnx
```

The script auto-detects:
- Model architecture (unet_se_small, seresnext_unet, etc.)
- Input channels (1 for grayscale, 3 for RGB)
- Number of classes
- Image size
- Base channel count

### 3. `run_demo.sh` - Run webcam demo

With model (auto-detects config):

```bash
./scripts/run_demo.sh model.onnx
```

Without model (only face/eye detection):

```bash
./scripts/run_demo.sh
```

## Full Workflow Example

### From scratch to running demo:

```bash
# 1. Train a model
./scripts/train_from_config.sh irispupilnet/configs/example_grayscale.yaml

# 2. Export to ONNX
./scripts/export_model.sh runs/grayscale_unet_se/best.pt model.onnx

# 3. Run demo
./scripts/run_demo.sh model.onnx
```

### Fine-tune existing model:

```bash
# 1. Create custom config (or edit existing)
cp irispupilnet/configs/example_finetune.yaml myconfig.yaml
# Edit myconfig.yaml: set resume path, data_root, csv, etc.

# 2. Train
./scripts/train_from_config.sh myconfig.yaml

# 3. Export and demo
./scripts/export_model.sh runs/finetuned_model/best.pt finetuned.onnx
./scripts/run_demo.sh finetuned.onnx
```

## Advanced Usage

You can still use the Python scripts directly for more control:

### Training

```bash
python -m irispupilnet.train --config myconfig.yaml --epochs 200
```

### Export

```bash
# Auto-detect from checkpoint
python export/export_to_onnx.py --checkpoint runs/exp/best.pt --out model.onnx

# Override specific settings
python export/export_to_onnx.py \
  --checkpoint runs/exp/best.pt \
  --out model.onnx \
  --size 224 \
  --in-channels 3
```

### Demo

```bash
# Auto-detect from ONNX
python demo/webcam_demo.py --model model.onnx

# Override settings
python demo/webcam_demo.py --model model.onnx --rgb --img-size 224
```

## See Also

- [docs/CONFIG.md](../docs/CONFIG.md) - Configuration file documentation
- [docs/METRICS.md](../docs/METRICS.md) - Metrics documentation
- [irispupilnet/configs/](../irispupilnet/configs/) - Example config files
