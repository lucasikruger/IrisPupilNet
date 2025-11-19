# Configuration Files

IrisPupilNet supports YAML configuration files for easier experiment management and reproducibility.

## Quick Start

### Using a config file

```bash
python -m irispupilnet.train --config irispupilnet/configs/example_grayscale.yaml
```

### Overriding config values via CLI

Command-line arguments always take precedence over config file values:

```bash
# Use config but override epochs and learning rate
python -m irispupilnet.train \
  --config irispupilnet/configs/example_grayscale.yaml \
  --epochs 100 \
  --lr 5e-4
```

### Traditional CLI-only (no config file)

```bash
python -m irispupilnet.train \
  --data-root /path/to/data \
  --csv dataset.csv \
  --model unet_se_small \
  --epochs 50
```

## Configuration Parameters

### Dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | `csv_seg` | Dataset type (always use `csv_seg`) |
| `data_root` | str | **required** | Base directory for dataset |
| `csv` | str | **required** | Path to CSV file with dataset splits |
| `default_format` | str | `mobius_3c` | Mask format converter (mobius_3c, tayed_3c, etc.) |

### Model Architecture

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `unet_se_small` | Model architecture (unet_se_small, seresnext_unet) |
| `num_classes` | int | `3` | Number of classes (3: bg/iris/pupil, 2: iris/pupil) |
| `base` | int | `32` | Base channel count for model |
| `in_channels` | int | `3` | Input channels (1=grayscale, 3=RGB) |
| `color` | bool | `false` | If true, use RGB; if false, convert to grayscale |

### Input/Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img_size` | int | `160` | Input image size (for square: H=W) |
| `img_width` | int | `null` | Image width (if different from height, overrides img_size) |
| `img_height` | int | `null` | Image height (if different from width, overrides img_size) |
| `out` | str | `runs/mobius_unet_se` | Base output directory (timestamped subfolder created automatically) |

**Output directory structure:**
- When you specify `out: runs/my_experiment`, the system automatically creates a timestamped subfolder like `runs/my_experiment/2025-11-19_14-30-45/`
- This allows you to run multiple experiments with the same base configuration without overwriting results
- Inside each run folder:
  - `config.yaml` - Copy of the configuration used for this run
  - `metrics.csv` - Epoch-by-epoch metrics
  - `best.pt`, `checkpoint_epoch_XXX.pt` - Model checkpoints
  - `best_metrics.yaml` - Best epoch metrics (if `save_best_metrics: true`)
  - `plots/` - Subfolder containing all visualizations
    - `metrics.png` - Training curves (Loss, IoU, Dice, HD95)
    - `examples_epoch_XXX.png` - Validation examples (if `show_examples > 0`)

**Note on image dimensions:** Currently, the dataset loader requires square images. Support for `img_width` != `img_height` is prepared but not yet fully implemented. If you specify different width/height, the system will warn you and use `img_size`.

### Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | `20` | Total number of training epochs |
| `batch_size` | int | `32` | Batch size for training |
| `lr` | float | `1e-3` | Learning rate for AdamW optimizer |
| `weight_decay` | float | `1e-4` | Weight decay for AdamW |
| `workers` | int | `2` | Number of data loading workers |

### Validation & Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `val_every` | int | `1` | Run validation every N epochs |
| `save_every` | int | `0` | Save checkpoint every N epochs (0 = only best) |
| `resume` | str | `null` | Path to checkpoint to resume from (null = start fresh) |

### Metrics & Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics_csv` | str | `null` | Custom path for metrics CSV (null = auto: out/metrics.csv) |
| `metrics_plot` | str | `null` | Custom path for metrics plot (null = auto: out/plots/metrics.png) |
| `log_every` | int | `0` | Log training progress every N batches (0 = disable batch logging) |
| `save_best_metrics` | bool | `false` | Save best_metrics.yaml with best epoch results |
| `tensorboard` | bool | `false` | Enable TensorBoard logging (requires tensorboard package) |

**Logging options explained:**
- **`log_every`**: Set to a positive number (e.g., 50) to see batch-level progress during training. Useful for debugging or monitoring long training runs. Set to 0 to only see epoch-level summaries.
- **`save_best_metrics`**: When enabled, saves a separate YAML file with all metrics from the best epoch. Useful for quick reference and experiment comparison.
- **`tensorboard`**: Enables TensorBoard integration. You can then view training curves in real-time with `tensorboard --logdir runs/`
- **Plots are now generated every validation epoch** and saved to the `plots/` subfolder automatically

### Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `show_examples` | int | `0` | Number of random validation examples to visualize (0 = disable) |
| `show_examples_every` | int | `1` | Generate examples visualization every N epochs |

**Visualization options explained:**
- **`show_examples`**: When set to a positive number (e.g., 4), generates a visualization showing N random validation samples with their images, ground truth masks, and predicted masks side-by-side. Saved as `plots/examples_epoch_XXX.png`.
- **`show_examples_every`**: Controls how frequently to generate visualizations. Set to 1 to visualize every validation epoch, or higher values to reduce disk usage for long training runs.

**Metrics Display:**
The training now shows per-class metrics during training:
```
epoch 01 | train 0.7857 | val 0.4771 | IoU iris:0.512 pupil:0.316 mean:0.414 | Dice iris:0.677 pupil:0.455 mean:0.566 | HD95 iris:15.23 pupil:12.45px
```
- **IoU (Intersection over Union)**: Shown per class (iris, pupil) and mean
- **Dice Score**: Shown per class (iris, pupil) and mean
- **HD95 (95% Hausdorff Distance)**: Shown per class (iris, pupil) in pixels
- All metrics (including center distance) are saved to `metrics.csv` for detailed analysis

## Example Configurations

### 1. Grayscale Training (Fast, Recommended)

File: [`irispupilnet/configs/example_grayscale.yaml`](../irispupilnet/configs/example_grayscale.yaml)

```yaml
model: unet_se_small
num_classes: 3
in_channels: 1
color: false
epochs: 50
batch_size: 32
lr: 1.0e-3
val_every: 1
save_every: 10
```

**Use case:** Standard training with grayscale images, faster and uses less memory.

**Run:**
```bash
python -m irispupilnet.train --config irispupilnet/configs/example_grayscale.yaml
```

### 2. RGB Training with Larger Model

File: [`irispupilnet/configs/example_rgb.yaml`](../irispupilnet/configs/example_rgb.yaml)

```yaml
model: seresnext_unet
num_classes: 3
in_channels: 3
color: true
epochs: 100
batch_size: 16  # Smaller batch for larger model
lr: 5.0e-4      # Lower LR
val_every: 2
save_every: 20
```

**Use case:** Training with RGB images and larger SEResNext-UNet model for potentially better accuracy.

**Run:**
```bash
python -m irispupilnet.train --config irispupilnet/configs/example_rgb.yaml
```

### 3. Fine-tuning from Pretrained Model

File: [`irispupilnet/configs/example_finetune.yaml`](../irispupilnet/configs/example_finetune.yaml)

```yaml
model: unet_se_small
epochs: 30
lr: 1.0e-4      # Lower LR for fine-tuning
resume: runs/pretrained_model/best.pt
```

**Use case:** Continue training from a pretrained checkpoint, useful for transfer learning or training on new data.

**Run:**
```bash
python -m irispupilnet.train --config irispupilnet/configs/example_finetune.yaml
```

## Creating Your Own Config

1. Copy an example config:
```bash
cp irispupilnet/configs/example_grayscale.yaml irispupilnet/configs/my_experiment.yaml
```

2. Edit the parameters:
```yaml
# my_experiment.yaml
data_root: /my/data/path
csv: dataset/my_dataset.csv
out: runs/my_experiment
epochs: 100
lr: 1.0e-3
```

3. Run training:
```bash
python -m irispupilnet.train --config irispupilnet/configs/my_experiment.yaml
```

## Resume Training

To resume from a checkpoint:

### Method 1: In config file

```yaml
resume: runs/experiment/best.pt
```

### Method 2: Via CLI

```bash
python -m irispupilnet.train \
  --config myconfig.yaml \
  --resume runs/experiment/checkpoint_epoch_050.pt
```

When resuming:
- Model weights and optimizer state are restored
- Training continues from the next epoch
- Best IoU metric is preserved

## Checkpoint Format

Checkpoints saved include:

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": current_epoch,
    "best_iou": best_validation_iou,
    "args": training_arguments
}
```

### Checkpoint Types

1. **Best checkpoint** (`best.pt`): Saved when validation IoU improves
2. **Periodic checkpoints** (`checkpoint_epoch_XXX.pt`): Saved every `save_every` epochs
3. **Best metrics** (`best_metrics.yaml`): Human-readable metrics from best epoch (if `save_best_metrics: true`)

### Best Metrics YAML Example

When `save_best_metrics: true`, a `best_metrics.yaml` file is saved with the best epoch's results:

```yaml
epoch: 42
train_loss: 0.087432
val_loss: 0.092145
val_iou: 0.923456
dice_iris: 0.945123
dice_pupil: 0.938765
dice_mean: 0.941944
iou_iris: 0.897234
iou_pupil: 0.884567
iou_mean: 0.890901
center_dist_iris_px: 2.345678
center_dist_pupil_px: 1.123456
hd95_iris: 5.678901
hd95_pupil: 3.456789
```

This file is useful for:
- Quick comparison between experiments
- Reporting final results
- Automated experiment tracking systems

## Tips

### Memory Optimization

- Use grayscale (`color: false`) to reduce memory by ~3x
- Reduce `batch_size` if you run out of GPU memory
- Reduce `workers` if CPU memory is limited

### Training Speed

- Increase `val_every` to skip some validation epochs (faster but less monitoring)
- Use more `workers` for faster data loading (if you have CPU cores available)
- Use `save_every: 0` to only save best checkpoint (saves disk I/O)

### Experiment Tracking

- Create a separate config file for each experiment
- Use descriptive `out` paths: `runs/2024-01-15-rgb-seresnext`
- Keep config files in version control

### Best Practices

```yaml
# Good: Descriptive output path with date
out: runs/2024-01-15-grayscale-mobius-unet

# Good: Conservative save_every for long training
save_every: 20  # For 100+ epoch training

# Good: Validate frequently at start, less later
val_every: 1  # Can change to 2-5 for very long training
```

## Command-Line Reference

### Priority Rules

1. **CLI arguments** (highest priority)
2. **Config file values**
3. **Argparse defaults** (lowest priority)

### Example: Mixed Usage

```bash
# Config file has: epochs=50, lr=1e-3
# This will use: epochs=100 (CLI override), lr=1e-3 (from config)
python -m irispupilnet.train \
  --config myconfig.yaml \
  --epochs 100
```

## Troubleshooting

### "data-root is required"

Make sure your config file or CLI has `data_root` set:
```yaml
data_root: /path/to/your/data
```

### "csv is required"

Make sure your config file or CLI has `csv` set:
```yaml
csv: dataset/merged/all_datasets.csv
```

### Resume checkpoint not found

Check the path in your config:
```yaml
resume: runs/experiment/best.pt  # Must exist!
```

Or use `resume: null` to start fresh.

### Config file not found

Check the path:
```bash
ls irispupilnet/configs/your_config.yaml
```
