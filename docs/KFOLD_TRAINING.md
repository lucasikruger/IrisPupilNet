# K-Fold Cross-Validation Training

This guide explains how to use k-fold cross-validation for training and evaluating iris/pupil segmentation models.

## What is K-Fold Cross-Validation?

K-fold cross-validation is a model evaluation technique that:
1. Splits the dataset into k equal-sized folds
2. Trains k separate models, each using a different fold for validation
3. Aggregates metrics across all folds to assess model robustness

This provides a more reliable estimate of model performance than a single train/val split, especially useful for:
- Hyperparameter tuning
- Model architecture comparison
- Small dataset evaluation
- Assessing model stability

## Quick Start

### Basic Usage

```bash
python -m irispupilnet.train_kfold \
  --data-root dataset \
  --csv dataset/mobius.csv \
  --n-folds 5
```

### With Config File (Recommended)

```bash
python -m irispupilnet.train_kfold --config configs/kfold_example.yaml
```

## Configuration Options

### K-Fold Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-folds` | 5 | Number of folds for cross-validation |
| `--shuffle` | false | Shuffle data before splitting into folds |
| `--random-seed` | 42 | Random seed for reproducible fold splitting |
| `--use-val` | false | Combine train+val splits before k-fold |

### Standard Training Parameters

All standard training parameters from `train.py` are supported:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | unet_se_small | Model architecture |
| `--img-size` | 160 | Square image size |
| `--epochs` | 20 | Training epochs per fold |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--color` | false | Use RGB (true) or grayscale (false) |
| `--workers` | 2 | DataLoader workers |

### Dataset Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--data-root` | Yes | Base directory for dataset paths |
| `--csv` | Yes | CSV file with image/mask paths and splits |
| `--default-format` | mobius_3c | Default mask format |

## Example Workflows

### 1. Standard 5-Fold Cross-Validation

```bash
python -m irispupilnet.train_kfold \
  --data-root dataset \
  --csv dataset/mobius.csv \
  --n-folds 5 \
  --shuffle \
  --epochs 30 \
  --batch-size 32
```

### 2. Using Train + Val Combined

If you want to use both train and validation splits for k-fold:

```bash
python -m irispupilnet.train_kfold \
  --data-root dataset \
  --csv dataset/mobius.csv \
  --n-folds 5 \
  --use-val \
  --shuffle
```

This combines train+val before splitting, useful when you want to maximize training data.

### 3. Hyperparameter Search with K-Fold

```bash
# Test different learning rates
for lr in 1e-4 5e-4 1e-3; do
  python -m irispupilnet.train_kfold \
    --data-root dataset \
    --csv dataset/mobius.csv \
    --n-folds 3 \
    --lr $lr \
    --out runs/lr_search_${lr}
done
```

### 4. Model Architecture Comparison

```bash
# Compare different models
for model in unet_se_small seresnext_unet unet_base; do
  python -m irispupilnet.train_kfold \
    --data-root dataset \
    --csv dataset/mobius.csv \
    --n-folds 5 \
    --model $model \
    --out runs/model_comparison_${model}
done
```

## Output Structure

K-fold creates a timestamped directory with results:

```
runs/kfold_experiment/kfold_5_2025-12-16_14-30-45/
├── config.yaml              # Configuration used
├── kfold_summary.yaml       # Aggregated metrics across folds
├── kfold_comparison.png     # Bar plots comparing folds
├── fold_1/
│   ├── best.pt              # Best model for fold 1
│   └── metrics.yaml         # Final metrics for fold 1
├── fold_2/
│   ├── best.pt
│   └── metrics.yaml
├── fold_3/
│   ├── best.pt
│   └── metrics.yaml
├── fold_4/
│   ├── best.pt
│   └── metrics.yaml
└── fold_5/
    ├── best.pt
    └── metrics.yaml
```

## Understanding Results

### Summary Metrics (kfold_summary.yaml)

The summary file contains aggregated statistics for each metric:

```yaml
val_iou_mean: 0.9245
val_iou_std: 0.0123
val_iou_min: 0.9087
val_iou_max: 0.9401

dice_mean_mean: 0.9589
dice_mean_std: 0.0089
...
```

**Interpretation:**
- `*_mean`: Average across all folds (primary metric)
- `*_std`: Standard deviation (lower = more stable)
- `*_min` / `*_max`: Range of values across folds

### Comparison Plot (kfold_comparison.png)

Bar plots showing key metrics for each fold:
- Validation IoU
- Dice scores (iris, pupil, mean)
- mAP (mean Average Precision)
- HD95 (Hausdorff Distance 95th percentile)

Red dashed line indicates mean across folds.

### Individual Fold Results (fold_N/metrics.yaml)

Each fold's best validation metrics:

```yaml
fold: 1
best_epoch: 18
best_iou: 0.9287
val_iou: 0.9287
dice_mean: 0.9612
map: 0.8945
...
```

## Best Practices

### Choosing Number of Folds

- **3-5 folds**: Good balance, faster training
- **5-10 folds**: More robust evaluation, slower
- **10+ folds**: For very small datasets (leave-one-out)

**Rule of thumb:** Use 5 folds unless you have specific reasons for more/less.

### When to Use `--use-val`

Use `--use-val` when:
- You want maximum training data in each fold
- You don't have a separate test set
- You're doing final model selection

Don't use `--use-val` when:
- You have a fixed validation set you want to preserve
- You're doing preliminary experiments

### Shuffle Recommendations

- **Use `--shuffle`** for most cases (ensures random distribution)
- **Don't shuffle** if data has temporal ordering you want to preserve
- Always set `--random-seed` for reproducibility

### Managing Training Time

K-fold trains k separate models, so total time = single training × k.

**Strategies to reduce time:**
1. Reduce `--epochs` (e.g., 15-20 instead of 50)
2. Use fewer folds (3 instead of 5)
3. Reduce `--img-size` for initial experiments
4. Use faster model (`unet_se_small` vs `seresnext_unet`)

## Example Config File

Create `configs/my_kfold.yaml`:

```yaml
# Dataset
dataset: csv_seg
data_root: dataset
csv: dataset/mobius.csv
default_format: mobius_3c
use_val: false

# K-Fold
n_folds: 5
shuffle: true
random_seed: 42

# Model
model: unet_se_small
img_size: 160
num_classes: 3
base: 32
color: false

# Training
batch_size: 32
epochs: 20
lr: 0.001
weight_decay: 0.0001
workers: 4

# Output
out: runs/kfold_mobius
```

Run with:
```bash
python -m irispupilnet.train_kfold --config configs/my_kfold.yaml
```

## Analyzing Results

### 1. Check Summary

```bash
cat runs/kfold_experiment/kfold_5_YYYY-MM-DD_HH-MM-SS/kfold_summary.yaml
```

Look for:
- High `val_iou_mean` (target: > 0.90)
- Low `val_iou_std` (target: < 0.02) → stable model
- Consistent `dice_mean` across folds

### 2. Visualize Fold Comparison

```bash
open runs/kfold_experiment/kfold_5_YYYY-MM-DD_HH-MM-SS/kfold_comparison.png
```

Check for:
- Bars close to mean line (low variance)
- No outlier folds (all bars similar height)
- High absolute values for IoU/Dice, low for HD95

### 3. Identify Best Fold

```bash
# Find fold with highest IoU
grep "best_iou" runs/kfold_experiment/kfold_5_*/fold_*/metrics.yaml
```

Use the best fold's model for deployment:
```bash
cp runs/kfold_experiment/kfold_5_*/fold_3/best.pt models/best_model.pt
```

## Troubleshooting

### "No samples found for split=train"

- Check that your CSV has rows with `split=train`
- Verify `--data-root` points to correct directory
- Check file paths exist: `dataset_base_dir / rel_image_path`

### High variance across folds

**Possible causes:**
- Small dataset (use more augmentation)
- Data imbalance (check class distribution)
- Model too complex (reduce capacity)
- Learning rate too high (reduce `--lr`)

**Solutions:**
- Increase `--epochs` for better convergence
- Add more data augmentation
- Try different `--random-seed` to verify randomness
- Use `--shuffle` to ensure random fold assignment

### Out of memory

**Solutions:**
- Reduce `--batch-size`
- Reduce `--img-size`
- Reduce `--workers`
- Train folds sequentially (this is default)

### Very slow training

**Solutions:**
- Check `--workers` (try 4-8)
- Ensure GPU is being used (check console output)
- Reduce `--img-size` for experiments
- Use grayscale (`--color false`)

## Comparison with Standard Training

| Aspect | Standard Training | K-Fold Training |
|--------|-------------------|-----------------|
| **Time** | 1× | k× (trains k models) |
| **Evaluation** | Single train/val split | k different splits |
| **Robustness** | May overfit to val set | More reliable estimate |
| **Output** | 1 best model | k models + aggregated metrics |
| **Use Case** | Final training | Model evaluation & selection |

**Recommendation:** Use k-fold for evaluation/comparison, then retrain best configuration on full dataset with standard training.

## Advanced Usage

### Custom Fold Assignment

If you need custom control over fold assignment, you can:
1. Pre-assign fold numbers in your CSV
2. Create k separate CSVs manually
3. Run standard `train.py` k times with different CSVs

### Combining Results

After k-fold, you might want to:

**Option 1: Use best fold's model**
```bash
# Find and use the fold with highest IoU
```

**Option 2: Ensemble prediction** (not implemented, but possible)
```python
# Load all k models and average predictions
```

**Option 3: Retrain on full dataset**
```bash
# Use k-fold results to pick best hyperparameters
# Then train on full train+val with standard train.py
```

## References

- Main training script: `irispupilnet/train.py`
- K-fold script: `irispupilnet/train_kfold.py`
- Example config: `configs/kfold_example.yaml`
- Metrics documentation: `docs/COMPLETE_METRICS_GUIDE.md`
