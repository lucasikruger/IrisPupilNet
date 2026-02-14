# Analysis Tools

This directory contains tools for analyzing and visualizing model performance metrics.

## Available Tools

### 1. `plot_metrics.py` - Plot Training Metrics

Generate comprehensive visualizations from training metrics CSV files.

**Usage:**

```bash
# Generate all individual plots + combined overview
python tools/analyze/plot_metrics.py \
  --csv runs/experiment/metrics.csv \
  --out plots/experiment/

# Generate only combined overview plot
python tools/analyze/plot_metrics.py \
  --csv runs/experiment/metrics.csv \
  --out plots/ \
  --combined-only
```

**Generated Plots:**

- `loss.png` - Training and validation loss
- `dice.png` - Dice scores per class (iris, pupil, mean)
- `iou.png` - IoU scores per class
- `center_distance.png` - Center distances in pixels
- `hd95.png` - 95% Hausdorff distance
- `precision_recall.png` - Precision and Recall per class
- `boundary_metrics.png` - ASSD, MASD, NSD, Boundary IoU
- `shape_metrics.png` - Radius error and area relative error
- `ap_metrics.png` - Average Precision (AP) per class and mAP
- `all_metrics_combined.png` - Overview of all key metrics

**Metrics Included:**

The script supports all metrics logged during training:

**Overlap metrics:**
- Dice coefficient (per class)
- IoU/Jaccard index (per class)

**Pixel-wise classification:**
- Precision (per class)
- Recall (per class)

**Centroid-based:**
- Center distance in pixels

**Boundary distance:**
- HD95 (95% Hausdorff distance)
- ASSD (Average Symmetric Surface Distance)
- MASD (Maximum Average Surface Distance)

**Boundary agreement:**
- Boundary IoU (with tolerance)
- NSD (Normalized Surface Dice)

**Shape metrics:**
- Radius error (pixels)
- Area relative error

**Average Precision:**
- AP per class (iris, pupil)
- mAP (mean Average Precision)

---

### 2. `plot_pr_curves.py` - Generate Precision-Recall Curves

Generate Precision-Recall curves from a trained model by evaluating it on a dataset.

**Usage:**

```bash
python tools/analyze/plot_pr_curves.py \
  --checkpoint runs/experiment/best.pt \
  --data-root /path/to/data \
  --csv dataset/merged/all_datasets.csv \
  --out pr_curves/ \
  --split val \
  --num-thresholds 100 \
  --batch-size 16 \
  --device cuda
```

**Parameters:**

- `--checkpoint`: Path to trained model checkpoint (.pt file)
- `--data-root`: Base directory for dataset
- `--csv`: Path to dataset CSV file
- `--out`: Output directory for plots (default: `pr_curves/`)
- `--split`: Dataset split to use: train, val, or test (default: val)
- `--num-thresholds`: Number of threshold points for PR curve (default: 100)
- `--batch-size`: Batch size for inference (default: 8)
- `--device`: Device to run on: cpu or cuda (default: auto-detect)

**Generated Outputs:**

- `pr_curves_separate.png` - Separate PR curves for iris and pupil
- `pr_curves_combined.png` - Both PR curves on the same plot
- `pr_curves_data.npz` - Raw PR curve data (NumPy archive)

**PR Curve Data:**

The `.npz` file contains:
- `thresholds` - Threshold values (0 to 1)
- `precision_iris`, `recall_iris`, `f1_iris` - Iris metrics
- `precision_pupil`, `recall_pupil`, `f1_pupil` - Pupil metrics

---

### 3. `plot_mobius_summary.py` - Dataset Distribution Analysis

Analyze and visualize dataset distribution and statistics.

**Usage:**

```bash
python tools/analyze/plot_mobius_summary.py \
  --csv dataset/mobius_output/mobius.csv \
  --out plots/mobius_summary.png
```

This script generates visualizations of:
- Split distribution (train/val/test)
- Class distribution
- Subject distribution
- Image size distribution

---

## Workflow Examples

### Complete Analysis Workflow

```bash
# 1. Train a model
python -m irispupilnet.train --config irispupilnet/configs/example_grayscale.yaml

# 2. Generate training metrics plots
python tools/analyze/plot_metrics.py \
  --csv runs/grayscale_unet_se/2025-12-16_14-30-45/metrics.csv \
  --out analysis/training_plots/

# 3. Generate PR curves from best checkpoint
python tools/analyze/plot_pr_curves.py \
  --checkpoint runs/grayscale_unet_se/2025-12-16_14-30-45/best.pt \
  --data-root data \
  --csv dataset/merged/all_datasets.csv \
  --out analysis/pr_curves/ \
  --split val

# 4. Analyze dataset distribution
python tools/analyze/plot_mobius_summary.py \
  --csv dataset/merged/all_datasets.csv \
  --out analysis/dataset_distribution.png
```

### Quick Metrics Check

```bash
# Just generate the combined overview plot
python tools/analyze/plot_metrics.py \
  --csv runs/experiment/metrics.csv \
  --out . \
  --combined-only
```

### Compare Multiple Experiments

```bash
# Generate plots for each experiment
for exp in runs/*/2025-*/; do
  exp_name=$(basename $(dirname $exp))
  python tools/analyze/plot_metrics.py \
    --csv "$exp/metrics.csv" \
    --out "comparison/$exp_name/"
done
```

---

## Loading PR Curve Data in Python

You can load the saved PR curve data for custom analysis:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load PR curve data
data = np.load("pr_curves/pr_curves_data.npz")

# Access data
thresholds = data["thresholds"]
precision_iris = data["precision_iris"]
recall_iris = data["recall_iris"]

# Custom plotting
plt.plot(recall_iris, precision_iris)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Custom PR Curve")
plt.show()
```

---

## Notes

- All scripts automatically create output directories if they don't exist
- PNG plots are saved with 150 DPI for high quality
- PR curve computation can be slow for large datasets (progress bar shown)
- GPU inference is recommended for PR curve generation (`--device cuda`)
- Metrics CSV files are automatically generated during training

---

## Troubleshooting

### "CSV file not found"

Make sure the metrics CSV file exists. It should be created automatically during training at `runs/<experiment>/metrics.csv`.

### "No module named 'pandas'"

Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### PR curves script fails with "model not found"

Ensure the checkpoint path is correct and contains a valid trained model:
```bash
ls -lh runs/experiment/best.pt
```

### Memory error when generating PR curves

Reduce batch size:
```bash
python tools/analyze/plot_pr_curves.py ... --batch-size 4
```
