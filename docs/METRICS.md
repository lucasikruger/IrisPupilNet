# Segmentation Metrics

This document describes the comprehensive metrics used to evaluate iris and pupil segmentation models in IrisPupilNet.

## Overview

The training pipeline now computes multiple metrics beyond the basic IoU:

- **Dice Score** (per class): Measures overlap between prediction and ground truth
- **IoU/Jaccard Index** (per class): Another overlap metric
- **Center Distance**: Euclidean distance between predicted and GT centers
- **HD95**: 95% Hausdorff distance, a robust boundary distance metric

## Class Mapping

- Class 0: Background
- Class 1: Iris
- Class 2: Pupil

## Metrics Details

### 1. Dice Score (per class)

Measures the overlap between prediction and ground truth.

**Formula:**
```
Dice_k = 2 * |P_k ∩ G_k| / (|P_k| + |G_k| + ε)
```

Where:
- `P_k`: set of pixels predicted as class k
- `G_k`: set of pixels where GT is class k
- `ε`: small constant for numerical stability

**Range:** [0, 1] (1 = perfect match)

**Logged metrics:**
- `dice_iris`: Dice score for iris
- `dice_pupil`: Dice score for pupil
- `dice_mean`: Average of iris and pupil Dice

### 2. IoU (Jaccard Index, per class)

Another common overlap metric, slightly more strict than Dice.

**Formula:**
```
IoU_k = |P_k ∩ G_k| / |P_k ∪ G_k| + ε
```

**Range:** [0, 1] (1 = perfect match)

**Logged metrics:**
- `iou_iris`: IoU for iris
- `iou_pupil`: IoU for pupil
- `iou_mean`: Average of iris and pupil IoU

### 3. Center Distance

Measures how accurately the model locates the center of each structure.

**Process:**
1. For each class k (iris, pupil), extract binary mask
2. Compute center of mass: `c_k = (x_k, y_k) = (1/N Σ x_i, 1/N Σ y_i)`
3. Compute Euclidean distance: `d_k = sqrt((x_k^pred - x_k^gt)^2 + (y_k^pred - y_k^gt)^2)`

**Logged metrics:**
- `center_dist_iris_px`: Center distance for iris (in pixels)
- `center_dist_pupil_px`: Center distance for pupil (in pixels)

**Interpretation:** Lower is better. 0 pixels = perfect center alignment.

### 4. HD95 (95% Hausdorff Distance)

Measures boundary quality by computing distances between predicted and GT boundaries.

**Why HD95?**
- Standard Hausdorff (max distance) is too sensitive to single outliers
- HD95 takes the 95th percentile, making it more robust
- Standard metric in medical image segmentation

**Process:**
1. Extract boundary pixels from binary masks via morphological gradient
2. For each boundary pixel in prediction, find distance to nearest GT boundary pixel
3. For each boundary pixel in GT, find distance to nearest predicted boundary pixel
4. Take 95th percentile of all distances

**Logged metrics:**
- `hd95_iris`: HD95 for iris boundary (in pixels)
- `hd95_pupil`: HD95 for pupil boundary (in pixels)

**Interpretation:** Lower is better. Smaller values indicate better boundary alignment.

## Usage

### During Training

Metrics are automatically computed during validation and logged to:
- **Console output:** Shows summary metrics (Dice mean, HD95 pupil) per epoch
- **CSV file:** `runs/<experiment>/metrics.csv` contains all metrics
- **PNG plot:** `runs/<experiment>/metrics.png` visualizes key metrics

Example training command:
```bash
python -m irispupilnet.train \
  --data-root /path/to/data \
  --csv dataset.csv \
  --epochs 50 \
  --val-every 1
```

### Standalone Plotting

Generate detailed plots from saved metrics CSV:

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

This creates:
- `loss.png`: Training and validation loss
- `dice.png`: Dice scores per class
- `iou.png`: IoU per class
- `center_distance.png`: Center distances
- `hd95.png`: HD95 distances
- `all_metrics_combined.png`: All metrics in one figure

## Metrics CSV Format

The `metrics.csv` file contains the following columns:

```
epoch,train_loss,val_loss,val_iou,dice_iris,dice_pupil,dice_mean,iou_iris,iou_pupil,iou_mean,center_dist_iris_px,center_dist_pupil_px,hd95_iris,hd95_pupil
```

## Implementation Details

All metrics are implemented in `irispupilnet/utils/segmentation_metrics.py`:

- **Dice and IoU:** Computed using one-hot encoding and tensor operations (fast on GPU)
- **Center distance:** Uses PyTorch's `nonzero` for efficiency
- **HD95:** Uses scipy's `distance_transform_edt` for fast distance computation

### Performance Considerations

- HD95 computation requires converting tensors to numpy (runs on CPU)
- For large batches, metrics add ~10-20% overhead to validation time
- All metrics use batch averaging for consistent reporting

## Interpretation Guidelines

### Good Model Performance

- **Dice > 0.90** for both iris and pupil
- **IoU > 0.85** for both classes
- **Center distance < 5 pixels** for pupil, < 10 pixels for iris
- **HD95 < 10 pixels** for both classes

### Model Comparison

When comparing models:
1. **Primary metric:** Dice mean or IoU mean (overall segmentation quality)
2. **Secondary metrics:**
   - Center distance (important for gaze estimation applications)
   - HD95 (important for precise boundary requirements)

### Expected Trends During Training

- Loss, center distance, and HD95 should **decrease**
- Dice and IoU should **increase**
- Metrics may plateau after ~15-20 epochs depending on dataset size
- HD95 is typically noisier than Dice/IoU across epochs

## References

- Dice coefficient: Dice, L. R. (1945). "Measures of the amount of ecologic association between species"
- IoU/Jaccard: Jaccard, P. (1901). "Distribution de la flore alpine dans le bassin des Dranses et dans quelques régions voisines"
- Hausdorff distance: Huttenlocher et al. (1993). "Comparing images using the Hausdorff distance"
- HD95 in medical imaging: Taha & Hanbury (2015). "Metrics for evaluating 3D medical image segmentation"
