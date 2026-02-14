# Complete Metrics Guide for IrisPupilNet

This guide provides a comprehensive overview of all metrics implemented in IrisPupilNet for evaluating iris and pupil segmentation models.

## Table of Contents

1. [Overview](#overview)
2. [Metrics Categories](#metrics-categories)
3. [Metrics During Training](#metrics-during-training)
4. [Precision-Recall Curves](#precision-recall-curves)
5. [Visualization](#visualization)
6. [Interpreting Results](#interpreting-results)

---

## Overview

IrisPupilNet computes **29 different metrics** during validation to comprehensively evaluate segmentation performance. These metrics are automatically calculated every validation epoch and saved to `metrics.csv`.

**Class mapping:**
- Class 0: Background
- Class 1: Iris
- Class 2: Pupil

---

## Metrics Categories

### 1. Overlap Metrics

Measure the similarity between predicted and ground truth regions.

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **Dice Score** | Harmonic mean of precision and recall<br>`2 * |P ∩ G| / (|P| + |G|)` | [0, 1] | Higher |
| **IoU (Jaccard)** | Intersection over union<br>`|P ∩ G| / |P ∪ G|` | [0, 1] | Higher |

Computed per class: `dice_iris`, `dice_pupil`, `dice_mean`, `iou_iris`, `iou_pupil`, `iou_mean`

**Good values:** Dice > 0.90, IoU > 0.85

---

### 2. Pixel-wise Classification

Standard classification metrics applied to pixels.

| Metric | Formula | Range | Better |
|--------|---------|-------|--------|
| **Precision** | TP / (TP + FP) | [0, 1] | Higher |
| **Recall** | TP / (TP + FN) | [0, 1] | Higher |

Computed per class: `precision_iris`, `recall_iris`, `precision_pupil`, `recall_pupil`

**Good values:** Precision > 0.90, Recall > 0.90

---

### 3. Centroid-based Metrics

Measure accuracy of structure localization.

| Metric | Description | Unit | Better |
|--------|-------------|------|--------|
| **Center Distance** | Euclidean distance between predicted and GT centers | pixels | Lower |

Computed per class: `center_dist_iris_px`, `center_dist_pupil_px`

**Good values:** < 5 pixels for pupil, < 10 pixels for iris

---

### 4. Boundary Distance Metrics

Measure boundary quality using distance transforms.

| Metric | Description | Unit | Better |
|--------|-------------|------|--------|
| **HD95** | 95th percentile of Hausdorff distance (robust to outliers) | pixels | Lower |
| **ASSD** | Average Symmetric Surface Distance | pixels | Lower |
| **MASD** | Maximum Average Surface Distance | pixels | Lower |

Computed per class: `hd95_iris`, `hd95_pupil`, `assd_iris`, `assd_pupil`, `masd_iris`, `masd_pupil`

**Good values:** HD95 < 10 pixels, ASSD < 5 pixels

---

### 5. Boundary Agreement Metrics

Measure boundary accuracy with tolerance.

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **Boundary IoU** | IoU computed on dilated boundary regions (1px tolerance) | [0, 1] | Higher |
| **NSD** | Normalized Surface Dice (% of boundary within 2px tolerance) | [0, 1] | Higher |

Computed per class: `boundary_iou_iris`, `boundary_iou_pupil`, `nsd_iris`, `nsd_pupil`

**Good values:** > 0.80 for both metrics

---

### 6. Shape Error Metrics

Measure geometric accuracy assuming circular structures.

| Metric | Formula | Unit/Range | Better |
|--------|---------|------------|--------|
| **Radius Error** | \|r_pred - r_gt\| where r = √(area/π) | pixels | Lower |
| **Area Relative Error** | \|A_pred - A_gt\| / A_gt | [0, ∞) | Lower |

Computed per class: `radius_error_iris`, `radius_error_pupil`, `area_rel_error_iris`, `area_rel_error_pupil`

**Good values:** Radius error < 5 pixels, Area error < 0.10 (10%)

---

### 7. Average Precision (AP)

Pixel-wise Average Precision computed from Precision-Recall curves.

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **AP** | Area under Precision-Recall curve | [0, 1] | Higher |
| **mAP** | Mean Average Precision (average of AP_iris and AP_pupil) | [0, 1] | Higher |

Computed: `ap_iris`, `ap_pupil`, `map`

**Good values:** mAP > 0.90

**Note:** AP computation is expensive (100 threshold evaluations per batch). It can be disabled by setting `compute_ap=False` in `compute_all_metrics()`.

---

## Metrics During Training

### Automatic Computation

All metrics are computed automatically during validation epochs:

```python
# In train.py, during validation:
batch_metrics = compute_all_metrics(logits, y, compute_ap=True, ap_num_thresholds=100)
```

### Console Output

Training progress shows key metrics:

```
epoch 01 | train 0.7857 | val 0.4771 | IoU iris:0.512 pupil:0.316 mean:0.414 | Dice iris:0.677 pupil:0.455 mean:0.566 | HD95 iris:15.23 pupil:12.45px | mAP:0.523
```

### CSV Storage

All metrics are saved to `runs/<experiment>/<timestamp>/metrics.csv`:

```csv
epoch,train_loss,val_loss,val_iou,dice_iris,dice_pupil,dice_mean,...,ap_iris,ap_pupil,map
1,0.785,0.477,0.414,0.677,0.455,0.566,...,0.534,0.512,0.523
```

---

## Precision-Recall Curves

### During Training (via AP)

AP metrics give you a summary of the PR curve (area under curve), computed during each validation epoch.

### Post-Training PR Curves

For detailed PR analysis, generate full curves from a trained model:

```bash
python tools/analyze/plot_pr_curves.py \
  --checkpoint runs/experiment/best.pt \
  --data-root data \
  --csv dataset/merged/all_datasets.csv \
  --out pr_curves/ \
  --num-thresholds 100
```

**Outputs:**
- `pr_curves_separate.png` - Individual PR curves for iris and pupil
- `pr_curves_combined.png` - Both curves on same plot with AP scores
- `pr_curves_data.npz` - Raw data (thresholds, precision, recall, F1)

**How it works:**
1. Runs model on entire validation/test set
2. Collects all pixel-level predictions and probabilities
3. Sweeps threshold from 0 to 1
4. Computes precision and recall at each threshold
5. Plots the curve and computes AP (area under curve)

---

## Visualization

### Automated Plots During Training

Plots are generated every validation epoch to `runs/<experiment>/<timestamp>/plots/metrics.png`:

- Loss curves (train and val)
- IoU per class
- Dice per class
- HD95 per class
- Center distance per class
- Best epoch marker (red line)

### Post-Training Analysis

Generate comprehensive plots from saved CSV:

```bash
# All plots
python tools/analyze/plot_metrics.py --csv runs/exp/metrics.csv --out plots/

# Combined overview only
python tools/analyze/plot_metrics.py --csv runs/exp/metrics.csv --out . --combined-only
```

**Generated plots:**
- `loss.png` - Loss curves
- `dice.png` - Dice scores
- `iou.png` - IoU scores
- `center_distance.png` - Center distances
- `hd95.png` - Hausdorff distances
- `precision_recall.png` - Precision and Recall
- `boundary_metrics.png` - Boundary-specific metrics (ASSD, MASD, NSD, Boundary IoU)
- `shape_metrics.png` - Shape errors (radius, area)
- `ap_metrics.png` - Average Precision over epochs
- `all_metrics_combined.png` - Overview of all key metrics

---

## Interpreting Results

### What makes a good model?

For iris and pupil segmentation, aim for:

| Metric Category | Target Values |
|----------------|---------------|
| **Overlap** | Dice > 0.90, IoU > 0.85 |
| **Precision/Recall** | Both > 0.90 |
| **Localization** | Center distance < 5px (pupil), < 10px (iris) |
| **Boundary** | HD95 < 10px, ASSD < 5px, NSD > 0.80 |
| **Shape** | Radius error < 5px, Area error < 10% |
| **Overall** | mAP > 0.90 |

### Metric Relationships

- **Dice ≈ F1:** Dice score is equivalent to F1 score for binary classification
- **Dice > IoU:** Dice is always higher than IoU for the same segmentation
  - Relationship: `Dice = 2*IoU / (1 + IoU)`
- **Precision vs Recall tradeoff:** Higher precision may come at cost of recall and vice versa
- **HD95 vs ASSD:** HD95 is more robust to outliers than standard Hausdorff or ASSD

### Common Issues

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| High loss, low Dice | Underfitting | Train longer, increase model capacity |
| Good Dice, poor HD95 | Rough boundaries | Check augmentation, may need boundary-focused loss |
| Good IoU, high center distance | Shifted predictions | Check alignment, may need spatial augmentation |
| High precision, low recall | Conservative predictions | Adjust decision threshold, check class balance |
| Unbalanced iris vs pupil | Class imbalance | Use weighted loss, oversample minority class |
| NaN in metrics | Empty masks in data | Check dataset, filter empty samples |

### Monitoring Training

**Early training (epochs 1-5):**
- Expect rapid improvement in all metrics
- Loss should decrease quickly
- Dice should reach ~0.5-0.7

**Mid training (epochs 5-20):**
- Metrics continue improving but slower
- Watch for overfitting (val loss increases while train loss decreases)
- Dice should reach ~0.8-0.9

**Late training (epochs 20+):**
- Metrics plateau
- Small fluctuations are normal
- Focus on validation metrics, not training

**When to stop:**
- Validation loss stops improving for 10+ epochs
- Validation metrics plateau
- Signs of overfitting (increasing gap between train and val)

---

## Metric Computation Details

### Computational Cost

Metrics are computed on CPU (except Dice/IoU which can use GPU):

| Metric Group | Relative Cost | Notes |
|--------------|---------------|-------|
| Dice, IoU | Fast | GPU tensor operations |
| Precision, Recall | Fast | GPU tensor operations |
| Center Distance | Fast | Simple centroid computation |
| HD95, ASSD, MASD | Moderate | Requires distance transforms (CPU) |
| Boundary IoU, NSD | Moderate | Morphological operations (CPU) |
| Shape Errors | Fast | Simple area/radius calculations |
| **AP (100 thresholds)** | **Expensive** | 100x precision/recall computations |

**Tip:** For faster training, you can disable AP computation:
```python
compute_all_metrics(logits, y, compute_ap=False)  # Skip AP during training
```

Then generate PR curves post-training using `plot_pr_curves.py`.

### Handling Edge Cases

- **Empty masks:** Return NaN, excluded from averaging with `np.nanmean()`
- **Perfect matches:** Return ideal values (1.0 for overlap, 0.0 for distances)
- **Single-class samples:** Only compute metrics for present classes

---

## References

- **Dice Coefficient:** Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between Species
- **IoU/Jaccard:** Jaccard, P. (1901). Distribution de la Flore Alpine
- **Hausdorff Distance:** Huttenlocher et al. (1993). Comparing Images Using the Hausdorff Distance
- **Medical Image Segmentation Metrics:** Taha & Hanbury (2015). Metrics for Evaluating 3D Medical Image Segmentation
- **Normalized Surface Dice:** Nikolov et al. (2018). Deep Learning to Achieve Clinically Applicable Segmentation

---

## Summary

IrisPupilNet provides **comprehensive multi-faceted evaluation** through:

✅ **7 categories of metrics** (29 metrics total)
✅ **Automatic computation** during training
✅ **CSV logging** for all epochs
✅ **Real-time plotting** with best epoch tracking
✅ **Post-hoc analysis** tools for detailed inspection
✅ **Precision-Recall curves** for threshold analysis

This ensures you can:
- **Track training progress** in real-time
- **Compare experiments** systematically
- **Diagnose issues** quickly
- **Report results** comprehensively
- **Optimize models** for specific applications (e.g., boundary quality vs overall overlap)
