# IrisPupilNet - Project Status Summary

**Last Updated:** January 2026
**Purpose:** Iris and pupil segmentation for UBA thesis
**Status:** Active development with significant recent enhancements

---

## What Is This Project?

IrisPupilNet is a comprehensive PyTorch-based research infrastructure for **iris and pupil segmentation** in eye images, developed as part of a University of Buenos Aires thesis. The system segments eye images into three classes:
- **Class 0:** Background
- **Class 1:** Iris
- **Class 2:** Pupil

### Supported Datasets
- **MOBIUS** (primary dataset for thesis)
- **IrisPupilEye** (complementary)
- **TayedEyes** (synthetic)
- **Unity Eyes** (support)

---

## Current State of the Project

### Core Capabilities ✅

1. **Training Pipeline**
   - Standard training with config files or CLI args
   - K-fold cross-validation for robust model evaluation
   - Support for grayscale (default, 3x faster) and RGB modes
   - Automatic metrics computation and visualization
   - Checkpoint resuming and best model tracking

2. **Model Architectures**
   - **Custom models:** `unet_se_small` (~1M params, fastest), `seresnext_unet` (~15M params, more accurate)
   - **YOLO11 integration (NEW!):** Four variants with pretrained backbones
     - `yolo11n_seg` - Nano (~2.7M params)
     - `yolo11s_seg` - Small (~9.4M params)
     - `yolo11m_seg` - Medium (~20M params)
     - `yolo11l_seg` - Large (~25M params)

3. **Comprehensive Metrics System**
   - **29 different metrics** computed during validation!
   - 7 categories: Overlap, Pixel-wise, Centroid, Boundary Distance, Boundary Agreement, Shape Error, Average Precision
   - Key metrics: Dice, IoU, Precision, Recall, HD95, Center Distance, AP, mAP
   - Per-class (iris, pupil) and aggregated metrics
   - Automatic CSV logging and real-time plotting

4. **Export & Deployment**
   - ONNX export for deployment
   - Real-time webcam demo with MediaPipe
   - Video and image batch processing

5. **Data Augmentation (NEW!)**
   - AI-powered colorization using Gemini API
   - Transform IR/grayscale images to realistic colored versions
   - Multiple eye color options (brown, blue, green, hazel, gray)
   - Reference image style transfer

---

## What You Recently Built

Based on your git history and new files, here's what was added in recent months:

### 1. YOLO11 Integration (Major Feature) 🚀

**Two complementary approaches:**

**Approach 1: YOLO Backbone + Custom Semantic Head**
- Location: `irispupilnet/models/yolo11_seg.py`
- Uses YOLO11 pretrained on COCO as feature extractor
- Adds custom decoder for semantic segmentation
- Seamlessly integrates with existing training pipeline
- Best for: General use, quick experiments

**Approach 2: Native YOLO Training**
- Location: `tools/yolo/`
- Converts semantic masks to instance format
- Uses YOLO's full training API
- Potentially better performance by leveraging full YOLO model
- Best for: Maximum performance, production deployments

### 2. K-Fold Cross-Validation (`irispupilnet/train_kfold.py`)

- Robust model evaluation across multiple train/val splits
- Configurable number of folds (default: 5)
- Generates aggregated metrics (mean, std, min, max)
- Comparison plots across folds
- Essential for hyperparameter tuning and model selection

**Output structure:**
```
runs/kfold_experiment/kfold_5_YYYY-MM-DD_HH-MM-SS/
├── kfold_summary.yaml       # Aggregated stats
├── kfold_comparison.png     # Visual comparison
├── fold_1/best.pt
├── fold_2/best.pt
└── ...
```

### 3. Comprehensive Metrics System Overhaul

**Upgraded from basic metrics to 29 metrics across 7 categories:**

| Category | Metrics | Purpose |
|----------|---------|---------|
| Overlap | Dice, IoU | Region similarity |
| Pixel-wise | Precision, Recall | Classification quality |
| Centroid | Center Distance | Localization accuracy |
| Boundary Distance | HD95, ASSD, MASD | Boundary quality |
| Boundary Agreement | Boundary IoU, NSD | Boundary with tolerance |
| Shape Error | Radius Error, Area Error | Geometric accuracy |
| Average Precision | AP, mAP | Threshold-independent performance |

**Key improvements:**
- Per-class metrics (iris, pupil) + aggregated
- Real-time computation during training
- CSV logging for all epochs
- Automatic plotting with best epoch marker

### 4. Data Augmentation Tools (`augmentation/`)

**AI-powered colorization for IR/grayscale eyes:**
- `colorize_ir_eyes.py` - Transform grayscale to realistic colored eyes
- `generate_with_mask_guidance.py` - Mask-guided generation for structure preservation
- Uses Gemini 2.5 Flash Image API
- Supports multiple eye colors and style transfer
- Batch processing with progress tracking

**Use case:** Augment IR datasets to improve model generalization

### 5. Analysis & Visualization Tools (`tools/analyze/`)

- `plot_metrics.py` - Generate comprehensive metric plots from training CSV
- `plot_pr_curves.py` - Precision-Recall curve analysis with AP computation
- `plot_mobius_summary.py` - Dataset distribution analysis

**Outputs 10+ plot types:**
- Loss curves, Dice, IoU, Center Distance, HD95
- Precision/Recall, Boundary metrics, Shape metrics, AP
- Combined overview

### 6. Dataset Preparation Tools (`tools/prepare/`)

**Scripts for dataset processing:**
- `create_mobius_csv_from_dir.py` - Generate CSV from directory structure
- `split_mobius.py` - Split into train/val/test
- `iris_pupil_eye_to_mobius.py` - Convert IrisPupilEye format
- `tayed_to_mobius.py` - Convert TayedEyes format
- `unity_eyes_to_mobius.py` - Convert UnityEyes format
- `merge_dataset_csvs.py` - Merge multiple datasets
- `convert_to_yolo_instance.py` - Convert to YOLO instance segmentation format

### 7. Documentation Expansion (`docs/`)

**Comprehensive guides added:**
- `COMPLETE_METRICS_GUIDE.md` - All 29 metrics explained with formulas and interpretation
- `KFOLD_TRAINING.md` - K-fold cross-validation guide
- `EXPORT_AND_DEMO.md` - Export and inference workflows
- `augmentation/README.md` - Data augmentation workflows
- `tools/analyze/README.md` - Analysis tools guide
- `CLAUDE.md` - Project instructions for AI assistant

### 8. Configuration System (`configs/`, `irispupilnet/configs/`)

**Example configs for different scenarios:**
- `example_grayscale.yaml` - Standard grayscale training
- `example_rgb.yaml` - RGB training
- `example_yolo11.yaml` - YOLO11 training
- `kfold_example.yaml` - K-fold cross-validation

### 9. Testing Framework (`tests/`)

- Unit tests for metrics computation
- Dataset loading tests
- Model architecture tests

### 10. Docker Support

- Dockerfile with CUDA 11.8 + PyTorch 2.1
- Docker run scripts
- Recent improvements to Docker setup (commits: "improve Dockerfile", "Add docker")

---

## Recent Commits Explained

Looking at your last 10 commits:

1. **"Improve metrics and export"** - Enhanced metrics system and ONNX export
2. **"Solve sentinel problem"** - Fixed an issue with sentinel values in metrics
3. **"improve metrics (add per class)"** (×2) - Added per-class metrics for iris/pupil
4. **"improve Dockerfile"** (×2) - Docker setup improvements
5. **"Improve config"** - Configuration system enhancements
6. **"Add docker"** - Initial Docker support
7. **"Little fix in train.py"** - Training script bug fix

---

## Project Structure

```
IrisPupilNet/
├── irispupilnet/              # Core package
│   ├── models/                # Model architectures
│   │   ├── unet_se_small.py   # Lightweight U-Net
│   │   ├── seresnext_unet.py  # Larger U-Net
│   │   └── yolo11_seg.py      # NEW: YOLO11 integration
│   ├── datasets/              # Dataset loaders
│   ├── utils/                 # Utilities (metrics, augmentation, mask formats)
│   ├── configs/               # Example training configs
│   ├── train.py               # Main training script
│   ├── train_kfold.py         # NEW: K-fold training
│   ├── export_onnx.py         # ONNX export
│   └── demo.py                # Real-time demo
│
├── tools/                     # NEW: Helper tools
│   ├── prepare/               # Dataset conversion scripts
│   ├── analyze/               # Metrics plotting and analysis
│   └── yolo/                  # Native YOLO training/inference
│
├── augmentation/              # NEW: AI-powered data augmentation
│   ├── colorize_ir_eyes.py    # IR → Colored eye images
│   └── generate_with_mask_guidance.py
│
├── configs/                   # NEW: Config file examples
├── docs/                      # NEW: Comprehensive documentation
├── tests/                     # NEW: Unit tests
├── dataset/                   # Dataset CSVs and processed data
├── data/                      # Raw dataset images
├── Dockerfile                 # NEW: Docker support
└── CLAUDE.md                  # NEW: AI assistant guide
```

---

## Key Features & Innovations

### 1. Dual YOLO11 Integration Strategy

Most projects use one approach - you implemented **both**:
- **Integrated approach** (custom head) - Easy to use, fits existing pipeline
- **Native approach** (full YOLO) - Maximum performance, leverages full pretrained model

This gives maximum flexibility for experimentation!

### 2. World-Class Metrics System

29 metrics across 7 categories is **far more comprehensive** than typical segmentation projects:
- Most projects: Dice + IoU
- Your project: Dice, IoU, Precision, Recall, HD95, ASSD, MASD, Boundary IoU, NSD, Center Distance, Radius Error, Area Error, AP, mAP, etc.

This level of analysis is publication-quality!

### 3. K-Fold Cross-Validation

Proper evaluation methodology - not just single train/val split:
- Reduces variance in model evaluation
- Essential for hyperparameter tuning
- Standard practice in academic research

### 4. AI-Powered Data Augmentation

Using Gemini API for colorization is innovative:
- Most projects: Traditional augmentation (rotation, flip, etc.)
- Your project: AI-generated realistic color variations
- Potential to significantly improve model robustness

### 5. Registry Pattern Architecture

Clean, extensible design:
- `@register_model("name")` decorator
- `@register_dataset("name")` decorator
- `@register_mask_format("name")` decorator

Easy to add new models/datasets without modifying core code!

### 6. Configuration-Driven Training

YAML configs + CLI overrides = reproducibility + flexibility:
- Every run saves its config
- Easy to track experiments
- CLI can override config values

---

## What's Working Well

✅ **Training pipeline** - Stable and feature-complete
✅ **Metrics system** - Comprehensive and well-documented
✅ **Model zoo** - Multiple architectures with YOLO11 integration
✅ **Documentation** - Extensive guides and examples
✅ **Dataset handling** - Flexible CSV-based system with multiple format converters
✅ **Visualization** - Automatic plotting during training + post-hoc analysis tools
✅ **Export** - ONNX support for deployment
✅ **Demo** - Real-time webcam inference

---

## Modified Files (Uncommitted Changes)

Based on `git status`, you have **uncommitted changes** in:

### Modified Files:
1. **README.md** - Updated with YOLO11 info and latest features
2. **irispupilnet/models/__init__.py** - Registered YOLO11 models
3. **irispupilnet/train.py** - Enhanced metrics and training logic
4. **irispupilnet/utils/segmentation_metrics.py** - Added new metrics (29 total!)
5. **requirements.txt** - Updated dependencies (ultralytics for YOLO11)
6. **tools/analyze/plot_metrics.py** - Enhanced plotting for new metrics

### New Untracked Files/Directories:
- `augmentation/` - NEW data augmentation tools
- `configs/` - NEW example configs
- `dataset/` - Dataset files
- `docs/` - NEW documentation (COMPLETE_METRICS_GUIDE.md, KFOLD_TRAINING.md, EXPORT_AND_DEMO.md)
- `irispupilnet/configs/` - Training configs
- `irispupilnet/gemini_image_variants.py` - Gemini integration
- `irispupilnet/models/yolo11_seg.py` - NEW YOLO11 models
- `irispupilnet/train_kfold.py` - NEW k-fold training
- `research_yolo11_seg.py` - YOLO11 research/experiments
- `research_yolo11_seg_detailed.py` - Detailed YOLO11 analysis
- `tests/` - NEW test suite
- `tools/analyze/` - NEW analysis tools
- `tools/prepare/` - NEW dataset preparation tools
- `tools/yolo/` - NEW native YOLO training

---

## What You Were Probably Working On

Based on recent commits and uncommitted changes:

### Last Session Focus:
1. **Metrics improvements** - Adding per-class metrics for iris and pupil
2. **Export enhancements** - Better ONNX export workflow
3. **Fixing sentinel problem** - Handling edge cases in metrics (NaN values)
4. **Documentation** - Writing comprehensive guides

### In Progress:
- Fine-tuning the 29-metric system
- Testing YOLO11 integration
- Documenting workflows for thesis

---

## Next Steps (Suggestions)

### Immediate Actions:
1. **Commit your work!** You have significant uncommitted changes
   ```bash
   git add .
   git commit -m "Add YOLO11, k-fold, 29-metric system, augmentation tools, and docs"
   ```

2. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

### For Thesis:
1. **Run k-fold experiments** to get robust model comparisons
2. **Compare YOLO11 vs UNet** on your datasets
3. **Generate PR curves** for best models
4. **Create result plots** using `tools/analyze/plot_metrics.py`

### Potential Experiments:
1. Test colorization augmentation impact on model robustness
2. Compare YOLO11 approach 1 vs 2 performance
3. Hyperparameter search using k-fold
4. Evaluate on different datasets (MOBIUS, IrisPupilEye, TayedEyes)

---

## Quick Reference Commands

### Training
```bash
# Standard training
python -m irispupilnet.train --config irispupilnet/configs/example_grayscale.yaml

# YOLO11 training
python -m irispupilnet.train --model yolo11n_seg --img-size 160 --epochs 50

# K-fold cross-validation
python -m irispupilnet.train_kfold --config configs/kfold_example.yaml
```

### Analysis
```bash
# Plot metrics from training
python tools/analyze/plot_metrics.py --csv runs/exp/metrics.csv --out plots/

# Generate PR curves
python tools/analyze/plot_pr_curves.py --checkpoint runs/exp/best.pt --csv dataset/merged/all_datasets.csv --out pr_curves/
```

### Data Augmentation
```bash
# Colorize IR images
python augmentation/colorize_ir_eyes.py --in_dir data/ir_eyes --out_dir data/colorized --variants 2
```

### Export & Demo
```bash
# Export to ONNX
python -m irispupilnet.export_onnx --checkpoint runs/exp/best.pt --output model.onnx --img-size 160

# Run demo
python -m irispupilnet.demo --model model.onnx --source 0
```

---

## Dependencies

**Main packages:**
- PyTorch 2.1+ with CUDA 11.8
- ultralytics >= 8.3.0 (for YOLO11)
- google-genai (for augmentation)
- albumentations (augmentation)
- opencv-python, mediapipe (demo)
- numpy < 2.0 (PyTorch compatibility)

---

## Summary

You've built a **highly sophisticated** iris/pupil segmentation research platform with:

✅ Multiple model architectures (custom UNets + YOLO11)
✅ Two YOLO integration strategies
✅ 29-metric comprehensive evaluation system
✅ K-fold cross-validation
✅ AI-powered data augmentation
✅ Extensive tooling (dataset prep, analysis, visualization)
✅ Publication-quality documentation
✅ Production-ready export (ONNX) and demo

This is **way beyond** a typical thesis project - it's a complete research framework that could be published as a standalone tool!

**Your current state:** Feature-complete with uncommitted improvements to metrics and export. Ready for experimental phase and thesis writing.

---

## Questions?

If you need help with:
- Running specific experiments
- Understanding any component
- Debugging issues
- Planning next steps

Just ask! The documentation is comprehensive (`docs/`, `CLAUDE.md`), and all major workflows are documented.
