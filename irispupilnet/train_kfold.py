import argparse, csv, sys, yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold


def _ensure_package_imports():
    """
    Allow running as `python -m irispupilnet.train_kfold` (preferred) **and**
    `python irispupilnet/train_kfold.py` by ensuring the project root is on sys.path
    before importing package modules.
    """
    if __package__ in (None, ""):
        project_root = Path(__file__).resolve().parents[1]
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)


_ensure_package_imports()

from irispupilnet import datasets as ds_init  # noqa: F401 (side-effects for registry)
from irispupilnet import models as md_init  # noqa: F401
from irispupilnet.models import MODEL_REGISTRY
from irispupilnet.datasets import DATASET_REGISTRY
from irispupilnet.utils.metrics import mean_iou_ignore_bg
from irispupilnet.utils.segmentation_metrics import compute_all_metrics

try:
    import albumentations as A  # just to ensure requirement is installed
except Exception:
    pass


def build_model(model_name: str, in_channels: int, n_classes: int, base: int):
    ModelCtor = MODEL_REGISTRY[model_name]
    return ModelCtor(in_channels=in_channels, n_classes=n_classes, base=base)


def train_one_epoch(model, dl, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in tqdm(dl, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)


@torch.no_grad()
def evaluate(model, dl, loss_fn, device, num_classes: int):
    model.eval()
    tot_loss, tot_iou, n = 0.0, 0.0, 0

    # Initialize metric accumulators
    metrics_sum = {}
    n_batches = 0

    for x, y in tqdm(dl, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y).item()
        iou = mean_iou_ignore_bg(logits, y, num_classes=num_classes)

        # Compute all segmentation metrics (including AP)
        batch_metrics = compute_all_metrics(logits, y, compute_ap=True, ap_num_thresholds=100)
        for k, v in batch_metrics.items():
            if k not in metrics_sum:
                metrics_sum[k] = 0.0
            metrics_sum[k] += v
        n_batches += 1

        bs = x.size(0)
        tot_loss += loss * bs
        tot_iou  += iou  * bs
        n += bs

    # Average metrics over batches
    metrics_avg = {k: v / n_batches for k, v in metrics_sum.items()}

    return tot_loss / n, tot_iou / n, metrics_avg


def create_fold_csvs(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
                     temp_dir: Path) -> tuple[Path, Path]:
    """Create temporary CSV files for train and val folds."""
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # Set split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'

    train_csv = temp_dir / 'train_fold.csv'
    val_csv = temp_dir / 'val_fold.csv'

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Combine for dataset loader
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    combined_csv = temp_dir / 'fold.csv'
    combined_df.to_csv(combined_csv, index=False)

    return combined_csv, train_csv, val_csv


def build_dataloaders(dataset_name: str,
                      data_root: str,
                      csv_path: str,
                      img_size: int,
                      batch_size: int,
                      workers: int,
                      default_format: str,
                      convert_to_grayscale: bool):
    DatasetCls = DATASET_REGISTRY[dataset_name]
    if dataset_name == "csv_seg":
        train_ds = DatasetCls(dataset_base_dir=data_root, csv_path=csv_path, split="train",
                              img_size=img_size, default_format=default_format,
                              convert_to_grayscale=convert_to_grayscale)
        val_ds   = DatasetCls(dataset_base_dir=data_root, csv_path=csv_path, split="val",
                              img_size=img_size, default_format=default_format,
                              convert_to_grayscale=convert_to_grayscale)
    else:
        raise ValueError("Use dataset=csv_seg for the CSV-driven loader.")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_dl, val_dl


def train_single_fold(fold_num: int, train_dl: DataLoader, val_dl: DataLoader,
                     args: argparse.Namespace, device: torch.device,
                     fold_dir: Path) -> Dict[str, float]:
    """Train a single fold and return final validation metrics."""

    print(f"\n{'='*80}")
    print(f"Training Fold {fold_num}/{args.n_folds}")
    print(f"{'='*80}")

    # Build fresh model for this fold
    model = build_model(args.model, args.in_channels, args.num_classes, args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_iou = -1.0
    best_epoch = 0
    best_metrics = None

    # Training loop
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_dl, opt, loss_fn, device)
        val_loss, val_iou, val_metrics = evaluate(model, val_dl, loss_fn, device, num_classes=args.num_classes)

        msg = f"  Epoch {ep:02d}/{args.epochs} | train {tr_loss:.4f} | val {val_loss:.4f}"
        msg += f" | IoU {val_iou:.3f} | Dice {val_metrics['dice_mean']:.3f} | mAP {val_metrics['map']:.3f}"
        print(msg)

        # Save best checkpoint for this fold
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = ep
            best_metrics = val_metrics.copy()
            best_metrics['val_iou'] = val_iou
            best_metrics['val_loss'] = val_loss
            best_metrics['train_loss'] = tr_loss

            ckpt = fold_dir / "best.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": ep,
                "best_iou": best_iou,
                "args": vars(args),
                "fold": fold_num
            }, ckpt)

    print(f"  ✓ Fold {fold_num} complete: best IoU = {best_iou:.4f} at epoch {best_epoch}")

    # Save final metrics for this fold
    metrics_path = fold_dir / "metrics.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump({
            'fold': fold_num,
            'best_epoch': best_epoch,
            'best_iou': float(best_iou),
            **{k: float(v) for k, v in best_metrics.items()}
        }, f, default_flow_style=False, sort_keys=False)

    return best_metrics


def aggregate_fold_results(fold_results: List[Dict[str, float]], output_dir: Path):
    """Aggregate results across all folds and save summary."""

    # Collect all metric keys
    all_keys = set()
    for result in fold_results:
        all_keys.update(result.keys())

    # Compute mean and std for each metric
    summary = {}
    for key in sorted(all_keys):
        values = [r[key] for r in fold_results if key in r]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))

    # Save summary
    summary_path = output_dir / "kfold_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*80}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Number of folds: {len(fold_results)}")
    print(f"\nKey Metrics (mean ± std):")
    print(f"  IoU:        {summary['val_iou_mean']:.4f} ± {summary['val_iou_std']:.4f}")
    print(f"  Dice:       {summary['dice_mean_mean']:.4f} ± {summary['dice_mean_std']:.4f}")
    print(f"  mAP:        {summary['map_mean']:.4f} ± {summary['map_std']:.4f}")
    print(f"  HD95 (avg): {(summary['hd95_iris_mean'] + summary['hd95_pupil_mean'])/2:.2f} ± {(summary['hd95_iris_std'] + summary['hd95_pupil_std'])/2:.2f} px")
    print(f"\nDetailed summary saved to: {summary_path}")
    print(f"{'='*80}\n")

    return summary


def plot_fold_comparison(fold_results: List[Dict[str, float]], output_dir: Path):
    """Create visualization comparing metrics across folds."""

    n_folds = len(fold_results)
    fold_nums = list(range(1, n_folds + 1))

    # Extract key metrics
    metrics_to_plot = [
        ('val_iou', 'Validation IoU', 'IoU'),
        ('dice_mean', 'Mean Dice Score', 'Dice'),
        ('map', 'Mean Average Precision', 'mAP'),
        ('hd95_iris', 'HD95 Iris', 'HD95 (px)'),
        ('hd95_pupil', 'HD95 Pupil', 'HD95 (px)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break

        values = [r.get(metric_key, 0) for r in fold_results]
        mean_val = np.mean(values)

        axes[idx].bar(fold_nums, values, alpha=0.7, color='steelblue')
        axes[idx].axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        axes[idx].set_xlabel('Fold')
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_title(title)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_xticks(fold_nums)

    # Hide extra subplot
    if len(metrics_to_plot) < len(axes):
        axes[-1].axis('off')

    fig.tight_layout()
    plot_path = output_dir / "kfold_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved fold comparison plot: {plot_path}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge YAML config with command-line arguments.
    Command-line arguments take precedence over config file.
    """
    parser = create_argument_parser()
    defaults = vars(parser.parse_args([]))

    for key, value in config.items():
        if hasattr(args, key):
            if getattr(args, key) == defaults.get(key):
                if key in ('out', 'config', 'csv') and value is not None:
                    value = Path(value)
                setattr(args, key, value)
        else:
            if key in ('out', 'config', 'csv') and value is not None:
                value = Path(value)
            setattr(args, key, value)

    return args


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    ap = argparse.ArgumentParser(description="K-Fold Cross-Validation for iris/pupil segmentation")

    # Config file
    ap.add_argument("--config", type=Path, default=None,
                    help="Path to YAML config file. CLI args override config values.")

    # Dataset
    ap.add_argument("--dataset", default="csv_seg", choices=list(DATASET_REGISTRY.keys()))
    ap.add_argument("--data-root", required=False, help="Base dir for relative paths in CSV.")
    ap.add_argument("--csv", required=False, help="CSV with rel_image_path, rel_mask_path, split[, dataset_format].")
    ap.add_argument("--default-format", default="mobius_3c", help="Fallback dataset_format if CSV row lacks it.")
    ap.add_argument("--use-val", action="store_true",
                    help="Include validation split in k-fold (combines train+val before splitting).")

    # K-Fold parameters
    ap.add_argument("--n-folds", type=int, default=5, help="Number of folds for cross-validation.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle data before splitting into folds.")
    ap.add_argument("--random-seed", type=int, default=42, help="Random seed for fold splitting.")

    # Model
    ap.add_argument("--model", default="unet_se_small", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--img-size", type=int, default=160, help="Image size (square: H=W).")
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--num-classes", type=int, default=3, help="0=bg, 1=iris, 2=pupil")
    ap.add_argument("--base", type=int, default=32)

    # Training
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--color", action="store_true", help="Keep RGB images (default is grayscale).")

    # Output & checkpointing
    ap.add_argument("--out", type=Path, default=Path("runs/kfold_mobius_unet_se"))

    return ap


def main():
    ap = create_argument_parser()
    args = ap.parse_args()

    # Load config file if provided
    if args.config is not None:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
        print(f"Loaded configuration from {args.config}")

    # Validate required arguments
    if not hasattr(args, 'data_root') or args.data_root is None:
        raise ValueError("--data-root is required (either via CLI or config file)")
    if not hasattr(args, 'csv') or args.csv is None:
        raise ValueError("--csv is required (either via CLI or config file)")

    convert_to_grayscale = not args.color
    if convert_to_grayscale:
        args.in_channels = 1

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_out = args.out
    args.out = base_out / f"kfold_{args.n_folds}_{timestamp}"
    args.out.mkdir(parents=True, exist_ok=True)

    # Save config to run directory
    config_save_path = args.out / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)
    print(f"Saved config to {config_save_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "cpu"
    if device.type == "cuda":
        try:
            device_name = torch.cuda.get_device_name(device.index if device.index is not None else 0)
        except Exception:
            device_name = "cuda"
    print(f"Using device: {device} ({device_name})")

    # Load CSV and prepare data for k-fold
    print(f"\nLoading data from {args.csv}")
    df = pd.read_csv(args.csv)

    # Filter for train (and optionally val) splits
    if args.use_val:
        print("Combining train + val splits for k-fold cross-validation")
        data_df = df[df['split'].str.lower().isin(['train', 'val'])].copy()
    else:
        print("Using only train split for k-fold cross-validation")
        data_df = df[df['split'].str.lower() == 'train'].copy()

    print(f"Total samples for k-fold: {len(data_df)}")

    # Initialize k-fold splitter
    kfold = KFold(n_splits=args.n_folds, shuffle=args.shuffle, random_state=args.random_seed if args.shuffle else None)

    print(f"\nStarting {args.n_folds}-Fold Cross-Validation")
    print(f"  model={args.model}, img_size={args.img_size}, in_channels={args.in_channels}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  image mode = {'grayscale' if convert_to_grayscale else 'RGB'}")
    print(f"  output directory: {args.out}")

    fold_results = []

    # Create temporary directory for fold CSVs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Train each fold
        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(data_df), 1):
            # Create fold directory
            fold_dir = args.out / f"fold_{fold_num}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary CSV files for this fold
            fold_csv, train_csv, val_csv = create_fold_csvs(data_df, train_idx, val_idx, temp_path)

            print(f"\nFold {fold_num}: {len(train_idx)} train samples, {len(val_idx)} val samples")

            # Build dataloaders
            train_dl, val_dl = build_dataloaders(
                args.dataset, args.data_root, str(fold_csv),
                args.img_size, args.batch_size, args.workers,
                default_format=args.default_format,
                convert_to_grayscale=convert_to_grayscale
            )

            # Train this fold
            fold_metrics = train_single_fold(
                fold_num, train_dl, val_dl, args, device, fold_dir
            )

            fold_results.append(fold_metrics)

    # Aggregate and save results
    summary = aggregate_fold_results(fold_results, args.out)

    # Create comparison plots
    plot_fold_comparison(fold_results, args.out)

    print(f"\nK-Fold Cross-Validation Complete!")
    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
