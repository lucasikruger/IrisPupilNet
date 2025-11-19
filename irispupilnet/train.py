import argparse, csv, sys, yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _ensure_package_imports():
    """
    Allow running as `python -m irispupilnet.train` (preferred) **and**
    `python irispupilnet/train.py` by ensuring the project root is on sys.path
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


def render_metrics_plot(metrics: List[Dict[str, Any]], plot_path: Path):
    if not metrics:
        return

    epochs = [m["epoch"] for m in metrics]
    train_losses = [m["train_loss"] for m in metrics]
    val_losses = [np.nan if m["val_loss"] is None else m["val_loss"] for m in metrics]
    val_ious = [np.nan if m["val_iou"] is None else m["val_iou"] for m in metrics]

    # Extract new metrics (with NaN for epochs where validation was skipped)
    dice_iris = [np.nan if "dice_iris" not in m else m["dice_iris"] for m in metrics]
    dice_pupil = [np.nan if "dice_pupil" not in m else m["dice_pupil"] for m in metrics]
    hd95_pupil = [np.nan if "hd95_pupil" not in m else m["hd95_pupil"] for m in metrics]
    hd95_iris = [np.nan if "hd95_iris" not in m else m["hd95_iris"] for m in metrics]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Plot 1: Loss
    axes[0, 0].plot(epochs, train_losses, label="train loss", marker="o", linewidth=2)
    if not all(np.isnan(val_losses)):
        axes[0, 0].plot(epochs, val_losses, label="val loss", marker="s", linewidth=2)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].set_title("Loss")

    # Plot 2: IoU (original metric)
    if not all(np.isnan(val_ious)):
        axes[0, 1].plot(epochs, val_ious, label="IoU (mean)", marker="d", color="#2c7fb8", linewidth=2)
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].set_title("IoU (Iris + Pupil)")

    # Plot 3: Dice scores
    if not all(np.isnan(dice_iris)):
        axes[1, 0].plot(epochs, dice_iris, label="Dice Iris", marker="^", color="#31a354", linewidth=2)
        axes[1, 0].plot(epochs, dice_pupil, label="Dice Pupil", marker="v", color="#756bb1", linewidth=2)
    axes[1, 0].set_ylabel("Dice Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].set_title("Dice Score per Class")

    # Plot 4: HD95 distance
    if not all(np.isnan(hd95_pupil)):
        axes[1, 1].plot(epochs, hd95_iris, label="HD95 Iris", marker="^", color="#e6550d", linewidth=2)
        axes[1, 1].plot(epochs, hd95_pupil, label="HD95 Pupil", marker="v", color="#fd8d3c", linewidth=2)
    axes[1, 1].set_ylabel("HD95 (pixels)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].set_title("95% Hausdorff Distance")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def init_metrics_writer(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "epoch", "train_loss", "val_loss", "val_iou",
        "dice_iris", "dice_pupil", "dice_mean",
        "iou_iris", "iou_pupil", "iou_mean",
        "center_dist_iris_px", "center_dist_pupil_px",
        "hd95_iris", "hd95_pupil"
    ])
    return csv_file, writer

def build_model(model_name: str, in_channels: int, n_classes: int, base: int):
    ModelCtor = MODEL_REGISTRY[model_name]
    return ModelCtor(in_channels=in_channels, n_classes=n_classes, base=base)

def train_one_epoch(model, dl, optimizer, loss_fn, device, epoch: int = None):
    model.train()
    total = 0.0
    desc = f"Train epoch {epoch:02d}" if epoch is not None else "Train"
    for x, y in tqdm(dl, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, loss_fn, device, num_classes: int, epoch: int = None):
    model.eval()
    tot_loss, tot_iou, n = 0.0, 0.0, 0

    # Initialize metric accumulators
    metrics_sum = {
        "dice_iris": 0.0,
        "dice_pupil": 0.0,
        "dice_mean": 0.0,
        "iou_iris": 0.0,
        "iou_pupil": 0.0,
        "iou_mean": 0.0,
        "center_dist_iris_px": 0.0,
        "center_dist_pupil_px": 0.0,
        "hd95_iris": 0.0,
        "hd95_pupil": 0.0,
    }
    n_batches = 0

    desc = f"Val epoch {epoch:02d}" if epoch is not None else "Val"
    for x, y in tqdm(dl, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y).item()
        iou = mean_iou_ignore_bg(logits, y, num_classes=num_classes)

        # Compute all segmentation metrics
        batch_metrics = compute_all_metrics(logits, y)
        for k, v in batch_metrics.items():
            metrics_sum[k] += v
        n_batches += 1

        bs = x.size(0)
        tot_loss += loss * bs
        tot_iou  += iou  * bs
        n += bs

    # Average metrics over batches
    metrics_avg = {k: v / n_batches for k, v in metrics_sum.items()}

    return tot_loss / n, tot_iou / n, metrics_avg


@torch.no_grad()
def show_examples(model, dl, device, num_examples: int, save_path: Path, epoch: int = None):
    """
    Show random validation examples with image, ground truth mask, and predicted mask.

    Args:
        model: The model to use for predictions
        dl: DataLoader to sample from
        device: Device to run inference on
        num_examples: Number of examples to show
        save_path: Path to save the visualization
        epoch: Optional epoch number for title
    """
    model.eval()

    # Get one batch
    x, y = next(iter(dl))
    x, y = x.to(device), y.to(device)

    # Run inference
    logits = model(x)
    pred = logits.argmax(1)  # [batch, H, W]

    # Move to CPU and convert to numpy
    x = x.cpu().permute(0, 2, 3, 1).numpy()  # [batch, H, W, C]
    y = y.cpu().numpy()  # [batch, H, W]
    pred = pred.cpu().numpy()  # [batch, H, W]

    # Normalize images to [0, 1] for display
    # If grayscale (C=1), we need to handle it
    if x.shape[-1] == 1:
        # Grayscale: squeeze channel dimension for display
        x_display = x.squeeze(-1)  # [batch, H, W]
    else:
        # RGB: clip to [0, 1]
        x_display = x.clip(0, 1)

    # Limit number of examples
    rows = min(num_examples, x.shape[0])

    fig, axes = plt.subplots(rows, 3, figsize=(9, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array even for single row

    for i in range(rows):
        # Column 0: Image
        if x.shape[-1] == 1:
            axes[i, 0].imshow(x_display[i], cmap='gray')
        else:
            axes[i, 0].imshow(x_display[i])
        axes[i, 0].set_title("Image")
        axes[i, 0].axis('off')

        # Column 1: Ground truth mask
        axes[i, 1].imshow(y[i], vmin=0, vmax=2, cmap='viridis')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        # Column 2: Predicted mask
        axes[i, 2].imshow(pred[i], vmin=0, vmax=2, cmap='viridis')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')

    title = f"Validation Examples - Epoch {epoch}" if epoch is not None else "Validation Examples"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ↳ saved examples visualization: {save_path}")


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
    # Get the defaults from argparse to know what was explicitly set
    parser = create_argument_parser()
    defaults = vars(parser.parse_args([]))

    # For each config key, set it in args if not explicitly provided via CLI
    for key, value in config.items():
        if hasattr(args, key):
            # Check if this arg was explicitly set (differs from default)
            if getattr(args, key) == defaults.get(key):
                # Use config value (CLI didn't override it)
                setattr(args, key, value)
        else:
            # Add new attribute from config
            setattr(args, key, value)

    return args


def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                   device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """
    Load checkpoint and restore model (and optionally optimizer) state.

    Returns:
        dict with keys: 'epoch', 'best_iou', 'args' (if available)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    safe_classes = [Path, type(None)]
    with torch.serialization.safe_globals(safe_classes):
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            print("  ↳ Loaded model state")
        else:
            # Assume the dict itself is the state_dict
            model.load_state_dict(checkpoint)
            print("  ↳ Loaded model state (direct state_dict)")

        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("  ↳ Loaded optimizer state")

        resume_info = {
            "epoch": checkpoint.get("epoch", 0),
            "best_iou": checkpoint.get("best_iou", -1.0),
            "args": checkpoint.get("args", {}),
        }
    else:
        # Old format: just state_dict
        model.load_state_dict(checkpoint)
        print("  ↳ Loaded model state (legacy format)")
        resume_info = {"epoch": 0, "best_iou": -1.0, "args": {}}

    return resume_info


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    ap = argparse.ArgumentParser(description="Train iris/pupil segmentation model")

    # Config file
    ap.add_argument("--config", type=Path, default=None,
                    help="Path to YAML config file. CLI args override config values.")

    # Dataset
    ap.add_argument("--dataset", default="csv_seg", choices=list(DATASET_REGISTRY.keys()))
    ap.add_argument("--data-root", required=False, help="Base dir for relative paths in CSV.")
    ap.add_argument("--csv", required=False, help="CSV with rel_image_path, rel_mask_path, split[, dataset_format].")
    ap.add_argument("--default-format", default="mobius_3c", help="Fallback dataset_format if CSV row lacks it.")

    # Model
    ap.add_argument("--model", default="unet_se_small", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--img-size", type=int, default=160, help="Image size (square: H=W). Overridden by img-width/img-height if provided.")
    ap.add_argument("--img-width", type=int, default=None, help="Image width (if different from height)")
    ap.add_argument("--img-height", type=int, default=None, help="Image height (if different from width)")
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
    ap.add_argument("--out", type=Path, default=Path("runs/mobius_unet_se"))
    ap.add_argument("--val-every", type=int, default=1, help="Evaluate on validation every N epochs.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Save checkpoint every N epochs (0 = only save best).")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Path to checkpoint to resume training from.")

    # Metrics & logging
    ap.add_argument("--metrics-csv", type=Path, default=None, help="Where to write epoch metrics CSV.")
    ap.add_argument("--metrics-plot", type=Path, default=None, help="Where to save the metrics plot PNG.")
    ap.add_argument("--log-every", type=int, default=0, help="Log batch progress every N batches (0=disable).")
    ap.add_argument("--save-best-metrics", action="store_true", help="Save best_metrics.yaml with best results.")
    ap.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")

    # Visualization
    ap.add_argument("--show-examples", type=int, default=0,
                    help="Show N random validation examples with predictions (0=disable).")
    ap.add_argument("--show-examples-every", type=int, default=1,
                    help="Generate examples visualization every N epochs.")

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

    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1")

    # Handle image size (support separate width/height or square img_size)
    if hasattr(args, 'img_width') and args.img_width is not None:
        img_width = args.img_width
    else:
        img_width = args.img_size

    if hasattr(args, 'img_height') and args.img_height is not None:
        img_height = args.img_height
    else:
        img_height = args.img_size

    # For now, datasets still need square images (this is a TODO for future)
    # So we'll use img_size but warn if width != height
    if img_width != img_height:
        print(f"WARNING: img_width ({img_width}) != img_height ({img_height})")
        print(f"  Current dataset loader requires square images. Using img_size={args.img_size}")
    args.img_size = args.img_size if img_width == img_height else args.img_size

    convert_to_grayscale = not args.color
    if convert_to_grayscale:
        args.in_channels = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    device_name = "cpu"
    if device.type == "cuda":
        try:
            device_name = torch.cuda.get_device_name(device.index if device.index is not None else 0)
        except Exception:
            device_name = "cuda"
    print(f"Using device: {device} ({device_name})")

    train_dl, val_dl = build_dataloaders(args.dataset, args.data_root, args.csv,
                                         args.img_size, args.batch_size, args.workers,
                                         default_format=args.default_format,
                                         convert_to_grayscale=convert_to_grayscale)
    model = build_model(args.model, args.in_channels, args.num_classes, args.base).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Load checkpoint if resuming
    start_epoch = 1
    best_iou = -1.0
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        resume_info = load_checkpoint(args.resume, model, opt, device)
        start_epoch = resume_info["epoch"] + 1
        best_iou = resume_info["best_iou"]
        print(f"Resuming from epoch {resume_info['epoch']}, best IoU so far: {best_iou:.4f}")

    metrics_csv_path = args.metrics_csv if args.metrics_csv is not None else (args.out / "metrics.csv")
    metrics_plot_path = args.metrics_plot if args.metrics_plot is not None else (args.out / "metrics.png")

    print("Starting training run")
    print(f"  dataset={args.dataset}, csv={args.csv}, data_root={args.data_root}")
    print(f"  model={args.model}, img_size={args.img_size}, in_channels={args.in_channels}")
    print(f"  num_classes={args.num_classes}, base={args.base}")
    print(f"  out={args.out}  val every {args.val_every} epoch(s)")
    if args.save_every > 0:
        print(f"  save checkpoint every {args.save_every} epoch(s)")
    print(f"  metrics CSV -> {metrics_csv_path}")
    print(f"  metrics plot -> {metrics_plot_path}")
    print(f"  image mode = {'grayscale' if convert_to_grayscale else 'RGB'}")
    if args.resume:
        print(f"  resuming from epoch {start_epoch}")

    metrics_file, metrics_writer = init_metrics_writer(metrics_csv_path)
    metrics: List[Dict[str, Any]] = []
    try:
        for ep in range(start_epoch, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_dl, opt, loss_fn, device, epoch=ep)
            should_eval = (ep % args.val_every == 0)

            val_loss = None
            val_iou = None
            val_metrics = None
            if should_eval:
                val_loss, val_iou, val_metrics = evaluate(model, val_dl, loss_fn, device, num_classes=args.num_classes, epoch=ep)
            msg = f"epoch {ep:02d} | train {tr_loss:.4f}"
            if should_eval:
                msg += f" | val {val_loss:.4f} | IoU {val_iou:.3f}"
                msg += f" | Dice {val_metrics['dice_mean']:.3f}"
                msg += f" | HD95 pupil {val_metrics['hd95_pupil']:.2f}px"
            else:
                msg += " | val skipped"
            print(msg)

            # Save best checkpoint
            if should_eval and val_iou is not None and val_iou > best_iou:
                best_iou = val_iou
                ckpt = args.out / "best.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": ep,
                    "best_iou": best_iou,
                    "args": vars(args)
                }, ckpt)
                print(f"  ↳ saved best checkpoint: {ckpt}")

                # Save best metrics to YAML if enabled
                if hasattr(args, 'save_best_metrics') and args.save_best_metrics and val_metrics is not None:
                    best_metrics_path = args.out / "best_metrics.yaml"
                    best_metrics_data = {
                        "epoch": ep,
                        "train_loss": float(tr_loss),
                        "val_loss": float(val_loss) if val_loss is not None else None,
                        "val_iou": float(val_iou),
                        **{k: float(v) for k, v in val_metrics.items()}
                    }
                    with open(best_metrics_path, 'w') as f:
                        yaml.dump(best_metrics_data, f, default_flow_style=False, sort_keys=False)
                    print(f"  ↳ saved best metrics: {best_metrics_path}")

            # Save periodic checkpoint
            if args.save_every > 0 and ep % args.save_every == 0:
                ckpt = args.out / f"checkpoint_epoch_{ep:03d}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": ep,
                    "best_iou": best_iou,
                    "args": vars(args)
                }, ckpt)
                print(f"  ↳ saved periodic checkpoint: {ckpt}")

            # Store metrics for plotting
            epoch_data = {
                "epoch": ep,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }
            if val_metrics is not None:
                epoch_data.update(val_metrics)
            metrics.append(epoch_data)

            # Write to CSV
            csv_row = [
                ep,
                f"{tr_loss:.6f}",
                f"{val_loss:.6f}" if val_loss is not None else "",
                f"{val_iou:.6f}" if val_iou is not None else "",
            ]
            if val_metrics is not None:
                csv_row.extend([
                    f"{val_metrics['dice_iris']:.6f}",
                    f"{val_metrics['dice_pupil']:.6f}",
                    f"{val_metrics['dice_mean']:.6f}",
                    f"{val_metrics['iou_iris']:.6f}",
                    f"{val_metrics['iou_pupil']:.6f}",
                    f"{val_metrics['iou_mean']:.6f}",
                    f"{val_metrics['center_dist_iris_px']:.6f}",
                    f"{val_metrics['center_dist_pupil_px']:.6f}",
                    f"{val_metrics['hd95_iris']:.6f}",
                    f"{val_metrics['hd95_pupil']:.6f}",
                ])
            else:
                csv_row.extend([""] * 10)  # Empty columns for skipped validation
            metrics_writer.writerow(csv_row)
            metrics_file.flush()

            render_metrics_plot(metrics, metrics_plot_path)

            # Generate examples visualization if enabled
            if hasattr(args, 'show_examples') and args.show_examples > 0:
                should_show_examples = (ep % args.show_examples_every == 0) if hasattr(args, 'show_examples_every') else True
                if should_show_examples:
                    examples_path = args.out / f"examples_epoch_{ep:03d}.png"
                    show_examples(model, val_dl, device, args.show_examples, examples_path, epoch=ep)
    finally:
        metrics_file.close()

    print("Training complete")
    print(f"Metrics CSV saved to {metrics_csv_path}")
    print(f"Metrics plot saved to {metrics_plot_path}")

if __name__ == "__main__":
    main()
