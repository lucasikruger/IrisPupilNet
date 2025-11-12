import argparse, csv, sys
from pathlib import Path
from typing import Dict, Any, List

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

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(epochs, train_losses, label="train loss", marker="o")
    if not all(np.isnan(val_losses)):
        axes[0].plot(epochs, val_losses, label="val loss", marker="s")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    has_val_iou = not all(np.isnan(val_ious))
    if has_val_iou:
        axes[1].plot(epochs, val_ious, label="val IoU (iris+pupil)", marker="d", color="#2c7fb8")
    axes[1].set_ylabel("IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    if has_val_iou:
        axes[1].legend(loc="upper left")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)


def init_metrics_writer(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "val_iou"])
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
    desc = f"Val epoch {epoch:02d}" if epoch is not None else "Val"
    for x, y in tqdm(dl, desc=desc, leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y).item()
        iou = mean_iou_ignore_bg(logits, y, num_classes=num_classes)
        bs = x.size(0)
        tot_loss += loss * bs
        tot_iou  += iou  * bs
        n += bs
    return tot_loss / n, tot_iou / n


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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="csv_seg", choices=list(DATASET_REGISTRY.keys()))
    ap.add_argument("--data-root", required=True, help="Base dir for relative paths in CSV.")
    ap.add_argument("--csv", required=True, help="CSV with rel_image_path, rel_mask_path, split[, dataset_format].")
    ap.add_argument("--default-format", default="mobius_3c", help="Fallback dataset_format if CSV row lacks it.")

    ap.add_argument("--model", default="unet_se_small", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--num-classes", type=int, default=3, help="0=bg, 1=iris, 2=pupil")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("runs/mobius_unet_se"))
    ap.add_argument("--val-every", type=int, default=1, help="Evaluate on validation every N epochs.")
    ap.add_argument("--metrics-csv", type=Path, default=None, help="Where to write epoch metrics CSV.")
    ap.add_argument("--metrics-plot", type=Path, default=None, help="Where to save the metrics plot PNG.")
    ap.add_argument("--color", action="store_true", help="Keep RGB images (default is grayscale).")
    args = ap.parse_args()

    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1")

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

    metrics_csv_path = args.metrics_csv if args.metrics_csv is not None else (args.out / "metrics.csv")
    metrics_plot_path = args.metrics_plot if args.metrics_plot is not None else (args.out / "metrics.png")

    print("Starting training run")
    print(f"  dataset={args.dataset}, csv={args.csv}, data_root={args.data_root}")
    print(f"  model={args.model}, img_size={args.img_size}, in_channels={args.in_channels}")
    print(f"  out={args.out}  val every {args.val_every} epoch(s)")
    print(f"  metrics CSV -> {metrics_csv_path}")
    print(f"  metrics plot -> {metrics_plot_path}")
    print(f"  image mode = {'grayscale' if convert_to_grayscale else 'RGB'}")

    metrics_file, metrics_writer = init_metrics_writer(metrics_csv_path)
    metrics: List[Dict[str, Any]] = []
    best_iou = -1.0
    try:
        for ep in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_dl, opt, loss_fn, device, epoch=ep)
            should_eval = (ep % args.val_every == 0)

            val_loss = None
            val_iou = None
            if should_eval:
                val_loss, val_iou = evaluate(model, val_dl, loss_fn, device, num_classes=args.num_classes, epoch=ep)
            msg = f"epoch {ep:02d} | train {tr_loss:.4f}"
            if should_eval:
                msg += f" | val {val_loss:.4f} | IoU(iris+pupil) {val_iou:.3f}"
            else:
                msg += " | val skipped"
            print(msg)

            if should_eval and val_iou is not None and val_iou > best_iou:
                best_iou = val_iou
                ckpt = args.out / "best.pt"
                torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
                print(f"  ↳ saved {ckpt}")

            metrics.append({
                "epoch": ep,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
            })
            metrics_writer.writerow([
                ep,
                f"{tr_loss:.6f}",
                f"{val_loss:.6f}" if val_loss is not None else "",
                f"{val_iou:.6f}" if val_iou is not None else "",
            ])
            metrics_file.flush()

            render_metrics_plot(metrics, metrics_plot_path)
    finally:
        metrics_file.close()

    print("Training complete")
    print(f"Metrics CSV saved to {metrics_csv_path}")
    print(f"Metrics plot saved to {metrics_plot_path}")

if __name__ == "__main__":
    main()
