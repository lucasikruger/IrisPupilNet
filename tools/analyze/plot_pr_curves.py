"""
Script to generate Precision-Recall curves from a trained model.

This script loads a trained model, runs it on a validation/test dataset,
and generates PR curves for iris and pupil segmentation.

Usage:
    python tools/analyze/plot_pr_curves.py --model path/to/model.pth --config config.yaml --out pr_curves.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _ensure_package_imports():
    """Ensure package imports work correctly."""
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_package_imports()

from irispupilnet import datasets as ds_init  # noqa: F401
from irispupilnet import models as md_init  # noqa: F401
from irispupilnet.datasets import DATASET_REGISTRY
from irispupilnet.models import MODEL_REGISTRY
from irispupilnet.utils.segmentation_metrics import precision_recall_curves_from_logits


def load_model(checkpoint_path: Path, num_classes: int = 3, in_channels: int = 1, base: int = 32) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model config from checkpoint args
    if "args" in checkpoint:
        args = checkpoint["args"]
        model_name = args.get("model", "unet_se_small")
        num_classes = args.get("num_classes", num_classes)
        in_channels = args.get("in_channels", in_channels)
        base = args.get("base", base)
    else:
        # Fallback to defaults if no args found
        model_name = "unet_se_small"

    # Build model
    model = MODEL_REGISTRY[model_name](in_channels=in_channels, n_classes=num_classes, base=base)

    # Load weights
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def compute_pr_curves_from_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    num_thresholds: int = 100,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute PR curves by aggregating predictions over entire dataset.

    Args:
        model: trained segmentation model
        dataloader: validation/test dataloader
        device: device to run model on
        num_thresholds: number of threshold points for PR curve

    Returns:
        Dictionary with keys: pr_iris, pr_pupil
        Each contains: {thresholds, precision, recall, f1}
    """
    model = model.to(device)
    model.eval()

    # Collect all logits and targets
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Computing predictions"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            all_logits.append(logits.cpu())
            all_targets.append(masks.cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute PR curves
    pr_curves = precision_recall_curves_from_logits(all_logits, all_targets, num_thresholds=num_thresholds)

    return pr_curves


def plot_pr_curves(pr_curves: Dict[str, Dict[str, np.ndarray]], out_path: Path):
    """
    Plot Precision-Recall curves for iris and pupil.

    Args:
        pr_curves: Dictionary with pr_iris and pr_pupil curves
        out_path: output file path for plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: PR Curve for Iris
    pr_iris = pr_curves["pr_iris"]
    axes[0].plot(pr_iris["recall"], pr_iris["precision"], linewidth=2.5, color="#31a354", label="Iris")
    axes[0].fill_between(pr_iris["recall"], pr_iris["precision"], alpha=0.2, color="#31a354")
    axes[0].set_xlabel("Recall", fontsize=13)
    axes[0].set_ylabel("Precision", fontsize=13)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Precision-Recall Curve: Iris", fontsize=15, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=12)

    # Add AP to title if available
    ap_iris = np.trapz(pr_iris["precision"][np.argsort(pr_iris["recall"])],
                       pr_iris["recall"][np.argsort(pr_iris["recall"])])
    axes[0].text(0.05, 0.95, f'AP = {ap_iris:.3f}',
                transform=axes[0].transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: PR Curve for Pupil
    pr_pupil = pr_curves["pr_pupil"]
    axes[1].plot(pr_pupil["recall"], pr_pupil["precision"], linewidth=2.5, color="#756bb1", label="Pupil")
    axes[1].fill_between(pr_pupil["recall"], pr_pupil["precision"], alpha=0.2, color="#756bb1")
    axes[1].set_xlabel("Recall", fontsize=13)
    axes[1].set_ylabel("Precision", fontsize=13)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Precision-Recall Curve: Pupil", fontsize=15, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=12)

    # Add AP to title
    ap_pupil = np.trapz(pr_pupil["precision"][np.argsort(pr_pupil["recall"])],
                        pr_pupil["recall"][np.argsort(pr_pupil["recall"])])
    axes[1].text(0.05, 0.95, f'AP = {ap_pupil:.3f}',
                transform=axes[1].transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PR curves to {out_path}")


def plot_combined_pr_curve(pr_curves: Dict[str, Dict[str, np.ndarray]], out_path: Path):
    """
    Plot both PR curves on the same plot for comparison.

    Args:
        pr_curves: Dictionary with pr_iris and pr_pupil curves
        out_path: output file path for plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    pr_iris = pr_curves["pr_iris"]
    pr_pupil = pr_curves["pr_pupil"]

    # Plot both curves
    ax.plot(pr_iris["recall"], pr_iris["precision"], linewidth=2.5, color="#31a354",
            label="Iris", marker="o", markersize=4, markevery=10)
    ax.plot(pr_pupil["recall"], pr_pupil["precision"], linewidth=2.5, color="#756bb1",
            label="Pupil", marker="s", markersize=4, markevery=10)

    # Fill areas
    ax.fill_between(pr_iris["recall"], pr_iris["precision"], alpha=0.15, color="#31a354")
    ax.fill_between(pr_pupil["recall"], pr_pupil["precision"], alpha=0.15, color="#756bb1")

    # Compute APs
    ap_iris = np.trapz(pr_iris["precision"][np.argsort(pr_iris["recall"])],
                       pr_iris["recall"][np.argsort(pr_iris["recall"])])
    ap_pupil = np.trapz(pr_pupil["precision"][np.argsort(pr_pupil["recall"])],
                        pr_pupil["recall"][np.argsort(pr_pupil["recall"])])
    map_score = (ap_iris + ap_pupil) / 2.0

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision-Recall Curves: Iris vs Pupil", fontsize=16, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=13, loc="lower left")

    # Add text box with AP scores
    textstr = f'AP Iris: {ap_iris:.3f}\nAP Pupil: {ap_pupil:.3f}\nmAP: {map_score:.3f}'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined PR curve to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Precision-Recall curves from trained model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset base directory")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--out", type=Path, default=Path("pr_curves"), help="Output directory")
    parser.add_argument("--num-thresholds", type=int, default=100, help="Number of thresholds for PR curve")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to use (default: val)")
    args = parser.parse_args()

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    print(f"Model loaded successfully")

    # Extract config from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    img_size = ckpt_args.get("img_size", 160)
    default_format = ckpt_args.get("default_format", "mobius_3c")
    in_channels = ckpt_args.get("in_channels", 1)
    convert_to_grayscale = (in_channels == 1)

    # Create dataset
    print(f"Loading dataset from {args.csv}...")
    dataset = DATASET_REGISTRY["csv_seg"](
        dataset_base_dir=args.data_root,
        csv_path=args.csv,
        split=args.split,
        img_size=img_size,
        default_format=default_format,
        convert_to_grayscale=convert_to_grayscale
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Compute PR curves
    print(f"Computing PR curves with {args.num_thresholds} thresholds...")
    pr_curves = compute_pr_curves_from_dataset(model, dataloader, args.device, args.num_thresholds)

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_pr_curves(pr_curves, args.out / "pr_curves_separate.png")
    plot_combined_pr_curve(pr_curves, args.out / "pr_curves_combined.png")

    # Save PR curve data as NPZ for later analysis
    np.savez(
        args.out / "pr_curves_data.npz",
        thresholds=pr_curves["pr_iris"]["thresholds"],
        precision_iris=pr_curves["pr_iris"]["precision"],
        recall_iris=pr_curves["pr_iris"]["recall"],
        f1_iris=pr_curves["pr_iris"]["f1"],
        precision_pupil=pr_curves["pr_pupil"]["precision"],
        recall_pupil=pr_curves["pr_pupil"]["recall"],
        f1_pupil=pr_curves["pr_pupil"]["f1"],
    )
    print(f"Saved PR curve data to {args.out / 'pr_curves_data.npz'}")

    print(f"\nAll outputs saved to {args.out}/")


if __name__ == "__main__":
    main()
