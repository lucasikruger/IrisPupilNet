"""
Standalone script to plot detailed segmentation metrics from training CSV.

Usage:
    python tools/analyze/plot_metrics.py --csv runs/experiment/metrics.csv --out plots/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_dice(df: pd.DataFrame, out_path: Path):
    """Plot Dice scores over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if "dice_iris" in df.columns:
        ax.plot(df["epoch"], df["dice_iris"], label="Dice Iris", marker="^", linewidth=2, markersize=6, color="#31a354")
        ax.plot(df["epoch"], df["dice_pupil"], label="Dice Pupil", marker="v", linewidth=2, markersize=6, color="#756bb1")
        ax.plot(df["epoch"], df["dice_mean"], label="Dice Mean", marker="o", linewidth=2, markersize=6, color="#3182bd", linestyle="--")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Dice Score per Class Over Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Dice plot to {out_path}")


def plot_iou(df: pd.DataFrame, out_path: Path):
    """Plot IoU scores over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if "iou_iris" in df.columns:
        ax.plot(df["epoch"], df["iou_iris"], label="IoU Iris", marker="^", linewidth=2, markersize=6, color="#31a354")
        ax.plot(df["epoch"], df["iou_pupil"], label="IoU Pupil", marker="v", linewidth=2, markersize=6, color="#756bb1")
        ax.plot(df["epoch"], df["iou_mean"], label="IoU Mean", marker="o", linewidth=2, markersize=6, color="#3182bd", linestyle="--")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("IoU (Jaccard Index)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("IoU per Class Over Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved IoU plot to {out_path}")


def plot_center_distance(df: pd.DataFrame, out_path: Path):
    """Plot center distances over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if "center_dist_iris_px" in df.columns:
        ax.plot(df["epoch"], df["center_dist_iris_px"], label="Center Dist Iris", marker="^", linewidth=2, markersize=6, color="#e6550d")
        ax.plot(df["epoch"], df["center_dist_pupil_px"], label="Center Dist Pupil", marker="v", linewidth=2, markersize=6, color="#fd8d3c")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Distance (pixels)", fontsize=12)
    ax.set_title("Center Distance (Pred vs GT) Over Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved center distance plot to {out_path}")


def plot_hd95(df: pd.DataFrame, out_path: Path):
    """Plot HD95 (95% Hausdorff distance) over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if "hd95_iris" in df.columns:
        ax.plot(df["epoch"], df["hd95_iris"], label="HD95 Iris", marker="^", linewidth=2, markersize=6, color="#e6550d")
        ax.plot(df["epoch"], df["hd95_pupil"], label="HD95 Pupil", marker="v", linewidth=2, markersize=6, color="#fd8d3c")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("HD95 (pixels)", fontsize=12)
    ax.set_title("95% Hausdorff Distance Over Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved HD95 plot to {out_path}")


def plot_loss(df: pd.DataFrame, out_path: Path):
    """Plot training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", linewidth=2, markersize=6, color="#3182bd")
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        ax.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s", linewidth=2, markersize=6, color="#e6550d")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot to {out_path}")


def plot_all_metrics_combined(df: pd.DataFrame, out_path: Path):
    """Create a comprehensive figure with all metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Plot 1: Loss
    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train", marker="o", linewidth=2)
    if "val_loss" in df.columns and df["val_loss"].notna().any():
        axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val", marker="s", linewidth=2)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    # Plot 2: Dice
    if "dice_iris" in df.columns:
        axes[0, 1].plot(df["epoch"], df["dice_iris"], label="Iris", marker="^", linewidth=2, color="#31a354")
        axes[0, 1].plot(df["epoch"], df["dice_pupil"], label="Pupil", marker="v", linewidth=2, color="#756bb1")
        axes[0, 1].plot(df["epoch"], df["dice_mean"], label="Mean", marker="o", linewidth=2, color="#3182bd", linestyle="--")
    axes[0, 1].set_ylabel("Dice Score")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Dice Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    # Plot 3: IoU
    if "iou_iris" in df.columns:
        axes[0, 2].plot(df["epoch"], df["iou_iris"], label="Iris", marker="^", linewidth=2, color="#31a354")
        axes[0, 2].plot(df["epoch"], df["iou_pupil"], label="Pupil", marker="v", linewidth=2, color="#756bb1")
        axes[0, 2].plot(df["epoch"], df["iou_mean"], label="Mean", marker="o", linewidth=2, color="#3182bd", linestyle="--")
    axes[0, 2].set_ylabel("IoU")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_title("IoU (Jaccard Index)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, linestyle="--", alpha=0.4)

    # Plot 4: Center Distance
    if "center_dist_iris_px" in df.columns:
        axes[1, 0].plot(df["epoch"], df["center_dist_iris_px"], label="Iris", marker="^", linewidth=2, color="#e6550d")
        axes[1, 0].plot(df["epoch"], df["center_dist_pupil_px"], label="Pupil", marker="v", linewidth=2, color="#fd8d3c")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Distance (px)")
    axes[1, 0].set_title("Center Distance")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    # Plot 5: HD95
    if "hd95_iris" in df.columns:
        axes[1, 1].plot(df["epoch"], df["hd95_iris"], label="Iris", marker="^", linewidth=2, color="#e6550d")
        axes[1, 1].plot(df["epoch"], df["hd95_pupil"], label="Pupil", marker="v", linewidth=2, color="#fd8d3c")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("HD95 (px)")
    axes[1, 1].set_title("95% Hausdorff Distance")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    # Plot 6: Summary bar (best epoch metrics)
    if "dice_mean" in df.columns:
        best_idx = df["dice_mean"].idxmax()
        best_row = df.loc[best_idx]
        metrics_names = ["Dice\nMean", "IoU\nMean", "Center\nDist\n(pupil)"]
        metrics_values = [
            best_row.get("dice_mean", 0),
            best_row.get("iou_mean", 0),
            min(best_row.get("center_dist_pupil_px", 100) / 10, 1.0)  # Normalize for vis
        ]
        colors = ["#31a354", "#3182bd", "#fd8d3c"]
        axes[1, 2].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_ylabel("Score")
        axes[1, 2].set_title(f"Best Epoch ({int(best_row['epoch'])})")
        axes[1, 2].grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle("Training Metrics Overview", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined metrics plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot segmentation metrics from training CSV")
    parser.add_argument("--csv", type=Path, required=True, help="Path to metrics CSV file")
    parser.add_argument("--out", type=Path, default=Path("plots"), help="Output directory for plots")
    parser.add_argument("--combined-only", action="store_true", help="Only generate combined plot")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.csv)
    print(f"Loaded metrics from {args.csv}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Epochs: {len(df)}")

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Generate plots
    if args.combined_only:
        plot_all_metrics_combined(df, args.out / "all_metrics_combined.png")
    else:
        plot_loss(df, args.out / "loss.png")
        plot_dice(df, args.out / "dice.png")
        plot_iou(df, args.out / "iou.png")
        plot_center_distance(df, args.out / "center_distance.png")
        plot_hd95(df, args.out / "hd95.png")
        plot_all_metrics_combined(df, args.out / "all_metrics_combined.png")

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
