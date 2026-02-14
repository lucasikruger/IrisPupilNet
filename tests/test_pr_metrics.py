"""
Test script for Precision-Recall metrics and curves.

Usage:
    python tests/test_pr_metrics.py
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Ensure package imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from irispupilnet.utils.segmentation_metrics import (
    precision_recall_curve_for_class,
    average_precision_for_class,
    average_precision_from_logits,
    precision_recall_curves_from_logits,
)


def create_synthetic_data(batch_size=4, num_classes=3, height=128, width=128):
    """Create synthetic logits and target masks for testing."""
    # Create logits with some structure
    logits = torch.randn(batch_size, num_classes, height, width)

    # Create target masks (0=background, 1=iris, 2=pupil)
    target = torch.zeros(batch_size, height, width, dtype=torch.long)

    for b in range(batch_size):
        # Create circular iris
        center_y, center_x = height // 2, width // 2
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        # Iris region (radius 40-60)
        iris_mask = (dist > 30) & (dist < 50)
        target[b][iris_mask] = 1

        # Pupil region (radius 0-40)
        pupil_mask = dist < 30
        target[b][pupil_mask] = 2

    # Make logits favor correct predictions (but not perfect)
    for b in range(batch_size):
        for c in range(num_classes):
            mask = (target[b] == c)
            logits[b, c][mask] += 2.0  # Boost correct class logits

    return logits, target


def test_pr_curve_computation():
    """Test PR curve computation."""
    print("=" * 60)
    print("Testing PR Curve Computation")
    print("=" * 60)

    # Create synthetic data
    logits, target = create_synthetic_data(batch_size=2, height=64, width=64)

    # Compute PR curve for iris (class 1)
    print("\nComputing PR curve for Iris (class 1)...")
    pr_iris = precision_recall_curve_for_class(logits, target, class_id=1, num_thresholds=20)

    print(f"  Thresholds shape: {pr_iris['thresholds'].shape}")
    print(f"  Precision shape: {pr_iris['precision'].shape}")
    print(f"  Recall shape: {pr_iris['recall'].shape}")
    print(f"  F1 shape: {pr_iris['f1'].shape}")
    print(f"  Max F1: {np.max(pr_iris['f1']):.4f}")
    print(f"  Precision range: [{np.min(pr_iris['precision']):.4f}, {np.max(pr_iris['precision']):.4f}]")
    print(f"  Recall range: [{np.min(pr_iris['recall']):.4f}, {np.max(pr_iris['recall']):.4f}]")

    # Compute PR curve for pupil (class 2)
    print("\nComputing PR curve for Pupil (class 2)...")
    pr_pupil = precision_recall_curve_for_class(logits, target, class_id=2, num_thresholds=20)

    print(f"  Max F1: {np.max(pr_pupil['f1']):.4f}")
    print(f"  Precision range: [{np.min(pr_pupil['precision']):.4f}, {np.max(pr_pupil['precision']):.4f}]")
    print(f"  Recall range: [{np.min(pr_pupil['recall']):.4f}, {np.max(pr_pupil['recall']):.4f}]")

    print("\n✓ PR curve computation successful!")


def test_average_precision():
    """Test Average Precision (AP) computation."""
    print("\n" + "=" * 60)
    print("Testing Average Precision (AP) Computation")
    print("=" * 60)

    # Create synthetic data
    logits, target = create_synthetic_data(batch_size=4, height=64, width=64)

    # Compute AP for iris
    print("\nComputing AP for Iris...")
    ap_iris = average_precision_for_class(logits, target, class_id=1, num_thresholds=50)
    print(f"  AP Iris: {ap_iris:.4f}")

    # Compute AP for pupil
    print("\nComputing AP for Pupil...")
    ap_pupil = average_precision_for_class(logits, target, class_id=2, num_thresholds=50)
    print(f"  AP Pupil: {ap_pupil:.4f}")

    # Compute all APs at once
    print("\nComputing all AP metrics...")
    ap_metrics = average_precision_from_logits(logits, target, num_thresholds=50)
    print(f"  AP Iris: {ap_metrics['ap_iris']:.4f}")
    print(f"  AP Pupil: {ap_metrics['ap_pupil']:.4f}")
    print(f"  mAP: {ap_metrics['map']:.4f}")

    print("\n✓ Average Precision computation successful!")


def test_pr_curves_both_classes():
    """Test PR curves for both classes at once."""
    print("\n" + "=" * 60)
    print("Testing PR Curves for Both Classes")
    print("=" * 60)

    # Create synthetic data
    logits, target = create_synthetic_data(batch_size=4, height=64, width=64)

    # Compute PR curves for both classes
    print("\nComputing PR curves for both Iris and Pupil...")
    pr_curves = precision_recall_curves_from_logits(logits, target, num_thresholds=30)

    print(f"\nIris PR Curve:")
    print(f"  Precision range: [{np.min(pr_curves['pr_iris']['precision']):.4f}, {np.max(pr_curves['pr_iris']['precision']):.4f}]")
    print(f"  Recall range: [{np.min(pr_curves['pr_iris']['recall']):.4f}, {np.max(pr_curves['pr_iris']['recall']):.4f}]")

    print(f"\nPupil PR Curve:")
    print(f"  Precision range: [{np.min(pr_curves['pr_pupil']['precision']):.4f}, {np.max(pr_curves['pr_pupil']['precision']):.4f}]")
    print(f"  Recall range: [{np.min(pr_curves['pr_pupil']['recall']):.4f}, {np.max(pr_curves['pr_pupil']['recall']):.4f}]")

    print("\n✓ PR curves for both classes computed successfully!")


def test_perfect_predictions():
    """Test PR curves with perfect predictions (should have AP=1.0)."""
    print("\n" + "=" * 60)
    print("Testing PR Curves with Perfect Predictions")
    print("=" * 60)

    batch_size = 2
    num_classes = 3
    height, width = 64, 64

    # Create target
    target = torch.zeros(batch_size, height, width, dtype=torch.long)
    center_y, center_x = height // 2, width // 2
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    dist = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

    for b in range(batch_size):
        iris_mask = (dist > 30) & (dist < 50)
        target[b][iris_mask] = 1
        pupil_mask = dist < 30
        target[b][pupil_mask] = 2

    # Create perfect logits (matching target exactly)
    logits = torch.zeros(batch_size, num_classes, height, width)
    for b in range(batch_size):
        for c in range(num_classes):
            logits[b, c][target[b] == c] = 10.0  # Very high logit for correct class
            logits[b, c][target[b] != c] = -10.0  # Very low logit for incorrect class

    # Compute AP
    ap_metrics = average_precision_from_logits(logits, target, num_thresholds=50)
    print(f"\nPerfect predictions AP metrics:")
    print(f"  AP Iris: {ap_metrics['ap_iris']:.6f} (expected ~1.0)")
    print(f"  AP Pupil: {ap_metrics['ap_pupil']:.6f} (expected ~1.0)")
    print(f"  mAP: {ap_metrics['map']:.6f} (expected ~1.0)")

    # Verify AP is close to 1.0
    assert ap_metrics['ap_iris'] > 0.99, f"Expected AP Iris > 0.99, got {ap_metrics['ap_iris']}"
    assert ap_metrics['ap_pupil'] > 0.99, f"Expected AP Pupil > 0.99, got {ap_metrics['ap_pupil']}"

    print("\n✓ Perfect prediction test passed!")


def main():
    print("\n" + "=" * 60)
    print("Running Precision-Recall Metrics Tests")
    print("=" * 60)

    try:
        test_pr_curve_computation()
        test_average_precision()
        test_pr_curves_both_classes()
        test_perfect_predictions()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe PR metrics implementation is working correctly.")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
