"""
Segmentation metrics for iris and pupil segmentation.

Metrics implemented:
- Dice score (per class): Measures overlap between prediction and ground truth
- IoU/Jaccard (per class): Another overlap metric
- Center distance: Euclidean distance between predicted and GT centers
- HD95 (95% Hausdorff distance): Robust boundary distance metric

Class mapping: 0=background, 1=iris, 2=pupil
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from scipy.ndimage import binary_erosion, distance_transform_edt


# ============================================================================
# Basic helpers
# ============================================================================

def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert model logits (B, C, H, W) to predicted labels (B, H, W) via argmax.

    Args:
        logits: (B, C, H, W) tensor of class logits

    Returns:
        (B, H, W) tensor of predicted class labels
    """
    return torch.argmax(logits, dim=1)


def one_hot_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer labels (B, H, W) to one-hot (B, C, H, W).

    Args:
        labels: (B, H, W) integer class labels
        num_classes: total number of classes

    Returns:
        (B, C, H, W) one-hot encoded tensor
    """
    b, h, w = labels.shape
    one_hot = F.one_hot(labels.long(), num_classes=num_classes)  # (B, H, W, C)
    return one_hot.permute(0, 3, 1, 2).float()


# ============================================================================
# Dice and IoU metrics
# ============================================================================

def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute Dice coefficient per class.

    Dice_k = 2 * |P_k ∩ G_k| / (|P_k| + |G_k| + ε)

    Args:
        pred: (B, H, W) predicted labels
        target: (B, H, W) ground truth labels
        num_classes: total number of classes
        eps: small constant for numerical stability

    Returns:
        (C,) tensor with Dice score for each class 0..C-1
    """
    pred_1h = one_hot_from_labels(pred, num_classes)      # (B, C, H, W)
    target_1h = one_hot_from_labels(target, num_classes)  # (B, C, H, W)

    dims = (0, 2, 3)  # sum over batch + spatial dimensions
    intersection = (pred_1h * target_1h).sum(dim=dims)
    pred_sum = pred_1h.sum(dim=dims)
    target_sum = target_1h.sum(dim=dims)

    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    return dice  # (C,)


def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute IoU (Jaccard index) per class.

    IoU_k = |P_k ∩ G_k| / |P_k ∪ G_k| + ε

    Args:
        pred: (B, H, W) predicted labels
        target: (B, H, W) ground truth labels
        num_classes: total number of classes
        eps: small constant for numerical stability

    Returns:
        (C,) tensor with IoU for each class 0..C-1
    """
    pred_1h = one_hot_from_labels(pred, num_classes)
    target_1h = one_hot_from_labels(target, num_classes)

    dims = (0, 2, 3)
    intersection = (pred_1h * target_1h).sum(dim=dims)
    union = pred_1h.sum(dim=dims) + target_1h.sum(dim=dims) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou  # (C,)


def iris_pupil_scores_from_logits(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute Dice/IoU for iris and pupil given logits and labels.

    Assumes class mapping: 0=background, 1=iris, 2=pupil

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary with keys: dice_iris, dice_pupil, dice_mean,
                             iou_iris, iou_pupil, iou_mean
    """
    num_classes = logits.shape[1]
    pred = logits_to_predictions(logits)
    dice = dice_per_class(pred, target, num_classes)
    iou = iou_per_class(pred, target, num_classes)

    dice_iris = dice[1].item()
    dice_pupil = dice[2].item()
    iou_iris = iou[1].item()
    iou_pupil = iou[2].item()

    return {
        "dice_iris": dice_iris,
        "dice_pupil": dice_pupil,
        "dice_mean": float((dice_iris + dice_pupil) / 2.0),
        "iou_iris": iou_iris,
        "iou_pupil": iou_pupil,
        "iou_mean": float((iou_iris + iou_pupil) / 2.0),
    }


# ============================================================================
# Center distance metrics
# ============================================================================

def center_of_mass(binary_mask: torch.Tensor) -> Tuple[float, float]:
    """
    Compute (x, y) center of mass for a binary mask in pixel coordinates.

    c_k = (x_k, y_k) = (1/N Σ x_i, 1/N Σ y_i)

    Args:
        binary_mask: (H, W) binary tensor

    Returns:
        (cx, cy) coordinates. Returns (-1, -1) if mask is empty.
    """
    ys, xs = torch.nonzero(binary_mask, as_tuple=True)
    if ys.numel() == 0:
        return -1.0, -1.0
    cx = xs.float().mean().item()
    cy = ys.float().mean().item()
    return cx, cy


def center_distance_for_class(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> float:
    """
    Compute Euclidean distance between predicted and GT centers for a specific class.

    d_k = sqrt((x_k^pred - x_k^gt)^2 + (y_k^pred - y_k^gt)^2)

    Args:
        pred: (H, W) integer labels (prediction)
        target: (H, W) integer labels (ground truth)
        class_id: which class to compute center for (1=iris, 2=pupil)

    Returns:
        Euclidean distance in pixels. Returns -1.0 if either mask is empty.
    """
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)

    cx_pred, cy_pred = center_of_mass(pred_mask)
    cx_gt, cy_gt = center_of_mass(target_mask)

    if cx_pred < 0 or cx_gt < 0:
        return -1.0

    dx = cx_pred - cx_gt
    dy = cy_pred - cy_gt
    return (dx * dx + dy * dy) ** 0.5


def center_distances_from_logits(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute average center distance (over batch) for iris and pupil.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary with keys: center_dist_iris_px, center_dist_pupil_px
    """
    pred = logits_to_predictions(logits)  # (B, H, W)
    b = pred.shape[0]

    d_iris_list = []
    d_pupil_list = []

    for i in range(b):
        d_iris = center_distance_for_class(pred[i], target[i], class_id=1)
        d_pupil = center_distance_for_class(pred[i], target[i], class_id=2)
        if d_iris >= 0:
            d_iris_list.append(d_iris)
        if d_pupil >= 0:
            d_pupil_list.append(d_pupil)

    d_iris_mean = float(sum(d_iris_list) / max(len(d_iris_list), 1))
    d_pupil_mean = float(sum(d_pupil_list) / max(len(d_pupil_list), 1))

    return {
        "center_dist_iris_px": d_iris_mean,
        "center_dist_pupil_px": d_pupil_mean,
    }


# ============================================================================
# HD95 (95% Hausdorff distance) on boundaries
# ============================================================================

def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Extract boundary pixels from a binary mask.

    Boundary = mask XOR erode(mask)

    Args:
        mask: (H, W) binary array

    Returns:
        (H, W) binary array of boundary pixels
    """
    eroded = binary_erosion(mask, border_value=0)
    boundary = mask ^ eroded
    return boundary


def hd95_for_class(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """
    Compute 95% Hausdorff distance for a specific class.

    HD95 is more robust than standard Hausdorff distance (which uses max).
    It computes the 95th percentile of distances between boundary points.

    Args:
        pred: (H, W) integer mask (prediction)
        target: (H, W) integer mask (ground truth)
        class_id: 1=iris, 2=pupil

    Returns:
        HD95 distance in pixels. Returns 0.0 if both masks empty,
        1e6 if only one mask is empty.
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)

    if not pred_mask.any() and not target_mask.any():
        return 0.0
    if not pred_mask.any() or not target_mask.any():
        return 1e6  # sentinel for missing mask

    pred_boundary = extract_boundary(pred_mask)
    target_boundary = extract_boundary(target_mask)

    # Distance transform: distances from any pixel to nearest boundary pixel
    dt_pred = distance_transform_edt(~pred_boundary)
    dt_target = distance_transform_edt(~target_boundary)

    # Distances: pred→target and target→pred
    pred_to_target = dt_target[pred_boundary]
    target_to_pred = dt_pred[target_boundary]

    all_dists = np.concatenate([pred_to_target, target_to_pred])
    if all_dists.size == 0:
        return 0.0

    return float(np.percentile(all_dists, 95))


def hd95_batch(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute mean HD95 for iris and pupil over a batch.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary with keys: hd95_iris, hd95_pupil
    """
    pred = logits_to_predictions(logits).cpu().numpy()
    gt = target.cpu().numpy()

    hd_iris = []
    hd_pupil = []

    for i in range(pred.shape[0]):
        hd_iris.append(hd95_for_class(pred[i], gt[i], class_id=1))
        hd_pupil.append(hd95_for_class(pred[i], gt[i], class_id=2))

    return {
        "hd95_iris": float(np.mean(hd_iris)),
        "hd95_pupil": float(np.mean(hd_pupil)),
    }


# ============================================================================
# Combined metric computation
# ============================================================================

def compute_all_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute all segmentation metrics in one call.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary with all metric keys:
        - dice_iris, dice_pupil, dice_mean
        - iou_iris, iou_pupil, iou_mean
        - center_dist_iris_px, center_dist_pupil_px
        - hd95_iris, hd95_pupil
    """
    metrics_overlap = iris_pupil_scores_from_logits(logits, target)
    metrics_center = center_distances_from_logits(logits, target)
    metrics_hd = hd95_batch(logits, target)

    return {**metrics_overlap, **metrics_center, **metrics_hd}
