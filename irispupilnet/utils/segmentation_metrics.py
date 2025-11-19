"""
Segmentation metrics for iris and pupil segmentation.

Metrics implemented:

**Overlap metrics:**
- Dice score (per class): Measures overlap between prediction and ground truth
- IoU/Jaccard (per class): Another overlap metric, IoU = TP/(TP+FP+FN)

**Pixel-wise classification:**
- Precision (per class): TP/(TP+FP), measures fraction of predicted positives that are correct
- Recall (per class): TP/(TP+FN), measures fraction of actual positives that are detected

**Centroid-based:**
- Center distance: Euclidean distance (in pixels) between predicted and GT centroids

**Boundary distance metrics:**
- HD95 (95% Hausdorff distance): Robust boundary distance, ignores outliers
- ASSD (Average Symmetric Surface Distance): Mean of bidirectional surface distances
- MASD (Maximum Average Surface Distance): Max of bidirectional surface distances

**Boundary agreement:**
- Boundary IoU: IoU computed only on dilated boundary regions (tolerance-based)
- NSD (Normalized Surface Dice): Dice-like metric on boundary points within tolerance (tau)

**Shape metrics:**
- Radius error: |r_pred - r_gt| assuming circular structures
- Area relative error: |area_pred - area_gt| / area_gt

All metrics return np.nan for invalid cases (e.g., empty masks) and use np.nanmean for averaging.

Class mapping: 0=background, 1=iris, 2=pupil
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt


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
# Precision, Recall, F1 (pixel-wise classification metrics)
# ============================================================================

def precision_recall_f1_for_class(pred: torch.Tensor, target: torch.Tensor, class_id: int, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute pixel-wise Precision, Recall, and F1 for a specific class.

    TP = pixels where pred = class_id AND target = class_id
    FP = pixels where pred = class_id AND target ≠ class_id
    FN = pixels where pred ≠ class_id AND target = class_id

    Precision = TP / (TP + FP + ε)
    Recall = TP / (TP + FN + ε)
    F1 = 2 * Precision * Recall / (Precision + Recall + ε)

    Args:
        pred: (B, H, W) predicted labels
        target: (B, H, W) ground truth labels
        class_id: which class to compute for (1=iris, 2=pupil)
        eps: small constant for numerical stability

    Returns:
        Dictionary with keys: precision, recall, f1
    """
    pred_positive = (pred == class_id)
    target_positive = (target == class_id)

    tp = (pred_positive & target_positive).sum().float()
    fp = (pred_positive & ~target_positive).sum().float()
    fn = (~pred_positive & target_positive).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def precision_recall_f1_from_logits(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1 for iris and pupil from logits.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary with keys: precision_iris, recall_iris, f1_iris,
                             precision_pupil, recall_pupil, f1_pupil
    """
    pred = logits_to_predictions(logits)

    metrics_iris = precision_recall_f1_for_class(pred, target, class_id=1)
    metrics_pupil = precision_recall_f1_for_class(pred, target, class_id=2)

    return {
        "precision_iris": metrics_iris["precision"],
        "recall_iris": metrics_iris["recall"],
        "f1_iris": metrics_iris["f1"],
        "precision_pupil": metrics_pupil["precision"],
        "recall_pupil": metrics_pupil["recall"],
        "f1_pupil": metrics_pupil["f1"],
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
        np.nan if only one mask is empty (cannot compute HD95).
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)

    # Case 1: both masks empty → no structure in GT or pred, consider perfect
    if not pred_mask.any() and not target_mask.any():
        return 0.0

    # Case 2: only one mask is empty → cannot compute HD95, return NaN
    if not pred_mask.any() or not target_mask.any():
        return float("nan")

    pred_boundary = extract_boundary(pred_mask)
    target_boundary = extract_boundary(target_mask)

    # Distance transform: distances from any pixel to nearest boundary pixel
    dt_pred = distance_transform_edt(np.logical_not(pred_boundary))
    dt_target = distance_transform_edt(np.logical_not(target_boundary))

    # Distances: pred→target and target→pred
    pred_to_target = dt_target[pred_boundary]
    target_to_pred = dt_pred[target_boundary]

    all_dists = np.concatenate([pred_to_target, target_to_pred])
    if all_dists.size == 0:
        return 0.0

    return float(np.percentile(all_dists, 95))


def hd95_batch(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute mean HD95 for iris and pupil over a batch, ignoring cases
    where one of the masks is empty (NaN values).

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
        h_i = hd95_for_class(pred[i], gt[i], class_id=1)
        h_p = hd95_for_class(pred[i], gt[i], class_id=2)
        hd_iris.append(h_i)
        hd_pupil.append(h_p)

    hd_iris = np.array(hd_iris, dtype=float)
    hd_pupil = np.array(hd_pupil, dtype=float)

    # Use nanmean to ignore NaN values (cases where mask is missing)
    iris_mean = float(np.nanmean(hd_iris)) if not np.all(np.isnan(hd_iris)) else 0.0
    pupil_mean = float(np.nanmean(hd_pupil)) if not np.all(np.isnan(hd_pupil)) else 0.0

    return {
        "hd95_iris": iris_mean,
        "hd95_pupil": pupil_mean,
    }


# ============================================================================
# ASSD and MASD (Average Surface Distance)
# ============================================================================

def assd_masd_for_class(pred: np.ndarray, target: np.ndarray, class_id: int) -> Dict[str, float]:
    """
    Compute ASSD (Average Symmetric Surface Distance) and MASD (Maximum ASD) for a class.

    ASSD = (ASD_pred→gt + ASD_gt→pred) / 2
    MASD = max(ASD_pred→gt, ASD_gt→pred)

    where ASD_pred→gt is the mean distance from predicted boundary to GT boundary.

    Args:
        pred: (H, W) integer mask (prediction)
        target: (H, W) integer mask (ground truth)
        class_id: 1=iris, 2=pupil

    Returns:
        Dictionary with keys: assd, masd. Returns NaN if one mask is empty.
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)

    # Case 1: both masks empty → perfect match
    if not pred_mask.any() and not target_mask.any():
        return {"assd": 0.0, "masd": 0.0}

    # Case 2: only one mask empty → cannot compute, return NaN
    if not pred_mask.any() or not target_mask.any():
        return {"assd": float("nan"), "masd": float("nan")}

    pred_boundary = extract_boundary(pred_mask)
    target_boundary = extract_boundary(target_mask)

    # Distance transforms
    dt_pred = distance_transform_edt(np.logical_not(pred_boundary))
    dt_target = distance_transform_edt(np.logical_not(target_boundary))

    # Distances: pred→target and target→pred
    pred_to_target = dt_target[pred_boundary]
    target_to_pred = dt_pred[target_boundary]

    if pred_to_target.size == 0 or target_to_pred.size == 0:
        return {"assd": 0.0, "masd": 0.0}

    asd_pred_to_gt = float(np.mean(pred_to_target))
    asd_gt_to_pred = float(np.mean(target_to_pred))

    assd = (asd_pred_to_gt + asd_gt_to_pred) / 2.0
    masd = max(asd_pred_to_gt, asd_gt_to_pred)

    return {"assd": assd, "masd": masd}


def assd_batch(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute mean ASSD for iris and pupil over a batch.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary with keys: assd_iris, assd_pupil
    """
    pred = logits_to_predictions(logits).cpu().numpy()
    gt = target.cpu().numpy()

    assd_iris_list = []
    assd_pupil_list = []

    for i in range(pred.shape[0]):
        metrics_iris = assd_masd_for_class(pred[i], gt[i], class_id=1)
        metrics_pupil = assd_masd_for_class(pred[i], gt[i], class_id=2)
        assd_iris_list.append(metrics_iris["assd"])
        assd_pupil_list.append(metrics_pupil["assd"])

    assd_iris = np.array(assd_iris_list, dtype=float)
    assd_pupil = np.array(assd_pupil_list, dtype=float)

    iris_mean = float(np.nanmean(assd_iris)) if not np.all(np.isnan(assd_iris)) else 0.0
    pupil_mean = float(np.nanmean(assd_pupil)) if not np.all(np.isnan(assd_pupil)) else 0.0

    return {
        "assd_iris": iris_mean,
        "assd_pupil": pupil_mean,
    }


def masd_batch(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute mean MASD for iris and pupil over a batch.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary with keys: masd_iris, masd_pupil
    """
    pred = logits_to_predictions(logits).cpu().numpy()
    gt = target.cpu().numpy()

    masd_iris_list = []
    masd_pupil_list = []

    for i in range(pred.shape[0]):
        metrics_iris = assd_masd_for_class(pred[i], gt[i], class_id=1)
        metrics_pupil = assd_masd_for_class(pred[i], gt[i], class_id=2)
        masd_iris_list.append(metrics_iris["masd"])
        masd_pupil_list.append(metrics_pupil["masd"])

    masd_iris = np.array(masd_iris_list, dtype=float)
    masd_pupil = np.array(masd_pupil_list, dtype=float)

    iris_mean = float(np.nanmean(masd_iris)) if not np.all(np.isnan(masd_iris)) else 0.0
    pupil_mean = float(np.nanmean(masd_pupil)) if not np.all(np.isnan(masd_pupil)) else 0.0

    return {
        "masd_iris": iris_mean,
        "masd_pupil": pupil_mean,
    }


# ============================================================================
# Boundary IoU
# ============================================================================

def boundary_iou_for_class(pred: np.ndarray, target: np.ndarray, class_id: int, dilation_radius: int = 1, eps: float = 1e-6) -> float:
    """
    Compute IoU on boundary regions only (with optional dilation for tolerance).

    BoundaryIoU = |boundary_pred ∩ boundary_gt| / |boundary_pred ∪ boundary_gt|

    Args:
        pred: (H, W) integer mask (prediction)
        target: (H, W) integer mask (ground truth)
        class_id: 1=iris, 2=pupil
        dilation_radius: pixels to dilate boundaries (for tolerance), default 1
        eps: small constant for numerical stability

    Returns:
        Boundary IoU score. Returns 0.0 if both boundaries empty, NaN if only one empty.
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)

    if not pred_mask.any() and not target_mask.any():
        return 0.0
    if not pred_mask.any() or not target_mask.any():
        return float("nan")

    pred_boundary = extract_boundary(pred_mask)
    target_boundary = extract_boundary(target_mask)

    # Optional: dilate boundaries to allow small tolerance
    if dilation_radius > 0:
        structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
        pred_boundary = binary_dilation(pred_boundary, structure=structure)
        target_boundary = binary_dilation(target_boundary, structure=structure)

    intersection = np.logical_and(pred_boundary, target_boundary).sum()
    union = np.logical_or(pred_boundary, target_boundary).sum()

    if union == 0:
        return 0.0

    return float(intersection / (union + eps))


def boundary_iou_batch(logits: torch.Tensor, target: torch.Tensor, dilation_radius: int = 1) -> Dict[str, float]:
    """
    Compute mean Boundary IoU for iris and pupil over a batch.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels
        dilation_radius: pixels to dilate boundaries for tolerance

    Returns:
        Dictionary with keys: boundary_iou_iris, boundary_iou_pupil
    """
    pred = logits_to_predictions(logits).cpu().numpy()
    gt = target.cpu().numpy()

    biou_iris_list = []
    biou_pupil_list = []

    for i in range(pred.shape[0]):
        biou_i = boundary_iou_for_class(pred[i], gt[i], class_id=1, dilation_radius=dilation_radius)
        biou_p = boundary_iou_for_class(pred[i], gt[i], class_id=2, dilation_radius=dilation_radius)
        biou_iris_list.append(biou_i)
        biou_pupil_list.append(biou_p)

    biou_iris = np.array(biou_iris_list, dtype=float)
    biou_pupil = np.array(biou_pupil_list, dtype=float)

    iris_mean = float(np.nanmean(biou_iris)) if not np.all(np.isnan(biou_iris)) else 0.0
    pupil_mean = float(np.nanmean(biou_pupil)) if not np.all(np.isnan(biou_pupil)) else 0.0

    return {
        "boundary_iou_iris": iris_mean,
        "boundary_iou_pupil": pupil_mean,
    }


# ============================================================================
# NSD (Normalized Surface Dice)
# ============================================================================

def nsd_for_class(pred: np.ndarray, target: np.ndarray, class_id: int, tau_px: float = 2.0, eps: float = 1e-6) -> float:
    """
    Compute Normalized Surface Dice with tolerance threshold tau.

    NSD = 2 * TP_surface / (2 * TP_surface + FP_surface + FN_surface)

    where:
    - TP_surface: boundary points within tau pixels of GT boundary
    - FP_surface: boundary pred points > tau from GT
    - FN_surface: boundary GT points > tau from pred

    Args:
        pred: (H, W) integer mask (prediction)
        target: (H, W) integer mask (ground truth)
        class_id: 1=iris, 2=pupil
        tau_px: tolerance threshold in pixels (default 2.0)
        eps: small constant for numerical stability

    Returns:
        NSD score. Returns 0.0 if both boundaries empty, NaN if only one empty.
    """
    pred_mask = (pred == class_id).astype(bool)
    target_mask = (target == class_id).astype(bool)

    if not pred_mask.any() and not target_mask.any():
        return 1.0  # Perfect match
    if not pred_mask.any() or not target_mask.any():
        return float("nan")

    pred_boundary = extract_boundary(pred_mask)
    target_boundary = extract_boundary(target_mask)

    # Distance transforms
    dt_pred = distance_transform_edt(np.logical_not(pred_boundary))
    dt_target = distance_transform_edt(np.logical_not(target_boundary))

    # Distances: pred→target and target→pred
    pred_to_target = dt_target[pred_boundary]
    target_to_pred = dt_pred[target_boundary]

    if pred_to_target.size == 0 and target_to_pred.size == 0:
        return 0.0

    # Count points within tolerance
    tp_pred = (pred_to_target <= tau_px).sum()  # Pred boundary points close to GT
    fp_pred = (pred_to_target > tau_px).sum()   # Pred boundary points far from GT

    tp_gt = (target_to_pred <= tau_px).sum()    # GT boundary points close to pred
    fn_gt = (target_to_pred > tau_px).sum()     # GT boundary points far from pred

    # NSD is symmetric Dice on surface
    # TP_surface = agreement between both directions
    # We approximate: TP ≈ (tp_pred + tp_gt) / 2, FP ≈ fp_pred, FN ≈ fn_gt
    tp_surface = (tp_pred + tp_gt) / 2.0
    fp_surface = fp_pred
    fn_surface = fn_gt

    nsd = (2 * tp_surface) / (2 * tp_surface + fp_surface + fn_surface + eps)

    return float(nsd)


def nsd_batch(logits: torch.Tensor, target: torch.Tensor, tau_px: float = 2.0) -> Dict[str, float]:
    """
    Compute mean NSD for iris and pupil over a batch.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels
        tau_px: tolerance threshold in pixels

    Returns:
        Dictionary with keys: nsd_iris, nsd_pupil
    """
    pred = logits_to_predictions(logits).cpu().numpy()
    gt = target.cpu().numpy()

    nsd_iris_list = []
    nsd_pupil_list = []

    for i in range(pred.shape[0]):
        nsd_i = nsd_for_class(pred[i], gt[i], class_id=1, tau_px=tau_px)
        nsd_p = nsd_for_class(pred[i], gt[i], class_id=2, tau_px=tau_px)
        nsd_iris_list.append(nsd_i)
        nsd_pupil_list.append(nsd_p)

    nsd_iris = np.array(nsd_iris_list, dtype=float)
    nsd_pupil = np.array(nsd_pupil_list, dtype=float)

    iris_mean = float(np.nanmean(nsd_iris)) if not np.all(np.isnan(nsd_iris)) else 0.0
    pupil_mean = float(np.nanmean(nsd_pupil)) if not np.all(np.isnan(nsd_pupil)) else 0.0

    return {
        "nsd_iris": iris_mean,
        "nsd_pupil": pupil_mean,
    }


# ============================================================================
# Shape errors: radius and area
# ============================================================================

def shape_errors_for_class(pred: np.ndarray, target: np.ndarray, class_id: int, eps: float = 1e-6) -> Dict[str, float]:
    """
    Compute radius error and relative area error for a class (assuming roughly circular).

    radius_error = |r_pred - r_gt|
    area_rel_error = |A_pred - A_gt| / (A_gt + ε)

    where radius = sqrt(area / π)

    Args:
        pred: (H, W) integer mask (prediction)
        target: (H, W) integer mask (ground truth)
        class_id: 1=iris, 2=pupil
        eps: small constant for numerical stability

    Returns:
        Dictionary with keys: radius_error, area_rel_error
        Returns NaN if GT area is 0.
    """
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)

    area_gt = target_mask.sum()
    area_pred = pred_mask.sum()

    if area_gt == 0:
        return {"radius_error": float("nan"), "area_rel_error": float("nan")}

    # Equivalent radius assuming circle
    radius_gt = np.sqrt(area_gt / np.pi)
    radius_pred = np.sqrt(area_pred / np.pi) if area_pred > 0 else 0.0

    radius_error = abs(radius_pred - radius_gt)
    area_rel_error = abs(area_pred - area_gt) / (area_gt + eps)

    return {
        "radius_error": float(radius_error),
        "area_rel_error": float(area_rel_error),
    }


def shape_errors_batch(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute mean shape errors for iris and pupil over a batch.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels

    Returns:
        Dictionary with keys: radius_error_iris, radius_error_pupil,
                             area_rel_error_iris, area_rel_error_pupil
    """
    pred = logits_to_predictions(logits).cpu().numpy()
    gt = target.cpu().numpy()

    radius_err_iris_list = []
    radius_err_pupil_list = []
    area_err_iris_list = []
    area_err_pupil_list = []

    for i in range(pred.shape[0]):
        errors_iris = shape_errors_for_class(pred[i], gt[i], class_id=1)
        errors_pupil = shape_errors_for_class(pred[i], gt[i], class_id=2)

        radius_err_iris_list.append(errors_iris["radius_error"])
        radius_err_pupil_list.append(errors_pupil["radius_error"])
        area_err_iris_list.append(errors_iris["area_rel_error"])
        area_err_pupil_list.append(errors_pupil["area_rel_error"])

    radius_err_iris = np.array(radius_err_iris_list, dtype=float)
    radius_err_pupil = np.array(radius_err_pupil_list, dtype=float)
    area_err_iris = np.array(area_err_iris_list, dtype=float)
    area_err_pupil = np.array(area_err_pupil_list, dtype=float)

    return {
        "radius_error_iris": float(np.nanmean(radius_err_iris)) if not np.all(np.isnan(radius_err_iris)) else 0.0,
        "radius_error_pupil": float(np.nanmean(radius_err_pupil)) if not np.all(np.isnan(radius_err_pupil)) else 0.0,
        "area_rel_error_iris": float(np.nanmean(area_err_iris)) if not np.all(np.isnan(area_err_iris)) else 0.0,
        "area_rel_error_pupil": float(np.nanmean(area_err_pupil)) if not np.all(np.isnan(area_err_pupil)) else 0.0,
    }


# ============================================================================
# Combined metric computation
# ============================================================================

def compute_all_metrics(logits: torch.Tensor, target: torch.Tensor,
                       tau_px: float = 2.0, boundary_dilation: int = 1) -> Dict[str, float]:
    """
    Compute all segmentation metrics in one call.

    Args:
        logits: (B, C, H, W) model output logits
        target: (B, H, W) ground truth labels
        tau_px: tolerance threshold for NSD in pixels (default 2.0)
        boundary_dilation: dilation radius for Boundary IoU (default 1)

    Returns:
        Dictionary with all metric keys:
        - Overlap: dice_iris, dice_pupil, dice_mean, iou_iris, iou_pupil, iou_mean
        - Pixel-wise classification: precision_iris, recall_iris, precision_pupil, recall_pupil
        - Center distances: center_dist_iris_px, center_dist_pupil_px
        - Boundary distances: hd95_iris, hd95_pupil, assd_iris, assd_pupil, masd_iris, masd_pupil
        - Boundary agreement: boundary_iou_iris, boundary_iou_pupil, nsd_iris, nsd_pupil
        - Shape errors: radius_error_iris, radius_error_pupil, area_rel_error_iris, area_rel_error_pupil
    """
    metrics_overlap = iris_pupil_scores_from_logits(logits, target)
    metrics_precision_recall = precision_recall_f1_from_logits(logits, target)
    metrics_center = center_distances_from_logits(logits, target)
    metrics_hd95 = hd95_batch(logits, target)
    metrics_assd = assd_batch(logits, target)
    metrics_masd = masd_batch(logits, target)
    metrics_boundary_iou = boundary_iou_batch(logits, target, dilation_radius=boundary_dilation)
    metrics_nsd = nsd_batch(logits, target, tau_px=tau_px)
    metrics_shape = shape_errors_batch(logits, target)

    # Note: we exclude f1_iris and f1_pupil from precision_recall_f1_from_logits
    # since F1 = Dice (already in metrics_overlap)
    metrics_precision_recall_filtered = {
        k: v for k, v in metrics_precision_recall.items() if not k.startswith("f1_")
    }

    return {
        **metrics_overlap,
        **metrics_precision_recall_filtered,
        **metrics_center,
        **metrics_hd95,
        **metrics_assd,
        **metrics_masd,
        **metrics_boundary_iou,
        **metrics_nsd,
        **metrics_shape,
    }
