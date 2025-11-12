import torch
import numpy as np

@torch.no_grad()
def mean_iou_ignore_bg(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int = 3) -> float:
    preds = logits.argmax(1)
    ious = []
    for c in range(1, num_classes):  # ignore class 0 (background)
        inter = ((preds==c) & (y_true==c)).sum().item()
        union = ((preds==c) | (y_true==c)).sum().item()
        ious.append(1.0 if union==0 else inter/union)
    return float(np.mean(ious)) if ious else 0.0
