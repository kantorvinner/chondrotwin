from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_per_class(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6
) -> torch.Tensor:
    pred_oh = F.one_hot(pred, num_classes).movedim(-1, 1).float()
    target_oh = F.one_hot(target, num_classes).movedim(-1, 1).float()
    dims = (0, *range(2, pred_oh.dim()))
    intersection = (pred_oh * target_oh).sum(dim=dims)
    cardinality = pred_oh.sum(dim=dims) + target_oh.sum(dim=dims)
    return (2.0 * intersection + eps) / (cardinality + eps)


def mean_foreground_dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    num_classes = logits.shape[1]
    pred = logits.argmax(dim=1)
    scores = dice_per_class(pred, target, num_classes)
    return float(scores[1:].mean())
