from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    num_classes = logits.shape[1]
    prob = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes).movedim(-1, 1).float()
    dims = (0, *range(2, prob.dim()))
    intersection = (prob * onehot).sum(dim=dims)
    cardinality = prob.sum(dim=dims) + onehot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def segmentation_objective(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return soft_dice(logits, target) + F.cross_entropy(logits, target)
