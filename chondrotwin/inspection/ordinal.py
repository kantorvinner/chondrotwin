from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..assay.dirichlet import expected_probability


def damage_probability(alpha: torch.Tensor) -> torch.Tensor:
    prob = expected_probability(alpha)
    return 1.0 - prob[..., 0]


def detection_auroc(
    alpha: torch.Tensor, grade: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    score = damage_probability(alpha).reshape(-1).detach().cpu().numpy()
    positive = (grade.reshape(-1) >= 1).long().detach().cpu().numpy()
    if mask is not None:
        keep = mask.reshape(-1).detach().cpu().numpy().astype(bool)
        score, positive = score[keep], positive[keep]
    if positive.min() == positive.max():
        return float("nan")
    return float(roc_auc_score(positive, score))


def ordinal_mae(alpha: torch.Tensor, grade: torch.Tensor) -> float:
    pred = expected_probability(alpha).argmax(dim=-1)
    return float((pred - grade).abs().float().mean())


def grade_accuracy(alpha: torch.Tensor, grade: torch.Tensor) -> float:
    pred = expected_probability(alpha).argmax(dim=-1)
    return float((pred == grade).float().mean())


def quadratic_weighted_kappa(alpha: torch.Tensor, grade: torch.Tensor, num_grades: int) -> float:
    pred = expected_probability(alpha).argmax(dim=-1).reshape(-1).cpu().numpy()
    truth = grade.reshape(-1).cpu().numpy()
    observed = np.zeros((num_grades, num_grades))
    for p, t in zip(pred, truth, strict=True):
        observed[t, p] += 1
    weights = np.zeros((num_grades, num_grades))
    for i in range(num_grades):
        for j in range(num_grades):
            weights[i, j] = ((i - j) ** 2) / ((num_grades - 1) ** 2)
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col) / max(observed.sum(), 1.0)
    denom = (weights * expected).sum()
    if denom == 0:
        return 1.0
    return float(1.0 - (weights * observed).sum() / denom)
