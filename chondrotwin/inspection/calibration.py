from __future__ import annotations

from itertools import pairwise

import torch


def expected_calibration_error(prob: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    confidence, prediction = prob.max(dim=-1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    total = confidence.numel()
    ece = torch.zeros(())
    for lower, upper in pairwise(edges):
        in_bin = (confidence > lower) & (confidence <= upper)
        count = in_bin.sum()
        if count > 0:
            avg_conf = confidence[in_bin].mean()
            avg_acc = correct[in_bin].mean()
            ece += (count / total) * (avg_conf - avg_acc).abs()
    return float(ece)


def maximum_calibration_error(prob: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    confidence, prediction = prob.max(dim=-1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0.0, 1.0, bins + 1)
    worst = torch.zeros(())
    for lower, upper in pairwise(edges):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.sum() > 0:
            gap = (confidence[in_bin].mean() - correct[in_bin].mean()).abs()
            worst = torch.maximum(worst, gap)
    return float(worst)
