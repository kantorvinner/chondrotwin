from __future__ import annotations

from typing import NamedTuple

import torch

from .dirichlet import expected_probability, strength


class UncertaintySplit(NamedTuple):
    total: torch.Tensor
    aleatoric: torch.Tensor
    epistemic: torch.Tensor


def _entropy(prob: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return -(prob * torch.log(prob + eps)).sum(dim=-1)


def decompose(alpha: torch.Tensor) -> UncertaintySplit:
    s = strength(alpha)
    mean = expected_probability(alpha)
    total = _entropy(mean)
    digamma_term = torch.digamma(alpha + 1.0) - torch.digamma(s + 1.0)
    aleatoric = -(mean * digamma_term).sum(dim=-1)
    epistemic = (total - aleatoric).clamp_min(0.0)
    return UncertaintySplit(total=total, aleatoric=aleatoric, epistemic=epistemic)
