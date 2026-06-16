from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dirichlet import kl_to_uniform, strength


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    num_grades: int = 4
    anneal_epochs: int = 10


def anneal_coefficient(epoch: int, anneal_epochs: int) -> float:
    if anneal_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(anneal_epochs))


def bayes_risk(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    s = strength(alpha)
    prob = alpha / s
    onehot = F.one_hot(target, num_classes=alpha.shape[-1]).float()
    err = ((onehot - prob) ** 2).sum(dim=-1)
    var = (prob * (1.0 - prob) / (s + 1.0)).sum(dim=-1)
    return err + var


def evidential_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    cfg: EvidenceConfig,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    risk = bayes_risk(alpha, target)
    onehot = F.one_hot(target, num_classes=alpha.shape[-1]).float()
    evidence_removed = onehot + (1.0 - onehot) * alpha
    penalty = kl_to_uniform(evidence_removed)
    coeff = anneal_coefficient(epoch, cfg.anneal_epochs)
    per_node = risk + coeff * penalty
    if mask is not None:
        weight = mask.float()
        return (per_node * weight).sum() / weight.sum().clamp_min(1.0)
    return per_node.mean()
