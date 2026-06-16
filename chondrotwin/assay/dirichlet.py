from __future__ import annotations

import torch


def strength(alpha: torch.Tensor) -> torch.Tensor:
    return alpha.sum(dim=-1, keepdim=True)


def expected_probability(alpha: torch.Tensor) -> torch.Tensor:
    return alpha / strength(alpha)


def vacuity(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[-1]
    return k / strength(alpha).squeeze(-1)


def belief(alpha: torch.Tensor) -> torch.Tensor:
    evidence = alpha - 1.0
    return evidence / strength(alpha)


def kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[-1]
    s = alpha.sum(dim=-1)
    ones = torch.ones_like(alpha)
    log_norm = (
        torch.lgamma(s) - torch.lgamma(torch.tensor(float(k))) - torch.lgamma(alpha).sum(dim=-1)
    )
    digamma = torch.digamma(alpha) - torch.digamma(s.unsqueeze(-1))
    aligned = ((alpha - ones) * digamma).sum(dim=-1)
    return log_norm + aligned
