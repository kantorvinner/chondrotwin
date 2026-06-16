from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, slots=True)
class HeadConfig:
    in_dim: int = 256
    hidden_dim: int = 128
    num_grades: int = 4


class DirichletHead(nn.Module):
    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_grades),
        )

    def evidence(self, node_features: torch.Tensor) -> torch.Tensor:
        logits = self.net(node_features)
        return F.softplus(logits)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.evidence(node_features) + 1.0
