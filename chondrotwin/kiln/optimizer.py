from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-2
    betas: tuple[float, float] = (0.9, 0.999)


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
