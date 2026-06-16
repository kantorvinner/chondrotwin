from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    warmup_epochs: int = 5
    total_epochs: int = 120
    peak_lr: float = 2.0e-4
    floor_lr: float = 1.0e-6


def lr_at(epoch: float, cfg: ScheduleConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.peak_lr * (epoch + 1) / cfg.warmup_epochs
    span = max(1, cfg.total_epochs - cfg.warmup_epochs)
    progress = min(1.0, (epoch - cfg.warmup_epochs) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.floor_lr + (cfg.peak_lr - cfg.floor_lr) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: ScheduleConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(epoch: int) -> float:
        return lr_at(float(epoch), cfg) / cfg.peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
