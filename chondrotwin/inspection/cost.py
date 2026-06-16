from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class CostRecord:
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int

    @property
    def trainable_fraction(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.trainable_parameters / self.total_parameters


def profile_parameters(model: nn.Module) -> CostRecord:
    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return CostRecord(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
    )
