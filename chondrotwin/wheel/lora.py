from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    targets: tuple[str, ...] = ("q", "v")
    num_layers: int = 24


class LoRALinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: float, bias: bool = True
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen = self.base(x)
        update = (x @ self.lora_a.t()) @ self.lora_b.t()
        return frozen + self.scaling * update

    def trainable_parameters(self) -> int:
        return self.lora_a.numel() + self.lora_b.numel()


def lora_parameter_count(cfg: LoRAConfig, hidden: int) -> int:
    per_proj = cfg.rank * hidden + hidden * cfg.rank
    return per_proj * len(cfg.targets) * cfg.num_layers
