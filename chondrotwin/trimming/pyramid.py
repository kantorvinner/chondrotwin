from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FeaturePyramid(nn.Module):
    def __init__(self, in_dim: int, channels: int, levels: int = 4) -> None:
        super().__init__()
        self.levels = levels
        self.downsample = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    in_dim, in_dim, kernel_size=3, stride=2 if lvl > 0 else 1, padding=1, bias=False
                ),
                nn.GroupNorm(8, in_dim),
                nn.GELU(),
            )
            for lvl in range(levels)
        )
        self.lateral = nn.ModuleList(
            nn.Conv2d(in_dim, channels, kernel_size=1) for _ in range(levels)
        )
        self.smooth = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(levels)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        current = x
        for lvl in range(self.levels):
            current = self.downsample[lvl](current)
            feats.append(self.lateral[lvl](current))
        merged = [feats[-1]]
        for lvl in range(self.levels - 2, -1, -1):
            up = F.interpolate(
                merged[0], size=feats[lvl].shape[-2:], mode="bilinear", align_corners=False
            )
            merged.insert(0, feats[lvl] + up)
        return [self.smooth[lvl](merged[lvl]) for lvl in range(self.levels)]
