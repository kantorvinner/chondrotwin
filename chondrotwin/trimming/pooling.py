from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PyramidPooling(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bins: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.bins = bins
        self.stages = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, out_dim),
                nn.GELU(),
            )
            for _ in bins
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim + out_dim * len(bins), out_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        pooled = [x]
        for bin_size, stage in zip(self.bins, self.stages, strict=True):
            size = max(1, min(bin_size, h, w))
            kh = max(1, h // size)
            kw = max(1, w // size)
            feat = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw), ceil_mode=True)
            feat = stage(feat)
            pooled.append(F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(pooled, dim=1))
