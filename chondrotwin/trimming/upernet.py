from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .pooling import PyramidPooling
from .pyramid import FeaturePyramid


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    embed_dim: int = 1024
    channels: int = 256
    levels: int = 4
    num_classes: int = 7


class UPerHead(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ppm = PyramidPooling(cfg.embed_dim, cfg.channels)
        self.fpn = FeaturePyramid(cfg.embed_dim, cfg.channels, cfg.levels)
        self.fuse = nn.Sequential(
            nn.Conv2d(
                cfg.channels * (cfg.levels + 1), cfg.channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(8, cfg.channels),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(cfg.channels, cfg.num_classes, kernel_size=1)

    def forward(self, feature_volume: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
        b, _c, d, _gh, _gw = feature_volume.shape
        planar = rearrange(feature_volume, "b c d gh gw -> (b d) c gh gw")
        context = self.ppm(planar)
        pyramid = self.fpn(planar)
        size = pyramid[0].shape[-2:]
        gathered = [F.interpolate(context, size=size, mode="bilinear", align_corners=False)]
        gathered.extend(
            F.interpolate(level, size=size, mode="bilinear", align_corners=False)
            for level in pyramid
        )
        fused = self.fuse(torch.cat(gathered, dim=1))
        logits = self.classifier(fused)
        logits = rearrange(logits, "(b d) k gh gw -> b k d gh gw", b=b, d=d)
        return F.interpolate(logits, size=target, mode="trilinear", align_corners=False)
