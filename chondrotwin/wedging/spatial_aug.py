from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class SpatialConfig:
    rotation_deg: float = 10.0
    scale_low: float = 0.9
    scale_high: float = 1.1
    translate_voxels: float = 15.0
    elastic_sigma: float = 4.0
    elastic_alpha: float = 8.0


def _affine_theta(rotation: float, scale: float, tx: float, ty: float) -> torch.Tensor:
    cos = math.cos(rotation)
    sin = math.sin(rotation)
    return torch.tensor(
        [[cos * scale, -sin * scale, tx], [sin * scale, cos * scale, ty]],
        dtype=torch.float,
    )


def affine_slicewise(
    volume: torch.Tensor, cfg: SpatialConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    c, d, h, w = volume.shape
    rotation = math.radians(
        float(torch.empty(1).uniform_(-cfg.rotation_deg, cfg.rotation_deg, generator=generator))
    )
    scale = float(torch.empty(1).uniform_(cfg.scale_low, cfg.scale_high, generator=generator))
    tx = (
        float(
            torch.empty(1).uniform_(
                -cfg.translate_voxels, cfg.translate_voxels, generator=generator
            )
        )
        / w
    )
    ty = (
        float(
            torch.empty(1).uniform_(
                -cfg.translate_voxels, cfg.translate_voxels, generator=generator
            )
        )
        / h
    )
    theta = _affine_theta(rotation, scale, tx, ty).unsqueeze(0).expand(d, -1, -1)
    planar = volume.permute(1, 0, 2, 3)
    grid = F.affine_grid(theta, [d, c, h, w], align_corners=False)
    warped = F.grid_sample(
        planar, grid, align_corners=False, mode="bilinear", padding_mode="border"
    )
    return warped.permute(1, 0, 2, 3)


def elastic_slicewise(
    volume: torch.Tensor, cfg: SpatialConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    c, d, h, w = volume.shape
    disp = torch.randn(1, 2, h, w, generator=generator)
    kernel = _gaussian_kernel(cfg.elastic_sigma)
    disp = _smooth(disp, kernel) * cfg.elastic_alpha
    base = F.affine_grid(torch.eye(2, 3).unsqueeze(0), [1, c, h, w], align_corners=False)
    flow = disp.permute(0, 2, 3, 1)
    flow = flow * torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)])
    grid = (base + flow).expand(d, -1, -1, -1)
    planar = volume.permute(1, 0, 2, 3)
    warped = F.grid_sample(
        planar, grid, align_corners=False, mode="bilinear", padding_mode="border"
    )
    return warped.permute(1, 0, 2, 3)


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1).float()
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g).view(1, 1, 2 * radius + 1, 2 * radius + 1)


def _smooth(disp: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = kernel.shape[-1] // 2
    channels = disp.shape[1]
    weight = kernel.expand(channels, 1, -1, -1)
    return F.conv2d(disp, weight, padding=pad, groups=channels)
