from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DepthAwarePositional(nn.Module):
    def __init__(
        self, embed_dim: int, base_grid: int, depth_tokens: int, depth_stride: int = 4
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.base_grid = base_grid
        self.depth_stride = depth_stride
        self.planar = nn.Parameter(torch.zeros(1, embed_dim, base_grid, base_grid))
        nn.init.trunc_normal_(self.planar, std=0.02)
        self.depth = nn.Parameter(torch.zeros(depth_tokens, embed_dim))
        nn.init.trunc_normal_(self.depth, std=0.02)

    def planar_for(self, grid_h: int, grid_w: int) -> torch.Tensor:
        resized = F.interpolate(
            self.planar, size=(grid_h, grid_w), mode="bicubic", align_corners=False
        )
        return resized.flatten(2).transpose(1, 2)

    def depth_for(self, depth: int) -> torch.Tensor:
        reduced = max(1, depth // self.depth_stride)
        source = self.depth.t().unsqueeze(0)
        resized = F.interpolate(source, size=reduced, mode="linear", align_corners=False)
        upsampled = F.interpolate(resized, size=depth, mode="linear", align_corners=False)
        return upsampled.squeeze(0).t()

    def forward(
        self, tokens: torch.Tensor, depth: int, grid_h: int, grid_w: int, use_depth: bool = True
    ) -> torch.Tensor:
        _b, _n, c = tokens.shape
        planar = self.planar_for(grid_h, grid_w)
        per_slice = grid_h * grid_w
        planar = planar.expand(depth, -1, -1).reshape(1, depth * per_slice, c)
        out = tokens + planar
        if use_depth:
            depth_emb = self.depth_for(depth)
            depth_rep = depth_emb.repeat_interleave(per_slice, dim=0).unsqueeze(0)
            out = out + depth_rep
        return out
