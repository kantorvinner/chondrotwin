from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .depth_posembed import DepthAwarePositional
from .lora import LoRAConfig, LoRALinear


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    embed_dim: int = 1024
    depth: int = 24
    heads: int = 16
    patch: int = 16
    mlp_ratio: float = 4.0
    base_grid: int = 24
    depth_tokens: int = 40
    depth_stride: int = 4
    use_depth: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)


class LoRAAttention(nn.Module):
    def __init__(self, dim: int, heads: int, lora: LoRAConfig) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.query = LoRALinear(dim, dim, lora.rank, lora.alpha)
        self.key = nn.Linear(dim, dim)
        self.key.weight.requires_grad_(False)
        if self.key.bias is not None:
            self.key.bias.requires_grad_(False)
        self.value = LoRALinear(dim, dim, lora.rank, lora.alpha)
        self.project = nn.Linear(dim, dim)
        self.project.weight.requires_grad_(False)
        if self.project.bias is not None:
            self.project.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.query(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.key(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.value(x), "b n (h d) -> b h n d", h=self.heads)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = rearrange(attn @ v, "b h n d -> b n (h d)")
        return self.project(out)


class FrozenMLP(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: float, lora: LoRAConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LoRAAttention(dim, heads, lora)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FrozenMLP(dim, ratio)
        for module in (self.norm1, self.norm2):
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, patch_embed: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = patch_embed
        self.positional = DepthAwarePositional(
            cfg.embed_dim, cfg.base_grid, cfg.depth_tokens, cfg.depth_stride
        )
        self.blocks = nn.ModuleList(
            EncoderBlock(cfg.embed_dim, cfg.heads, cfg.mlp_ratio, cfg.lora)
            for _ in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        for param in self.norm.parameters():
            param.requires_grad_(False)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        b, _, d, _h, _w = volume.shape
        slices = rearrange(volume, "b c d h w -> (b d) c h w")
        feats = self.patch_embed(slices)
        grid_h, grid_w = feats.shape[-2:]
        tokens = rearrange(feats, "(b d) c gh gw -> b (d gh gw) c", b=b, d=d)
        tokens = self.positional(tokens, d, grid_h, grid_w, self.cfg.use_depth)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return rearrange(tokens, "b (d gh gw) c -> b c d gh gw", d=d, gh=grid_h, gw=grid_w)

    def enable_full_finetune(self) -> None:
        for param in self.parameters():
            param.requires_grad_(True)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
