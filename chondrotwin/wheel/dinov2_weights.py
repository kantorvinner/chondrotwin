from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .vit_adapter import CrossModalEncoder, EncoderConfig


@dataclass(frozen=True, slots=True)
class BackboneConfig:
    use_pretrained: bool = False
    weights_name: str = "vit_large_patch16_224.dinov2"
    in_channels: int = 1


class OrthonormalPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)
        weight = self.proj.weight.data.reshape(embed_dim, -1)
        q, _ = torch.linalg.qr(weight.t())
        self.proj.weight.data = q.t().reshape(self.proj.weight.shape).contiguous()
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RetfoundUnavailable(RuntimeError):
    pass


def _load_pretrained_patch_embed(cfg: BackboneConfig, enc: EncoderConfig) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise RetfoundUnavailable(
            "pretrained DINOv2 backbone requires the 'backbone' extra (timm); "
            "install it or set use_pretrained=false to use the frozen orthonormal stand-in"
        ) from exc
    model = timm.create_model(cfg.weights_name, pretrained=True, in_chans=cfg.in_channels)
    patch = cast(nn.Module, model.patch_embed)
    for param in patch.parameters():
        param.requires_grad_(False)
    return patch


def build_encoder(cfg: BackboneConfig, enc: EncoderConfig) -> CrossModalEncoder:
    if cfg.use_pretrained:
        patch_embed = _load_pretrained_patch_embed(cfg, enc)
    else:
        patch_embed = OrthonormalPatchEmbed(cfg.in_channels, enc.embed_dim, enc.patch)
    return CrossModalEncoder(enc, patch_embed)
