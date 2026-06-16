from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

BaselineFactory = Callable[[int, int], nn.Module]


class TinyVNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 16) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv3d(in_channels, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv3d(width, num_classes, 1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(volume))


class SlicewiseUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 16) -> None:
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=1), nn.GELU())
        self.up = nn.Conv2d(width, num_classes, 1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = volume.shape
        planar = volume.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        logits = self.up(F.gelu(self.down(planar)))
        return logits.reshape(b, d, -1, h, w).permute(0, 2, 1, 3, 4)


class BaselineRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BaselineFactory] = {}

    def register(self, name: str, factory: BaselineFactory) -> None:
        self._factories[name] = factory

    def build(self, name: str, in_channels: int, num_classes: int) -> nn.Module:
        if name not in self._factories:
            raise KeyError(f"unknown baseline '{name}'; registered: {sorted(self._factories)}")
        return self._factories[name](in_channels, num_classes)

    def names(self) -> list[str]:
        return sorted(self._factories)


def default_registry() -> BaselineRegistry:
    registry = BaselineRegistry()
    registry.register("vnet3d", lambda c, k: TinyVNet(c, k))
    registry.register("slicewise_unet", lambda c, k: SlicewiseUNet(c, k))
    return registry
