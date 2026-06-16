from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class IntensityConfig:
    gaussian_sigma: float = 0.02
    gain_low: float = 0.95
    gain_high: float = 1.05
    eps: float = 1e-6


def zscore(volume: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / (std + eps)


def additive_gaussian(
    volume: torch.Tensor, sigma: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    noise = torch.randn(volume.shape, generator=generator) * sigma
    return volume + noise


def multiplicative_gain(
    volume: torch.Tensor, low: float, high: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    gain = torch.empty(1).uniform_(low, high, generator=generator)
    return volume * gain


def photometric(
    volume: torch.Tensor, cfg: IntensityConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    out = additive_gaussian(volume, cfg.gaussian_sigma, generator)
    out = multiplicative_gain(out, cfg.gain_low, cfg.gain_high, generator)
    return zscore(out, cfg.eps)
