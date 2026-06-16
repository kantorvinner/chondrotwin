from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
    }
    if extra is not None:
        payload["extra"] = extra
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    loaded: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    return loaded
