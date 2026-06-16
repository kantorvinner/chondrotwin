from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..bench import ChondroTwin
from ..clay.schema import Batch
from .objective import LossTerms, ObjectiveConfig, joint_loss


@dataclass(frozen=True, slots=True)
class TrainConfig:
    grad_accum: int = 2
    max_steps: int | None = None
    amp: bool = False
    amp_dtype: str = "bfloat16"
    log_every: int = 10


@dataclass(slots=True)
class EpochReport:
    epoch: int
    total: float
    segmentation: float
    evidential: float
    steps: int


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        volume=batch["volume"].to(device),
        seg=batch["seg"].to(device),
        grade=batch["grade"].to(device),
        node_mask=batch["node_mask"].to(device),
        subject_id=batch["subject_id"],
    )


class Trainer:
    def __init__(
        self,
        model: ChondroTwin,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        objective: ObjectiveConfig,
        train_cfg: TrainConfig,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.objective = objective
        self.train_cfg = train_cfg
        self._amp_dtype = getattr(torch, train_cfg.amp_dtype)

    def _forward_loss(self, batch: Batch, epoch: int) -> LossTerms:
        output = self.model(batch["volume"])
        return joint_loss(
            output, batch["seg"], batch["grade"], batch["node_mask"], epoch, self.objective
        )

    def run_epoch(self, loader: Iterable[Batch], epoch: int) -> EpochReport:
        self.model.train()
        totals = torch.zeros(3)
        steps = 0
        self.optimizer.zero_grad(set_to_none=True)
        use_amp = self.train_cfg.amp and self.device.type == "cuda"
        for index, raw in enumerate(loader):
            batch = _to_device(raw, self.device)
            with torch.autocast(
                device_type=self.device.type, dtype=self._amp_dtype, enabled=use_amp
            ):
                terms = self._forward_loss(batch, epoch)
            (terms.total / self.train_cfg.grad_accum).backward()
            if (index + 1) % self.train_cfg.grad_accum == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            totals += torch.tensor(
                [terms.total.detach(), terms.segmentation.detach(), terms.evidential.detach()]
            )
            steps += 1
            if self.train_cfg.max_steps is not None and steps >= self.train_cfg.max_steps:
                break
        self.scheduler.step()
        mean = totals / max(steps, 1)
        return EpochReport(
            epoch=epoch,
            total=float(mean[0]),
            segmentation=float(mean[1]),
            evidential=float(mean[2]),
            steps=steps,
        )

    def fit(self, loader: DataLoader[Batch], epochs: int) -> list[EpochReport]:
        history: list[EpochReport] = []
        for epoch in range(epochs):
            history.append(self.run_epoch(loader, epoch))
        return history


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
