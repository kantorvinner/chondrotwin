from __future__ import annotations

import torch

from chondrotwin import build_model, tiny_config
from chondrotwin.gallery.experiments import build_experiments
from chondrotwin.gallery.runtime import (
    build_loader,
    materialize_model,
    materialize_objective,
    materialize_optimizer,
    materialize_schedule,
    materialize_train,
)
from chondrotwin.kiln.optimizer import build_optimizer
from chondrotwin.kiln.schedule import build_scheduler
from chondrotwin.kiln.trainer import Trainer


def test_trainer_runs_two_steps_and_reduces_loss() -> None:
    torch.manual_seed(0)
    cfg = build_experiments()["_unittest"]
    spec = materialize_model(cfg.model).spec
    network = build_model(tiny_config())
    optimizer = build_optimizer(network, materialize_optimizer(cfg.optim))
    scheduler = build_scheduler(optimizer, materialize_schedule(cfg.sched))
    objective = materialize_objective(cfg.objective, spec.num_grades)
    train_cfg = materialize_train(cfg.train)
    trainer = Trainer(network, optimizer, scheduler, objective, train_cfg)
    loader = build_loader(cfg.data, spec, "train", 2)
    report = trainer.run_epoch(loader, epoch=0)
    assert report.steps == 2
    assert report.total > 0.0


def test_ski10_transfer_loader_builds() -> None:
    cfg = build_experiments()["transfer_ski10"]
    spec = materialize_model(cfg.model).spec
    loader = build_loader(cfg.data, spec, "test", 1)
    batch = next(iter(loader))
    assert batch["volume"].shape[0] == 1
