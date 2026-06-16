from __future__ import annotations

from pathlib import Path
from typing import Any

from ..kiln.checkpointing import save_checkpoint
from ..kiln.optimizer import build_optimizer
from ..kiln.reproducibility import set_seed
from ..kiln.schedule import build_scheduler
from ..kiln.trainer import Trainer
from .experiments import run_with_hydra
from .runtime import (
    assemble_model,
    build_loader,
    materialize_model,
    materialize_objective,
    materialize_optimizer,
    materialize_schedule,
    materialize_train,
)


def train_task(
    model: Any,
    train: Any,
    optim: Any,
    sched: Any,
    objective: Any,
    data: Any,
    output_dir: str,
) -> None:
    set_seed(int(train.seed))
    model_cfg = materialize_model(model)
    network = assemble_model(model)
    optimizer = build_optimizer(network, materialize_optimizer(optim))
    scheduler = build_scheduler(optimizer, materialize_schedule(sched))
    objective_cfg = materialize_objective(objective, model_cfg.spec.num_grades)
    trainer = Trainer(network, optimizer, scheduler, objective_cfg, materialize_train(train))
    loader = build_loader(data, model_cfg.spec, "train", int(train.batch_size))
    history = trainer.fit(loader, int(train.epochs))
    first, last = history[0], history[-1]
    print(f"steps={last.steps} loss[0]={first.total:.4f} loss[-1]={last.total:.4f}")
    trainable = network.trainable_parameters()
    print(f"seg={last.segmentation:.4f} edl={last.evidential:.4f} trainable={trainable}")
    target = Path(output_dir) / "checkpoint.pt"
    save_checkpoint(target, network, optimizer, int(train.epochs), int(train.seed))


def main() -> None:
    run_with_hydra(train_task, "train")


if __name__ == "__main__":
    main()
