from __future__ import annotations

from typing import Any

import torch

from ..assay.decompose import decompose
from ..assay.dirichlet import expected_probability
from ..inspection.calibration import expected_calibration_error
from ..inspection.ordinal import detection_auroc, grade_accuracy, ordinal_mae
from ..inspection.overlap import mean_foreground_dice
from ..inspection.selective import area_under_risk_coverage
from ..kiln.reproducibility import set_seed
from .experiments import run_with_hydra
from .runtime import assemble_model, build_loader, materialize_model


@torch.no_grad()
def evaluate_loader(network: Any, loader: Any) -> dict[str, float]:
    network.eval()
    dice: list[float] = []
    alphas: list[torch.Tensor] = []
    grades: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch in loader:
        output = network(batch["volume"])
        dice.append(mean_foreground_dice(output.seg_logits, batch["seg"]))
        alphas.append(output.alpha)
        grades.append(batch["grade"])
        masks.append(batch["node_mask"])
    alpha = torch.cat(alphas, dim=0)
    grade = torch.cat(grades, dim=0)
    mask = torch.cat(masks, dim=0)
    prob = expected_probability(alpha)
    correct = (prob.argmax(dim=-1) == grade).reshape(-1)
    uncertainty = decompose(alpha).total.reshape(-1)
    return {
        "dice": sum(dice) / max(len(dice), 1),
        "auroc": detection_auroc(alpha, grade, mask),
        "grade_accuracy": grade_accuracy(alpha, grade),
        "ordinal_mae": ordinal_mae(alpha, grade),
        "ece": expected_calibration_error(prob.reshape(-1, prob.shape[-1]), grade.reshape(-1)),
        "aurc": area_under_risk_coverage(uncertainty, correct),
    }


def evaluate_task(
    model: Any,
    train: Any,
    optim: Any,
    sched: Any,
    objective: Any,
    data: Any,
    output_dir: str,
) -> None:
    set_seed(int(train.seed))
    spec = materialize_model(model).spec
    network = assemble_model(model)
    loader = build_loader(data, spec, "test", int(train.batch_size))
    metrics = evaluate_loader(network, loader)
    for name, value in metrics.items():
        print(f"{name}={value:.4f}")


def main() -> None:
    run_with_hydra(evaluate_task, "evaluate")


if __name__ == "__main__":
    main()
