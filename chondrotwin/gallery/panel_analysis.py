from __future__ import annotations

from typing import Any

import torch

from ..assay.dirichlet import expected_probability
from ..clay.schema import REGION_NODES
from ..kiln.reproducibility import set_seed
from .experiments import run_with_hydra
from .runtime import assemble_model, build_loader, materialize_model


@torch.no_grad()
def compartment_grade_matrix(
    network: Any, loader: Any, num_nodes: int, num_grades: int
) -> torch.Tensor:
    counts = torch.zeros(num_nodes, num_grades)
    for batch in loader:
        output = network(batch["volume"])
        pred = expected_probability(output.alpha).argmax(dim=-1)
        for node in range(num_nodes):
            for grade in range(num_grades):
                counts[node, grade] += float((pred[:, node] == grade).sum())
    return counts


def panel_task(
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
    network.eval()
    loader = build_loader(data, spec, "test", int(train.batch_size))
    matrix = compartment_grade_matrix(network, loader, spec.num_nodes, spec.num_grades)
    totals = matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
    fractions = matrix / totals
    header = "region".ljust(24) + "".join(f"  g{g}".rjust(7) for g in range(spec.num_grades))
    print(header)
    for node in range(spec.num_nodes):
        name = REGION_NODES[node] if node < len(REGION_NODES) else f"node_{node}"
        row = "".join(f"{float(fractions[node, g]):7.3f}" for g in range(spec.num_grades))
        print(name.ljust(24) + row)


def main() -> None:
    run_with_hydra(panel_task, "panel_analysis")


if __name__ == "__main__":
    main()
