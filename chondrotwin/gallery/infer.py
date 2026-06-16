from __future__ import annotations

from typing import Any

import torch

from ..assay.decompose import decompose
from ..assay.dirichlet import expected_probability
from ..clay.schema import REGION_NODES
from ..kiln.reproducibility import set_seed
from .experiments import run_with_hydra
from .runtime import assemble_model, build_loader, materialize_model


@torch.no_grad()
def infer_task(
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
    loader = build_loader(data, spec, "test", 1)
    batch = next(iter(loader))
    output = network(batch["volume"])
    prob = expected_probability(output.alpha)[0]
    split = decompose(output.alpha)
    grade = prob.argmax(dim=-1)
    print(f"subject={batch['subject_id'][0]}")
    for node in range(spec.num_nodes):
        name = REGION_NODES[node] if node < len(REGION_NODES) else f"node_{node}"
        ale = float(split.aleatoric[0, node])
        epi = float(split.epistemic[0, node])
        print(f"{name:24s} grade={int(grade[node])} u_ale={ale:.3f} u_epi={epi:.3f}")


def main() -> None:
    run_with_hydra(infer_task, "infer")


if __name__ == "__main__":
    main()
