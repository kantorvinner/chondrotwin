from __future__ import annotations

from typing import Any

import torch

from ..assay.decompose import decompose
from ..clay.schema import REGION_NODES
from ..kiln.reproducibility import set_seed
from .experiments import run_with_hydra
from .runtime import assemble_model, build_loader, materialize_model


@torch.no_grad()
def uncertainty_summary(
    network: Any, loader: Any, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ale = torch.zeros(num_nodes)
    epi = torch.zeros(num_nodes)
    seen = 0
    for batch in loader:
        output = network(batch["volume"])
        split = decompose(output.alpha)
        ale += split.aleatoric.sum(dim=0)
        epi += split.epistemic.sum(dim=0)
        seen += output.alpha.shape[0]
    denom = max(seen, 1)
    return ale / denom, epi / denom


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
    ale, epi = uncertainty_summary(network, loader, spec.num_nodes)
    print("region".ljust(24) + "u_ale".rjust(9) + "u_epi".rjust(9))
    for node in range(spec.num_nodes):
        name = REGION_NODES[node] if node < len(REGION_NODES) else f"node_{node}"
        print(name.ljust(24) + f"{float(ale[node]):9.3f}" + f"{float(epi[node]):9.3f}")


def main() -> None:
    run_with_hydra(panel_task, "panel_uncertainty")


if __name__ == "__main__":
    main()
