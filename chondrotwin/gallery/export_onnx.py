from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..bench import ChondroTwin
from ..kiln.reproducibility import set_seed
from .experiments import run_with_hydra
from .runtime import assemble_model, materialize_model


class SegmentationExport(nn.Module):
    def __init__(self, network: ChondroTwin) -> None:
        super().__init__()
        self.network = network

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        target = (volume.shape[-3], volume.shape[-2], volume.shape[-1])
        feature_volume = self.network.encoder(volume)
        return self.network.decoder(feature_volume, target)


def export_task(
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
    wrapper = SegmentationExport(network)
    dummy = torch.randn(1, 1, spec.depth, spec.height, spec.width)
    target = Path(output_dir) / "chondrotwin.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(target),
        input_names=["volume"],
        output_names=["seg_logits"],
        dynamic_axes={"volume": {0: "batch"}, "seg_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {target}")


def main() -> None:
    run_with_hydra(export_task, "export")


if __name__ == "__main__":
    main()
