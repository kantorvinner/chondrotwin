from __future__ import annotations

import torch

from chondrotwin import ChondroTwin, tiny_config
from chondrotwin.clay.schema import Batch


def test_segmentation_logits_shape(model: ChondroTwin, batch: Batch) -> None:
    out = model(batch["volume"])
    spec = tiny_config().spec
    expected = (2, spec.num_seg_classes, spec.depth, spec.height, spec.width)
    assert tuple(out.seg_logits.shape) == expected


def test_alpha_shape_and_support(model: ChondroTwin, batch: Batch) -> None:
    out = model(batch["volume"])
    spec = tiny_config().spec
    assert tuple(out.alpha.shape) == (2, spec.num_nodes, spec.num_grades)
    assert bool((out.alpha >= 1.0).all())


def test_uncertainty_shape(model: ChondroTwin, batch: Batch) -> None:
    out = model(batch["volume"])
    spec = tiny_config().spec
    assert tuple(out.uncertainty.total.shape) == (2, spec.num_nodes)
    assert tuple(out.uncertainty.aleatoric.shape) == (2, spec.num_nodes)
    assert tuple(out.uncertainty.epistemic.shape) == (2, spec.num_nodes)


def test_batch_independence(model: ChondroTwin, batch: Batch) -> None:
    model.eval()
    with torch.no_grad():
        joint = model(batch["volume"]).seg_logits
        single = model(batch["volume"][:1]).seg_logits
    assert torch.allclose(joint[:1], single, atol=1e-4)
