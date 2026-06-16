from __future__ import annotations

import torch

from chondrotwin import build_model, tiny_config
from chondrotwin.assay.edl import EvidenceConfig
from chondrotwin.clay import SyntheticKneeCohort, collate
from chondrotwin.kiln.objective import ObjectiveConfig, joint_loss


def test_loss_drops_on_repeated_batch() -> None:
    torch.manual_seed(0)
    model = build_model(tiny_config())
    spec = tiny_config().spec
    data = SyntheticKneeCohort(spec, length=2, split="train")
    batch = collate([data[0], data[1]])
    cfg = ObjectiveConfig(edl_weight=0.5, evidence=EvidenceConfig(num_grades=4, anneal_epochs=2))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    losses = []
    for step in range(40):
        out = model(batch["volume"])
        terms = joint_loss(out, batch["seg"], batch["grade"], batch["node_mask"], step, cfg)
        optimizer.zero_grad()
        terms.total.backward()
        optimizer.step()
        losses.append(float(terms.total.detach()))
    assert losses[-1] < losses[0] * 0.7


def test_all_trainable_receive_gradient_after_warm_step() -> None:
    torch.manual_seed(0)
    model = build_model(tiny_config())
    spec = tiny_config().spec
    data = SyntheticKneeCohort(spec, length=2, split="train")
    batch = collate([data[0], data[1]])
    cfg = ObjectiveConfig(edl_weight=0.5, evidence=EvidenceConfig(num_grades=4, anneal_epochs=2))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for step in range(2):
        out = model(batch["volume"])
        terms = joint_loss(out, batch["seg"], batch["grade"], batch["node_mask"], step, cfg)
        optimizer.zero_grad()
        terms.total.backward()
        optimizer.step()
    missing = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)
    ]
    assert missing == []
