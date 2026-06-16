from __future__ import annotations

import torch

from chondrotwin.armature.anatomy_graph import GraphTopology
from chondrotwin.armature.graph_sage import AnatomyGraphSage, TwinConfig
from chondrotwin.wheel.lora import LoRALinear


def test_lora_update_is_zero_at_init() -> None:
    layer = LoRALinear(16, 16, rank=4, alpha=8.0)
    x = torch.randn(2, 16)
    assert torch.allclose(layer(x), layer.base(x), atol=1e-6)


def test_lora_base_is_frozen() -> None:
    layer = LoRALinear(16, 16, rank=4, alpha=8.0)
    assert not layer.base.weight.requires_grad
    assert layer.lora_a.requires_grad
    assert layer.lora_b.requires_grad


def test_lora_rank_limits_update() -> None:
    layer = LoRALinear(16, 16, rank=4, alpha=8.0)
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn_like(layer.lora_b))
    delta = layer.lora_b @ layer.lora_a
    assert torch.linalg.matrix_rank(delta) <= 4


def test_edge_weights_are_frozen_buffer() -> None:
    graph = GraphTopology(num_nodes=14)
    twin = AnatomyGraphSage(TwinConfig(feature_dim=8, hidden_dim=8, layers=2), graph)
    assert not twin.edge_weight.requires_grad
    before = twin.edge_weight.clone()
    out = twin(torch.randn(2, 14, 8))
    out.sum().backward()
    assert torch.allclose(before, twin.edge_weight)


def test_contact_weights_can_be_replaced() -> None:
    graph = GraphTopology(num_nodes=14)
    twin = AnatomyGraphSage(TwinConfig(feature_dim=8, hidden_dim=8, layers=2), graph)
    new = torch.arange(graph.num_edges, dtype=torch.float) + 1.0
    twin.set_contact_weights(new)
    assert torch.allclose(twin.edge_weight, new)
