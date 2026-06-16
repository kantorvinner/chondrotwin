from __future__ import annotations

import torch

from chondrotwin.armature.anatomy_graph import GraphTopology
from chondrotwin.armature.graph_sage import AnatomyGraphSage, TwinConfig
from chondrotwin.inspection.reference_models import default_registry
from chondrotwin.wheel.depth_posembed import DepthAwarePositional


def test_topology_node_and_edge_counts() -> None:
    graph = GraphTopology(num_nodes=14)
    assert graph.num_nodes == 14
    assert graph.num_edges == 26
    assert graph.edge_index().shape == (2, 26)


def test_depth_positional_adds_two_terms() -> None:
    pos = DepthAwarePositional(embed_dim=8, base_grid=4, depth_tokens=8, depth_stride=2)
    tokens = torch.zeros(1, 6 * 9, 8)
    out = pos(tokens, depth=6, grid_h=3, grid_w=3, use_depth=True)
    flat = pos(tokens, depth=6, grid_h=3, grid_w=3, use_depth=False)
    assert out.shape == tokens.shape
    assert not torch.allclose(out, flat)


def test_graph_sage_output_dim() -> None:
    graph = GraphTopology(num_nodes=14)
    twin = AnatomyGraphSage(TwinConfig(feature_dim=16, hidden_dim=10, layers=3), graph)
    out = twin(torch.randn(2, 14, 16))
    assert tuple(out.shape) == (2, 14, 10)


def test_baseline_registry_shapes() -> None:
    registry = default_registry()
    volume = torch.randn(1, 1, 6, 16, 16)
    for name in registry.names():
        net = registry.build(name, 1, 7)
        logits = net(volume)
        assert logits.shape == (1, 7, 6, 16, 16)
