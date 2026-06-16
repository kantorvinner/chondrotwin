from __future__ import annotations

import torch

from ..clay.schema import AnatomyTopology
from .anatomy_graph import GraphTopology


def _compartment_index(topology: AnatomyTopology) -> list[int]:
    compartments = list(dict.fromkeys(topology.compartment_of.values()))
    return [compartments.index(topology.compartment_of[node]) for node in topology.nodes]


def contact_from_segmentation(
    mean_seg: torch.Tensor, graph: GraphTopology, num_classes: int, smooth: float = 1e-3
) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for cls in range(num_classes):
        counts[cls] = float((mean_seg == cls).sum())
    weights = torch.empty(graph.num_edges)
    edges = graph.directed_edges()
    anatomy = AnatomyTopology()
    comp = _compartment_index(anatomy)
    for idx, (src, dst) in enumerate(edges):
        a = counts[min(comp[src] + 1, num_classes - 1)]
        b = counts[min(comp[dst] + 1, num_classes - 1)]
        overlap = torch.sqrt(a * b)
        weights[idx] = overlap / (a + b + smooth)
    return weights / (weights.sum() + smooth) * graph.num_edges


def default_contact_weights(graph: GraphTopology) -> torch.Tensor:
    anatomy = AnatomyTopology()
    comp = _compartment_index(anatomy)
    edges = graph.directed_edges()
    weights = torch.empty(graph.num_edges)
    for idx, (src, dst) in enumerate(edges):
        weights[idx] = 1.0 + 0.1 * float(abs(comp[src] - comp[dst]))
    return weights / weights.mean()
