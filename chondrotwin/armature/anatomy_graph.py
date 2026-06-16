from __future__ import annotations

from dataclasses import dataclass

import torch

UNDIRECTED_EDGES: tuple[tuple[int, int], ...] = (
    (0, 4),
    (1, 7),
    (2, 11),
    (2, 9),
    (2, 10),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (12, 4),
    (13, 7),
    (9, 11),
    (10, 11),
)


@dataclass(frozen=True, slots=True)
class GraphTopology:
    num_nodes: int = 14

    def directed_edges(self) -> tuple[tuple[int, int], ...]:
        edges: list[tuple[int, int]] = []
        for src, dst in UNDIRECTED_EDGES:
            edges.append((src, dst))
            edges.append((dst, src))
        return tuple(edges)

    @property
    def num_edges(self) -> int:
        return len(self.directed_edges())

    def edge_index(self) -> torch.Tensor:
        edges = self.directed_edges()
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def adjacency(self) -> torch.Tensor:
        mat = torch.zeros(self.num_nodes, self.num_nodes)
        for src, dst in self.directed_edges():
            mat[dst, src] = 1.0
        return mat
