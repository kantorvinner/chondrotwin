from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .anatomy_graph import GraphTopology
from .contact_weights import default_contact_weights


@dataclass(frozen=True, slots=True)
class TwinConfig:
    feature_dim: int = 1024
    hidden_dim: int = 256
    layers: int = 3
    edge_mode: str = "contact"


class SageLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return self.act(self.self_proj(x) + self.neigh_proj(agg))


class AnatomyGraphSage(nn.Module):
    def __init__(self, cfg: TwinConfig, graph: GraphTopology) -> None:
        super().__init__()
        self.cfg = cfg
        self.graph = graph
        edge_index, edge_weight = self._wiring(cfg.edge_mode, graph)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.edge_index: torch.Tensor
        self.edge_weight: torch.Tensor
        dims = [cfg.feature_dim] + [cfg.hidden_dim] * cfg.layers
        self.layers = nn.ModuleList(SageLayer(dims[i], dims[i + 1]) for i in range(cfg.layers))

    @staticmethod
    def _wiring(edge_mode: str, graph: GraphTopology) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_mode == "fc":
            n = graph.num_nodes
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            return edge_index, torch.ones(edge_index.shape[1])
        edge_index = graph.edge_index()
        if edge_mode == "uniform":
            return edge_index, torch.ones(graph.num_edges)
        return edge_index, default_contact_weights(graph)

    def set_contact_weights(self, weights: torch.Tensor) -> None:
        if weights.shape != self.edge_weight.shape:
            raise ValueError("contact weight shape mismatch")
        self.edge_weight.copy_(weights.detach())
        self.edge_weight.requires_grad_(False)

    def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        src = self.edge_index[0]
        dst = self.edge_index[1]
        messages = x[:, src, :] * self.edge_weight.view(1, -1, 1)
        agg = torch.zeros(b, n, c, dtype=x.dtype, device=x.device)
        index = dst.view(1, -1, 1).expand(b, -1, c)
        agg = agg.scatter_add(1, index, messages)
        degree = torch.zeros(n, dtype=x.dtype, device=x.device)
        degree = degree.scatter_add(0, dst, self.edge_weight).clamp_min(1e-6)
        return agg / degree.view(1, n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            agg = self._aggregate(x)
            x = layer(x, agg)
        return x
