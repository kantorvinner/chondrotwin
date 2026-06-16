from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..clay.schema import AnatomyTopology


class RegionPooling(nn.Module):
    def __init__(self, feature_dim: int, num_nodes: int, num_compartments: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        anatomy = AnatomyTopology()
        compartments = list(dict.fromkeys(anatomy.compartment_of.values()))
        comp_idx = [compartments.index(anatomy.compartment_of[node]) for node in anatomy.nodes]
        self.register_buffer("compartment_of", torch.tensor(comp_idx, dtype=torch.long))
        self.compartment_of: torch.Tensor
        self.subregion = nn.Parameter(torch.zeros(num_nodes, feature_dim))
        nn.init.trunc_normal_(self.subregion, std=0.02)

    def forward(self, feature_volume: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
        _b, _c, d, gh, gw = feature_volume.shape
        prob = torch.softmax(seg_logits, dim=1)
        prob = F.interpolate(prob, size=(d, gh, gw), mode="trilinear", align_corners=False)
        num_classes = seg_logits.shape[1]
        nodes = []
        for node in range(self.num_nodes):
            cls = min(int(self.compartment_of[node].item()) + 1, num_classes - 1)
            weight = prob[:, cls : cls + 1]
            pooled = (feature_volume * weight).flatten(2).sum(-1)
            denom = weight.flatten(2).sum(-1).clamp_min(1e-4)
            nodes.append(pooled / denom + self.subregion[node])
        return torch.stack(nodes, dim=1)
