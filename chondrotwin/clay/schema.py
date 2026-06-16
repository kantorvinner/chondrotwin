from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import torch

COMPARTMENTS: tuple[str, ...] = (
    "femoral",
    "medial_tibial",
    "lateral_tibial",
    "patellar",
    "medial_meniscus",
    "lateral_meniscus",
)

REGION_NODES: tuple[str, ...] = (
    "femoral_medial",
    "femoral_lateral",
    "femoral_trochlear",
    "medial_tibial_anterior",
    "medial_tibial_central",
    "medial_tibial_posterior",
    "lateral_tibial_anterior",
    "lateral_tibial_central",
    "lateral_tibial_posterior",
    "patellar_medial",
    "patellar_lateral",
    "patellar_central",
    "medial_meniscus_body",
    "lateral_meniscus_body",
)


class Sample(TypedDict):
    volume: torch.Tensor
    seg: torch.Tensor
    grade: torch.Tensor
    node_mask: torch.Tensor
    subject_id: str


class Batch(TypedDict):
    volume: torch.Tensor
    seg: torch.Tensor
    grade: torch.Tensor
    node_mask: torch.Tensor
    subject_id: list[str]


@dataclass(frozen=True, slots=True)
class VolumeSpec:
    depth: int = 160
    height: int = 384
    width: int = 384
    spacing: tuple[float, float, float] = (0.7, 0.365, 0.365)
    num_seg_classes: int = len(COMPARTMENTS) + 1
    num_nodes: int = len(REGION_NODES)
    num_grades: int = 4

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.depth, self.height, self.width)


@dataclass(frozen=True, slots=True)
class SplitCounts:
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    name: str
    role: str
    counts: SplitCounts
    access: str
    license: str
    resolves: bool = True


def collate(samples: list[Sample]) -> Batch:
    return Batch(
        volume=torch.stack([s["volume"] for s in samples], dim=0),
        seg=torch.stack([s["seg"] for s in samples], dim=0),
        grade=torch.stack([s["grade"] for s in samples], dim=0),
        node_mask=torch.stack([s["node_mask"] for s in samples], dim=0),
        subject_id=[s["subject_id"] for s in samples],
    )


@dataclass(frozen=True, slots=True)
class AnatomyTopology:
    nodes: tuple[str, ...] = REGION_NODES
    compartment_of: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.compartment_of:
            mapping = {
                "femoral_medial": "femoral",
                "femoral_lateral": "femoral",
                "femoral_trochlear": "femoral",
                "medial_tibial_anterior": "medial_tibial",
                "medial_tibial_central": "medial_tibial",
                "medial_tibial_posterior": "medial_tibial",
                "lateral_tibial_anterior": "lateral_tibial",
                "lateral_tibial_central": "lateral_tibial",
                "lateral_tibial_posterior": "lateral_tibial",
                "patellar_medial": "patellar",
                "patellar_lateral": "patellar",
                "patellar_central": "patellar",
                "medial_meniscus_body": "medial_meniscus",
                "lateral_meniscus_body": "lateral_meniscus",
            }
            object.__setattr__(self, "compartment_of", mapping)
