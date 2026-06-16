from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from ..assay.edl import EvidenceConfig, evidential_loss
from ..bench import TwinOutput
from .segmentation_loss import segmentation_objective


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    edl_weight: float = 0.5
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)


class LossTerms(NamedTuple):
    total: torch.Tensor
    segmentation: torch.Tensor
    evidential: torch.Tensor


def joint_loss(
    output: TwinOutput,
    seg_target: torch.Tensor,
    grade_target: torch.Tensor,
    node_mask: torch.Tensor,
    epoch: int,
    cfg: ObjectiveConfig,
) -> LossTerms:
    seg = segmentation_objective(output.seg_logits, seg_target)
    edl = evidential_loss(output.alpha, grade_target, epoch, cfg.evidence, node_mask)
    total = seg + cfg.edl_weight * edl
    return LossTerms(total=total, segmentation=seg, evidential=edl)
