from __future__ import annotations

import hashlib

import torch
from torch.utils.data import Dataset

from .schema import Sample, VolumeSpec


def _seed_of(tag: str) -> int:
    digest = hashlib.blake2b(tag.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _spherical_mask(
    shape: tuple[int, int, int], center: torch.Tensor, radius: float
) -> torch.Tensor:
    d, h, w = shape
    zz = torch.arange(d).view(d, 1, 1).float()
    yy = torch.arange(h).view(1, h, 1).float()
    xx = torch.arange(w).view(1, 1, w).float()
    dist = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return dist <= radius**2


class SyntheticKneeCohort(Dataset[Sample]):
    def __init__(self, spec: VolumeSpec, length: int, split: str = "train") -> None:
        self.spec = spec
        self.length = length
        self.split = split

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Sample:
        subject = f"{self.split}-{index:05d}"
        gen = torch.Generator().manual_seed(_seed_of(subject))
        shape = self.spec.shape
        volume = torch.randn(1, *shape, generator=gen) * 0.2
        seg = torch.zeros(shape, dtype=torch.long)
        grade = torch.zeros(self.spec.num_nodes, dtype=torch.long)
        node_mask = torch.ones(self.spec.num_nodes, dtype=torch.bool)

        for cls in range(1, self.spec.num_seg_classes):
            center = torch.tensor(
                [
                    torch.randint(0, shape[0], (1,), generator=gen).item(),
                    torch.randint(0, shape[1], (1,), generator=gen).item(),
                    torch.randint(0, shape[2], (1,), generator=gen).item(),
                ],
                dtype=torch.float,
            )
            radius = float(min(shape)) / 6.0
            mask = _spherical_mask(shape, center, radius)
            seg[mask] = cls
            volume[0][mask] += 0.6 + 0.2 * cls

        for node in range(self.spec.num_nodes):
            logits = torch.randn(self.spec.num_grades, generator=gen)
            logits[0] += 1.5
            grade[node] = int(torch.argmax(logits).item())

        volume = (volume - volume.mean()) / (volume.std() + 1e-6)
        return Sample(
            volume=volume,
            seg=seg,
            grade=grade,
            node_mask=node_mask,
            subject_id=subject,
        )
