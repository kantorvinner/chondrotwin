from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import Sample, VolumeSpec
from .synthetic import SyntheticKneeCohort


class ManifestCohort(Dataset[Sample]):
    def __init__(self, root: Path, spec: VolumeSpec, split: str) -> None:
        self.root = Path(root)
        self.spec = spec
        self.split = split
        self.items = sorted((self.root / split).glob("*.npz"))
        if not self.items:
            raise FileNotFoundError(f"no .npz shards under {self.root / split}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Sample:
        with np.load(self.items[index]) as blob:
            volume = torch.from_numpy(blob["volume"].astype("float32")).unsqueeze(0)
            seg = torch.from_numpy(blob["seg"].astype("int64"))
            grade = torch.from_numpy(blob["grade"].astype("int64"))
            node_mask = torch.from_numpy(blob["node_mask"].astype("bool"))
        return Sample(
            volume=volume,
            seg=seg,
            grade=grade,
            node_mask=node_mask,
            subject_id=self.items[index].stem,
        )


def _build(root: str | None, spec: VolumeSpec, split: str, synthetic_len: int) -> Dataset[Sample]:
    if root is None:
        return SyntheticKneeCohort(spec, length=synthetic_len, split=split)
    return ManifestCohort(Path(root), spec, split)


def oai_dess(
    spec: VolumeSpec, split: str, root: str | None = None, synthetic_len: int = 8
) -> Dataset[Sample]:
    return _build(root, spec, split, synthetic_len)


def oai_zib(
    spec: VolumeSpec, split: str, root: str | None = None, synthetic_len: int = 8
) -> Dataset[Sample]:
    return _build(root, spec, split, synthetic_len)


def ski10(
    spec: VolumeSpec, split: str, root: str | None = None, synthetic_len: int = 6
) -> Dataset[Sample]:
    return _build(root, spec, split, synthetic_len)


def iwoai_split(
    spec: VolumeSpec, split: str, root: str | None = None, synthetic_len: int = 6
) -> Dataset[Sample]:
    return _build(root, spec, split, synthetic_len)
