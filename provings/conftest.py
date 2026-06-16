from __future__ import annotations

import pytest
import torch

from chondrotwin import ChondroTwin, build_model, tiny_config
from chondrotwin.clay import SyntheticKneeCohort, collate
from chondrotwin.clay.schema import Batch


@pytest.fixture
def model() -> ChondroTwin:
    torch.manual_seed(0)
    return build_model(tiny_config())


@pytest.fixture
def batch() -> Batch:
    cfg = tiny_config()
    dataset = SyntheticKneeCohort(cfg.spec, length=2, split="train")
    return collate([dataset[0], dataset[1]])
