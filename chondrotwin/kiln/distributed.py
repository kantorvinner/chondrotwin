from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class WorldInfo:
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def discover_world() -> WorldInfo:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return WorldInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def initialize(world: WorldInfo, backend: str = "nccl") -> None:
    if world.distributed and not dist.is_initialized():
        chosen = backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=chosen, rank=world.rank, world_size=world.world_size)


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
