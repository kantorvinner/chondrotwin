from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset

from ..armature.graph_sage import TwinConfig
from ..assay.dirichlet_head import HeadConfig
from ..assay.edl import EvidenceConfig
from ..bench import AblationConfig, ChondroTwin, ChondroTwinConfig, build_model
from ..clay import collate, iwoai_split, oai_dess, oai_zib, ski10
from ..clay.schema import Batch, VolumeSpec
from ..kiln.objective import ObjectiveConfig
from ..kiln.optimizer import OptimizerConfig
from ..kiln.schedule import ScheduleConfig
from ..kiln.trainer import TrainConfig
from ..trimming.upernet import DecoderConfig
from ..wheel.dinov2_weights import BackboneConfig
from ..wheel.lora import LoRAConfig
from ..wheel.vit_adapter import EncoderConfig

_CORPUS = {"oai": oai_dess, "oai_zib": oai_zib, "ski10": ski10, "iwoai": iwoai_split}


@dataclass
class ModelGroup:
    depth: int = 160
    height: int = 384
    width: int = 384
    num_seg_classes: int = 7
    num_nodes: int = 14
    num_grades: int = 4
    embed_dim: int = 1024
    vit_depth: int = 24
    heads: int = 16
    patch: int = 16
    mlp_ratio: float = 4.0
    base_grid: int = 24
    depth_tokens: int = 40
    depth_stride: int = 4
    use_depth: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0
    decoder_channels: int = 256
    decoder_levels: int = 4
    twin_hidden: int = 256
    twin_layers: int = 3
    edge_mode: str = "contact"
    head_hidden: int = 128
    use_pretrained: bool = False
    transfer: str = "lora"
    use_gnn: bool = True
    head_kind: str = "dirichlet"


@dataclass
class TrainGroup:
    seed: int = 42
    epochs: int = 120
    grad_accum: int = 2
    max_steps: int | None = None
    amp: bool = True
    amp_dtype: str = "bfloat16"
    batch_size: int = 4


@dataclass
class OptimGroup:
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-2
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class SchedGroup:
    warmup_epochs: int = 5
    total_epochs: int = 120
    peak_lr: float = 2.0e-4
    floor_lr: float = 1.0e-6


@dataclass
class ObjectiveGroup:
    edl_weight: float = 0.5
    anneal_epochs: int = 10


@dataclass
class DataGroup:
    corpus: str = "oai_zib"
    root: str | None = None
    train_len: int = 8
    val_len: int = 4


@dataclass
class RunConfig:
    model: ModelGroup = field(default_factory=ModelGroup)
    train: TrainGroup = field(default_factory=TrainGroup)
    optim: OptimGroup = field(default_factory=OptimGroup)
    sched: SchedGroup = field(default_factory=SchedGroup)
    objective: ObjectiveGroup = field(default_factory=ObjectiveGroup)
    data: DataGroup = field(default_factory=DataGroup)
    output_dir: str = "runs"


def _get(node: Any, key: str) -> Any:
    return getattr(node, key)


def materialize_model(model: Any) -> ChondroTwinConfig:
    spec = VolumeSpec(
        depth=_get(model, "depth"),
        height=_get(model, "height"),
        width=_get(model, "width"),
        num_seg_classes=_get(model, "num_seg_classes"),
        num_nodes=_get(model, "num_nodes"),
        num_grades=_get(model, "num_grades"),
    )
    encoder = EncoderConfig(
        embed_dim=_get(model, "embed_dim"),
        depth=_get(model, "vit_depth"),
        heads=_get(model, "heads"),
        patch=_get(model, "patch"),
        mlp_ratio=_get(model, "mlp_ratio"),
        base_grid=_get(model, "base_grid"),
        depth_tokens=_get(model, "depth_tokens"),
        depth_stride=_get(model, "depth_stride"),
        use_depth=_get(model, "use_depth"),
        lora=LoRAConfig(
            rank=_get(model, "lora_rank"),
            alpha=_get(model, "lora_alpha"),
            num_layers=_get(model, "vit_depth"),
        ),
    )
    decoder = DecoderConfig(
        embed_dim=_get(model, "embed_dim"),
        channels=_get(model, "decoder_channels"),
        levels=_get(model, "decoder_levels"),
        num_classes=_get(model, "num_seg_classes"),
    )
    twin = TwinConfig(
        feature_dim=_get(model, "embed_dim"),
        hidden_dim=_get(model, "twin_hidden"),
        layers=_get(model, "twin_layers"),
        edge_mode=_get(model, "edge_mode"),
    )
    head = HeadConfig(
        in_dim=_get(model, "twin_hidden"),
        hidden_dim=_get(model, "head_hidden"),
        num_grades=_get(model, "num_grades"),
    )
    ablation = AblationConfig(
        transfer=_get(model, "transfer"),
        use_gnn=_get(model, "use_gnn"),
        head_kind=_get(model, "head_kind"),
    )
    return ChondroTwinConfig(
        spec=spec,
        encoder=encoder,
        backbone=BackboneConfig(use_pretrained=_get(model, "use_pretrained")),
        decoder=decoder,
        twin=twin,
        head=head,
        ablation=ablation,
    )


def materialize_optimizer(optim: Any) -> OptimizerConfig:
    return OptimizerConfig(
        lr=_get(optim, "lr"),
        weight_decay=_get(optim, "weight_decay"),
        betas=(_get(optim, "beta1"), _get(optim, "beta2")),
    )


def materialize_schedule(sched: Any) -> ScheduleConfig:
    return ScheduleConfig(
        warmup_epochs=_get(sched, "warmup_epochs"),
        total_epochs=_get(sched, "total_epochs"),
        peak_lr=_get(sched, "peak_lr"),
        floor_lr=_get(sched, "floor_lr"),
    )


def materialize_objective(objective: Any, num_grades: int) -> ObjectiveConfig:
    return ObjectiveConfig(
        edl_weight=_get(objective, "edl_weight"),
        evidence=EvidenceConfig(
            num_grades=num_grades, anneal_epochs=_get(objective, "anneal_epochs")
        ),
    )


def materialize_train(train: Any) -> TrainConfig:
    return TrainConfig(
        grad_accum=_get(train, "grad_accum"),
        max_steps=_get(train, "max_steps"),
        amp=_get(train, "amp"),
        amp_dtype=_get(train, "amp_dtype"),
    )


def build_loader(data: Any, spec: VolumeSpec, split: str, batch_size: int) -> DataLoader[Batch]:
    corpus = _CORPUS[_get(data, "corpus")]
    length = _get(data, "train_len") if split == "train" else _get(data, "val_len")
    raw = corpus(spec, split, root=_get(data, "root"), synthetic_len=length)
    dataset = cast(Dataset[Batch], raw)
    shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def assemble_model(model_group: Any) -> ChondroTwin:
    return build_model(materialize_model(model_group))


def as_dict(cfg: RunConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
