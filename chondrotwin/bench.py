from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import nn

from .armature.anatomy_graph import GraphTopology
from .armature.graph_sage import AnatomyGraphSage, TwinConfig
from .armature.region_features import RegionPooling
from .assay.decompose import UncertaintySplit, decompose
from .assay.dirichlet_head import DirichletHead, HeadConfig
from .clay.schema import VolumeSpec
from .trimming.upernet import DecoderConfig, UPerHead
from .wheel.dinov2_weights import BackboneConfig, build_encoder
from .wheel.lora import LoRAConfig
from .wheel.vit_adapter import EncoderConfig


class TwinOutput(NamedTuple):
    seg_logits: torch.Tensor
    alpha: torch.Tensor
    uncertainty: UncertaintySplit


@dataclass(frozen=True, slots=True)
class AblationConfig:
    transfer: str = "lora"
    use_gnn: bool = True
    head_kind: str = "dirichlet"


@dataclass(frozen=True, slots=True)
class ChondroTwinConfig:
    spec: VolumeSpec = field(default_factory=VolumeSpec)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    twin: TwinConfig = field(default_factory=TwinConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def reconciled(self) -> ChondroTwinConfig:
        embed = self.encoder.embed_dim
        decoder = DecoderConfig(
            embed_dim=embed,
            channels=self.decoder.channels,
            levels=self.decoder.levels,
            num_classes=self.spec.num_seg_classes,
        )
        twin = TwinConfig(
            feature_dim=embed,
            hidden_dim=self.twin.hidden_dim,
            layers=self.twin.layers,
            edge_mode=self.twin.edge_mode,
        )
        head_in = twin.hidden_dim if self.ablation.use_gnn else embed
        head = HeadConfig(
            in_dim=head_in, hidden_dim=self.head.hidden_dim, num_grades=self.spec.num_grades
        )
        return ChondroTwinConfig(
            spec=self.spec,
            encoder=self.encoder,
            backbone=self.backbone,
            decoder=decoder,
            twin=twin,
            head=head,
            ablation=self.ablation,
        )


class SoftmaxHead(nn.Module):
    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_grades),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(node_features), dim=-1) * float(node_features.shape[-1]) + 1.0


class ChondroTwin(nn.Module):
    def __init__(self, cfg: ChondroTwinConfig) -> None:
        super().__init__()
        cfg = cfg.reconciled()
        self.cfg = cfg
        self.graph = GraphTopology(num_nodes=cfg.spec.num_nodes)
        self.encoder = build_encoder(cfg.backbone, cfg.encoder)
        if cfg.ablation.transfer == "full_ft":
            self.encoder.enable_full_finetune()
        self.decoder = UPerHead(cfg.decoder)
        num_compartments = cfg.spec.num_seg_classes - 1
        self.region = RegionPooling(cfg.encoder.embed_dim, cfg.spec.num_nodes, num_compartments)
        self.use_gnn = cfg.ablation.use_gnn
        self.twin = AnatomyGraphSage(cfg.twin, self.graph)
        if cfg.ablation.head_kind == "softmax":
            self.head: nn.Module = SoftmaxHead(cfg.head)
        else:
            self.head = DirichletHead(cfg.head)

    def forward(self, volume: torch.Tensor) -> TwinOutput:
        target = (volume.shape[-3], volume.shape[-2], volume.shape[-1])
        feature_volume = self.encoder(volume)
        seg_logits = self.decoder(feature_volume, target)
        node_features = self.region(feature_volume, seg_logits)
        twin_features = self.twin(node_features) if self.use_gnn else node_features
        alpha = self.head(twin_features)
        return TwinOutput(seg_logits=seg_logits, alpha=alpha, uncertainty=decompose(alpha))

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: ChondroTwinConfig) -> ChondroTwin:
    return ChondroTwin(cfg)


def tiny_config() -> ChondroTwinConfig:
    spec = VolumeSpec(depth=8, height=32, width=32, num_seg_classes=7, num_nodes=14, num_grades=4)
    encoder = EncoderConfig(
        embed_dim=32,
        depth=2,
        heads=4,
        patch=8,
        mlp_ratio=2.0,
        base_grid=4,
        depth_tokens=8,
        depth_stride=2,
        lora=LoRAConfig(rank=4, alpha=8.0, num_layers=2),
    )
    decoder = DecoderConfig(embed_dim=32, channels=16, levels=3, num_classes=7)
    twin = TwinConfig(feature_dim=32, hidden_dim=24, layers=3)
    head = HeadConfig(in_dim=24, hidden_dim=16, num_grades=4)
    return ChondroTwinConfig(
        spec=spec,
        encoder=encoder,
        backbone=BackboneConfig(use_pretrained=False),
        decoder=decoder,
        twin=twin,
        head=head,
    )
