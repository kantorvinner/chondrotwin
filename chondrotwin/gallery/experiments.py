from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from .runtime import (
    DataGroup,
    ModelGroup,
    ObjectiveGroup,
    RunConfig,
    SchedGroup,
    TrainGroup,
)


def _variant(**model_overrides: Any) -> RunConfig:
    base = RunConfig()
    model = dataclasses.replace(base.model, **model_overrides)
    return dataclasses.replace(base, model=model)


def _unittest() -> RunConfig:
    model = ModelGroup(
        depth=8,
        height=32,
        width=32,
        embed_dim=32,
        vit_depth=2,
        heads=4,
        patch=8,
        mlp_ratio=2.0,
        base_grid=4,
        depth_tokens=8,
        depth_stride=2,
        lora_rank=4,
        lora_alpha=8.0,
        decoder_channels=16,
        decoder_levels=3,
        twin_hidden=24,
        twin_layers=3,
        head_hidden=16,
    )
    return RunConfig(
        model=model,
        train=TrainGroup(seed=7, epochs=2, grad_accum=1, max_steps=2, amp=False, batch_size=2),
        sched=SchedGroup(warmup_epochs=1, total_epochs=2, peak_lr=2.0e-4, floor_lr=1.0e-6),
        objective=ObjectiveGroup(edl_weight=0.5, anneal_epochs=2),
        data=DataGroup(corpus="oai_zib", train_len=4, val_len=2),
        output_dir="runs/_unittest",
    )


def _transfer_ski10() -> RunConfig:
    base = RunConfig()
    return dataclasses.replace(base, data=DataGroup(corpus="ski10", train_len=8, val_len=4))


def build_experiments() -> dict[str, RunConfig]:
    catalogue: dict[str, RunConfig] = {
        "main": RunConfig(),
        "_unittest": _unittest(),
        "transfer_ski10": _transfer_ski10(),
        "ablation_transfer_full_ft": _variant(transfer="full_ft"),
        "ablation_transfer_lora_no_depth": _variant(use_depth=False),
        "ablation_transfer_lora_r4": _variant(lora_rank=4),
        "ablation_transfer_lora_r64": _variant(lora_rank=64),
        "ablation_transfer_patch3d": _variant(depth_stride=1),
        "ablation_transfer_slice_wise_late": _variant(use_depth=False, depth_stride=1),
        "ablation_gnn_fc_graph": _variant(edge_mode="fc"),
        "ablation_gnn_indep_mlp": _variant(use_gnn=False),
        "ablation_gnn_self_attn": _variant(edge_mode="fc", twin_layers=1),
        "ablation_gnn_concat_mlp": _variant(use_gnn=False, twin_hidden=512),
        "ablation_gnn_biomech": _variant(edge_mode="contact"),
        "ablation_component_no_lora": _variant(transfer="full_ft"),
        "ablation_component_no_depth_interp": _variant(use_depth=False),
        "ablation_component_no_gnn": _variant(use_gnn=False),
        "ablation_component_no_joint": dataclasses.replace(
            RunConfig(), objective=ObjectiveGroup(edl_weight=1.0, anneal_epochs=10)
        ),
        "ablation_component_softmax_head": _variant(head_kind="softmax"),
        "ablation_component_uniform_edges": _variant(edge_mode="uniform"),
        "ablation_component_fc_edges": _variant(edge_mode="fc"),
    }
    return catalogue


def run_with_hydra(task: Callable[..., None], task_name: str) -> None:
    from hydra_zen import store, zen

    experiment_store = store(group="experiment", package="_global_")
    for name, cfg in build_experiments().items():
        experiment_store(cfg, name=name)

    base = store(
        {"hydra": {"job": {"chdir": False}}},
        hydra_defaults=["_self_", {"experiment": "main"}],
        name=task_name,
    )
    del base
    store.add_to_hydra_store(overwrite_ok=True)
    zen(task).hydra_main(config_name=task_name, config_path=None, version_base="1.3")
