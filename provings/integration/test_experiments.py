from __future__ import annotations

from chondrotwin.bench import ChondroTwinConfig
from chondrotwin.gallery.experiments import build_experiments
from chondrotwin.gallery.runtime import (
    assemble_model,
    build_loader,
    materialize_model,
    materialize_objective,
    materialize_optimizer,
    materialize_schedule,
)


def test_all_experiments_materialize() -> None:
    experiments = build_experiments()
    assert len(experiments) >= 20
    for cfg in experiments.values():
        model_cfg = materialize_model(cfg.model)
        assert isinstance(model_cfg, ChondroTwinConfig)
        assert materialize_optimizer(cfg.optim).lr > 0
        assert materialize_schedule(cfg.sched).peak_lr > 0
        assert materialize_objective(cfg.objective, model_cfg.spec.num_grades).edl_weight >= 0


def test_unittest_experiment_builds_and_runs() -> None:
    cfg = build_experiments()["_unittest"]
    network = assemble_model(cfg.model)
    loader = build_loader(cfg.data, materialize_model(cfg.model).spec, "train", 2)
    batch = next(iter(loader))
    out = network(batch["volume"])
    assert out.seg_logits.shape[0] == batch["volume"].shape[0]


def test_ablation_switches_change_parameter_count() -> None:
    experiments = build_experiments()
    full = assemble_model(experiments["_unittest"].model)
    no_gnn_cfg = experiments["_unittest"].model
    no_gnn_cfg = type(no_gnn_cfg)(**{**no_gnn_cfg.__dict__, "use_gnn": False})
    reduced = assemble_model(no_gnn_cfg)
    assert full.trainable_parameters() != reduced.trainable_parameters()
