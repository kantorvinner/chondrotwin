from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from chondrotwin.gallery.experiments import build_experiments
from chondrotwin.gallery.runtime import RunConfig, materialize_model

RECIPES = Path(__file__).resolve().parents[2] / "recipes"


def test_every_experiment_has_a_recipe_file() -> None:
    names = set(build_experiments())
    files = {p.stem for p in RECIPES.glob("*.yaml")}
    assert names == files


def test_recipes_parse_into_runconfig() -> None:
    for path in sorted(RECIPES.glob("*.yaml")):
        merged = OmegaConf.merge(OmegaConf.structured(RunConfig), OmegaConf.load(path))
        cfg = OmegaConf.to_object(merged)
        assert isinstance(cfg, RunConfig)
        spec = materialize_model(cfg.model).spec
        assert spec.num_nodes == 14


def test_main_recipe_matches_manuscript_defaults() -> None:
    main = OmegaConf.load(RECIPES / "main.yaml")
    assert main.optim.lr == 2.0e-4
    assert main.optim.weight_decay == 1.0e-2
    assert main.train.epochs == 120
    assert main.model.lora_rank == 16
    assert main.model.lora_alpha == 32.0
    assert main.train.batch_size * main.train.grad_accum * 4 == 32
