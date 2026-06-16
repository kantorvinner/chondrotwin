# Repository Plan

## Layout

```
chondrotwin/
  clay/        corpus records, schema, synthetic cohort, .npz loaders
  wedging/     intensity normalisation and spatial / photometric augmentation
  wheel/       LoRA, depth-aware positional encoding, ViT-L adapter, backbone loader
  trimming/    pyramid pooling, feature pyramid, UPerNet head
  armature/    anatomy graph, frozen contact weights, region pooling, GraphSAGE
  assay/       Dirichlet head and closed forms, EDL loss, uncertainty decomposition
  kiln/        objective, optimiser, schedule, losses, trainer, DDP, seeds, checkpoints
  inspection/  Dice, ordinal, calibration, selective referral, significance, cost, baselines
  gallery/     hydra-zen runtime, experiment registry, train / evaluate / infer / export / panels
  bench.py     assembled ChondroTwin and configuration
recipes/       one YAML per reported experiment and ablation
provings/      shape / invariant / overfit / regression / integration suites
throwing/      launch and data-preparation shell scripts
docs/          this plan, project context, implementation map, deviations
```

## Module responsibilities

- `clay` owns the data contract (`Sample`, `Batch`, `VolumeSpec`) and never imports model code.
- `wheel → trimming → armature → assay` form the forward chain; `bench.ChondroTwin` is the only place they
  are wired together.
- `kiln` holds everything stateful about optimisation; it depends on `bench` and `assay` but not on
  `gallery`.
- `inspection` is import-light (numpy / torch / sklearn / statsmodels) and read-only.
- `gallery` is the only package that imports hydra-zen; tasks are plain functions wrapped by `zen()`.

## Dependencies

Runtime: torch ≥ 2.1, torchvision, einops, numpy, scipy, scikit-learn, statsmodels, hydra-zen,
hydra-core, omegaconf, pyyaml. Optional extras: timm / transformers (`backbone`), monai / SimpleITK
(`medio`), onnx (`export`).

## Testing

The proving suite spans tensor-shape contracts, Dirichlet-simplex and frozen-edge invariants, single-batch
overfitting and gradient flow, numerical agreement against torch.distributions / scikit-learn / monai,
schedule and calibration numerics, configuration composition, a style guard (no comments, no docstrings, no
forbidden phrasing or emoji), and a two-step training smoke run.
