# Implementation Map

The source carries no inline comments or docstrings; this table is the single place that ties each
module to the manuscript section, equation, table, or algorithm box it realises.

| Manuscript anchor                         | Equation / Table | File / object | Notes |
| ----------------------------------------- | ---------------- | ------------- | ----- |
| Sec III.A — inputs, region taxonomy       | —                | `chondrotwin/clay/schema.py` | `VolumeSpec`, six compartments, 14 region nodes, K = 4 grades |
| Sec IV.A — datasets, splits               | Table II         | `chondrotwin/clay/catalog.py`, `clay/loaders.py` | OAI / OAI-ZIB / SKI10 / IWOAI records and `.npz` loaders; synthetic stand-in for offline runs |
| Sec V.C — augmentation                    | —                | `chondrotwin/wedging/` | z-score (`intensity_norm`), affine + elastic + photometric (`spatial_aug`) |
| Sec III.C-1 — LoRA injection              | Eq 1             | `chondrotwin/wheel/lora.py` | `LoRALinear`: frozen base, low-rank A/B on Q and V, scaling α/r |
| Sec III.C-2 — depth-wise positional       | Eq 2             | `chondrotwin/wheel/depth_posembed.py` | bicubic interpolation of the planar grid + learned depth embedding, stride s_d |
| Sec III.C — frozen DINOv2 ViT-L backbone  | Table I          | `chondrotwin/wheel/vit_adapter.py`, `wheel/dinov2_weights.py` | LoRA attention blocks; timm/transformers loader or a frozen orthonormal stand-in |
| Sec III.C-3 — UPerNet decoder             | Eq 3             | `chondrotwin/trimming/` | `PyramidPooling` (PPM), `FeaturePyramid` (FPN), `UPerHead` |
| Sec III.D-1 — anatomy graph               | —                | `chondrotwin/armature/anatomy_graph.py` | 14 nodes, 13 undirected → 26 directed contact edges |
| Sec III.D-2 — frozen contact weights      | —                | `chondrotwin/armature/contact_weights.py` | contact-area ratio from the mean training segmentation, frozen thereafter |
| Sec III.D — region feature pooling        | —                | `chondrotwin/armature/region_features.py` | segmentation-probability-weighted pooling per region node |
| Sec III.D — GraphSAGE aggregation         | Eq 4             | `chondrotwin/armature/graph_sage.py` | mean aggregator weighted by frozen contact edges, L = 3, hidden 256 |
| Sec III.E-1 — Dirichlet head              | Eq 5–6           | `chondrotwin/assay/dirichlet_head.py` | Softplus(·) + 1 → α |
| Sec III.E — Dirichlet closed forms        | Eq 6             | `chondrotwin/assay/dirichlet.py` | strength, expected probability, vacuity, KL(Dir‖Dir(1)) |
| Sec III.E-2 — uncertainty decomposition   | Eq 7–8           | `chondrotwin/assay/decompose.py` | total = aleatoric (expected entropy) + epistemic (mutual information) |
| Sec III.E-3 — EDL objective               | Eq 9             | `chondrotwin/assay/edl.py` | Bayes risk + annealed KL, λ_t = min(1, t / T_anneal) |
| Sec III.F — joint objective               | Eq 10            | `chondrotwin/kiln/objective.py`, `kiln/segmentation_loss.py` | L = L_seg + μ L_EDL; soft-Dice + cross-entropy |
| Sec V.C — optimiser / schedule            | —                | `chondrotwin/kiln/optimizer.py`, `kiln/schedule.py` | AdamW; warmup 0→2e-4 then cosine→1e-6 |
| Sec V.C — training loop, AMP, DDP, seeds  | —                | `chondrotwin/kiln/trainer.py`, `kiln/distributed.py`, `kiln/reproducibility.py`, `kiln/checkpointing.py` | grad-accum, bf16 autocast, atomic checkpoints carrying the seed |
| Sec V — assembled model                   | Fig 1            | `chondrotwin/bench.py` | `ChondroTwin` ties encoder → decoder → twin → evidence head |
| Table III — segmentation                  | Table III        | `chondrotwin/inspection/overlap.py` | per-class and foreground Dice |
| Table IV — detection + grading            | Table IV         | `chondrotwin/inspection/ordinal.py` | damage AUROC, ordinal MAE, accuracy, quadratic-weighted kappa |
| Table V — calibration                     | Table V          | `chondrotwin/inspection/calibration.py` | expected / maximum calibration error |
| Sec V — selective referral                | —                | `chondrotwin/inspection/selective.py` | risk–coverage curve, area under risk–coverage |
| Sec V — significance testing              | —                | `chondrotwin/inspection/significance.py` | paired bootstrap (n = 10 000), Holm–Bonferroni |
| Table X — compute cost                    | Table X          | `chondrotwin/inspection/cost.py` | parameter accounting (total / trainable / frozen) |
| Sec V — baselines                         | Table III–IV     | `chondrotwin/inspection/reference_models.py` | V-Net and slice-wise baselines behind a registry |
| Table VI — transfer ablations             | Table VI         | `recipes/ablation_transfer_*.yaml` | full fine-tune, no-depth, rank 4 / 64, patch / slice variants |
| Table VII — GNN ablations                 | Table VII        | `recipes/ablation_gnn_*.yaml` | fully connected, independent MLP, self-attention, concat, biomechanical edges |
| Table VIII — component ablations          | Table VIII       | `recipes/ablation_component_*.yaml` | remove LoRA / depth / GNN / joint loss, softmax head, uniform / fully connected edges |
| Table IX — zero-shot SKI10                | Table IX         | `recipes/transfer_ski10.yaml` | zero-shot femur/tibia transfer |
| Fig 4 — analysis panels                   | Fig 4            | `chondrotwin/gallery/panel_analysis.py`, `gallery/panel_uncertainty.py` | per-region grade distribution and uncertainty summaries |

## Code availability statement (verbatim, manuscript)

> To support the reproducibility claims, the full implementation of CartiDT, including training scripts,
> evaluation pipelines, configuration files, and pre-trained checkpoints sufficient to reproduce all
> results in Tables III–X, is publicly available at https://github.com/dmckenna7/cartidt-passes.

## Citation

```bibtex
@article{feng_cartidt_jbhi,
  title   = {Foundation Models Meet Musculoskeletal Digital Twins: Uncertainty-Aware Early Detection
             of Subclinical Cartilage Damage via Cross-Modal Transfer},
  author  = {Feng, Haoxuan and Peng, Dangbing},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026}
}
```
