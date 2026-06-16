# Project Context

Canonical values shared by every module. Anchors point to the manuscript; confidence is HIGH unless noted.

## Summary

| Field            | Value                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------- |
| package          | `chondrotwin`                                                                           |
| system           | CartiDT — cross-modal foundation-model transfer + anatomy-graph digital twin + evidential uncertainty |
| domain           | musculoskeletal 3D DESS knee MRI: six-compartment segmentation, per-region WORMS grading, decomposed uncertainty |
| framework        | PyTorch 2.x (Sec V.C states PyTorch 2.1); einops, timm/transformers for DINOv2, monai for Dice/elastic |
| venue            | IEEE Journal of Biomedical and Health Informatics                                       |
| compute          | 4× NVIDIA A100 80 GB, ~38 h wall-clock, 120 epochs                                      |
| precision        | bf16 autocast (A100 native; the manuscript leaves precision unstated — see deviations) |
| config / CLI     | hydra-zen structured configs over hydra-core; `zen()` task entrypoints                  |
| supplementary    | none (single 12-page submission)                                                        |

## Datasets

| Name           | Role                                   | Split (train/val/test) | Access (verified June 2026)                   | Terms |
| -------------- | -------------------------------------- | ---------------------- | --------------------------------------------- | ----- |
| OAI            | WORMS grading cohort                   | 3212 / 802 / 782       | https://nda.nih.gov/oai (resolves, HTTP 200)  | NIH Data Use Agreement; registration required |
| OAI-ZIB        | dense six-compartment segmentation     | 404 / 51 / 52          | https://pubdata.zib.de (resolves, HTTP 200)   | OAI DUA + ZIB redistribution terms |
| SKI10          | zero-shot femur/tibia transfer         | 100 / – / 50           | original host `ski10.org` no longer resolves (301 to an unrelated site) — obtain via the MICCAI 2010 challenge archive | challenge-distribution terms |
| IWOAI 2019     | held-out segmentation evaluation       | 60 / 14 / 14           | derived from OAI-ZIB; the URL printed in the manuscript (`github.com/IWOAI/knee-segmentation`) returns HTTP 404 and is dropped — see ref. Desai et al., Radiol. Artif. Intell. 3(3):e200078, 2021 | inherits OAI/OAI-ZIB terms |

Every dataset URL was checked. Two links that do not resolve (SKI10 host, the IWOAI repository) were removed from the README and code; the datasets themselves remain real and are reached through the channels listed above.

## Architecture and hyperparameters (Sec III, Sec V.C)

| Group            | Value |
| ---------------- | ----- |
| backbone         | DINOv2 ViT-Large/16, 304 M parameters, frozen (Sec III.C; Table I) |
| LoRA             | rank r = 16, α = 32 on Q and V projections of all 24 self-attention layers (Eq 1); 1.57 M trainable |
| depth encoding   | bicubic interpolation of the 2D positional grid plus a learned depth embedding, stride s_d = 4 (Eq 2) |
| decoder          | UPerNet (FPN + PPM) at full resolution, six cartilage/meniscus compartments + background (Eq 3) |
| twin graph       | GraphSAGE, L = 3, hidden 256, GELU; 14 region nodes, 26 directed edges (Eq 4) |
| edge weights     | contact-area ratio computed once from the mean training segmentation, then frozen (Sec III.D-2) |
| evidence head    | 2-layer MLP → Softplus(·) + 1 → Dirichlet α, K = 4 grades (Eq 5–6) |
| evidential loss  | Bayes-risk term + λ_t · KL(Dir(α̃)‖Dir(1)), λ_t = min(1, t / 10) (Eq 7–9) |
| joint loss       | L = L_seg + μ · L_EDL, μ = 0.5; L_seg = soft-Dice + cross-entropy (Eq 10) |
| optimiser        | AdamW (β1 = 0.9, β2 = 0.999, weight decay 1e-2) |
| schedule         | linear warmup 0 → 2e-4 over 5 epochs, cosine decay to 1e-6 over the remaining 115 |
| batch            | 4 per GPU × 2 grad-accum × 4 GPUs = effective 32 |
| epochs / seeds   | 120 epochs; seeds 42, 123, 7 (±std reported in the main tables) |
| volume           | 160 × 384 × 384, spacing 0.365 × 0.365 × 0.7 mm, per-volume z-score |
| augmentation     | rotation ±10°, scale 0.9–1.1, translation ±15 voxels, elastic, Gaussian σ = 0.02, gain ∈ [0.95, 1.05] |
| statistics       | paired bootstrap n = 10 000, Holm–Bonferroni across Tables III–IX |

## Reported artefacts

- Algorithm boxes: Module A (LoRA + depth interpolation), Module B (graph aggregation), Module C (Dirichlet head + EDL).
- Tables III (segmentation), IV (detection + grading), V (calibration), VI (transfer ablation, 6 variants), VII (GNN ablation, 5 variants), VIII (component ablation, 7 variants), IX (zero-shot SKI10), X (compute cost).
- Figures: architecture, mathematical-principle diagram, qualitative segmentation + uncertainty maps, multi-panel analysis.

## Ethics and funding (manuscript)

- Data are de-identified and previously collected; analysis ran under a Data Use Agreement, DUA protocol 2024-0347, dated 12 December 2024.
- No additional IRB review was required under university policy for secondary analysis of de-identified data.
- Funding: National Natural Science Foundation of China, No. 81800834.
