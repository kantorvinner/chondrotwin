# chondrotwin — a studio for cross-modal knee-cartilage digital twins

CartiDT cast as a working pottery studio: raw MRI clay is wedged, thrown on the foundation-model wheel,
trimmed into six cartilage compartments, set onto an anatomy armature, assayed for evidential
uncertainty, and fired into a calibrated digital twin of subclinical knee-cartilage damage.

---

## About the studio

`chondrotwin` implements the CartiDT pipeline for IEEE J-BHI: a frozen DINOv2 ViT-Large adapted with
depth-wise LoRA, a UPerNet decoder over 3D DESS volumes, a GraphSAGE twin across fourteen anatomical
regions with frozen contact-area edges, and a Dirichlet evidential head that grades WORMS damage while
separating aleatoric from epistemic uncertainty for selective referral.

## The clay body

```
  volume ─▶ wedging ─▶ wheel (DINOv2 + depth LoRA) ─┬─▶ trimming (UPerNet) ─▶ six-compartment masks
                                                     └─▶ armature (region pooling + GraphSAGE)
                                                              └─▶ assay (Dirichlet head) ─▶ grade + u_ale / u_epi
```

Every stage maps to one subpackage; the manuscript anchors live in `docs/implementation-map.md`.

## Tools and kiln

```bash
# the wheel and the kiln (runtime only)
pip install -e .

# add the DINOv2 wheel-head and the medical-imaging glazes
pip install -e ".[backbone,medio,export]"
```

```bash
# a fully stocked studio via conda
conda env create -f environment.yml
conda activate chondrotwin
```

```bash
# fire it inside a container
docker build -t chondrotwin:0.1.0 .
docker run --gpus all chondrotwin:0.1.0 +experiment=main
```

## Production sequence

| Bench verb | Command | What it makes |
| ---------- | ------- | ------------- |
| fire    | `python -m chondrotwin.gallery.train +experiment=main`            | trains the twin and writes a checkpoint |
| inspect | `python -m chondrotwin.gallery.evaluate +experiment=main`         | segmentation, detection, calibration, referral metrics |
| pull    | `python -m chondrotwin.gallery.infer +experiment=main`            | per-region grade with aleatoric / epistemic readings |
| cast    | `python -m chondrotwin.gallery.export_onnx +experiment=main`      | exports the segmentation path to ONNX |

Console entry points `chondrotwin-fire`, `chondrotwin-inspect`, `chondrotwin-pull`, and
`chondrotwin-cast` mirror the four verbs. Any field is overridable on the kiln shelf, for example
`+experiment=main model.lora_rank=8 train.amp=false`.

## Firing log

Targets are the values reported in the manuscript over the full 4×A100 run with seeds {42, 123, 7}. The
offline synthetic clay exercises the pipeline shape, not these numbers.

| Reading (manuscript table) | Command | Fired target |
| -------------------------- | ------- | ------------ |
| Mean DSC, IWOAI test (Table III) | `inspect +experiment=main`            | 0.879 ± 0.002 |
| Patellar DSC, hardest compartment | `inspect +experiment=main`           | 0.814 |
| Damage detection AUROC, OAI (Table IV) | `inspect +experiment=main`      | 0.861 |
| Expected calibration error (Table V) | `inspect +experiment=main`        | 0.034 |
| Area under risk–coverage | `inspect +experiment=main`                     | 0.096 |
| False-positive rate at 15% abstention | `inspect +experiment=main`       | 0.114 |
| Zero-shot SKI10 DSC (Table IX) | `inspect +experiment=transfer_ski10`     | 0.874 |

Ablation recipes reproduce Tables VI–VIII, for example
`inspect +experiment=ablation_component_no_gnn`.

## Kiln specifications

| Quantity | Value |
| -------- | ----- |
| Hardware | 4 × NVIDIA A100 80 GB |
| Wall-clock | ~38 h to convergence |
| Epochs | 120 |
| Effective batch | 4 per GPU × 2 grad-accum × 4 GPUs = 32 |
| Precision | bf16 autocast (see `docs/deviations.md`) |
| Trainable share | LoRA + decoder + twin + head ≈ 1.57 M LoRA parameters over a frozen 304 M backbone |

## Materials

| Corpus | Role | Access |
| ------ | ---- | ------ |
| OAI | WORMS grading cohort (3212 / 802 / 782) | https://nda.nih.gov/oai (Data Use Agreement) |
| OAI-ZIB | dense six-compartment masks (404 / 51 / 52) | https://pubdata.zib.de |
| SKI10 | zero-shot femur/tibia transfer | MICCAI 2010 challenge archive |
| IWOAI 2019 | held-out segmentation evaluation | derived from OAI-ZIB |

`throwing/prepare_data.sh` prints the expected on-disk `.npz` layout. Without a data root the loaders fall
back to a deterministic synthetic cohort so the studio runs offline.

## Test tiles

```bash
make lint    # ruff, black --check, isort --check
make type    # mypy --strict
make test    # the proving suite
make smoke   # two firing steps on the unit-test recipe
```

The proving suite under `provings/` mixes shape contracts, Dirichlet-simplex and frozen-edge invariants,
single-batch overfitting and gradient flow, agreement with torch.distributions / scikit-learn / monai,
schedule and calibration numerics, configuration composition, and a style guard.

## Studio practice

Analysis used de-identified, previously collected data under Data Use Agreement 2024-0347 (dated
12 December 2024); secondary analysis of de-identified data required no further IRB review. Funded by the
National Natural Science Foundation of China, No. 81800834. Checkpoints are available from the
corresponding author on reasonable request.
