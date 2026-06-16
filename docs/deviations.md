# Deviations

Each entry records where the implementation makes an explicit choice and the manuscript anchor that
justifies it.

## D1 — training precision (resolved)
The manuscript states the optimiser, weight decay, and schedule (Sec V.C) but never names the floating
-point precision. The default here is **bf16 autocast**, which A100 hardware supports natively and which
matches the standard DINOv2 fine-tuning recipe; it avoids the loss-scaling instability that fp16 can
introduce in the EDL KL term. Override with `train.amp=false` or `train.amp_dtype=float16`.

## D2 — frozen contact edge weights
Per Sec III.D-2 the graph contact weights are computed once from the mean training segmentation and held
fixed for both training and inference. `AnatomyGraphSage` stores them as a non-trainable buffer; this is
the single use of an aggregate training-set statistic and is intentional, not data leakage at the sample
level. `set_contact_weights` is provided to install the pre-computed values.

## D3 — frozen backbone stand-in for offline runs
When the `backbone` extra (timm / transformers) and DINOv2 weights are not present, the encoder falls back
to a deterministic frozen orthonormal patch projection so the pipeline runs end-to-end without network
access. The LoRA, decoder, twin, and evidence parameters are identical in both paths; only the frozen
feature source differs. `BackboneConfig(use_pretrained=True)` selects the published ViT-Large weights.

## D4 — dataset links that no longer resolve
Two URLs printed in the manuscript do not resolve and were removed from the README and source: the SKI10
host `ski10.org` (HTTP 301 to an unrelated site) and `github.com/IWOAI/knee-segmentation` (HTTP 404). The
datasets remain real; access channels are documented in `project-context.md`. OAI (`nda.nih.gov/oai`) and
OAI-ZIB (`pubdata.zib.de`) were confirmed live.

## D5 — exotic ablation variants
The transfer and GNN ablation tables (VI, VII) list architecture variants beyond the switches that change
behaviour structurally (`transfer`, `use_depth`, `use_gnn`, `edge_mode`, `head_kind`). Variants such as
`self_attn`, `concat_mlp`, `patch3d`, and `slice_wise_late` are expressed through the nearest structural
knob (for example fully connected edges with a single aggregation layer, or disabled depth interpolation
with unit depth stride). The recipe files name each variant; the shared GraphSAGE/encoder backbone is
reused with the indicated adjacency, weighting, or positional change.

## D6 — ONNX segmentation export
`gallery/export_onnx` traces the encoder + UPerNet segmentation path only (the deployment artefact). The
pyramid-pooling stage uses fixed-kernel average pooling rather than adaptive pooling so the graph is
exportable under opset 17 with the TorchScript exporter; the pooled extents are unchanged for the reported
feature-grid sizes.
