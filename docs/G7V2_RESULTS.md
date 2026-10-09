# G7-v2: six-model GDDR6X bit-flip campaign — results summary

All numbers are pooled from the VERIFIED campaign runs under
`artifacts/g7/campaign` (derived at read time by
`tools/g5_faultinj/analyze_campaign.py` / `plot_accuracy_curve.py`;
per-model points CSVs sit next to each figure). Campaign completed
2026-09-29.

Setup: NVIDIA RTX 4090 (GDDR6X), six ImageNet-1K models as TensorRT
8.6.1 INT8 PTQ engines (explicit Q/DQ, per-channel weights, classifier
head quantized — the v2 engine family), 10,000-image eval split,
100 trials per BER level. One trial = B random bit flips (single-bit-up
60 % / multi-cell-up 40 % event mix) XOR-held on device memory through
the model's FULL evaluation of all 10,000 images, then restored and
verified. Seven frozen BER levels, 1e-7 … 1e-5; B = round(BER × R × 8)
per model from its frozen injection surface R (§4). Trials aborted by
DUE (first invalid output) or killed mid-trial (PROCESS_FATAL, §3) are
excluded from the accuracy mean and reported as rates — accuracy
conventions identical across all six models.

## 1. Top-1 accuracy (%) per BER level

| model | clean | L1 1e-7 | L2 5e-7 | L3 1e-6 | L4 3e-6 | L5 5e-6 | L6 7e-6 | L7 1e-5 | Δ (L7−clean) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | 78.42 | 78.34 | 77.13 | 76.68 | 73.08 | 66.62 | 67.14 | 61.23 | −17.19 |
| MobileNetV3-L | 75.04 | 74.58 | 74.34 | 71.60 | 67.13 | 53.38 | 51.58 | 47.27 | −27.77 |
| EfficientNet-B0 | 77.36 | 77.24 | 76.50 | 75.21 | 73.63 | 71.08 | 66.20 | 61.34 | −16.02 |
| DeiT-S | 78.73 | 78.51 | 78.47 | 78.35 | 75.54 | 68.48 | 64.63 | 61.11 | −17.62 |
| Swin-T | 81.30 | 81.25 | 79.62 | 77.92 | 74.20 | 74.82 | 62.25 | 58.84 | −22.46 |
| ViT-B | 78.21 | 76.80 | 77.72 | 77.54 | 77.08 | 73.85 | 70.86 | 71.73 | −6.48 |

ResNet-50 runs a NINE-level ladder; the seven columns above are its
L3–L9 (the same BERs as the five-model ladder, matched by BER value).
Its two lower points: L1 (1e-8) 78.44, L2 (5e-8) 78.39.

Reading: every model except ViT-B loses 16–28 pp by 1e-5.
MobileNetV3 collapses hardest (−27.8); ViT-B is the outlier — only
−6.5 pp, and its curve is non-monotonic (L6 70.86 → L7 71.73).

## 2. SDC-top1 rate (% of images whose top-1 class changes)

| model | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | 1.77 | 4.10 | 5.31 | 10.84 | 19.68 | 19.82 | 28.15 |
| MobileNetV3-L | 1.65 | 3.82 | 8.53 | 16.33 | 35.08 | 37.92 | 44.90 |
| EfficientNet-B0 | 1.49 | 3.33 | 5.93 | 9.48 | 14.48 | 21.13 | 28.89 |
| DeiT-S | 9.20 | 9.53 | 9.86 | 13.33 | 21.78 | 26.31 | 30.92 |
| Swin-T | 1.99 | 4.06 | 6.31 | 11.37 | 11.38 | 27.21 | 31.31 |
| ViT-B | 11.86 | 10.91 | 11.16 | 11.78 | 15.77 | 19.24 | 18.50 |

Numeric perturbation (probability change without class change) is
≈100 % of images at every level ≥ L2 for all models (L1: 40–72 %
depending on model) — single flips always perturb logits, only
sometimes enough to flip the argmax.

Reading: the CNNs and Swin-T grow roughly linearly with log(BER) and
converge near 28–45 % at 1e-5; DeiT-S starts high and plateaus
(L1–L3 ≈ 9–10 %); ViT-B is nearly BER-independent (11.9 % at L1, 18.5 %
at L7) — per-flip sensitivity is the highest of the six (mean |ΔP| at
L1: ViT-B 0.085 vs Swin-T 0.018), but the damage saturates instead of
compounding, so accuracy barely degrades.

## 3. Reliability: DUE and PROCESS_FATAL trials (of 100 attempted)

| model | L1 | L2 | L3 | L4 | L5 | L6 | L7 | crash total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 2/0 | 2/0 | 0 |
| MobileNetV3-L | 0/0 | 0/0 | 0/0 | 2/0 | 5/0 | 3/0 | 6/0 | 0 |
| EfficientNet-B0 | 0/0 | 0/1 | 0/1 | 0/6 | 2/7 | 3/6 | 4/3 | 24 |
| DeiT-S | 0/0 | 0/0 | 0/0 | 0/0 | 0/1 | 0/0 | 0/0 | 1 |
| Swin-T | 0/0 | 0/0 | 0/1 | 1/3 | 0/5 | 1/7 | 1/17 | 33 |
| ViT-B | 0/0 | 0/0 | 0/0 | 0/0 | 1/2 | 0/0 | 2/1 | 3 |

(each cell: DUE trials / PROCESS_FATAL crashes; the ResNet-50 row is
its L3–L9, so the last two columns carry its L8/L9 DUE trials)

PROCESS_FATAL = the runner process died mid-trial: a flip landed in
TensorRT runtime control state (a ctx-phase pool) and the next
injected inference killed the CUDA context. The restart protocol
verified the dead segment's evidence (contiguous completed-trial
prefix, exactly one open trial, complete flip-set tail) and relaunched
fresh gated segments for the remaining slots — 61 crashes across the
campaign, zero lost or fabricated trials. Two fatal CUDA signatures
are whitelisted (illegal memory access; operation not supported on
global/shared address space — same mechanism, different driver
string).

Reading: crash rate tracks each model's ctx-phase-pool share of R
(Swin-T largest at 28 % → 17 % crash at L7; MobileNetV3 scoped clean
→ 0). DUE is marginal everywhere except MobileNetV3 (up to 6 %), the
only model whose INT8 head-plus-depthwise stack produces NaN/inf
outputs rather than silent corruption.

## 4. Frozen injection surfaces (per-workload R)

| model | R (bytes) | bits @ L1 (1e-7) | bits @ L7 (1e-5) | engine |
| --- | --- | --- | --- | --- |
| ResNet-50 | 28,832,268 | 23 (its L3) | 2,307 (its L9) | int8 PTQ explicit Q/DQ, head quantized |
| MobileNetV3-L | 9,087,912 * | 7 | 727 | same |
| EfficientNet-B0 | 17,144,232 | 14 | 1,372 | same |
| DeiT-S | 26,010,832 | 21 | 2,081 | same |
| Swin-T | 43,799,616 | 35 | 3,504 | same (+ `--disable_mha_qdq`) |
| ViT-B | 93,867,728 | 75 | 7,509 | same |

\* MobileNetV3's frozen R is the EXCLUSION-SCOPED injection surface,
not the full measured residency: the bootstrap measured 9,336,232 B,
but 248,320 B (2.66 %) of that is the trt-internal-5/6 ctx-phase
pools, which are process-fatal on ANY hit (CUDA illegal memory access
on the first injected inference, reproduced offline on seeds 7/8/10).
User decision 2026-09-27: exclude those two pools from the sampling
surface for MobileNetV3 only; every other model injects over its FULL
surface and relies on the restart protocol to absorb the crashes
(EfficientNet-B0 was evaluated the same way and kept full surface —
24 crashes recovered, no data loss).

Level compositions (singles/doubles/triples) are frozen per workload
in `tools/g5_faultinj/fault_model.py` with an exact s+2d+3t = bits
identity; see each workload's `work_detail.json` for the per-trial
echo.

## 5. Where the artifacts live

- Unified figure: `artifacts/g7/campaign/fig_six_models/`
  (fig_six_models_accuracy_vs_ber.{png,pdf} + points CSV; generated by
  `tools/g5_faultinj/plot_six_models.py`)
- Per-model figures: `artifacts/g7/campaign/fig_{mobilenetv3,
  efficientnet,deit,swin,vit}/` and the ResNet-50 figure under
  `artifacts/g7/campaign/`
- Full per-level analysis text: `artifacts/g7/{mobilenet,
  resnet50}_analysis.txt` and the per-chain console logs
  (`artifacts/g7/campaign_chain_*_console.log`)
- Preprocessing cache groups (three, keyed by spec): MobileNetV3 +
  EfficientNet share 256-scale ImageNet-normalized; DeiT-S + Swin-T
  share 248-scale ImageNet-normalized; ViT-B is alone at 248-scale
  mean/std 0.5 — all under `artifacts/g7/image_cache/`
