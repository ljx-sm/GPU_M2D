# G7 preparation tooling (six models, ImageNet-1K)

Bulk base work for the G7 multi-model extension (plan doc §G7, user
decisions 2026-09-23): download all six models now, quantize all to
INT8 now, run FP32+INT8 clean on the 10K eval split — then the fault
campaigns proceed strictly one model at a time starting with
ResNet-50/ImageNet.

Model set (frozen): ResNet-50 / MobileNetV3-Large / EfficientNet-B0 /
ViT-B/16 / DeiT-S / Swin-T, all timm ImageNet-1k pretrained, all
224×224, INT8 PTQ via TensorRT 8.6.1.

## Environment (single source of truth wiring)

- Python: `/data1/luojx/miniforge3/envs/vit_fault/bin/python` (torch
  2.5.1+cu124, timm 1.0.26, onnx 1.17.0, cv2 4.10.0,
  **tensorrt_bindings 8.6.1**).
- TensorRT libs: `/data1/luojx/REMU/.local/deps/tensorrt-8.6.1/tensorrt_libs`
  + cuDNN 8.9.7 `/data1/luojx/REMU/.local/deps/cudnn-8.9.7.29/...` —
  same LD_LIBRARY_PATH wiring as REMU `run_stage13_int8_build_one.sh`
  (set by the two `.sh` wrappers below; not needed system-wide).
- Methodology inherited from the user's own REMU `tests/stage13/`
  scripts (export → entropy PTQ → clean eval); no external code copied.

## Pipeline

1. `build_imagenet_splits.py` — seed 7; calib = 1000 classes × 1,
   eval = 1000 × 10 from `/data1/luojx/datasets/imagenet1k` (every val
   image cross-checked against the official `val_map.txt`,
   fail-closed). Outputs under `<dataset>/splits/`:
   `g7_calib_1000_perclass1.csv`, `g7_eval_10000_perclass10.csv`,
   `g7_splits_manifest.json`.
2. `download_models.py` — timm weights + `model_meta.json`
   (mean/std/interpolation/input_size/crop_pct/…) into
   `/data1/luojx/g7_models/<timm-name>/`. The meta file is the only
   source of preprocessing facts; nothing else hardcodes per-model
   values.
3. `export_g7_onnx.py` — ONNX opset 17, fixed batch 1, three-binding
   contract `data (1,3,224,224) fp32 / prob (1,1) fp32 / index (1,1)
   int32` (the G5 runner contract), INT64_MAX slice-end sentinel
   rewrite, sha256 provenance. → `model.onnx` + `onnx_summary.json`.
4. `quantize_g7_qdq.sh <models|all>` — **explicit Q/DQ INT8 quantization
   (v2 canonical path)**: ModelOpt ONNX PTQ in the ISOLATED
   `/data1/luojx/REMU/.local/deps/modelopt-venv` (never install into
   vit_fault — modelopt would drag torch>=2.8/CUDA13 with it), entropy
   calibration on `calib_canonical.npy`, per-channel weight
   quantization. Per-model recipes (see the protocol section below) are
   encoded in the wrapper, which then runs the two graph post-steps:
   `bypass_g7_qdq_activations.py` (mobile/effnet only — swish-output
   activations back to FP32) and `fix_qdq_for_trt86.py` (all models —
   TRT 8.6.1 parser/shape-inference normalizations). → final
   `model_qdq.onnx`.
5. `dump_g7_calib_npy.py` — feeds the SAME 1000-image calibration set
   through the SAME `preprocess_image()` into
   `calib_canonical.npy` (1000,3,224,224) fp32 for ModelOpt.
6. `build_g7_engines.sh 0 --explicit [models...]` — engine build.
   Explicit path (canonical): `build_g7_explicit_engine.py` parses the
   Q/DQ ONNX with EXPLICIT_BATCH only (the Q/DQ nodes dictate
   precision; the kINT8 builder flag is STILL required — "int8 is not
   configured in the builder" otherwise — but no calibrator), Q/DQ
   census + per-channel weight verification written into the summary.
   Implicit path (fallback, `build_g7_int8_engine.py`):
   IInt8EntropyCalibrator2, batch 1, per-tensor symmetric weights.
   Both → `clean.engine` + `engine_summary.json` (zero-input smoke at
   build time). `--parse-only` validates the ONNX→TRT parse.
7. `eval_g7_clean.sh 0 [models...]` — FP32 (torch/timm) and INT8 (TRT)
   on the 10K eval split, identical preprocessing; INT8 acceptance =
   two full passes with byte-identical (prediction, probability).
   → `eval/{fp32,int8}_predictions*.csv`, `eval/clean_summary.json`
   (fp32_top1, int8_top1, quantization loss pp).

## Preprocessing contract v2 (canonical, 2026-09-24 — identical everywhere)

timm-canonical val transform replicated in cv2: aspect-preserving
resize of the shorter edge to `int(224 / crop_pct)` (torchvision-exact
int() truncation), **INTER_AREA on downscale** (cv2's antialiased
downscaler — plain INTER_CUBIC/INTER_LINEAR aliasing costs ~1 pp top-1
on ImageNet; measured resnet50 79.26 vs 80.55), the model's own
interpolation on upscale, torchvision-exact round() center crop 224,
BGR→RGB, /255, mean/std from `model_meta.json`, CHW float32. Defined
once in `build_g7_int8_engine.preprocess_image` and imported
everywhere (calibration dump, implicit calibrator, eval, future G5
runner parameterization). Under this contract all six models' FP32
top-1 returns to paper level (80.55/…/… measured; the earlier
square-resize v1 policy cost 0.8–3.7 pp and is retired).

## Quantization protocol v2 rationale (2026-09-24, user-approved)

The v1 implicit-entropy engines were verified correct (92–95% of
engine layers pure INT8 I/O; ONNX chain ≤0.04 pp faithful) but
per-tensor symmetric weight quantization — the deprecated implicit
path — costs 5.7–9.7 pp on MobileNetV3/EfficientNet/DeiT-S. Explicit
Q/DQ with per-channel weight quantization (ModelOpt) is the standard
<1–2 pp recipe and replaces it as the G7 canonical engine
(`int8_ptq_explicit_qdq` in engine_summary.json). Diagnostics and the
MinMax discriminator that established this are
`diag_verbose_build.py` / `diag_canonical_fp32.py` /
`diag_eval_engine.py` / `inspect_engine_precision.py`.

### Final per-model recipes and results (2026-09-24, 10K clean eval)

Quantization damage attribution (Q/DQ bypass surgery + ORT probes on a
2000-image held probe subset) showed per-channel INT8 **weights cost
~0 pp everywhere**; all remaining damage was activation-side and
concentrated in specific op families. Final contract per model
(all: entropy calibration, per-channel weights, opset 17, fp32
high-precision dtype):

| model | recipe beyond base | FP32 | INT8 | loss |
| --- | --- | --- | --- | --- |
| resnet50 | none (base recipe) | 80.61 | 78.50 | 2.11 |
| mobilenetv3_large_100 | `--op_types_to_quantize Conv Gemm MatMul` + HardSwish-output bypass | 75.64 | 75.15 | 0.49 |
| efficientnet_b0 | `--op_types_to_quantize Conv Gemm MatMul` + SiLU-output bypass | 77.96 | 77.32 | 0.64 |
| vit_base_patch16_224 | none (base recipe) | 79.41 | 78.22 | 1.19 |
| deit_small_patch16_224 | none (base recipe) | 80.26 | 78.75 | 1.51 |
| swin_tiny_patch4_window7_224 | `--op_types_to_quantize Conv Gemm MatMul --disable_mha_qdq` | 81.63 | 81.17 | 0.46 |

Evidence chain behind the two non-default choices:

- **Swish-output activations** (EffNet SiLU ×16, MobileNetV3
  HardSwish ×14 tensors) are catastrophically hostile to per-tensor
  symmetric INT8: bypassing only those tensors recovers -22.65→-0.30 pp
  (effnet) and -6.05→-0.30 pp (mobile) on the probe; everything else
  (SE/ReLU/Add/GAP activations, all weights) costs ~0.3 pp combined.
  `--use_zero_point` does NOT help (60.85 with vs without). The bypass
  keeps conv **weights INT8 and every other conv-input activation
  INT8**; only the swish/hard-swish outputs feed their convs in FP32
  (the "少量非线性算子不量化" exception).
- **Swin attention region**: TRT 8.6.1 mis-executes Q/DQ around the
  window-attention machinery (Reshape/Transpose/Squeeze/Softmax
  cluster) — the quantized graph is 91.0 % in ORT but 0.5 % in TRT;
  stripping all Q/DQ runs 92.7 % in TRT; bypassing only `attn/`
  activations restores 91.0 % in TRT. `--disable_mha_qdq` is the
  recipe-level fix (keeps MLP/patch/downsample MatMuls + all weights
  quantized).

### ModelOpt auto-excludes the batch-1 classifier head (GEMV heuristic, all six models)

Found 2026-09-24 during the G7 fault-surface census: the ONE weighted
op left FP32 in every `model_qdq.onnx` is the classifier-head Gemm —
in all six models, with NO exclusion flag anywhere in our recipes.
Cause (ModelOpt 0.47.0, in the isolated venv): `enable_gemv_detection_
for_trt` is **default-on** (`modelopt/onnx/quantization/int8.py:167`)
and is NOT exposed as a CLI flag, so our CLI run necessarily carried
it. `find_nodes_from_matmul_to_exclude` (`graph_utils.py:1413`) drops
any weighted MatMul/Gemm whose output is rank<3 with a dim==1 into
`nodes_to_exclude`, and `int8.py:249-251` filters those out of
`nodes_to_quantize` — "GEMV cannot utilize TensorCores; the perf of
adding Q/DQ layers is not good in TRT" (a perf heuristic, not a
precision policy) — **even when Gemm is explicitly requested** via
`--op_types_to_quantize Conv Gemm MatMul` (mobilenet/effnet/swin do;
their heads are still FP32). Our exports are fixed batch-1 (the G5
host contract, step 3), so every head Gemm outputs `(1,1000)` and
trips the rule.

Verified by counterfactual on the real artifacts (modelopt venv,
2026-09-24): production-equivalent defaults reproduce the shipped
`model_qdq.onnx` exactly (QL/DQL 108/108, head weight a raw FP32
initializer); the single-flag delta `enable_gemv_detection_for_trt=
False` yields 110/110 with `classifier.fc.weight` behind a
per-channel DequantizeLinear. That single-flag difference eliminates
every other candidate cause (op lists, post-processing, adjacency).

Accepted as-is (engines frozen; the losses in the table above were
measured WITH the FP32 head, so the head is not a damage source, and
anyone running ModelOpt ONNX PTQ on a batch-1 export gets the same
graph). Consequences to remember:

- unquantized head-weight bytes: resnet50 8,192,000 (97.1% of its
  FP32 constant surface; 23.4% of the campaign residency R=34,959,884)
  / mobilenet, effnet 5,120,000 each (LARGER than their entire INT8
  backbone) / vit, swin 3,072,000 / deit 1,536,000;
- the FP32 head is part of the G7 fault surface and one of the
  explicit-Q/DQ engine's overflow→NaN (DUE) pathways; fatal-surface
  decompositions are per-model and must be redone for each engine;
- a fully-quantized-head variant would need the python-API flag above
  or a batch>1 / dynamic-batch export — out of scope for G7.

### TRT 8.6.1 graph-compatibility notes (`fix_qdq_for_trt86.py`)

- SE `ReduceMean(axes const-input) → Q → DQ → Conv` defeats the
  parser's conv channel inference ("group and kernel shape misalign")
  → rewritten to native GlobalAveragePool (+Flatten when keepdims=0);
  bitwise equivalent, verified ORT==TRT.
- ModelOpt forces opset 19 and Constant-input axes; the swin head
  `ReduceMean → Gemm` then loses shape inference ("GEMM must have 2D
  inputs"). Axes-as-INITIALIZER does not help; axes-as-ATTRIBUTE alone
  is spec-INVALID at opset 19 and TRT executes it as
  reduce-over-all-dims (swin 81.63→0.51 top-1). Correct fix: restore
  the attribute AND downgrade the graph opset to 17 (the original
  export's form) — legal, and TRT 8.6 imports it natively.

## G7-v2 fault-surface scoping — MobileNetV3 only (2026-09-27, user decision)

**Incident.** The first MobileNetV3 L1 campaign (seed 7) aborted on
trial 0 with `GPU_M2D_G1_5_FAIL: synchronize inference stream failed:
an illegal memory access`; two diagnostic reruns (seeds 8, 10) died the
same way after 40 / 14 trials. Offline re-sampling reproduced every
crash site exactly.

**Root cause (proven, 3/3 runs).** `trt-internal-5` (246,272 B, one of
the two `create_execution_context`-phase TRT private buffers, 2.64 % of
R = 9,336,232 B) holds device-resident per-kernel execution parameters.
A flip at an address-bearing offset (observed fatal: 19 / 93,034 /
~121.6 K) corrupts a kernel argument → CUDA illegal memory access on
the FIRST injected inference → runner process death. Benign offsets
(observed: 100.2 K / 223 K / 228 K — scalar arguments) complete and
restore normally (seed 8 trials 6/16/37 each took 2 internal-5 hits and
survived). Every flip in the weights (`trt-internal-0`), scratch
(`trt-internal-7`), deserialize constants (`trt-internal-1/2/4`), and
the input binding survived and restored across all runs. Aggregate
internal-5 hit rate matched the event-based prediction exactly (0.2
bits/trial; 10.9 % of trials) — the sampler is behaving, the buffer is
simply process-fatal to hit. ResNet-50's counterpart buffer was
22,528 B (0.08 % of its R) and never fatal over 900 trials / ~230 K
sites, which is why the problem only surfaced now.

**Decision (user, 2026-09-27).** Shrink the injection surface for THIS
workload only: exclude the two ctx-phase TRT private control-state
allocations (`trt-internal-5` 246,272 B + `trt-internal-6` 2,048 B =
248,320 B = 2.66 % of R). They are runtime control state, not model
data; corrupting them measures a process-fatal reliability event, not
an output-observable fault, and at L4+ the full surface makes
P(fatal per trial) → 1 (no campaign possible). ResNet-50 stays as run
(no re-run needed); the remaining four extension models keep
FULL-surface injection until they individually show the same failure
(一个一个来 — no blanket policy). The full-surface protocol remains
the control experiment for the paper appendix: the shrink rationale
(this section) plus the three diagnostic run dirs
(`artifacts/g7/campaign/run_L1_gpu0_*`, 2026-09-27) are the evidence.

**Mechanics.** The scoping lives in the frozen workload entry:
`surface_excludes: ("trt-internal-5", "trt-internal-6")` +
`resident_bytes_nominal` = the INJECTION SURFACE R = 9,087,912 B
(full 9,336,232 − 248,320). `fault_model.surface_rows_for()` is the
single filter point (sampling in `sample_campaign`, the residency
guard and work_detail/summary echo in `run_g5_campaign.py` all go
through it; bootstrap records keep the FULL measured R).
`freeze_g7v2_workload.py` derives R_eff from the bootstrap's
`g1_5_allocations.csv` (fail-closed: sizes must reconcile with the
measured R) and refuses any silent re-freeze — this re-freeze went
through an explicit placeholder reset.

**Ladder shift** (BERs unchanged; bits = BER × R_eff × 8):
L1 7→7, L2 37→36, L3 75→73, L4 224→218, L5 373→364, L6 523→509,
L7 747→727. Final surface: weights `trt-internal-0` 6,414,240 B
(70.6 %) + deserialize constants `trt-internal-1/2/3/4` 246,272 B
(2.7 %) + scratch `trt-internal-7` 1,825,280 B (20.1 %) + input
binding 602,112 B (6.6 %) + prob/index 8 B.

### EfficientNet-B0 (2026-09-28): same failure class, 400x lower density -> full surface + restart protocol

Full-surface protocol ran as decided: **L1 completed** (100 trials,
VERIFIED, 14 bits/trial), **L2 died at trial 83** on its first injected
inference (`CUDA illegal memory access`; no summary.json — harness.log
is ground truth). The dying trial's 69 flips: 43x `trt-internal-4` +
22x `trt-internal-0` + 3x input binding + 1x `trt-internal-2`; trials
0-82 and all of L1 contributed 4,180 benign `internal-4` hits and
2,747 benign weight hits.

This engine's `create_execution_context`-phase private pool is
`trt-internal-2` (18,944 B) + `trt-internal-3` (2,048 B) +
**`trt-internal-4` (9,923,072 B)** — the counterpart of MobileNetV3's
246 KB `internal-5` is 9.9 MB here, **58.0 % of R**. Weights,
deserialize constants, and input bindings are data and structurally
cannot produce an IMA (2,747 / 4 / 217 benign hits respectively), so
the fatal flip is in the ctx-phase pool. Calibrated fatal density:
one death per ~4,228 exposed pool bits (~2.4e-4; L1 observed 0/100 vs
predicted 0.2, L2 observed 1/84 = 1.19 % vs predicted 0.94 %).

**Crash-rate evaluation vs MobileNetV3 (user decision 2026-09-28).**
MobileNetV3's pool density is ~1 fatal bit per handful (L4+ at 40-100 %
of trials) AND the pool is only 2.66 % of R — scoping there removed a
prohibitive hazard at negligible cost. EfficientNet-B0 is the opposite
regime: pool share huge, density ~400x lower. Estimated full-surface
per-trial crash rates: L1 0.19 %, L2 0.94 %, L3 1.9 %, L4 5.5 %,
L5 9.0 %, L6 12.3 %, L7 17.1 % (~18 process relaunches at L7). A
58 % exclusion (briefly frozen in commit b8bf6e5, reverted the same
day) would leave an injection surface that is 91 % weights by
construction — it changes the experiment's meaning, not just its
risk. Chosen protocol: **FULL surface + campaign restart** (runner
flushes results per trial; a process-fatal trial is recorded as a
PROCESS_FATAL outcome like DUE — excluded from the accuracy mean,
counted in its own column; the orchestrator relaunches the remaining
trials in a fresh gated process, which is statistically equivalent
because trials are independent spatial draws). The crash rate becomes
a measured reliability curve vs BER at zero discarded bits; the L1 run
of the reverted scoping is the full-surface data already in the pool.

Paper note: the ctx-pool share of residency varies wildly across
engines (ResNet-50 0.08 %, MobileNetV3 2.6 %, EfficientNet-B0 58 %) —
TensorRT sizes its device-side control state by kernel count and graph
complexity. That share, combined with the pool's fatal density,
decides the protocol: scope when the pool is small AND deadly
(mobilenet), restart when it is large AND sparse (efficientnet).

**Implementation (2026-09-28).** The restart protocol is landed and
self-tested. Runner (`apps/resnet50_int8_g1_5.cpp`): the three result
CSVs are streamed through `G5ResultStreams` and flushed after every
completed trial, so a trial is on disk iff it reached TRIAL_END.
Orchestrator (`tools/g5_faultinj/run_g5_campaign.py`): one BER level =
a segment loop (`execute_segment`); a mid-trial death is classified by
`analyze_process_death` (recoverable ONLY as the IMA signature + every
prior trial TRIAL_END + exactly one open trial whose complete flip set
is the event-stream tail), the dead segment's completed prefix is
fully re-verified, the dying trial becomes PROCESS_FATAL on its global
execution-order slot, and a fresh gated segment relaunches for the
remaining slots (seed `--seed + k`; a crashed segment's never-run work
tail is dropped at merge — those slots were re-sampled).
`merge_campaign_outputs` renumbers local trials to global slots
(single clean segment = byte-identical hardlink). `summary.json`
carries `process_fatal_trials`/`process_fatal_count`/`segments`;
`analyze_campaign.py` prints the per-run `pf` column and the pooled
P(trial crash) reliability column. Every other death shape fails the
level closed (exit 2).

**Host-image cache (same day).** Each restart segment's prelude
re-preprocessed the 10K ImageNet pass on the host (~3 min CPU), which
dominated the per-restart cost (~4.5 min). The runner now caches the
preprocessed CHW float32 buffer (5.6 GiB) at
`<output-root>/../image_cache/g5img_<key>.bin`, keyed by
SHA-256(sample CSV content + preprocessing spec + image count + tensor
size + the runner binary itself); the key, header fields, and blob
checksum are all verified on load, and ANY miss/mismatch/I-O error
falls back to fresh preprocessing and rewrites the file (pure
optimization, zero effect on the gated window or trial semantics).
The key includes the FULL preprocessing spec, and the five extension
models do NOT share one — three distinct specs exist (verified against
model_meta.json): mobilenet/efficientnet 256 + ImageNet mean/std,
deit/swin 248 + ImageNet, vit 248 + 0.5-mean (ResNet-50's 235 is a
fourth, already complete). Each spec gets its own ~5.6 GiB file, ~17
GiB total. SHA-256 is a self-contained FIPS implementation verified
against the standard known-answer vectors ("", "abc", 1M×'a'). A
relaunch now costs ~1.5 min (observer + snapshot + clean pass) instead
of ~4.5.
