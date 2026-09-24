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
| resnet50 | none (default: quantize everything quantizable) | 80.61 | 78.50 | 2.11 |
| mobilenetv3_large_100 | `--op_types_to_quantize Conv Gemm MatMul` + HardSwish-output bypass | 75.64 | 75.15 | 0.49 |
| efficientnet_b0 | `--op_types_to_quantize Conv Gemm MatMul` + SiLU-output bypass | 77.96 | 77.32 | 0.64 |
| vit_base_patch16_224 | none (default) | 79.41 | 78.22 | 1.19 |
| deit_small_patch16_224 | none (default) | 80.26 | 78.75 | 1.51 |
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
