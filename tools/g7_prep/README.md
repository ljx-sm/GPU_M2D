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
   quantization. → `model_qdq.onnx`.
5. `dump_g7_calib_npy.py` — feeds the SAME 1000-image calibration set
   through the SAME `preprocess_image()` into
   `calib_canonical.npy` (1000,3,224,224) fp32 for ModelOpt.
6. `build_g7_engines.sh 0 [models...] --explicit` — engine build.
   Explicit path (canonical): `build_g7_explicit_engine.py` parses the
   Q/DQ ONNX with EXPLICIT_BATCH|STRONGLY_TYPED (no calibrator, no
   kINT8 flag — the Q/DQ nodes dictate precision), Q/DQ census +
   per-channel weight verification written into the summary.
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
