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
   (mean/std/interpolation/input_size/…) into
   `/data1/luojx/g7_models/<timm-name>/`. The meta file is the only
   source of preprocessing facts; nothing else hardcodes per-model
   values.
3. `export_g7_onnx.py` — ONNX opset 17, fixed batch 1, three-binding
   contract `data (1,3,224,224) fp32 / prob (1,1) fp32 / index (1,1)
   int32` (the G5 runner contract), INT64_MAX slice-end sentinel
   rewrite, sha256 provenance. → `model.onnx` + `onnx_summary.json`.
4. `build_g7_engines.sh 0 [models...]` — INT8 PTQ engine build
   (IInt8EntropyCalibrator2, batch 1, 1000 calibration images, 4 GiB
   workspace, calibration cache keyed on
   onnx+calib+mean/std+interpolation). → `clean.engine` +
   `engine_summary.json` (+ zero-input smoke at build time).
   `--parse-only` validates the ONNX→TRT parse without building.
5. `eval_g7_clean.sh 0 [models...]` — FP32 (torch/timm) and INT8 (TRT)
   on the 10K eval split, identical preprocessing; INT8 acceptance =
   two full passes with byte-identical (prediction, probability).
   → `eval/{fp32,int8}_predictions*.csv`, `eval/clean_summary.json`
   (fp32_top1, int8_top1, quantization loss pp).

## Preprocessing contract (identical everywhere)

Square resize to 224 with the model's own interpolation (bicubic →
cv2.INTER_CUBIC, bilinear → cv2.INTER_LINEAR), BGR→RGB, /255, mean/std
from `model_meta.json`, CHW float32. Defined once in
`build_g7_int8_engine.preprocess_image` and imported by the eval
script; the future per-workload G5 runner parameterization must consume
`model_meta.json` the same way. Note: this is deliberately simpler than
timm's canonical Resize(256)+CenterCrop val pipeline — top-1 numbers
are comparable within G7 (same pixels for FP32 and INT8), not against
paper leaderboards.
