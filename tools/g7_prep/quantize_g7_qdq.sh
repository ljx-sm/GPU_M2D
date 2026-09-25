#!/usr/bin/env bash
# GPU_M2D G7: explicit Q/DQ quantization of the six ONNX models via
# NVIDIA ModelOpt ONNX PTQ (runs in the ISOLATED modelopt venv -- never in
# vit_fault: modelopt 0.47 would drag torch>=2.8/CUDA13 into it).
#
# v2 (2026-09-24, user decision): the classifier-head Gemm is quantized
# like every other weighted op. v1 drove the modelopt CLI, whose
# default-on enable_gemv_detection_for_trt heuristic silently kept the
# batch-1 head FP32 (no CLI flag exists); quantize_g7_qdq_head.py calls
# the same quantizer through the python API with that heuristic OFF --
# the only delta. Per-model recipes (see README "Final per-model
# recipes"; all share entropy calibration on calib_canonical.npy,
# per-channel INT8 weights, opset 17, fp32 high-precision dtype):
#   resnet50 / vit / deit      default (quantize everything quantizable)
#   mobilenetv3 / efficientnet --op_types_to_quantize Conv Gemm MatMul
#                              + swish-output activation bypass (post-step)
#   swin                       --op_types_to_quantize Conv Gemm MatMul
#                              --disable_mha_qdq (TRT 8.6.1 mis-executes Q/DQ
#                              around the window-attention machinery)
# After quantization the wrapper runs, for every model:
#   bypass_g7_qdq_activations.py  (registered models only)
#   fix_qdq_for_trt86.py          (TRT 8.6.1 parser normalizations)
# The head-quantized check inside quantize_g7_qdq_head.py fails closed.
#
# Inputs (per model):  model.onnx + calib_canonical.npy (dump_g7_calib_npy)
# Output:              model_qdq.onnx (final, engine-ready)
#
# Usage: quantize_g7_qdq.sh <model-name> [model-name ...]
#        quantize_g7_qdq.sh all
set -euo pipefail

MODELOPT_PY=/data1/luojx/REMU/.local/deps/modelopt-venv/bin/python
TOOLS=/data1/luojx/GPU_M2D/tools/g7_prep
ALL_MODELS="resnet50 mobilenetv3_large_100 efficientnet_b0 vit_base_patch16_224 deit_small_patch16_224 swin_tiny_patch4_window7_224"

if [ "${1:-}" = "all" ]; then
    set -- $ALL_MODELS
fi

for model in "$@"; do
    "${MODELOPT_PY}" "${TOOLS}/quantize_g7_qdq_head.py" "${model}"
    "${MODELOPT_PY}" "${TOOLS}/bypass_g7_qdq_activations.py" "${model}"
    "${MODELOPT_PY}" "${TOOLS}/fix_qdq_for_trt86.py" "${model}"
    echo "=== modelopt qdq DONE: ${model} ==="
done
