#!/usr/bin/env bash
# GPU_M2D G7: explicit Q/DQ quantization of the six ONNX models via
# NVIDIA ModelOpt ONNX PTQ (runs in the ISOLATED modelopt venv -- never in
# vit_fault: modelopt 0.47 would drag torch>=2.8/CUDA13 into it).
#
# Inputs (per model):  model.onnx + calib_canonical.npy (dump_g7_calib_npy)
# Output:              model_qdq.onnx (Q/DQ, per-channel weight quant)
#
# Usage: quantize_g7_qdq.sh <model-name> [model-name ...]
#        quantize_g7_qdq.sh all
set -euo pipefail

MODELOPT_PY=/data1/luojx/REMU/.local/deps/modelopt-venv/bin/python
WEIGHTS_ROOT=/data1/luojx/g7_models
ALL_MODELS="resnet50 mobilenetv3_large_100 efficientnet_b0 vit_base_patch16_224 deit_small_patch16_224 swin_tiny_patch4_window7_224"

if [ "${1:-}" = "all" ]; then
    set -- $ALL_MODELS
fi

for model in "$@"; do
    echo "=== modelopt qdq: ${model} ==="
    "${MODELOPT_PY}" -m modelopt.onnx.quantization.quantize \
        --onnx_path "${WEIGHTS_ROOT}/${model}/model.onnx" \
        --output_path "${WEIGHTS_ROOT}/${model}/model_qdq.onnx" \
        --calibration_data "${WEIGHTS_ROOT}/${model}/calib_canonical.npy" \
        --calibration_method entropy \
        --quantize_mode int8 \
        --keep_intermediate_files \
        --verbose
    test -s "${WEIGHTS_ROOT}/${model}/model_qdq.onnx"
    echo "=== modelopt qdq DONE: ${model} ==="
done
