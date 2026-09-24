#!/usr/bin/env bash
# GPU_M2D G7: explicit Q/DQ quantization of the six ONNX models via
# NVIDIA ModelOpt ONNX PTQ (runs in the ISOLATED modelopt venv -- never in
# vit_fault: modelopt 0.47 would drag torch>=2.8/CUDA13 into it).
#
# Per-model recipes (see README "Final per-model recipes"; all share entropy
# calibration on calib_canonical.npy, per-channel INT8 weights, opset 17,
# --high_precision_dtype fp32):
#   resnet50 / vit / deit      default (quantize everything quantizable)
#   mobilenetv3 / efficientnet --op_types_to_quantize Conv Gemm MatMul
#                              + swish-output activation bypass (post-step)
#   swin                       --op_types_to_quantize Conv Gemm MatMul
#                              --disable_mha_qdq (TRT 8.6.1 mis-executes Q/DQ
#                              around the window-attention machinery)
# After quantization the wrapper runs, for every model:
#   bypass_g7_qdq_activations.py  (registered models only)
#   fix_qdq_for_trt86.py          (TRT 8.6.1 parser normalizations)
#
# Inputs (per model):  model.onnx + calib_canonical.npy (dump_g7_calib_npy)
# Output:              model_qdq.onnx (final, engine-ready)
#
# Usage: quantize_g7_qdq.sh <model-name> [model-name ...]
#        quantize_g7_qdq.sh all
set -euo pipefail

MODELOPT_PY=/data1/luojx/REMU/.local/deps/modelopt-venv/bin/python
TOOLS=/data1/luojx/GPU_M2D/tools/g7_prep
WEIGHTS_ROOT=/data1/luojx/g7_models
ALL_MODELS="resnet50 mobilenetv3_large_100 efficientnet_b0 vit_base_patch16_224 deit_small_patch16_224 swin_tiny_patch4_window7_224"

if [ "${1:-}" = "all" ]; then
    set -- $ALL_MODELS
fi

recipe_extra() {
    case "$1" in
        mobilenetv3_large_100|efficientnet_b0)
            echo "--op_types_to_quantize Conv Gemm MatMul" ;;
        swin_tiny_patch4_window7_224)
            echo "--op_types_to_quantize Conv Gemm MatMul --disable_mha_qdq" ;;
        *) echo "" ;;
    esac
}

for model in "$@"; do
    echo "=== modelopt qdq: ${model} ==="
    # shellcheck disable=SC2086
    "${MODELOPT_PY}" -m modelopt.onnx.quantization \
        --onnx_path "${WEIGHTS_ROOT}/${model}/model.onnx" \
        --output_path "${WEIGHTS_ROOT}/${model}/model_qdq.onnx" \
        --calibration_data_path "${WEIGHTS_ROOT}/${model}/calib_canonical.npy" \
        --calibration_method entropy \
        --quantize_mode int8 \
        --opset 17 \
        --high_precision_dtype fp32 \
        --keep_intermediate_files \
        --log_level INFO \
        $(recipe_extra "${model}")
    test -s "${WEIGHTS_ROOT}/${model}/model_qdq.onnx"
    "${MODELOPT_PY}" "${TOOLS}/bypass_g7_qdq_activations.py" "${model}"
    "${MODELOPT_PY}" "${TOOLS}/fix_qdq_for_trt86.py" "${model}"
    echo "=== modelopt qdq DONE: ${model} ==="
done
