#!/usr/bin/env bash
# GPU_M2D G7: clean FP32+INT8 evaluation wiring (same env as
# build_g7_engines.sh).  usage: eval_g7_clean.sh PHYSICAL_GPU [MODEL ...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 PHYSICAL_GPU [MODEL ...]" >&2
    exit 2
fi
physical_gpu=$1
shift || true
case "${physical_gpu}" in
    0|1|2) ;;
    *) echo "physical GPU must be 0, 1, or 2" >&2; exit 2 ;;
esac

root=/data1/luojx/GPU_M2D
python=/data1/luojx/miniforge3/envs/vit_fault/bin/python
trt_lib=/data1/luojx/REMU/.local/deps/tensorrt-8.6.1/tensorrt_libs
cudnn_lib=/data1/luojx/REMU/.local/deps/cudnn-8.9.7.29/nvidia/cudnn/lib

[[ -x "${python}" ]] || { echo "missing Python: ${python}" >&2; exit 3; }
[[ -f "${trt_lib}/libnvinfer.so.8" ]] || { echo "missing TensorRT 8.6.1 libraries" >&2; exit 3; }

export CUDA_VISIBLE_DEVICES=${physical_gpu}
export LD_LIBRARY_PATH=${trt_lib}:${cudnn_lib}:/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "physical_gpu=${physical_gpu}"
nvidia-smi --id="${physical_gpu}" --query-gpu=index,uuid,name,memory.free,memory.total --format=csv,noheader,nounits

if [[ $# -ge 1 ]]; then
    exec "${python}" "${root}/tools/g7_prep/eval_g7_clean.py" \
        --device "${physical_gpu}" --models "$@"
else
    exec "${python}" "${root}/tools/g7_prep/eval_g7_clean.py" \
        --device "${physical_gpu}"
fi
