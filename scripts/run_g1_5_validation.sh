#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remu_root="${GPU_M2D_REMU_ROOT:-/data1/luojx/REMU}"
dataset_root="${GPU_M2D_DATASET_ROOT:-/data1/luojx/datasets/REMU_stage8}"
build_dir="${1:-${project_root}/build-g1.5}"
result_dir="${project_root}/artifacts/g1_5"

trt_header_root="${remu_root}/.local/deps/TensorRT-8.6.1"
trt_runtime_dir="${remu_root}/.local/deps/tensorrt-8.6.1/tensorrt_libs"
opencv_root="${remu_root}/.local/deps/conda"
engine="${remu_root}/artifacts/stage8/engines/paper_priority/resnet50_resisc45_int8_ptq.engine"
sample_csv="${dataset_root}/RESISC45/splits/original_repo_1000_eval.csv"

for required in \
    "${trt_header_root}/include/NvInfer.h" \
    "${trt_runtime_dir}/libnvinfer.so.8" \
    "${opencv_root}/lib/cmake/opencv4/OpenCVConfig.cmake" \
    "${engine}" \
    "${sample_csv}"; do
    if [[ ! -s "${required}" ]]; then
        echo "Missing required G1.5 asset: ${required}" >&2
        exit 2
    fi
done

mkdir -p "${result_dir}"

cmake -S "${project_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_M2D_BUILD_TENSORRT_RUNNER=ON \
    -DGPU_M2D_TENSORRT_ROOT="${trt_header_root}" \
    -DGPU_M2D_TENSORRT_LIBRARY_DIR="${trt_runtime_dir}" \
    -DOpenCV_DIR="${opencv_root}/lib/cmake/opencv4"
cmake --build "${build_dir}" --parallel

export LD_LIBRARY_PATH="${trt_runtime_dir}:${opencv_root}/lib:${LD_LIBRARY_PATH:-}"

device_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
for ((device = 0; device < device_count; ++device)); do
    "${build_dir}/gpu_m2d_resnet50_int8_g1_5" \
        --engine "${engine}" \
        --sample-csv "${sample_csv}" \
        --sample-index 0 \
        --device "${device}" \
        --element 0 \
        --bit 0 \
        --output-prefix "${result_dir}/gpu${device}"
done
