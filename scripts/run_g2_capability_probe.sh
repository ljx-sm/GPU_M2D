#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${project_root}/build-g2}"
result_dir="${project_root}/artifacts/g2/capability"

mkdir -p "${result_dir}"

cmake -S "${project_root}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel

device_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
for ((device = 0; device < device_count; ++device)); do
    "${build_dir}/gpu_m2d_g2_capability_probe" \
        --device "${device}" \
        --allocation-bytes 2097152 \
        | tee "${result_dir}/gpu${device}.txt"
done
