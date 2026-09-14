#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${project_root}/build}"

cmake -S "${project_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure

device_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
for ((device = 0; device < device_count; ++device)); do
    "${build_dir}/test_cuda_injector" --device "${device}"
done
