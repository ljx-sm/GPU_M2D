#!/usr/bin/env bash
set -euo pipefail

echo "captured_at=$(date --iso-8601=seconds)"
echo "kernel=$(uname -srmo)"
echo "cmake=$(cmake --version | head -n 1)"
echo "compiler=$(g++ --version | head -n 1)"
echo "nvcc_begin"
nvcc --version
echo "nvcc_end"
echo "gpu_begin"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id,compute_cap --format=csv,noheader
echo "gpu_end"
