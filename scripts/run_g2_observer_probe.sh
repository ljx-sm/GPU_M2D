#!/usr/bin/env bash
# GPU_M2D G2 observer scratch probe across the visible GPUs.
#
# Runs the read-only eBPF PTE observer with one scratch allocation per
# GPU and per allocation API. Must be executed with sudo from the
# research account; the CUDA child is dropped back to the invoking user
# and no other GPU process is touched.
#
# Usage:
#   sudo scripts/run_g2_observer_probe.sh [--api device|vmm] [--size-mib N] [--hold-seconds N]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${HERE}/.." && pwd)"
OBSERVER="${PROJECT}/tools/g2_observer"

API="device"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api) API="$2"; shift 2 ;;
        --size-mib|--hold-seconds) EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$(id -un)" != "root" ]]; then
    echo "run through sudo from the research account" >&2
    exit 1
fi

make -C "${OBSERVER}" all check

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
FAILURES=0
for DEVICE in $(seq 0 $((GPU_COUNT - 1))); do
    if ! /usr/bin/python3 "${OBSERVER}/run_g2_scratch_probe.py" \
        --api "${API}" --device "${DEVICE}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
        FAILURES=$((FAILURES + 1))
    fi
done

if [[ "${FAILURES}" -gt 0 ]]; then
    echo "G2_OBSERVER_PROBE_FAILED devices=${FAILURES}" >&2
    exit 2
fi
echo "G2_OBSERVER_PROBE_PASS api=${API} devices=${GPU_COUNT}"
