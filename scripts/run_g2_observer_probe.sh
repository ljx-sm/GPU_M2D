#!/usr/bin/env bash
# GPU_M2D G2 observer probes across the visible GPUs.
#
# Designed to be whitelisted for passwordless sudo (visudo) and to run
# as root only for the kprobe attachment:
#   - the CUDA harness / TensorRT runner child is dropped back to the
#     invoking user by the orchestrator, and no other GPU process is
#     touched;
#   - nothing is compiled as root: build the harnesses as the normal user
#     first (`make -C tools/g2_observer all check` and the G1.5 runner via
#     scripts/run_g1_5_validation.sh or cmake);
#   - all run artifacts are chowned back to the invoking user.
#
# Usage:
#   sudo scripts/run_g2_observer_probe.sh --api device   # cudaMalloc scratch
#   sudo scripts/run_g2_observer_probe.sh --api vmm      # CUDA VMM scratch
#   sudo scripts/run_g2_observer_probe.sh --api alias    # VMM alias double-mapping
#   sudo scripts/run_g2_observer_probe.sh --api tensorrt # full G1.5 workload
#
# Common options:
#   [--size-mib N] [--hold-seconds N]
# TensorRT-only options:
#   [--runner PATH]   prebuilt gpu_m2d_resnet50_int8_g1_5 binary
#                     (default: <project>/build-g1.5/gpu_m2d_resnet50_int8_g1_5)
#
# The tensorrt mode also regenerates artifacts/g2/gpu_va_pa_map.csv from the
# newest passing run of each GPU; mappings are never reused across runs.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${HERE}/.." && pwd)"
OBSERVER="${PROJECT}/tools/g2_observer"
OUTPUT_ROOT="${PROJECT}/artifacts/g2/observer"

API="device"
RUNNER="${PROJECT}/build-g1.5/gpu_m2d_resnet50_int8_g1_5"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api) API="$2"; shift 2 ;;
        --size-mib|--hold-seconds) EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        --runner) RUNNER="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
case "${API}" in
    device|vmm|alias|tensorrt) ;;
    *) echo "--api must be device, vmm, alias, or tensorrt" >&2; exit 1 ;;
esac

if [[ "$(id -un)" != "root" ]]; then
    echo "run through sudo; the CUDA child is dropped back to the invoking user" >&2
    exit 1
fi
INVOKING_UID="${SUDO_UID:-$(id -u)}"
INVOKING_GID="${SUDO_GID:-$(id -g)}"

if [[ "${API}" != "tensorrt" ]]; then
    HARNESS="g2_scratch_harness"
    [[ "${API}" == "alias" ]] && HARNESS="g2_alias_harness"
    if [[ ! -x "${OBSERVER}/${HARNESS}" ]]; then
        echo "harness missing; build it as the normal user first:" \
             "make -C tools/g2_observer all check" >&2
        exit 1
    fi
else
    if [[ ! -x "${RUNNER}" ]]; then
        echo "G1.5 runner missing; build it as the normal user first:" \
             "scripts/run_g1_5_validation.sh (or cmake with" \
             "-DGPU_M2D_BUILD_TENSORRT_RUNNER=ON)" >&2
        exit 1
    fi
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
mkdir -p "${OUTPUT_ROOT}"

FAILURES=0
for DEVICE in $(seq 0 $((GPU_COUNT - 1))); do
    case "${API}" in
        device|vmm)
            if ! /usr/bin/python3 "${OBSERVER}/run_g2_scratch_probe.py" \
                --api "${API}" --device "${DEVICE}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
                FAILURES=$((FAILURES + 1))
            fi
            ;;
        alias)
            if ! /usr/bin/python3 "${OBSERVER}/run_g2_alias_probe.py" \
                --device "${DEVICE}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
                FAILURES=$((FAILURES + 1))
            fi
            ;;
        tensorrt)
            if ! /usr/bin/python3 "${OBSERVER}/run_g2_tensorrt_probe.py" \
                --runner "${RUNNER}" --device "${DEVICE}" \
                "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
                FAILURES=$((FAILURES + 1))
            fi
            ;;
    esac
done

if [[ "${API}" == "tensorrt" && "${FAILURES}" -eq 0 ]]; then
    /usr/bin/python3 "${OBSERVER}/aggregate_va_pa_map.py" \
        --output "${PROJECT}/artifacts/g2/gpu_va_pa_map.csv"
fi

chown -R "${INVOKING_UID}:${INVOKING_GID}" "${OUTPUT_ROOT}" \
    "${PROJECT}/artifacts/g2/gpu_va_pa_map.csv" 2>/dev/null || true

if [[ "${FAILURES}" -gt 0 ]]; then
    echo "G2_OBSERVER_PROBE_FAILED api=${API} failed_devices=${FAILURES}" >&2
    exit 2
fi
echo "G2_OBSERVER_PROBE_PASS api=${API} devices=${GPU_COUNT}"
