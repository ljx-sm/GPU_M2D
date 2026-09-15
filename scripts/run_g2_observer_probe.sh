#!/usr/bin/env bash
# GPU_M2D G2 observer scratch probe across the visible GPUs.
#
# Designed to be whitelisted for passwordless sudo (visudo) and to run
# as root only for the kprobe attachment:
#   - the CUDA harness is dropped back to the invoking user by the
#     orchestrator, and no other GPU process is touched;
#   - nothing is compiled as root: build the harness as the normal user
#     first (`make -C tools/g2_observer all check`);
#   - all run artifacts are chowned back to the invoking user.
#
# Usage:
#   sudo scripts/run_g2_observer_probe.sh [--api device|vmm] [--size-mib N] [--hold-seconds N]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${HERE}/.." && pwd)"
OBSERVER="${PROJECT}/tools/g2_observer"
OUTPUT_ROOT="${PROJECT}/artifacts/g2/observer"

API="device"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api) API="$2"; shift 2 ;;
        --size-mib|--hold-seconds) EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
if [[ "${API}" != "device" && "${API}" != "vmm" ]]; then
    echo "--api must be device or vmm" >&2
    exit 1
fi

if [[ "$(id -un)" != "root" ]]; then
    echo "run through sudo; the CUDA child is dropped back to the invoking user" >&2
    exit 1
fi
INVOKING_UID="${SUDO_UID:-$(id -u)}"
INVOKING_GID="${SUDO_GID:-$(id -g)}"

if [[ ! -x "${OBSERVER}/g2_scratch_harness" ]]; then
    echo "harness missing; build it as the normal user first:" \
         "make -C tools/g2_observer all check" >&2
    exit 1
fi
if [[ ! -f "${OBSERVER}/g2_scratch_harness" ]]; then
    echo "harness not executable: ${OBSERVER}/g2_scratch_harness" >&2
    exit 1
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
mkdir -p "${OUTPUT_ROOT}"

FAILURES=0
for DEVICE in $(seq 0 $((GPU_COUNT - 1))); do
    if ! /usr/bin/python3 "${OBSERVER}/run_g2_scratch_probe.py" \
        --api "${API}" --device "${DEVICE}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
        FAILURES=$((FAILURES + 1))
    fi
done

chown -R "${INVOKING_UID}:${INVOKING_GID}" "${OUTPUT_ROOT}"

if [[ "${FAILURES}" -gt 0 ]]; then
    echo "G2_OBSERVER_PROBE_FAILED api=${API} failed_devices=${FAILURES}" >&2
    exit 2
fi
echo "G2_OBSERVER_PROBE_PASS api=${API} devices=${GPU_COUNT}"
