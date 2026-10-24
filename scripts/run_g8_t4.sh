#!/usr/bin/env bash
# G8-T4 driver: L2 cache-fault campaigns (frozen --cache-faults,
# BER_cache = rho x BER, rho = 1) for ONE G7-v2 workload on GPU 0, at its
# seven BER levels 1e-7 ... 1e-5, 100 trials x 10K images each -- the same
# engine, 10K split, preprocessing and seed as the G7-v2 DRAM-only runs.
# Two modes complete the three-way comparison with the G7-v2 DRAM-only runs:
#   --mode dram_sram (default)  DRAM + L2  -> artifacts/g8/t4/campaign/
#   --mode sram_only            L2 only    -> artifacts/g8/t4/sram_only/campaign/
#   --mode all                  dram_sram, then sram_only (stops on failure)
# --rho 300 (user, 2026-10-13): the same campaigns with the frozen cache
# rho = 300 -- DRAM levels unchanged, BER_cache = 300 x the level's BER --
# written to separate roots so the rho = 1 runs stay untouched:
#   dram_sram -> artifacts/g8/rho300/campaign/
#   sram_only -> artifacts/g8/rho300/sram_only/campaign/
# --rho 5000 (user, 2026-10-13, supersedes 300): the R1..R7 ladder
# (fault_model.R_LEVELS: DRAM BER 1e-10, 2e-10, 1e-9, 2e-9, 1e-8, 2e-8,
# 1e-7; cache BER 5000 x that = 5e-7 ... 5e-4) in all three modes, the
# G7-v2 DRAM-only runs no longer apply:
#   --mode dram_only -> artifacts/g8/rho5000/dram_only/campaign/
#   --mode dram_sram -> artifacts/g8/rho5000/campaign/
#   --mode sram_only -> artifacts/g8/rho5000/sram_only/campaign/
#   --mode all       -> dram_only, dram_sram, sram_only (stops on failure)
# --rho 10 (user, 2026-10-13, supersedes 300 and 5000): the G7-v2 DRAM
# levels 1e-7 ... 1e-5 unchanged (the G7-v2 DRAM-only runs stand), cache
# BER 10 x the level = 1e-6 ... 1e-4:
#   dram_sram -> artifacts/g8/rho10/campaign/
#   sram_only -> artifacts/g8/rho10/sram_only/campaign/
# --precision fp32|fp16 (user, 2026-10-22): the G8 precision study --
# workload g8p_imagenet1k_<model>_<precision> (ResNet-50, ViT-B; FP32 /
# FP16 engine, bootstrap under artifacts/g8/precision/bootstrap), rho 10
# only, the same seven levels and protocol as the INT8 runs:
#   --mode dram_sram -> artifacts/g8/rho10/precision/campaign/
#   --mode dram_only -> artifacts/g8/precision/evidence/campaign/ (the
#                       DRAM-only evidence run for cache_alloc_classes)
# Precision runs pass --timeout-seconds 43200 (an FP32 ViT-B level runs
# far past the 3600 s default inside one segment). Orchestrated end to end
# by scripts/run_g8_precision.sh.
# --precision w4a8|w16a8 (user, 2026-10-23): the G9 integer-precision study
# -- workload g9i_imagenet1k_<model>_<precision> (INT4 / INT16 weights
# packed by the M2DPackedWeight plugin), the same protocol; the runner is
# build-g9/gpu_m2d_resnet50_int8_g1_5 (opt-in --plugin-lib, carried by the
# bootstrap's runner passthrough), roots under artifacts/g9_intprec:
#   --mode dram_sram -> artifacts/g9_intprec/rho10/campaign/
#   --mode dram_only -> artifacts/g9_intprec/evidence/campaign/
# --surface native (user, 2026-10-24; with --precision w4a8|w16a8): the same
# engines on the NATIVE-EQUIVALENT injection surface -- workload
# g9n_imagenet1k_<model>_<precision> (FP32-compute-only regions excluded,
# freeze_g7v2_workload.py NATIVE_SURFACE), roots under
# artifacts/g9_intprec/native/{bootstrap,evidence,rho10}.
#
# Usage (from the repo root, in tmux; needs passwordless sudo like the
# other campaign entry points):
#   scripts/run_g8_t4.sh resnet50 [--mode dram_sram|sram_only|all]
#                        [--levels "L3 L4"] [--trials 100] [--rho 1|10|300|5000]
#                        [--precision fp32|fp16|w4a8|w16a8]
#                        [--surface native] [--dry-run]
#
# --dry-run prints the resolved levels and campaign commands only.
#
# Resumable: a level that already has a complete G5_CAMPAIGN_VERIFIED
# frozen-cache run under the output root is skipped -- complete = all
# requested trials either completed or PROCESS_FATAL (restart protocol).
# Stops (exit 1) at the first level that does not verify, so it can be
# inspected before going on. The driver logs itself (append) to
# artifacts/g8/t4/drivers/<model>.log, so no "| tee" is needed (a pipe
# would hide the exit status from a following "&&").
# Idle rule (user, 2026-10-04): every level runs with --require-idle-gpu --
# each segment waits for an idle GPU 0 and fails closed if its residency
# pass was not idle (tools/g8_cache/audit_idle.py audits finished runs).
# Per-level console logs: <mode root>/logs/<model>_<level>.log
set -u -o pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

MODEL="${1:?usage: $0 <model, e.g. resnet50> [--levels \"L3 L4\"] [--trials N]}"
shift
LEVELS=""
TRIALS=100
DRY_RUN=0
MODE=dram_sram
RHO=1
PRECISION=""
SURFACE=""
while (($#)); do
    case "$1" in
        --levels) LEVELS="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift 1 ;;
        --mode) MODE="$2"; shift 2 ;;
        --rho) RHO="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        --surface) SURFACE="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
WORKLOAD="g7v2_imagenet1k_${MODEL#g7v2_imagenet1k_}"
NAME="${MODEL#g7v2_imagenet1k_}"
BOOT_ROOT="artifacts/g7/campaign"
EXTRA_ARGS=()
if [[ -n "$PRECISION" ]]; then
    case "$PRECISION" in
        fp32|fp16|w4a8|w16a8) ;;
        *) echo "unknown --precision $PRECISION (fp32|fp16|w4a8|w16a8)" >&2; exit 2 ;;
    esac
    [[ "$RHO" == 10 ]] || { echo "--precision runs use --rho 10 only" >&2; exit 2; }
    case "$MODE" in
        dram_sram|dram_only) ;;
        *) echo "--precision supports --mode dram_sram|dram_only" >&2; exit 2 ;;
    esac
    WORKLOAD="g8p_imagenet1k_${NAME}_${PRECISION}"
    NAME="${NAME}_${PRECISION}"
    BOOT_ROOT="artifacts/g8/precision/bootstrap"
    EXTRA_ARGS=(--timeout-seconds 43200)
    if [[ "$PRECISION" == w4a8 || "$PRECISION" == w16a8 ]]; then
        WORKLOAD="g9i_${WORKLOAD#g8p_}"
        BOOT_ROOT="artifacts/g9_intprec/bootstrap"
        EXTRA_ARGS+=(--runner "$PROJECT/build-g9/gpu_m2d_resnet50_int8_g1_5")
        if [[ "$SURFACE" == native ]]; then
            WORKLOAD="g9n_${WORKLOAD#g9i_}"
            NAME="${NAME}_native"
            BOOT_ROOT="artifacts/g9_intprec/native/bootstrap"
        fi
    fi
fi
case "$SURFACE" in
    "") ;;
    native) [[ "$WORKLOAD" == g9n_* ]] || {
                echo "--surface native needs --precision w4a8|w16a8" >&2; exit 2; } ;;
    *) echo "unknown --surface $SURFACE (native)" >&2; exit 2 ;;
esac
case "$RHO" in
    1) BASE_ROOT="$PROJECT/artifacts/g8/t4"; RHO_ARGS=(); RHO_PY=1.0 ;;
    10) BASE_ROOT="$PROJECT/artifacts/g8/rho10"; RHO_ARGS=(--cache-rho 10)
        RHO_PY=10.0 ;;
    300) BASE_ROOT="$PROJECT/artifacts/g8/rho300"; RHO_ARGS=(--cache-rho 300)
         RHO_PY=300.0 ;;
    5000) BASE_ROOT="$PROJECT/artifacts/g8/rho5000"; RHO_ARGS=(--cache-rho 5000)
          RHO_PY=5000.0 ;;
    *) echo "unknown --rho $RHO (1|10|300|5000)" >&2; exit 2 ;;
esac
if [[ -n "$PRECISION" ]]; then
    if [[ "$MODE" == dram_only ]]; then
        BASE_ROOT="$PROJECT/artifacts/g8/precision/evidence"
    else
        BASE_ROOT="$PROJECT/artifacts/g8/rho10/precision"
    fi
    if [[ "$WORKLOAD" == g9i_* || "$WORKLOAD" == g9n_* ]]; then
        G9_ROOT="$PROJECT/artifacts/g9_intprec"
        [[ "$WORKLOAD" == g9n_* ]] && G9_ROOT="$G9_ROOT/native"
        if [[ "$MODE" == dram_only ]]; then
            BASE_ROOT="$G9_ROOT/evidence"
        else
            BASE_ROOT="$G9_ROOT/rho10"
        fi
    fi
fi

if ((!DRY_RUN)) && [[ -z "${G8_T4_LOGGING:-}" ]]; then
    export G8_T4_LOGGING=1
    DRIVER_LOG="$BASE_ROOT/drivers/${NAME}.log"
    mkdir -p "$(dirname "$DRIVER_LOG")"
    exec > >(tee -a "$DRIVER_LOG") 2>&1
fi
if [[ "$MODE" == all ]]; then
    FORWARD=(--trials "$TRIALS" --rho "$RHO")
    [[ -n "$LEVELS" ]] && FORWARD+=(--levels "$LEVELS")
    ((DRY_RUN)) && FORWARD+=(--dry-run)
    if [[ "$RHO" == 5000 ]]; then
        "$0" "$MODEL" --mode dram_only "${FORWARD[@]}" || exit 1
    fi
    "$0" "$MODEL" --mode dram_sram "${FORWARD[@]}" || exit 1
    "$0" "$MODEL" --mode sram_only "${FORWARD[@]}" || exit 1
    exit 0
fi

SEED=7
DEVICE=0
SAMPLE_CSV=/data1/luojx/datasets/imagenet1k/splits/g7_eval_10000_perclass10.csv
case "$MODE" in
    dram_only) [[ "$RHO" == 5000 || -n "$PRECISION" ]] || {
                   echo "--mode dram_only is for --rho 5000 or --precision (rho 1/10/300 DRAM-only = the G7-v2 runs)" >&2
                   exit 2; }
               MODE_ROOT="$BASE_ROOT/dram_only"
               [[ -n "$PRECISION" ]] && MODE_ROOT="$BASE_ROOT"
               MODE_ARGS=(--require-idle-gpu) ;;
    dram_sram) MODE_ROOT="$BASE_ROOT"
               MODE_ARGS=(--cache-faults "${RHO_ARGS[@]}" --require-idle-gpu) ;;
    sram_only) MODE_ROOT="$BASE_ROOT/sram_only"
               MODE_ARGS=(--cache-faults "${RHO_ARGS[@]}" --no-dram-faults
                          --require-idle-gpu) ;;
    *) echo "unknown --mode $MODE (dram_only|dram_sram|sram_only|all)" >&2; exit 2 ;;
esac
OUTPUT_ROOT="$MODE_ROOT/campaign"
IMAGE_CACHE="$PROJECT/artifacts/g8/image_cache"
LOG_DIR="$MODE_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

# Engine + preprocessing from the workload's newest G7 bootstrap; default
# levels = every frozen level with 1e-7 <= BER <= 1e-5 (rho 1, 300), or
# the R1..R7 ladder (rho 5000).
read -r -d '' PY <<'EOF'
import glob, json, sys
sys.path.insert(0, "tools/g5_faultinj")
import fault_model
workload, want, rho, boot_root = (sys.argv[1], sys.argv[2].split(),
                                  sys.argv[3], sys.argv[4])
boots = [json.load(open(p)) for p in sorted(glob.glob(
    f"{boot_root}/run_bootstrap_*/bootstrap.json"))]
boots = [b for b in boots if b["workload"] == workload]
if not boots:
    sys.exit(f"no bootstrap for {workload} under {boot_root}")
pt = list(boots[-1]["runner_passthrough"])
if "--image-cache-dir" in pt:
    i = pt.index("--image-cache-dir"); del pt[i:i + 2]
if rho == "5000":
    levels = [r["level"] for r in fault_model.R_LEVELS[workload]]
else:
    levels = [r["level"] for r in fault_model.WORKLOADS[workload]["levels"]
              if 1e-7 * (1 - 1e-9) <= r["ber"] <= 1e-5 * (1 + 1e-9)]
if want:
    bad = [lv for lv in want if lv not in levels]
    if bad:
        sys.exit(f"levels {bad} not in {workload}'s set {levels}")
    levels = want
print(" ".join(levels))
print(" ".join(["--engine", boots[-1]["engine_path"]] + pt))
EOF
mapfile -t CFG < <(python3 -c "$PY" "$WORKLOAD" "$LEVELS" "$RHO" "$BOOT_ROOT") || exit 2
[[ ${#CFG[@]} -eq 2 ]] || { echo "config lookup failed for $WORKLOAD" >&2; exit 2; }
read -r -a RUN_LEVELS <<<"${CFG[0]}"
read -r -a PASSTHROUGH <<<"${CFG[1]}"
if [[ "$WORKLOAD" == g9[in]_* && " ${PASSTHROUGH[*]} " != *" --plugin-lib "* ]]; then
    echo "$WORKLOAD: the bootstrap's runner passthrough has no --plugin-lib" >&2
    exit 2
fi

verified() {  # $1 = level -> exit 0 if a complete frozen-cache run exists
    python3 - "$OUTPUT_ROOT" "$WORKLOAD" "$1" "$TRIALS" "$MODE" "$RHO_PY" <<'EOF'
import glob, json, sys
root, workload, level, trials, mode = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
rho = float(sys.argv[6])
for p in glob.glob(f"{root}/run_{level}_gpu*/summary.json"):
    s = json.load(open(p))
    if (s.get("workload") == workload and s.get("level") == level
            and s.get("status") == "G5_CAMPAIGN_VERIFIED"
            and (mode == "dram_only" or (
                s.get("g8_cache_ber_frozen") is True
                and float(s.get("g8_cache_rho") or 0) == rho))
            and s.get("fault_mode", "dram_sram") == mode
            and s.get("trials_requested") == trials
            and s.get("trials_completed", 0)
            + s.get("process_fatal_count", 0) == trials):
        print(p.rsplit("/", 2)[-2]); sys.exit(0)
sys.exit(1)
EOF
}

echo "== G8-T4 [$MODE] rho=$RHO $WORKLOAD levels: ${RUN_LEVELS[*]} trials=$TRIALS seed=$SEED $(date '+%F %T')"
for LEVEL in "${RUN_LEVELS[@]}"; do
    if RUN=$(verified "$LEVEL"); then
        echo "== $LEVEL already VERIFIED ($RUN), skipped"
        continue
    fi
    LOG="$LOG_DIR/${NAME}_${LEVEL}.log"
    CMD=(sudo -n scripts/run_g2_observer_probe.sh --api g5campaign --device "$DEVICE"
        --workload "$WORKLOAD" "${PASSTHROUGH[@]}"
        --level "$LEVEL" --trials "$TRIALS" --seed "$SEED"
        --sample-csv "$SAMPLE_CSV" --output-root "$OUTPUT_ROOT"
        --image-cache-dir "$IMAGE_CACHE" "${MODE_ARGS[@]}"
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")
    if ((DRY_RUN)); then echo "== $LEVEL would run: ${CMD[*]} > $LOG"; continue; fi
    echo "== $LEVEL start $(date '+%F %T') (GPU $DEVICE other processes:" \
         "$(nvidia-smi -i "$DEVICE" --query-compute-apps=pid --format=csv,noheader | wc -l)) log: $LOG"
    "${CMD[@]}" >"$LOG" 2>&1
    RC=$?
    grep -h 'G8 residency map\|G8 cache sampler\|WARNING G8\|DISABLED\|PROCESS_FATAL\|status=' "$LOG" \
        | sed 's/^/   /'
    if RUN=$(verified "$LEVEL"); then
        echo "== $LEVEL VERIFIED ($RUN) $(date '+%F %T')"
    else
        echo "== $LEVEL did NOT verify (exit $RC) -- stopping; see $LOG" >&2
        exit 1
    fi
done
echo "== G8-T4 [$MODE] rho=$RHO $WORKLOAD done $(date '+%F %T')"
