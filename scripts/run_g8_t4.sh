#!/usr/bin/env bash
# G8-T4 driver: L2 cache-fault campaigns (frozen --cache-faults,
# BER_cache = rho x BER, rho = 1) for ONE G7-v2 workload on GPU 0, at its
# seven BER levels 1e-7 ... 1e-5, 100 trials x 10K images each -- the same
# engine, 10K split, preprocessing and seed as the G7-v2 DRAM-only runs.
# Two modes complete the three-way comparison with the G7-v2 DRAM-only runs:
#   --mode dram_sram (default)  DRAM + L2  -> artifacts/g8/t4/campaign/
#   --mode sram_only            L2 only    -> artifacts/g8/t4/sram_only/campaign/
#
# Usage (from the repo root, in tmux; needs passwordless sudo like the
# other campaign entry points):
#   scripts/run_g8_t4.sh resnet50 [--mode dram_sram|sram_only]
#                        [--levels "L3 L4"] [--trials 100] [--dry-run]
#
# --dry-run prints the resolved levels and campaign commands only.
#
# Resumable: a level that already has a G5_CAMPAIGN_VERIFIED frozen-cache
# run with all trials under the output root is skipped. Stops at the first
# level that does not verify, so it can be inspected before going on.
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
while (($#)); do
    case "$1" in
        --levels) LEVELS="$2"; shift 2 ;;
        --trials) TRIALS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift 1 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
WORKLOAD="g7v2_imagenet1k_${MODEL#g7v2_imagenet1k_}"

SEED=7
DEVICE=0
SAMPLE_CSV=/data1/luojx/datasets/imagenet1k/splits/g7_eval_10000_perclass10.csv
case "$MODE" in
    dram_sram) MODE_ROOT="$PROJECT/artifacts/g8/t4"; MODE_ARGS=(--cache-faults) ;;
    sram_only) MODE_ROOT="$PROJECT/artifacts/g8/t4/sram_only"
               MODE_ARGS=(--cache-faults --no-dram-faults) ;;
    *) echo "unknown --mode $MODE (dram_sram|sram_only)" >&2; exit 2 ;;
esac
OUTPUT_ROOT="$MODE_ROOT/campaign"
IMAGE_CACHE="$PROJECT/artifacts/g8/image_cache"
LOG_DIR="$MODE_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

# Engine + preprocessing from the workload's newest G7 bootstrap; default
# levels = every frozen level with 1e-7 <= BER <= 1e-5.
read -r -d '' PY <<'EOF'
import glob, json, sys
sys.path.insert(0, "tools/g5_faultinj")
import fault_model
workload, want = sys.argv[1], sys.argv[2].split()
boots = [json.load(open(p)) for p in sorted(glob.glob(
    "artifacts/g7/campaign/run_bootstrap_*/bootstrap.json"))]
boots = [b for b in boots if b["workload"] == workload]
if not boots:
    sys.exit(f"no G7 bootstrap for {workload}")
pt = list(boots[-1]["runner_passthrough"])
if "--image-cache-dir" in pt:
    i = pt.index("--image-cache-dir"); del pt[i:i + 2]
levels = [r["level"] for r in fault_model.WORKLOADS[workload]["levels"]
          if 1e-7 * (1 - 1e-9) <= r["ber"] <= 1e-5 * (1 + 1e-9)]
if want:
    bad = [lv for lv in want if lv not in levels]
    if bad:
        sys.exit(f"levels {bad} not in {workload}'s 1e-7..1e-5 set {levels}")
    levels = want
print(" ".join(levels))
print(" ".join(["--engine", boots[-1]["engine_path"]] + pt))
EOF
mapfile -t CFG < <(python3 -c "$PY" "$WORKLOAD" "$LEVELS") || exit 2
[[ ${#CFG[@]} -eq 2 ]] || { echo "config lookup failed for $WORKLOAD" >&2; exit 2; }
read -r -a RUN_LEVELS <<<"${CFG[0]}"
read -r -a PASSTHROUGH <<<"${CFG[1]}"

verified() {  # $1 = level -> exit 0 if a complete frozen-cache run exists
    python3 - "$OUTPUT_ROOT" "$WORKLOAD" "$1" "$TRIALS" "$MODE" <<'EOF'
import glob, json, sys
root, workload, level, trials, mode = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
for p in glob.glob(f"{root}/run_{level}_gpu*/summary.json"):
    s = json.load(open(p))
    if (s.get("workload") == workload and s.get("level") == level
            and s.get("status") == "G5_CAMPAIGN_VERIFIED"
            and s.get("g8_cache_ber_frozen") is True
            and s.get("fault_mode", "dram_sram") == mode
            and s.get("trials_completed") == trials):
        print(p.rsplit("/", 2)[-2]); sys.exit(0)
sys.exit(1)
EOF
}

echo "== G8-T4 [$MODE] $WORKLOAD levels: ${RUN_LEVELS[*]} trials=$TRIALS seed=$SEED $(date '+%F %T')"
for LEVEL in "${RUN_LEVELS[@]}"; do
    if RUN=$(verified "$LEVEL"); then
        echo "== $LEVEL already VERIFIED ($RUN), skipped"
        continue
    fi
    LOG="$LOG_DIR/${MODEL#g7v2_imagenet1k_}_${LEVEL}.log"
    CMD=(sudo -n scripts/run_g2_observer_probe.sh --api g5campaign --device "$DEVICE"
        --workload "$WORKLOAD" "${PASSTHROUGH[@]}"
        --level "$LEVEL" --trials "$TRIALS" --seed "$SEED"
        --sample-csv "$SAMPLE_CSV" --output-root "$OUTPUT_ROOT"
        --image-cache-dir "$IMAGE_CACHE" "${MODE_ARGS[@]}")
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
echo "== G8-T4 [$MODE] $WORKLOAD done $(date '+%F %T')"
