#!/usr/bin/env bash
# G8 precision study (user, 2026-10-22): ResNet-50 and ViT-B as FP32 and
# FP16 TensorRT engines, DRAM + SRAM mode at rho = 10 on the seven G7-v2
# levels 1e-7 ... 1e-5, 100 trials x 10K images -- everything as in the
# INT8 runs except the engine precision. Engines:
# /data1/luojx/g7_models/<model>/<precision>.engine
# (tools/g7_prep/build_g7_engines.sh 1 --float fp32|fp16, clean top-1 in
# <model>/eval/<precision>_clean_summary.json).
#
# Per workload g8p_imagenet1k_<model>_<precision>, each step resumable:
#   1. bootstrap: measure R of the engine on GPU 0 (skipped if a bootstrap
#      of the current engine exists) -> artifacts/g8/precision/bootstrap/
#   2. freeze R and the seven levels into fault_model.py
#      (freeze_g7v2_workload.py; idempotent)
#   3. DRAM-only evidence run, L7, 100 trials ->
#      artifacts/g8/precision/evidence/campaign/ -- the restore checks
#      classify every TRT-internal allocation read-only / engine-written,
#      which the cache flips' lifetimes need (G8-T2c)
#   4. insert cache_alloc_classes into fault_model.py (idempotent)
#   5. DRAM + SRAM, L1..L7 (scripts/run_g8_t4.sh --precision, rho 10)
#      -> artifacts/g8/rho10/precision/campaign/
# Steps 2 and 4 rewrite tools/g5_faultinj/fault_model.py; commit it after
# the run.
#
# Usage (repo root, in tmux; passwordless sudo like the other campaigns):
#   scripts/run_g8_precision.sh all               # the four, in order
#   scripts/run_g8_precision.sh resnet50 fp16     # one workload
# Order of "all": resnet50 fp16, resnet50 fp32, vit fp16, vit fp32.
# Stops at the first failure; rerunning resumes.
set -u -o pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

if [[ "${1:-}" == all ]]; then
    for pair in "resnet50 fp16" "resnet50 fp32" \
                "vit_base_patch16_224 fp16" "vit_base_patch16_224 fp32"; do
        # shellcheck disable=SC2086
        "$0" $pair || exit 1
    done
    echo "== G8 precision study: all four workloads done $(date '+%F %T')"
    exit 0
fi

MODEL="${1:?usage: $0 all | <resnet50|vit_base_patch16_224> <fp32|fp16>}"
PRECISION="${2:?usage: $0 all | <resnet50|vit_base_patch16_224> <fp32|fp16>}"
case "$MODEL" in resnet50|vit_base_patch16_224) ;;
    *) echo "model must be resnet50 or vit_base_patch16_224" >&2; exit 2 ;; esac
case "$PRECISION" in fp32|fp16) ;;
    *) echo "precision must be fp32 or fp16" >&2; exit 2 ;; esac

WORKLOAD="g8p_imagenet1k_${MODEL}_${PRECISION}"
ENGINE="/data1/luojx/g7_models/${MODEL}/${PRECISION}.engine"
BOOT_ROOT="$PROJECT/artifacts/g8/precision/bootstrap"
EVIDENCE_ROOT="$PROJECT/artifacts/g8/precision/evidence/campaign"
SAMPLE_CSV=/data1/luojx/datasets/imagenet1k/splits/g7_eval_10000_perclass10.csv
IMAGE_CACHE="$PROJECT/artifacts/g8/image_cache"
FREEZE=(/usr/bin/python3 tools/g7_prep/freeze_g7v2_workload.py --workload "$WORKLOAD")
DRIVER_LOG="$PROJECT/artifacts/g8/precision/drivers/${MODEL}_${PRECISION}.log"
mkdir -p "$BOOT_ROOT" "$(dirname "$DRIVER_LOG")"
exec > >(tee -a "$DRIVER_LOG") 2>&1

[[ -f "$ENGINE" ]] || { echo "missing engine $ENGINE" >&2; exit 2; }
echo "== G8 precision $WORKLOAD start $(date '+%F %T')"

# 1. bootstrap (skip if one exists for the CURRENT engine)
read -r -d '' PY <<'EOF'
import glob, hashlib, json, sys
workload, engine, model, boot_root = sys.argv[1:5]
sha = hashlib.sha256(open(engine, "rb").read()).hexdigest()
for p in sorted(glob.glob(f"{boot_root}/run_bootstrap_*/bootstrap.json")):
    b = json.load(open(p))
    if b.get("workload") == workload and b.get("engine_sha256") == sha:
        print("HAVE", p); sys.exit(0)
# runner passthrough (preprocessing) = the INT8 workload's G7 bootstrap
boots = [json.load(open(p)) for p in sorted(glob.glob(
    "artifacts/g7/campaign/run_bootstrap_*/bootstrap.json"))]
boots = [b for b in boots if b["workload"] == f"g7v2_imagenet1k_{model}"]
if not boots:
    sys.exit(f"no G7 bootstrap of g7v2_imagenet1k_{model}")
pt = list(boots[-1]["runner_passthrough"])
if "--image-cache-dir" in pt:
    i = pt.index("--image-cache-dir"); del pt[i:i + 2]
print("NEED", " ".join(pt))
EOF
BOOT_STATE=$(python3 -c "$PY" "$WORKLOAD" "$ENGINE" "$MODEL" "$BOOT_ROOT") || exit 2
if [[ "$BOOT_STATE" == HAVE* ]]; then
    echo "== bootstrap present: ${BOOT_STATE#HAVE }"
else
    read -r -a PASSTHROUGH <<<"${BOOT_STATE#NEED }"
    LOG="$PROJECT/artifacts/g8/precision/bootstrap_${MODEL}_${PRECISION}.log"
    echo "== bootstrap start $(date '+%F %T') log: $LOG"
    sudo -n scripts/run_g2_observer_probe.sh --api g5campaign --device 0 \
        --bootstrap --workload "$WORKLOAD" --engine "$ENGINE" "${PASSTHROUGH[@]}" \
        --sample-csv "$SAMPLE_CSV" --output-root "$BOOT_ROOT" \
        --image-cache-dir "$IMAGE_CACHE" --timeout-seconds 7200 >"$LOG" 2>&1
    RC=$?
    grep -h 'bootstrap:\|status=\|ERROR' "$LOG" | sed 's/^/   /'
    ((RC == 0)) || { echo "== bootstrap FAILED (exit $RC); see $LOG" >&2; exit 1; }
fi

# 2. freeze R + levels (idempotent)
"${FREEZE[@]}" || { echo "== freeze FAILED" >&2; exit 1; }

# 3. DRAM-only evidence run for the cache lifetime classes
scripts/run_g8_t4.sh "$MODEL" --precision "$PRECISION" --rho 10 \
    --mode dram_only --levels L7 || exit 1

# 4. cache_alloc_classes (idempotent)
"${FREEZE[@]}" --alloc-classes-root "$EVIDENCE_ROOT" \
    || { echo "== cache_alloc_classes FAILED" >&2; exit 1; }

# 5. DRAM + SRAM, seven levels
scripts/run_g8_t4.sh "$MODEL" --precision "$PRECISION" --rho 10 \
    --mode dram_sram || exit 1
echo "== G8 precision $WORKLOAD done $(date '+%F %T')"
