#!/usr/bin/env python3
"""GPU_M2D G8 precision study: clean top-1 of an FP32 / FP16 TensorRT
engine (build_g7_float_engine.py) on the 10K eval split, with the SAME
preprocessing and images as eval_g7_clean.py.

Acceptance (fail-closed): the engine matches its build summary (sha256),
two passes are bitwise identical, every output is valid. Reported against
the torch FP32 reference (<name>/eval/fp32_predictions.csv, written by
eval_g7_clean.py): top-1 and prediction agreement.

Writes <name>/eval/<precision>_predictions_run1.csv and
<name>/eval/<precision>_clean_summary.json.

  tools/g7_prep/eval_g7_clean.sh 1 --float fp32 resnet50 vit_base_patch16_224
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from build_g7_int8_engine import INPUT_SHAPE, INTERPOLATION_CV, CudaRuntime, atomic_text, sha256_file
from build_g7_float_engine import DEFAULT_MODELS, PRECISIONS
from eval_g7_clean import (
    WEIGHTS_ROOT,
    evaluation_rows,
    int8_pass_once,
    prediction_csv,
)

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt


def evaluate_one(name: str, precision: str) -> dict[str, object]:
    model_dir = WEIGHTS_ROOT / name
    build_summary = json.loads(
        (model_dir / f"{precision}_engine_summary.json").read_text(encoding="utf-8")
    )
    engine_path = Path(build_summary["engine_path"])
    if (
        build_summary.get("status") != "PASS"
        or build_summary.get("model_key") != name
        or build_summary.get("precision") != precision
        or sha256_file(engine_path) != build_summary.get("engine_sha256")
    ):
        raise RuntimeError(f"{name}: {precision} engine identity/provenance mismatch")

    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    mean = np.asarray(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(meta["std"], dtype=np.float32).reshape(1, 1, 3)
    interpolation = INTERPOLATION_CV[str(meta["interpolation"])]
    resize_scale = int(INPUT_SHAPE[2] // float(meta["crop_pct"]))
    rows_in = evaluation_rows()

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None or engine.num_bindings != 3:
        raise RuntimeError(f"{name}: {precision} engine does not match the three-binding contract")
    context = engine.create_execution_context()
    cuda = CudaRuntime()
    bindings = [cuda.malloc(3 * 224 * 224 * 4), cuda.malloc(4), cuda.malloc(4)]
    try:
        started = time.time()
        run1 = int8_pass_once(context, bindings, cuda, rows_in, mean, std,
                              interpolation, resize_scale)
        elapsed1 = time.time() - started
        run2 = int8_pass_once(context, bindings, cuda, rows_in, mean, std,
                              interpolation, resize_scale)
    finally:
        for pointer in bindings:
            cuda.free(pointer)
    if not all(a["prediction"] == b["prediction"] and a["probability"] == b["probability"]
               for a, b in zip(run1, run2, strict=True)):
        raise RuntimeError(f"{name}: {precision} repeated-evaluation acceptance failed")
    invalid = sum(not row["valid"] for row in run1)
    if invalid:
        raise RuntimeError(f"{name}: {precision} pass produced {invalid} invalid outputs")

    with (model_dir / "eval/fp32_predictions.csv").open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    if len(reference) != len(run1):
        raise RuntimeError(f"{name}: torch FP32 reference has {len(reference)} rows")
    torch_top1 = sum(row["correct"] in ("1", "True", "true") for row in reference) / len(reference)
    top1 = sum(bool(row["correct"]) for row in run1) / len(run1)
    agreement = sum(int(ref["prediction"]) == row["prediction"]
                    for ref, row in zip(reference, run1, strict=True)) / len(run1)

    eval_dir = model_dir / "eval"
    atomic_text(eval_dir / f"{precision}_predictions_run1.csv", prediction_csv(run1))
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "model_key": name,
        "precision": precision,
        "engine_path": str(engine_path),
        "engine_sha256": build_summary["engine_sha256"],
        "images": len(run1),
        "top1": top1,
        "torch_fp32_top1": torch_top1,
        "prediction_agreement_vs_torch_fp32": agreement,
        "repeated_pass_identical": True,
        "pass_elapsed_seconds": elapsed1,
    }
    atomic_text(eval_dir / f"{precision}_clean_summary.json",
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"{name}: {precision}_clean=PASS top1={100 * top1:.2f} "
          f"torch_fp32={100 * torch_top1:.2f} agreement={100 * agreement:.2f}% "
          f"pass_seconds={elapsed1:.1f}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--precision", required=True, choices=PRECISIONS)
    parser.add_argument("--device", type=int, default=0, choices=(0, 1, 2))
    args = parser.parse_args()
    for name in args.models:
        evaluate_one(name, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
