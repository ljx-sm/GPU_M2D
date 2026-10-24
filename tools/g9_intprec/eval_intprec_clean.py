#!/usr/bin/env python3
"""G9 integer-precision study: clean top-1 of a W4A8 / W16A8 engine
(build_intprec_engine.py) on the 10K eval split, same preprocessing and
images as eval_g7_clean.py, with the M2DPackedWeight plugin loaded.

Acceptance (fail-closed): the engine matches its build summary (sha256),
two passes are bitwise identical, every output is valid. Reported against
the torch FP32 reference (<name>/eval/fp32_predictions.csv) and the INT8
engine (<name>/eval/clean_summary.json).

Writes <name>/eval/<precision>_predictions_run1.csv and
<name>/eval/<precision>_clean_summary.json (new files only).
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools/g7_prep"))
from build_g7_int8_engine import (  # noqa: E402
    INPUT_SHAPE, INTERPOLATION_CV, CudaRuntime, atomic_text, sha256_file)
from eval_g7_clean import WEIGHTS_ROOT, evaluation_rows, int8_pass_once, prediction_csv  # noqa: E402

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

PLUGIN_LIB = PROJECT / "artifacts/g9_intprec/libm2d_packed_weight.so"


def evaluate(name: str, precision: str) -> dict:
    ctypes.CDLL(str(PLUGIN_LIB), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    model_dir = WEIGHTS_ROOT / name
    build = json.loads((model_dir / f"{precision}_engine_summary.json").read_text())
    engine_path = Path(build["engine_path"])
    if build.get("status") != "PASS" or sha256_file(engine_path) != build["engine_sha256"]:
        raise RuntimeError(f"{name}: {precision} engine identity mismatch")
    meta = json.loads((model_dir / "model_meta.json").read_text())
    mean = np.asarray(meta["mean"], np.float32).reshape(1, 1, 3)
    std = np.asarray(meta["std"], np.float32).reshape(1, 1, 3)
    interp = INTERPOLATION_CV[str(meta["interpolation"])]
    resize = int(INPUT_SHAPE[2] // float(meta["crop_pct"]))
    rows_in = evaluation_rows()
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    cuda = CudaRuntime()
    bindings = [cuda.malloc(3 * 224 * 224 * 4), cuda.malloc(4), cuda.malloc(4)]
    try:
        started = time.time()
        run1 = int8_pass_once(context, bindings, cuda, rows_in, mean, std, interp, resize)
        elapsed = time.time() - started
        run2 = int8_pass_once(context, bindings, cuda, rows_in, mean, std, interp, resize)
    finally:
        for pointer in bindings:
            cuda.free(pointer)
    if not all(a["prediction"] == b["prediction"] and a["probability"] == b["probability"]
               for a, b in zip(run1, run2, strict=True)):
        raise RuntimeError(f"{name}: {precision} repeated-evaluation acceptance failed")
    invalid = sum(not r["valid"] for r in run1)
    if invalid:
        raise RuntimeError(f"{name}: {precision} produced {invalid} invalid outputs")
    with (model_dir / "eval/fp32_predictions.csv").open(newline="") as handle:
        torch_rows = list(csv.DictReader(handle))
    int8 = json.loads((model_dir / "eval/clean_summary.json").read_text())
    top1 = sum(bool(r["correct"]) for r in run1) / len(run1)
    torch_top1 = sum(r["correct"] in ("1", "True", "true") for r in torch_rows) / len(torch_rows)
    agree = sum(int(t["prediction"]) == r["prediction"]
                for t, r in zip(torch_rows, run1, strict=True)) / len(run1)
    atomic_text(model_dir / f"eval/{precision}_predictions_run1.csv", prediction_csv(run1))
    summary = {"schema_version": 1, "status": "PASS", "model_key": name,
               "precision": precision, "engine_sha256": build["engine_sha256"],
               "images": len(run1), "top1": top1, "torch_fp32_top1": torch_top1,
               "int8_engine_top1": int8.get("int8_top1"),
               "prediction_agreement_vs_torch_fp32": agree,
               "repeated_pass_identical": True, "pass_elapsed_seconds": elapsed}
    atomic_text(model_dir / f"eval/{precision}_clean_summary.json",
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"{name}: {precision}_clean=PASS top1={100 * top1:.2f} torch_fp32={100 * torch_top1:.2f} "
          f"int8_engine={int8.get('int8_top1')} agreement={100 * agree:.2f}%", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--precision", required=True, choices=("w4a8", "w16a8"))
    args = parser.parse_args()
    for name in args.models:
        evaluate(name, args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
