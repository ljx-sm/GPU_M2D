#!/usr/bin/env python3
"""G7 diagnostic: evaluate an arbitrary engine file on the 10K split
with the SQUARE-RESIZE preprocessing (the G7 v1 policy), single pass,
reporting top-1 -- used to compare diagnostic engines (TRT-FP32,
MinMax-calibrated INT8) against the frozen baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_g7_int8_engine import CudaRuntime, INTERPOLATION_CV, preprocess_image
from eval_g7_clean import evaluation_rows, int8_pass_once

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

WEIGHTS_ROOT = Path("/data1/luojx/g7_models")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    model_key = None
    for meta_dir in WEIGHTS_ROOT.iterdir():
        if (meta_dir / "model_meta.json").is_file():
            model_key = meta_dir.name
    # model inferred from the engine path when it embeds a model name
    for candidate in WEIGHTS_ROOT.iterdir():
        if candidate.name in str(args.engine):
            model_key = candidate.name
    if model_key is None:
        raise RuntimeError("cannot infer model from engine path")

    import json
    meta = json.loads((WEIGHTS_ROOT / model_key / "model_meta.json").read_text())
    mean = np.asarray(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(meta["std"], dtype=np.float32).reshape(1, 1, 3)
    interpolation = INTERPOLATION_CV[str(meta["interpolation"])]

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("cannot deserialize engine")
    context = engine.create_execution_context()
    cuda = CudaRuntime()
    bindings = [cuda.malloc(3 * 224 * 224 * 4), cuda.malloc(4), cuda.malloc(4)]
    rows_in = evaluation_rows()
    try:
        rows = int8_pass_once(context, bindings, cuda, rows_in, mean, std, interpolation)
    finally:
        for pointer in bindings:
            cuda.free(pointer)
    top1 = sum(bool(row["correct"]) for row in rows) / len(rows)
    invalid = sum(not bool(row["valid"]) for row in rows)
    label = args.label or args.engine.name
    print(f"{label}: model={model_key} top1={top1:.4f} invalid={invalid}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
