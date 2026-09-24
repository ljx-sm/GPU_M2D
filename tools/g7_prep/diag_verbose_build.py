#!/usr/bin/env python3
"""G7 diagnostic: rebuild engines with VERBOSE logging to /tmp (never
touching the frozen artifacts) so the TRT build log's per-layer precision
lines can be counted -- which layers run INT8, which stay FP32.

Also supports --fp32-only (no INT8 flag, no calibrator) to produce an
FP32 TRT engine for ONNX-fidelity isolation.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

from build_g7_int8_engine import (
    EntropyCalibrator,
    INPUT_SHAPE,
    WEIGHTS_ROOT,
    atomic_bytes,
    parse_network,
    read_calibration_paths,
)

OUT_ROOT = Path("/tmp/g7_diag")


class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    """Same batching/preprocessing as the frozen EntropyCalibrator; only
    the range-estimation algorithm differs (diagnostic: is the INT8 loss
    driven by activation ranges or by weight granularity?)."""

    def __init__(self, inner: EntropyCalibrator) -> None:
        super().__init__()
        self.inner = inner
        self.cache_path = inner.cache_path

    def get_batch_size(self) -> int:
        return self.inner.get_batch_size()

    def get_batch(self, names):
        return self.inner.get_batch(names)

    def read_calibration_cache(self):
        return self.inner.read_calibration_cache()

    def write_calibration_cache(self, cache):
        atomic_bytes(self.cache_path, bytes(cache))
        print(f"minmax_calibration_cache=WRITTEN path={self.cache_path}", flush=True)


class CaptureLogger(trt.ILogger):
    """TRT logger that forwards to python print (capturable)."""

    def __init__(self, severity: int) -> None:
        super().__init__()
        self.severity = severity

    def log(self, message: str, severity) -> None:
        print(f"[TRT{severity}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--fp32-only", action="store_true")
    parser.add_argument("--calibrator", choices=("entropy", "minmax"),
                        default="entropy")
    args = parser.parse_args()

    model_dir = WEIGHTS_ROOT / args.model
    meta = json.loads((model_dir / "model_meta.json").read_text())
    onnx_path = model_dir / "model.onnx"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger = CaptureLogger(int(trt.Logger.VERBOSE))
    trt.init_libnvinfer_plugins(logger, "")
    builder, network = parse_network(onnx_path, logger)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024**3)
    if not args.fp32_only:
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = EntropyCalibrator(
            read_calibration_paths(),
            tuple(float(v) for v in meta["mean"]),
            tuple(float(v) for v in meta["std"]),
            {"bicubic": cv2.INTER_CUBIC,
             "bilinear": cv2.INTER_LINEAR}[str(meta["interpolation"])],
            OUT_ROOT / f"calib_{args.model}.cache",
        )
        if args.calibrator == "minmax":
            calibrator = MinMaxCalibrator(calibrator)
        config.int8_calibrator = calibrator

    tag = "fp32" if args.fp32_only else f"int8_{args.calibrator}"
    log_path = OUT_ROOT / f"build_{args.model}_{tag}.log"
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        serialized = builder.build_serialized_network(network, config)
    log_path.write_text(buffer.getvalue())
    print(f"log lines: {len(buffer.getvalue().splitlines())} -> {log_path}")
    if serialized is None:
        raise RuntimeError("build failed")
    engine_path = OUT_ROOT / f"{args.model}_{tag}.engine"
    engine_path.write_bytes(bytes(serialized))
    print(f"engine bytes: {engine_path.stat().st_size} -> {engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
