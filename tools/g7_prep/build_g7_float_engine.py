#!/usr/bin/env python3
"""GPU_M2D G8 precision study: build an FP32 or FP16 TensorRT 8.6.1 engine
from the SAME unquantized ONNX export the INT8 path starts from
(<weights-root>/<name>/model.onnx, export_g7_onnx.py: opset 17, fixed
batch 1, three-binding contract data/prob/index).

User decision 2026-10-22: ResNet-50 and ViT-B, DRAM + SRAM mode, the
G8 rho = 10 protocol unchanged -- only the engine precision changes from
INT8 to FP32 / FP16.

  --precision fp32   no reduced-precision flag; TF32 is CLEARED so every
                     FP32 layer computes in IEEE FP32 (TRT 8.6 enables
                     TF32 by default)
  --precision fp16   BuilderFlag.FP16 (TensorRT may still keep a layer in
                     FP32 where that is faster or required); TF32 cleared
                     for those FP32 layers. Bindings stay FP32: the
                     network I/O types come from the ONNX graph.

Writes <name>/<precision>.engine + <precision>_engine_summary.json (the
INT8 clean.engine / engine_summary.json are never touched). Idempotent:
an existing PASS summary with the same build identity and engine sha256
is kept.

Run under the vit_fault python with the TRT wiring:
  tools/g7_prep/build_g7_engines.sh 0 --float fp32 resnet50 vit_base_patch16_224
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

from build_g7_int8_engine import (
    WEIGHTS_ROOT,
    atomic_bytes,
    atomic_text,
    engine_bindings,
    gpu_identity,
    parse_network,
    sha256_file,
    zero_input_smoke,
)

WORKSPACE_BYTES = 4 * 1024**3
PRECISIONS = ("fp32", "fp16")
DEFAULT_MODELS = ["resnet50", "vit_base_patch16_224"]


def build(name: str, precision: str, physical_gpu: int) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly the requested physical GPU"
        )
    model_dir = WEIGHTS_ROOT / name
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.is_file():
        raise RuntimeError(f"{name}: missing {onnx_path} (run export_g7_onnx.py first)")
    onnx_sha256 = sha256_file(onnx_path)
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))

    flags = ["FP16"] if precision == "fp16" else []
    build_identity_source = {
        "onnx_sha256": onnx_sha256,
        "precision": precision,
        "builder_flags": flags,
        "tf32": False,
        "creation_flags": ["EXPLICIT_BATCH"],
        "workspace_bytes": WORKSPACE_BYTES,
        "tensorrt": trt.__version__,
    }
    build_identity = hashlib.sha256(
        json.dumps(build_identity_source, sort_keys=True).encode("utf-8")
    ).hexdigest()

    engine_path = model_dir / f"{precision}.engine"
    summary_path = model_dir / f"{precision}_engine_summary.json"
    if summary_path.is_file() and engine_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") == "PASS"
            and prior.get("precision") == precision
            and prior.get("build_identity") == build_identity
            and prior.get("engine_sha256") == sha256_file(engine_path)
        ):
            print(f"{name}: {precision}_engine_build=SKIP reason=eligible_existing_engine",
                  flush=True)
            return summary_path

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder, network = parse_network(onnx_path, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("platform reports no fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)

    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT returned no serialized {precision} engine")
    engine_bytes = bytes(serialized)
    if not engine_bytes:
        raise RuntimeError("TensorRT returned an empty serialized engine")
    atomic_bytes(engine_path, engine_bytes)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None or engine.has_implicit_batch_dimension:
        raise RuntimeError(f"{precision} engine failed deserialization sanity")
    contract = engine_bindings(engine)
    if [item["name"] for item in contract] != ["data", "prob", "index"]:
        raise RuntimeError(f"engine binding order changed: {contract}")
    if [item["dtype"] for item in contract] != [
        str(trt.float32), str(trt.float32), str(trt.int32)
    ]:
        raise RuntimeError(f"engine binding dtypes changed: {contract}")
    smoke = zero_input_smoke(engine_bytes, logger)

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "g8_precision",
        "model_key": name,
        "precision": precision,
        "onnx_path": str(onnx_path),
        "onnx_sha256": onnx_sha256,
        "preprocessing": {
            "policy": "canonical_aspect_resize_center_crop_v2",
            "mean": meta["mean"],
            "std": meta["std"],
            "interpolation": meta["interpolation"],
            "crop_pct": meta["crop_pct"],
        },
        "build_identity": build_identity,
        "build_identity_source": build_identity_source,
        "workspace_bytes": WORKSPACE_BYTES,
        "creation_flags": ["EXPLICIT_BATCH"],
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "engine_size_bytes": len(engine_bytes),
        "engine_bindings": contract,
        "zero_input_smoke": smoke,
        "build_elapsed_seconds": time.time() - started,
        "physical_gpu": gpu_identity(physical_gpu),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "tensorrt": trt.__version__,
        },
    }
    atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"{name}: {precision}_engine_build=PASS size_bytes={len(engine_bytes)} "
        f"elapsed_seconds={summary['build_elapsed_seconds']:.1f} "
        f"smoke_prob={smoke['prob']:.6f}",
        flush=True,
    )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--precision", required=True, choices=PRECISIONS)
    parser.add_argument("--device", type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    for name in args.models or DEFAULT_MODELS:
        build(name, args.precision, args.device)
    print(f"g8_{args.precision}_engine_build=PASS models={len(args.models or DEFAULT_MODELS)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
