#!/usr/bin/env python3
"""GPU_M2D G7: build the TensorRT 8.6.1 engine from an EXPLICIT-quantized
(Q/DQ, per-channel weights) ONNX produced by ModelOpt ONNX PTQ.

This replaces the implicit-entropy path (build_g7_int8_engine.py, kept as
fallback) as the G7 canonical engine: implicit PTQ quantizes weights
per-tensor symmetric (deprecated path) and costs 6-10 pp on
depthwise+SE/distilled architectures; explicit Q/DQ with per-channel
weight quantization is the standard <1-2 pp recipe.

Differences from the implicit builder:
  - network parsed with EXPLICIT_BATCH | STRONGLY_TYPED so the Q/DQ nodes
    dictate precision (no kINT8 builder flag, no calibrator);
  - build identity keyed on the Q/DQ ONNX sha256 instead of a calibration
    cache identity (calibration already happened inside ModelOpt);
  - Q/DQ node census written into the summary (weights per-channel is
    verified from the QuantizeLinear scale shapes, not assumed).

Expects <weights-root>/<name>/model_qdq.onnx + calib_canonical.json
(provenance of the ModelOpt calibration input).  Writes clean.engine +
engine_summary.json (same schema the eval consumes).

Run under the vit_fault python with the TRT wiring (build_g7_engines.sh).
"""

from __future__ import annotations

import argparse
import json
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
    INPUT_SHAPE,
    MODELS,
    WEIGHTS_ROOT,
    atomic_bytes,
    atomic_text,
    engine_bindings,
    gpu_identity,
    sha256_file,
    zero_input_smoke,
)

WORKSPACE_BYTES = 4 * 1024**3


def parse_network_strongly_typed(onnx_path: Path, logger):
    builder = trt.Builder(logger)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) | (
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parsing failed:\n" + "\n".join(errors))
    if network.num_inputs != 1 or network.num_outputs != 2:
        raise RuntimeError("parsed network does not expose one input and two outputs")
    input_tensor = network.get_input(0)
    outputs = {network.get_output(index).name for index in range(2)}
    if (
        input_tensor.name != "data"
        or tuple(input_tensor.shape) != INPUT_SHAPE
        or input_tensor.dtype != trt.float32
        or outputs != {"prob", "index"}
    ):
        raise RuntimeError("parsed TensorRT interface contract changed")
    return builder, network


def qdq_census(onnx_path: Path) -> dict[str, object]:
    """Count Q/DQ nodes and verify weight QuantizeLinear scales are
    per-output-channel (rank-1 with the channel dim), not per-tensor."""
    import onnx

    model = onnx.load(str(onnx_path))
    quantizers = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
    dequantizers = [n for n in model.graph.node if n.op_type == "DequantizeLinear"]
    initializers = {init.name: init for init in model.graph.initializer}
    per_channel = 0
    per_tensor = 0
    for node in quantizers:
        # a weight quantizer is one whose input is a graph initializer
        if node.input[0] in initializers:
            scale_init = initializers.get(node.input[1])
            if scale_init is None:
                continue
            scale_array = onnx.numpy_helper.to_array(scale_init)
            if scale_array.ndim == 1 and scale_array.size > 1:
                per_channel += 1
            else:
                per_tensor += 1
    return {
        "quantize_linear_nodes": len(quantizers),
        "dequantize_linear_nodes": len(dequantizers),
        "weight_quantizers_per_channel": per_channel,
        "weight_quantizers_per_tensor": per_tensor,
    }


def build(name: str, physical_gpu: int) -> Path:
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly the requested physical GPU"
        )
    model_dir = WEIGHTS_ROOT / name
    qdq_path = model_dir / "model_qdq.onnx"
    if not qdq_path.is_file():
        raise RuntimeError(f"{name}: missing {qdq_path} (run ModelOpt ONNX PTQ first)")
    qdq_sha256 = sha256_file(qdq_path)
    calib_provenance = json.loads(
        (model_dir / "calib_canonical.json").read_text(encoding="utf-8")
    )

    census = qdq_census(qdq_path)
    if census["quantize_linear_nodes"] == 0 or census["dequantize_linear_nodes"] == 0:
        raise RuntimeError(f"{name}: ONNX carries no Q/DQ nodes -- not explicit-quantized")

    build_identity_source = {
        "qdq_onnx_sha256": qdq_sha256,
        "calib_npy_sha256": calib_provenance["npy_sha256"],
        "creation_flags": ["EXPLICIT_BATCH", "STRONGLY_TYPED"],
        "workspace_bytes": WORKSPACE_BYTES,
        "tensorrt": trt.__version__,
    }
    import hashlib

    build_identity = hashlib.sha256(
        json.dumps(build_identity_source, sort_keys=True).encode("utf-8")
    ).hexdigest()

    engine_path = model_dir / "clean.engine"
    summary_path = model_dir / "engine_summary.json"
    if summary_path.is_file() and engine_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            prior.get("status") == "PASS"
            and prior.get("precision") == "int8_ptq_explicit_qdq"
            and prior.get("build_identity") == build_identity
            and prior.get("engine_sha256") == sha256_file(engine_path)
            and prior.get("engine_size_bytes") == engine_path.stat().st_size
        ):
            print(f"{name}: explicit_engine_build=SKIP reason=eligible_existing_engine",
                  flush=True)
            return summary_path

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder, network = parse_network_strongly_typed(qdq_path, logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)

    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized explicit-quantized engine")
    engine_bytes = bytes(serialized)
    if not engine_bytes:
        raise RuntimeError("TensorRT returned an empty serialized engine")
    atomic_bytes(engine_path, engine_bytes)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None or engine.has_implicit_batch_dimension:
        raise RuntimeError("explicit engine failed deserialization sanity")
    contract = engine_bindings(engine)
    if [item["name"] for item in contract] != ["data", "prob", "index"]:
        raise RuntimeError(f"engine binding order changed: {contract}")
    smoke = zero_input_smoke(engine_bytes, logger)

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "g7_imagenet1k",
        "model_key": name,
        "precision": "int8_ptq_explicit_qdq",
        "onnx_path": str(qdq_path),
        "onnx_sha256": qdq_sha256,
        "qdq_census": census,
        "calibration": {
            "manifest": calib_provenance.get("calibration_manifest"),
            "images": calib_provenance["images"],
            "npy_sha256": calib_provenance["npy_sha256"],
            "resize_scale": calib_provenance["resize_scale"],
            "tool": "nvidia-modelopt onnx ptq (independent venv)",
        },
        "preprocessing": {
            "policy": "canonical_aspect_resize_center_crop_v2",
            "mean": calib_provenance["mean"],
            "std": calib_provenance["std"],
            "interpolation": calib_provenance["interpolation"],
            "resize_scale": calib_provenance["resize_scale"],
        },
        "build_identity": build_identity,
        "build_identity_source": build_identity_source,
        "workspace_bytes": WORKSPACE_BYTES,
        "creation_flags": ["EXPLICIT_BATCH", "STRONGLY_TYPED"],
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
        f"{name}: explicit_engine_build=PASS size_bytes={len(engine_bytes)} "
        f"qdq=Q{census['quantize_linear_nodes']}/DQ{census['dequantize_linear_nodes']} "
        f"w_per_channel={census['weight_quantizers_per_channel']} "
        f"w_per_tensor={census['weight_quantizers_per_tensor']} "
        f"elapsed_seconds={summary['build_elapsed_seconds']:.1f} "
        f"smoke_prob={smoke['prob']:.6f}",
        flush=True,
    )
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=MODELS)
    parser.add_argument("--device", type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    if not args.models:
        parser.error("no models selected")
    for name in args.models:
        build(name, args.device)
    print(f"g7_explicit_engine_build=PASS models={len(args.models)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
