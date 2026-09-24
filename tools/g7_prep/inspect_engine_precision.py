#!/usr/bin/env python3
"""GPU_M2D G7 diagnostic: per-layer precision of the built INT8 engines.

Answers: which layers actually run INT8 (weights AND activations), which
stay FP32 (expected: non-linear ops without calibration relevance --
softmax, topk, pooling tails, etc.), so the quantization coverage claim
is read off the engine itself, not assumed.

Run with the TRT wiring (LD_LIBRARY_PATH from build_g7_engines.sh):

  .../vit_fault/bin/python tools/g7_prep/inspect_engine_precision.py

Prints per model: layer count by (op, in-precision, out-precision)
clusters + the FP32 layers by name, and writes
<weights-root>/<name>/eval/layer_precision.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

WEIGHTS_ROOT = Path("/data1/luojx/g7_models")
MODELS = [
    "resnet50",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "vit_base_patch16_224",
    "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
]


def cluster_name(layer_name: str) -> str:
    """Collapse TRT suffixed names to the ONNX-ish op cluster."""
    return re.sub(r" \(\d+\)$", "", layer_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)

    for name in args.models:
        model_dir = args.weights_root / name
        summary = json.loads((model_dir / "engine_summary.json").read_text())
        engine_path = Path(summary["engine_path"])
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"{name}: cannot deserialize engine")
        inspector = engine.create_engine_inspector()
        raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        info = json.loads(raw)

        layers = []
        for layer in info["Layers"]:
            name_raw = layer["Name"]
            inputs = layer["Inputs"]
            outputs = layer["Outputs"]
            in_formats = sorted({
                f"{tensor['Format']}:{tensor['DataType']}"
                for tensor in inputs
            })
            out_formats = sorted({
                f"{tensor['Format']}:{tensor['DataType']}"
                for tensor in outputs
            })
            layers.append({
                "layer": cluster_name(name_raw),
                "tactic": layer.get("TacticValue", ""),
                "inputs": in_formats,
                "outputs": out_formats,
                "any_fp32_io": any(
                    "FP32" in fmt for fmt in (*in_formats, *out_formats)
                ),
            })

        fp32_layers = [l for l in layers if l["any_fp32_io"]]
        clusters = Counter(l["layer"] for l in fp32_layers)
        report = {
            "model_key": name,
            "engine_sha256": summary["engine_sha256"],
            "total_layers": len(layers),
            "int8_layers": len(layers) - len(fp32_layers),
            "fp32_touching_layers": len(fp32_layers),
            "fp32_clusters": dict(clusters.most_common()),
            "layers": layers,
        }
        out_dir = model_dir / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "layer_precision.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            f"{name}: layers={len(layers)} int8_io={len(layers) - len(fp32_layers)} "
            f"fp32_touching={len(fp32_layers)} ({len(fp32_layers) / len(layers):.1%})"
        )
        for cluster, count in clusters.most_common(12):
            print(f"    fp32: {cluster} x{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
