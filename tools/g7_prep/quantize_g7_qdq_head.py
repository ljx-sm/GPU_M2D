#!/usr/bin/env python3
"""GPU_M2D G7: explicit Q/DQ INT8 quantization WITH the classifier head
quantized (v2 canonical fix, user decision 2026-09-24).

The G7 contract is "weights AND activations INT8, only a few nonlinear
ops FP32". The v1 wrapper drove ``python -m modelopt.onnx.quantization``;
ModelOpt 0.47.0's default-on ``enable_gemv_detection_for_trt`` heuristic
silently excluded the batch-1 classifier-head Gemm from quantization
(output (1,1000) counts as a GEMV; no CLI flag exists to disable it) --
so every v1 engine shipped an FP32 fc head (README "ModelOpt
auto-excludes the batch-1 classifier head"). This wrapper calls the SAME
quantizer through the python API with
``enable_gemv_detection_for_trt=False`` -- the ONE delta vs v1. Every
recipe (entropy calibration on calib_canonical.npy, per-channel INT8
weights, opset 17, fp32 high-precision dtype, per-model op-type lists,
swin's disable_mha_qdq) is identical to the v1 recipes.

After quantizing, the wrapper FAILS CLOSED unless every weighted Gemm in
the graph carries a DequantizeLinear on its weight input (i.e. the head
is genuinely INT8), and prints the Q/DQ census for the record.

Runs in the ISOLATED modelopt venv:

  /data1/luojx/REMU/.local/deps/modelopt-venv/bin/python \\
      tools/g7_prep/quantize_g7_qdq_head.py <model> [model ...]

Writes <weights-root>/<model>/model_qdq.onnx (engine-ready build
intermediate; bypass_g7_qdq_activations.py and fix_qdq_for_trt86.py run
afterwards, exactly as in v1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx

WEIGHTS_ROOT = Path("/data1/luojx/g7_models")

# v1 recipes verbatim (quantize_g7_qdq.sh recipe_extra), now expressed as
# python-API kwargs
RECIPES: dict[str, dict] = {
    "resnet50": {},
    "vit_base_patch16_224": {},
    "deit_small_patch16_224": {},
    "mobilenetv3_large_100": {"op_types_to_quantize": ["Conv", "Gemm",
                                                       "MatMul"]},
    "efficientnet_b0": {"op_types_to_quantize": ["Conv", "Gemm", "MatMul"]},
    "swin_tiny_patch4_window7_224": {"op_types_to_quantize": ["Conv", "Gemm",
                                                              "MatMul"],
                                     "disable_mha_qdq": True},
}


def quantize_model(model: str) -> None:
    from modelopt.onnx.quantization import quantize

    model_dir = WEIGHTS_ROOT / model
    calibration = np.load(model_dir / "calib_canonical.npy")
    settings = dict(
        onnx_path=str(model_dir / "model.onnx"),
        output_path=str(model_dir / "model_qdq.onnx"),
        quantize_mode="int8",
        calibration_data=calibration,
        calibration_method="entropy",
        opset=17,
        high_precision_dtype="fp32",
        keep_intermediate_files=True,
        log_level="INFO",
        # THE v2 fix: quantize the batch-1 head Gemm like every other
        # weighted op (CLI-equivalent of v1 plus this one flag)
        enable_gemv_detection_for_trt=False,
    )
    settings.update(RECIPES[model])
    print(f"=== modelopt qdq (head quantized): {model} ===")
    quantize(**settings)


def verify_head_quantized(model: str) -> int:
    """Fail closed unless every Gemm weight sits behind a DequantizeLinear."""
    path = WEIGHTS_ROOT / model / "model_qdq.onnx"
    graph = onnx.load(str(path)).graph
    producers = {out: node for node in graph.node for out in node.output}
    initializer = {init.name for init in graph.initializer}

    quantize_nodes = [n for n in graph.node if n.op_type == "QuantizeLinear"]
    dequantize_nodes = [n for n in graph.node
                        if n.op_type == "DequantizeLinear"]
    weight_quantizers = sum(1 for n in quantize_nodes
                            if n.input[0] in initializer)

    problems: list[str] = []
    gemm_report: list[str] = []
    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        weight = node.input[1] if len(node.input) > 1 else ""
        producer = producers.get(weight)
        quantized = producer is not None and \
            producer.op_type == "DequantizeLinear"
        gemm_report.append(f"{node.name}: weight {'DQ(int8)' if quantized else 'RAW-FP32'}")
        if not quantized:
            problems.append(f"{node.name} weight input {weight!r} has no "
                            "DequantizeLinear producer")
    for line in gemm_report:
        print(f"  head {line}")
    print(f"{model}: QuantizeLinear={len(quantize_nodes)} "
          f"DequantizeLinear={len(dequantize_nodes)} "
          f"weight_quantizers={weight_quantizers}")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        raise SystemExit(f"quantize_g7_qdq_head: {model}: head not quantized")
    return len(quantize_nodes)


def main() -> int:
    models = sys.argv[1:]
    if not models or any(m not in RECIPES for m in models):
        raise SystemExit(__doc__)
    for model in models:
        quantize_model(model)
        verify_head_quantized(model)
        print(f"=== modelopt qdq (head quantized) DONE: {model} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
