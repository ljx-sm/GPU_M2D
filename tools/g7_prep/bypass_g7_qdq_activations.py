#!/usr/bin/env python3
"""GPU_M2D G7: bypass the activation Q/DQ pairs on swish-family outputs
in a ModelOpt Q/DQ graph (weights and every other conv-input activation
stay INT8; the nonlinearity outputs feed their convs in FP32).

Why: per-tensor symmetric INT8 on swish outputs is catastrophically lossy
for the two swish-heavy architectures -- measured on the 2000-image probe
(vs FP32 ONNX 81.35 / 83.50):

  mobilenetv3_large_100  conv-only         75.30   (-6.05 pp)
  mobilenetv3_large_100  + HardSwish bypass 81.05  (-0.30 pp)
  efficientnet_b0        conv-only         60.85   (-22.65 pp)
  efficientnet_b0        + SiLU bypass     83.20   (-0.30 pp)

The residual -0.3 pp is the cost of EVERYTHING ELSE quantized (per-channel
INT8 weights: free; SE/ReLU/Add/GAP activations: ~free).  This is the
"少量非线性算子不量化" exception of the G7 contract: swish/hard-swish
outputs are the one activation family kept FP32.

Runs in the ISOLATED modelopt venv (onnx + onnx-graphsurgeon):

  /data1/luojx/REMU/.local/deps/modelopt-venv/bin/python \
      tools/g7_prep/bypass_g7_qdq_activations.py <model> [model ...]

Rewrites <weights-root>/<model>/model_qdq.onnx in place (build
intermediate; its sha256 feeds the explicit build identity).
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs

WEIGHTS_ROOT = Path("/data1/luojx/g7_models")

# quantization-hostile nonlinearity outputs, by model (regex on the
# activation tensor name; only ACTIVATION quantizers are ever bypassed --
# weight quantizers are never touched)
BYPASS_PATTERNS: dict[str, list[str]] = {
    "mobilenetv3_large_100": [r"act/HardSwish"],
    "efficientnet_b0": [r"act/Mul"],
}


def bypass_activations(model: str, patterns: list[str]) -> int:
    onnx_path = WEIGHTS_ROOT / model / "model_qdq.onnx"
    graph = gs.import_onnx(onnx.load(str(onnx_path)))
    const_names = {t.name for t in graph.tensors().values()
                   if isinstance(t, gs.Constant)}
    consumers: dict[str, list[gs.Node]] = {}
    for node in graph.nodes:
        for inp in node.inputs:
            consumers.setdefault(inp.name, []).append(node)

    import re

    matchers = [re.compile(p) for p in patterns]
    bypassed = 0
    for node in list(graph.nodes):
        if node.op != "QuantizeLinear":
            continue
        tensor = node.inputs[0]
        if tensor.name in const_names:  # never bypass weight quantizers
            continue
        if not any(m.search(tensor.name) for m in matchers):
            continue
        dq = next((c for c in consumers.get(node.outputs[0].name, [])
                   if c.op == "DequantizeLinear"), None)
        if dq is None:
            continue
        for consumer in list(graph.nodes):
            for i, inp in enumerate(consumer.inputs):
                if inp.name == dq.outputs[0].name:
                    consumer.inputs[i] = tensor
        node.outputs = []
        dq.outputs = []
        bypassed += 1

    if bypassed == 0:
        print(f"{model}: qdq_activation_bypass=NONE (no matching tensors)")
        return 0
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), str(onnx_path))
    print(f"{model}: qdq_activation_bypass=REWIRED tensors={bypassed} "
          f"patterns={patterns} -> {onnx_path}")
    return bypassed


def main() -> int:
    models = sys.argv[1:]
    if not models:
        raise SystemExit(__doc__)
    for model in models:
        patterns = BYPASS_PATTERNS.get(model)
        if patterns is None:
            print(f"{model}: qdq_activation_bypass=SKIP (no pattern registered)")
            continue
        bypass_activations(model, patterns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
