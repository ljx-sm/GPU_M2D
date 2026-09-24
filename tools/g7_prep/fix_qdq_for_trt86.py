#!/usr/bin/env python3
"""GPU_M2D G7: normalize ModelOpt Q/DQ ONNX graphs for the TensorRT
8.6.1 parser.

Two parser defects observed on the Q/DQ graphs (the ORIGINAL export
models parse fine; the quantized ones do not):

1. SE modules (MobileNetV3 / EfficientNet): ReduceMean(axes=[2,3],
   keepdims=1) -> Q -> DQ -> Conv makes the parser's convMultiInput
   pre-pass see an input with unknown channel count ("group and kernel
   shape misalign with the channel size").
2. Swin head: ReduceMean(axes=[2,3], keepdims=0) -> Gemm makes importGemm
   see a non-2D input ("GEMM must have 2D inputs").

Fix (bitwise-equivalent, Q/DQ chains untouched):
  A. spatial mean (NCHW, axes={2,3}): replace the ReduceMean with
     GlobalAveragePool (+ Flatten(axis=1) when keepdims=0) -- native to
     TRT 8.6, output shape always resolvable;
  B. any other axes (e.g. the swin head's axes=[1] on a 3D tensor):
     restore the axes as an ATTRIBUTE and downgrade the graph opset to
     17, where attribute form is legal and TRT 8.6 imports it natively
     (axes-as-input -- Constant node or initializer -- defeats TRT 8.6
     shape inference entirely).

Runs in the ISOLATED modelopt venv (needs onnx + onnx-graphsurgeon):

  /data1/luojx/REMU/.local/deps/modelopt-venv/bin/python \
      tools/g7_prep/fix_qdq_for_trt86.py <model> [model ...]

Rewrites <weights-root>/<model>/model_qdq.onnx in place (the file is a
build intermediate; its sha256 feeds the explicit build identity).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs


WEIGHTS_ROOT = Path("/data1/luojx/g7_models")


def tensor_rank(graph_inputs: dict, value_info: dict, name: str) -> int | None:
    """Rank of a tensor's declared shape; accepts graphsurgeon tensors
    (graph inputs) and raw onnx ValueInfoProto (shape-inferred)."""
    entry = graph_inputs.get(name)
    if entry is not None:
        shape = entry.shape
        if isinstance(shape, str):
            return None
        return len(shape)
    proto = value_info.get(name)
    if proto is None:
        return None
    return len(proto.type.tensor_type.shape.dim)


def fix_model(model: str) -> int:
    onnx_path = WEIGHTS_ROOT / model / "model_qdq.onnx"
    graph = gs.import_onnx(onnx.load(str(onnx_path)))

    # constant tensors map for axes lookup: initializers folded by
    # graphsurgeon plus outputs of Constant NODES (modelopt emits axes as
    # Constant nodes, which gs keeps as nodes)
    constants = {t.name: t for t in graph.tensors().values()
                 if isinstance(t, gs.Constant)}
    producers: dict[str, gs.Node] = {}
    for gs_node in graph.nodes:
        for out in gs_node.outputs:
            producers[out.name] = gs_node
    inferred = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    value_info = {
        v.name: v for v in list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    graph_inputs = {t.name: t for t in graph.inputs}

    replaced = 0
    attr_restored = 0
    for node in graph.nodes:
        if node.op != "ReduceMean":
            continue
        axes_name = node.inputs[1].name if len(node.inputs) > 1 else ""
        axes_const = constants.get(axes_name)
        if axes_const is None:
            producer = producers.get(axes_name)
            if producer is not None and producer.op == "Constant":
                axes_const = producer.attrs.get("value")
        if axes_const is None:
            continue
        axes = [int(a) for a in np.asarray(axes_const.values).flatten().tolist()]
        keepdims = int(node.attrs.get("keepdims", 1))
        rank = tensor_rank(graph_inputs, value_info, node.inputs[0].name)
        if rank == 4 and set(axes) == {2, 3}:
            # pattern A: mean over spatial dims of an NCHW tensor ->
            # GlobalAveragePool (+ Flatten for keepdims=0); bitwise
            # equivalent and natively shape-resolvable by TRT 8.6
            out = node.outputs[0]
            gap = gs.Node(op="GlobalAveragePool", name=node.name + "_gap",
                          inputs=[node.inputs[0]], outputs=[])
            if keepdims == 1:
                gap.outputs = [out]
                node.outputs = []
                node.inputs = []
            else:
                intermediate = gs.Variable(
                    f"{node.name}_gap_out", dtype=out.dtype, shape=None)
                gap.outputs = [intermediate]
                flatten = gs.Node(
                    op="Flatten", name=node.name + "_flatten",
                    attrs={"axis": 1},
                    inputs=[intermediate], outputs=[out])
                node.outputs = []
                node.inputs = []
                graph.nodes.append(flatten)
            graph.nodes.append(gap)
            replaced += 1
        else:
            # pattern B (e.g. swin head axes=[1] keepdims=0 on a 3D
            # tensor): restore the axes as an ATTRIBUTE and downgrade the
            # graph opset to 17, where that form is legal.  TRT 8.6
            # cannot shape-infer ReduceMean with axes-as-input at all
            # (Constant node OR initializer -> "GEMM must have 2D inputs"
            # downstream), while an axes attribute on an opset-19 graph is
            # spec-invalid and gets executed as reduce-over-all-dims (that
            # collapsed the swin head to 0.5% top-1).  Our original exports
            # ARE opset-17 attribute-form graphs, so downgrading restores
            # exactly the semantics TRT 8.6 imports natively.
            node.attrs["axes"] = axes
            node.inputs = [node.inputs[0]]
            attr_restored += 1

    if replaced == 0 and attr_restored == 0:
        print(f"{model}: reduce_mean_fix=NONE (no eligible nodes)")
        return 0
    graph.cleanup().toposort()
    exported = gs.export_onnx(graph)
    if attr_restored:
        # make the restored attribute-form axes legal (opset 18 removed
        # the ReduceMean axes attribute; our originals are opset 17)
        for entry in exported.opset_import:
            if entry.domain in ("", "ai.onnx"):
                entry.version = 17
    onnx.save(exported, str(onnx_path))
    print(f"{model}: reduce_mean_fix=REPLACED nodes={replaced} "
          f"axes_attr_opset17={attr_restored} -> {onnx_path}")
    return replaced


def main() -> int:
    models = sys.argv[1:]
    if not models:
        raise SystemExit(__doc__)
    for model in models:
        fix_model(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
