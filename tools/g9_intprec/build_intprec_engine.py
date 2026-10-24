#!/usr/bin/env python3
"""G9 integer-precision study (user decision 2026-10-23): build a
weight-INT4 (W4A8) or weight-INT16 (W16A8) TensorRT 8.6.1 engine for
ResNet-50 / ViT-B from the SAME Q/DQ graph as the G7-v2 INT8 engine
(<model>/model_qdq.onnx), so the activation quantization is identical
and only the weight storage changes.

Every weight of a Conv / Gemm / MatMul (Constant -> Quantize ->
Dequantize in the parsed network) is replaced by an M2DPackedWeight
plugin (libm2d_packed_weight.so) holding the weight PACKED in device
memory:
  w4a8   INT4, symmetric scales per GROUP of --group consecutive weights
         inside an output channel (default 128, MSE-optimal clipping);
         --keep8-json leaves the listed layers exactly as the INT8 engine
         has them (mixed precision, ResNet-50: int4_sensitivity.py)
  w16a8  INT16, per-output-channel scales
--compute fp32 (default, user decision 2026-10-23): the expanded weights
feed the layer directly, so TensorRT computes in FP32 -- exact for
INT16/INT4 x INT8 products (< 2^24), and INT4 group scales survive.
--compute int8 re-quantizes them per output channel to INT8 (TensorRT's
INT8 kernels); that erases small INT4 groups and is not the default.
Activations keep the INT8 engine's Q/DQ in both.
A weight MatMul (ViT-B linear layers, both classifier heads) is rewritten
as a 1x1 convolution over the tokens: TensorRT 8.6 has no INT8 MatMul
kernel for runtime per-channel weights (feasibility test), and a 1x1
convolution is the form it gives a constant-weight INT8 MatMul itself.
Activation x activation MatMuls (attention) are untouched.

Every PluginField array is held until create_plugin returns (the Python
PluginField does not own its array). The weight values are read from the
ONNX initializer feeding each weight QuantizeLinear (matched by node
name), never from IConstantLayer.weights: in the TensorRT 8.6 Python
bindings that view of a parsed constant does not hold the initializer's
values (ResNet-50 conv1: max |w| 0.0027 read back vs 2.65 in the ONNX).

Writes /data1/luojx/g7_models/<model>/<precision>.engine +
<precision>_engine_summary.json (new files; the INT8 clean.engine and
the FP16/FP32 engines are never touched).

  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=<trt>:<cudnn>:<cuda> \\
      /data1/luojx/miniforge3/envs/vit_fault/bin/python \\
      tools/g9_intprec/build_intprec_engine.py resnet50 --precision w4a8 --device 1
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools/g7_prep"))
from build_g7_int8_engine import (  # noqa: E402
    WEIGHTS_ROOT, atomic_bytes, atomic_text, engine_bindings, gpu_identity,
    sha256_file, zero_input_smoke)

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

PLUGIN_LIB = PROJECT / "artifacts/g9_intprec/libm2d_packed_weight.so"
PRECISIONS = {"w4a8": 4, "w16a8": 16}
WORKSPACE_BYTES = 4 * 1024**3
TACTIC_SOURCES = (1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)) | \
    (1 << int(trt.TacticSource.JIT_CONVOLUTIONS))


CLIP_RATIOS = np.linspace(1.0, 0.3, 15)


def quantize(w: np.ndarray, bits: int, group: int,
             clip: str = "max") -> tuple[np.ndarray, np.ndarray, int]:
    """w [C, ...] channel-major -> (q int32 [C, ...], scales [C * G], group).
    clip "max": scale = max|w| / qmax per group; "mse": the scale of
    CLIP_RATIOS x max|w| / qmax with the least squared error per group."""
    c = w.shape[0]
    inner = int(np.prod(w.shape[1:]))
    g = inner if group <= 0 or group > inner else group
    groups = (inner + g - 1) // g
    flat = w.reshape(c, inner).astype(np.float64)
    pad = groups * g - inner
    padded = np.pad(flat, ((0, 0), (0, pad))).reshape(c, groups, g)
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(padded).max(axis=2)
    amax[amax == 0] = 1.0
    scale = amax / qmax
    if clip == "mse":
        best = None
        for ratio in CLIP_RATIOS:
            s_r = amax * ratio / qmax
            q_r = np.clip(np.round(padded / s_r[:, :, None]), -qmax - 1, qmax)
            err = ((q_r * s_r[:, :, None] - padded) ** 2).sum(axis=2)
            if best is None:
                best, scale = err, s_r
            else:
                better = err < best
                best = np.where(better, err, best)
                scale = np.where(better, s_r, scale)
    q = np.clip(np.round(padded / scale[:, :, None]), -qmax - 1, qmax)
    q = q.reshape(c, groups * g)[:, :inner].astype(np.int32).reshape(w.shape)
    return q, scale.astype(np.float32).reshape(-1), g


def dequantize(q: np.ndarray, scales: np.ndarray, group: int) -> np.ndarray:
    c = q.shape[0]
    inner = int(np.prod(q.shape[1:]))
    idx = np.arange(inner) // group
    s = scales.reshape(c, -1)[:, idx]
    return (q.reshape(c, inner) * s).reshape(q.shape).astype(np.float32)


def pack(q: np.ndarray, bits: int) -> np.ndarray:
    flat = q.reshape(-1)
    if bits == 16:
        return flat.astype("<i2").view(np.uint8)
    nib = (flat & 0xF).astype(np.uint8)
    if nib.size % 2:
        nib = np.append(nib, np.uint8(0))
    return (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8)


def as_(layer, cls):
    layer.__class__ = cls
    return layer


class Surgery:
    def __init__(self, net, bits: int, group: int, onnx_path: Path,
                 compute: str = "int8", keep8: tuple[str, ...] = (), clip: str = "max"):
        self.net = net
        self.requant = compute == "int8"    # expanded weights -> INT8 Q/DQ
        self.keep8 = keep8                  # layer names left as the INT8 engine has them
        self.clip = clip
        graph = onnx.load(str(onnx_path)).graph
        inits = {i.name: i for i in graph.initializer}
        self.weights = {n.name: inits[n.input[0]] for n in graph.node
                        if n.op_type == "QuantizeLinear" and n.input[0] in inits}
        self.bits = bits
        self.group = group
        self.creator = trt.get_plugin_registry().get_plugin_creator("M2DPackedWeight", "1", "")
        if self.creator is None:
            raise RuntimeError("M2DPackedWeight plugin not registered")
        self.held = []      # every array handed to TensorRT stays alive
        self.anchor = net.get_input(0)
        self.stats = {"weights": 0, "params": 0, "packed_bytes": 0, "scale_bytes": 0,
                      "conv": 0, "matmul_as_conv": 0}
        self.producer, self.consumers = {}, {}
        for i in range(net.num_layers):
            layer = net.get_layer(i)
            for o in range(layer.num_outputs):
                self.producer[layer.get_output(o).name] = layer
            for k in range(layer.num_inputs):
                t = layer.get_input(k)
                if t is not None:
                    self.consumers.setdefault(t.name, []).append((layer, k))

    def const(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        self.held.append(arr)
        return self.net.add_constant(arr.shape, arr).get_output(0)

    def plugin(self, w_cm: np.ndarray, bits: int, anchor):
        """Packed plugin for a channel-major weight [C, ...] -> (FP32 tensor, dequantized).
        `anchor` is the consuming layer's activation input: the expansion then runs
        right before that layer and TensorRT reuses its output memory, so only one
        layer's expanded FP32 weights exist at a time (anchored on the network input,
        every expansion ran at the start and ResNet-50 held ~100 MB of FP32 copies)."""
        group = self.group if bits == 4 else 0
        q, scales, g = quantize(w_cm, bits, group, self.clip if bits == 4 else "max")
        packed = pack(q, bits).view(np.int8)
        held = [np.array([bits], np.int32), np.array(w_cm.shape, np.int32),
                np.array([0], np.int32), np.array([g], np.int32),
                np.ascontiguousarray(scales), np.ascontiguousarray(packed)]
        self.held.extend(held)
        fields = trt.PluginFieldCollection([
            trt.PluginField("bits", held[0], trt.PluginFieldType.INT32),
            trt.PluginField("dims", held[1], trt.PluginFieldType.INT32),
            trt.PluginField("axis", held[2], trt.PluginFieldType.INT32),
            trt.PluginField("group", held[3], trt.PluginFieldType.INT32),
            trt.PluginField("scales", held[4], trt.PluginFieldType.FLOAT32),
            trt.PluginField("packed", held[5], trt.PluginFieldType.INT8),
        ])
        plugin = self.creator.create_plugin(f"packed_{self.stats['weights']}", fields)
        layer = self.net.add_plugin_v2([anchor], plugin)
        layer.name = f"M2DPackedWeight_{self.stats['weights']}"
        self.stats["weights"] += 1
        self.stats["params"] += int(q.size)
        self.stats["packed_bytes"] += int(packed.size)
        self.stats["scale_bytes"] += int(scales.size * 4)
        return layer.get_output(0), dequantize(q, scales, g)

    def weight_of(self, chain) -> np.ndarray:
        """The weight values from the ONNX initializer of this QuantizeLinear."""
        init = self.weights.get(chain["q"].name)
        if init is None:
            raise RuntimeError(f"{chain['q'].name}: no ONNX weight initializer")
        w = numpy_helper.to_array(init).astype(np.float32)
        if tuple(w.shape) != tuple(chain["const"].shape):
            raise RuntimeError(f"{chain['q'].name}: ONNX weight {w.shape} != parsed "
                               f"constant {tuple(chain['const'].shape)}")
        return w

    def weight_chain(self, tensor):
        """tensor <- [Shuffle] <- Dequantize <- Quantize <- Constant: the layers, or None."""
        layer = self.producer.get(tensor.name)
        shuffle = None
        if layer is not None and layer.type == trt.LayerType.SHUFFLE:
            shuffle, layer = layer, self.producer.get(layer.get_input(0).name)
        if layer is None or layer.type != trt.LayerType.DEQUANTIZE:
            return None
        quant = self.producer.get(layer.get_input(0).name)
        if quant is None or quant.type != trt.LayerType.QUANTIZE:
            return None
        const = self.producer.get(quant.get_input(0).name)
        if const is None or const.type != trt.LayerType.CONSTANT:
            return None
        return {"dq": as_(layer, trt.IDequantizeLayer), "q": as_(quant, trt.IQuantizeLayer),
                "const": as_(const, trt.IConstantLayer), "shuffle": shuffle}

    def int8_route(self, chain, w_deq_layout: np.ndarray, axis: int):
        """Re-quantize the expanded weights per output channel (INT8)."""
        other = tuple(i for i in range(w_deq_layout.ndim) if i != axis)
        s8 = (np.abs(w_deq_layout).max(axis=other) / 127).astype(np.float32)
        s8[s8 == 0] = 1.0
        scale = self.const(s8)
        chain["q"].set_input(1, scale)
        chain["dq"].set_input(1, scale)
        chain["q"].axis = axis
        chain["dq"].axis = axis

    def conv(self, layer):
        layer = as_(layer, trt.IConvolutionLayer)
        chain = self.weight_chain(layer.get_input(1))
        if chain is None or chain["shuffle"] is not None:
            return
        if layer.name in self.keep8:
            return
        w = self.weight_of(chain)
        out, w_deq = self.plugin(w, self.bits, layer.get_input(0))
        if self.requant:
            chain["q"].set_input(0, out)
            self.int8_route(chain, w_deq, 0)
        else:
            layer.set_input(1, out)
        self.stats["conv"] += 1

    def matmul(self, layer):
        layer = as_(layer, trt.IMatrixMultiplyLayer)
        chain = self.weight_chain(layer.get_input(1))
        if chain is None or layer.name in self.keep8:
            return
        w = self.weight_of(chain)
        w2 = w.reshape(w.shape[-2:])
        b = w2.T if layer.op1 == trt.MatrixOperation.TRANSPOSE else w2      # [K, N]
        if layer.op0 != trt.MatrixOperation.NONE:
            raise RuntimeError(f"{layer.name}: transposed activation operand")
        wc = np.ascontiguousarray(b.T)[:, :, None, None]                      # [N, K, 1, 1]
        n_out, k_in = wc.shape[:2]
        out, w_deq = self.plugin(wc, self.bits, layer.get_input(0))
        if self.requant:
            # a fresh Q/DQ pair on the expanded [N, K, 1, 1] weights
            s8 = (np.abs(w_deq).max(axis=(1, 2, 3)) / 127).astype(np.float32)
            s8[s8 == 0] = 1.0
            scale = self.const(s8)
            q = self.net.add_quantize(out, scale)
            dq = self.net.add_dequantize(q.get_output(0), scale)
            q.axis = 0
            dq.axis = 0
            wt = dq.get_output(0)
        else:
            wt = out
        x = layer.get_input(0)
        shape = tuple(x.shape)
        if len(shape) == 3:                                   # [1, T, K]
            tokens = shape[1]
            sh = self.net.add_shuffle(x)
            sh.reshape_dims = (shape[0], tokens, k_in, 1)
            sh.second_transpose = (0, 2, 1, 3)
        elif len(shape) == 2:                                 # [1, K]
            tokens = None
            sh = self.net.add_shuffle(x)
            sh.reshape_dims = (shape[0], k_in, 1, 1)
        else:
            raise RuntimeError(f"{layer.name}: activation rank {len(shape)}")
        conv = self.net.add_convolution_nd(sh.get_output(0), n_out, (1, 1),
                                           trt.Weights(), trt.Weights())
        conv.set_input(1, wt)
        conv.name = f"{layer.name}_as_conv1x1"
        back = self.net.add_shuffle(conv.get_output(0))
        if tokens is not None:
            back.first_transpose = (0, 2, 1, 3)
            back.reshape_dims = (shape[0], tokens, n_out)
        else:
            back.reshape_dims = (shape[0], n_out)
        new = back.get_output(0)
        old = layer.get_output(0)
        users = self.consumers.get(old.name, [])
        for user, k in users:
            user.set_input(k, new)
        if old.is_network_output:
            raise RuntimeError(f"{layer.name}: weight MatMul output is a network output")
        if not users:
            raise RuntimeError(f"{layer.name}: weight MatMul output has no consumers")
        self.stats["matmul_as_conv"] += 1

    def run(self):
        layers = [self.net.get_layer(i) for i in range(self.net.num_layers)]
        for layer in layers:
            if layer.type == trt.LayerType.CONVOLUTION:
                self.conv(layer)
            elif layer.type == trt.LayerType.MATRIX_MULTIPLY:
                self.matmul(layer)
        return self.stats


def build(name: str, precision: str, group: int, physical_gpu: int, compute: str = "fp32",
          clip: str = "mse", keep8_json: Path | None = None) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly the requested physical GPU")
    ctypes.CDLL(str(PLUGIN_LIB), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    model_dir = WEIGHTS_ROOT / name
    onnx_path = model_dir / "model_qdq.onnx"
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    bits = PRECISIONS[precision]

    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("ONNX parse failed: " + "; ".join(
            str(parser.get_error(i)) for i in range(parser.num_errors)))
    keep8: tuple[str, ...] = ()
    if keep8_json is not None:
        keep8 = tuple(json.loads(keep8_json.read_text())["keep8_trt_names"])
        names = {net.get_layer(i).name for i in range(net.num_layers)}
        missing = [k for k in keep8 if k not in names]
        if missing:
            raise RuntimeError(f"--keep8-json layers not in the network: {missing}")
    surgery = Surgery(net, bits, group, onnx_path, compute=compute, keep8=keep8, clip=clip)
    stats = surgery.run()
    stats["kept_int8"] = list(keep8)
    if stats["weights"] == 0:
        raise RuntimeError("no weight was replaced")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    config.set_flag(trt.BuilderFlag.INT8)
    config.clear_flag(trt.BuilderFlag.TF32)
    # TensorRT-native kernels only: the campaign runner ships no cuDNN /
    # cuBLAS (the G7-v2 INT8 engines need none), and their handles' device
    # memory would sit outside the allocation registry
    config.set_tactic_sources(TACTIC_SOURCES)
    started = time.time()
    serialized = builder.build_serialized_network(net, config)
    if serialized is None:
        raise RuntimeError(f"{name} {precision}: TensorRT returned no engine")
    engine_bytes = bytes(serialized)
    engine_path = model_dir / f"{precision}.engine"
    atomic_bytes(engine_path, engine_bytes)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)
    contract = engine_bindings(engine)
    if [item["name"] for item in contract] != ["data", "prob", "index"]:
        raise RuntimeError(f"engine binding order changed: {contract}")
    smoke = zero_input_smoke(engine_bytes, logger)
    identity_source = {"onnx_sha256": sha256_file(onnx_path), "precision": precision,
                       "group": group if bits == 4 else 0, "compute": compute,
                       "clip": clip if bits == 4 else "max", "keep8": list(keep8),
                       "plugin_sha256": sha256_file(PLUGIN_LIB),
                       "tactic_sources": ["EDGE_MASK_CONVOLUTIONS", "JIT_CONVOLUTIONS"],
                       "plugin_source_sha256": sha256_file(
                           PROJECT / "tools/g9_intprec/m2d_packed_weight_plugin.cu"),
                       "tensorrt": trt.__version__}
    summary = {
        "schema_version": 1, "status": "PASS", "stage": "g9_intprec", "model_key": name,
        "precision": precision, "weight_bits": bits,
        "weight_group": group if bits == 4 else "per-channel",
        "weight_clip": clip if bits == 4 else "max", "compute": compute,
        "kept_int8_layers": list(keep8),
        "onnx_path": str(onnx_path), "plugin_lib": str(PLUGIN_LIB),
        "surgery": stats,
        "preprocessing": {"mean": meta["mean"], "std": meta["std"],
                          "interpolation": meta["interpolation"], "crop_pct": meta["crop_pct"]},
        "build_identity": hashlib.sha256(json.dumps(identity_source, sort_keys=True)
                                         .encode()).hexdigest(),
        "build_identity_source": identity_source,
        "engine_path": str(engine_path), "engine_sha256": sha256_file(engine_path),
        "engine_size_bytes": len(engine_bytes), "engine_bindings": contract,
        "zero_input_smoke": smoke, "build_elapsed_seconds": time.time() - started,
        "physical_gpu": gpu_identity(physical_gpu), "tensorrt": trt.__version__,
    }
    summary_path = model_dir / f"{precision}_engine_summary.json"
    atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"{name}: {precision}_engine_build=PASS size_bytes={len(engine_bytes)} "
          f"weights={stats['weights']} (conv {stats['conv']}, matmul->conv "
          f"{stats['matmul_as_conv']}) packed={stats['packed_bytes']} "
          f"scales={stats['scale_bytes']} kept_int8={len(keep8)} compute={compute} "
          f"smoke_prob={smoke['prob']:.6f}", flush=True)
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--precision", required=True, choices=sorted(PRECISIONS))
    parser.add_argument("--group", type=int, default=128,
                        help="INT4 weights per scale group inside a channel (w4a8)")
    parser.add_argument("--compute", choices=("fp32", "int8"), default="fp32",
                        help="fp32: expanded weights feed the layer (exact); int8: "
                             "re-quantized per output channel to INT8")
    parser.add_argument("--clip", choices=("mse", "max"), default="mse",
                        help="INT4 group scale: MSE-optimal clipping or max|w|")
    parser.add_argument("--keep8-json", type=Path, default=None,
                        help="int4_sensitivity.py output: layers left INT8")
    parser.add_argument("--device", type=int, required=True, choices=(0, 1, 2))
    args = parser.parse_args()
    for name in args.models:
        build(name, args.precision, args.group, args.device, args.compute, args.clip,
              args.keep8_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
