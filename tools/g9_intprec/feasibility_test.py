#!/usr/bin/env python3
"""G9 integer-precision feasibility test (user decision 2026-10-23).

Can TensorRT 8.6.1 run a layer whose weights are stored PACKED as INT16 or
INT4 (M2DPackedWeight plugin) while the activations stay INT8 (Q/DQ), as
the G7-v2 INT8 engines do? Two real layers, read from the unquantized
ONNX exports (read only):
  ResNet-50 layer1.0.conv2  Conv 64x64x3x3   (onnx::Conv_510)
  ViT-B blocks.0.attn.qkv   MatMul 768x2304  (onnx::MatMul_1104), built as a
                            1x1 convolution over the 197 tokens -- the form
                            TensorRT itself gives the constant-weight INT8
                            MatMul (its INT8 MatMul has no kernel for
                            per-channel runtime weights)
Three engines per layer, activations INT8 (Q/DQ, per-tensor) in all:
  W8A8   constant weights -> Q/DQ per-channel (the campaign engine path)
  W4A8   plugin(INT4)  -> Q/DQ per-channel with the INT4 scales
  W16A8  plugin(INT16) -> the layer directly
Checks per engine: (1) it builds; (2) TensorRT's per-layer precision
(engine inspector); (3) the plugin's packed buffer is allocated through
the runtime's IGpuAllocator (a tracking allocator, as in the G1.5
runner); (4) the output against a float64 NumPy reference of the same
quantized math, and the quantization error against unquantized FP32.

Every PluginField array is held in a variable until the plugin is
created: the Python PluginField does not own its array, and a freed
temporary corrupts the plugin's copy (seen as one wrong output channel).

Writes artifacts/g9_intprec/feasibility/{report.json,<case>_inspector.json}.
Run on an idle card other than the campaign GPU:
  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=<trt>:<cudnn>:<cuda> \\
      /data1/luojx/miniforge3/envs/vit_fault/bin/python tools/g9_intprec/feasibility_test.py
"""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools/g7_prep"))
from build_g7_int8_engine import CudaRuntime  # noqa: E402

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

OUT = PROJECT / "artifacts/g9_intprec/feasibility"
PLUGIN = PROJECT / "artifacts/g9_intprec/libm2d_packed_weight.so"
LAYERS = {
    "resnet50_conv": ("resnet50", "onnx::Conv_510", "conv"),
    "vit_qkv": ("vit_base_patch16_224", "onnx::MatMul_1104", "linear"),
}


def quantize(w: np.ndarray, bits: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel symmetric: q in [-(2^(b-1)), 2^(b-1)-1], scale = max|w| / qmax."""
    qmax = 2 ** (bits - 1) - 1
    other = tuple(i for i in range(w.ndim) if i != axis)
    scale = np.abs(w).max(axis=other) / qmax
    scale[scale == 0] = 1.0
    shape = [1] * w.ndim
    shape[axis] = -1
    q = np.clip(np.round(w / scale.reshape(shape)), -qmax - 1, qmax).astype(np.int32)
    return q, scale.astype(np.float32)


def dequant(q: np.ndarray, scale: np.ndarray, axis: int) -> np.ndarray:
    shape = [1] * q.ndim
    shape[axis] = -1
    return q.astype(np.float64) * scale.reshape(shape).astype(np.float64)


def pack(q: np.ndarray, bits: int) -> np.ndarray:
    flat = q.reshape(-1)
    if bits == 16:
        return flat.astype("<i2").view(np.uint8)
    nib = (flat & 0xF).astype(np.uint8)
    if nib.size % 2:
        nib = np.append(nib, np.uint8(0))
    return (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8)


def conv2d(x: np.ndarray, w: np.ndarray, pad: int) -> np.ndarray:
    n, c, h, wd = x.shape
    k = w.shape[2]
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.stack([xp[:, :, i:i + h, j:j + wd] for i in range(k) for j in range(k)],
                    axis=2)  # n, c, k*k, h, w
    return np.einsum("nckhw,ock->nohw", cols, w.reshape(w.shape[0], c, k * k))


class TrackingAllocator(trt.IGpuAllocator):
    def __init__(self, cuda: CudaRuntime):
        trt.IGpuAllocator.__init__(self)
        self.cuda = cuda
        self.live: dict[int, int] = {}
        self.log: list[int] = []

    def allocate(self, size, alignment, flags):
        ptr = self.cuda.malloc(max(int(size), 1))
        self.live[ptr] = int(size)
        self.log.append(int(size))
        return ptr

    def deallocate(self, memory):
        if memory in self.live:
            self.cuda.free(memory)
            del self.live[memory]
        return True

    def free(self, memory):
        self.deallocate(memory)


def build(case: str, kind: str, variant: str, x: np.ndarray, w: np.ndarray,
          sx: float, logger) -> tuple[bytes, dict]:
    axis = 0  # output channel: Conv [O, I, kh, kw], linear as 1x1 conv [N, K, 1, 1]
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    xin = net.add_input("x", trt.float32, x.shape)

    def const(arr):
        return net.add_constant(arr.shape, np.ascontiguousarray(arr)).get_output(0)

    def qdq(t, scale, ax=None):
        q = net.add_quantize(t, const(scale))
        dq = net.add_dequantize(q.get_output(0), const(scale))
        if ax is not None:
            q.axis = ax
            dq.axis = ax
        return dq.get_output(0)

    xdq = qdq(xin, np.array([sx], np.float32))
    meta = {}
    if variant == "W8A8":
        q8, s8 = quantize(w, 8, axis)
        wt = qdq(const(w.astype(np.float32)), s8, axis)
        meta["w_ref"] = dequant(q8, s8, axis)
    else:
        bits = 4 if variant == "W4A8" else 16
        q, s = quantize(w, bits, axis)
        packed = pack(q, bits)
        creator = trt.get_plugin_registry().get_plugin_creator("M2DPackedWeight", "1", "")
        # held until create_plugin returns: PluginField does not own its array
        held = [np.array([bits], np.int32), np.array(w.shape, np.int32),
                np.array([axis], np.int32), np.ascontiguousarray(s),
                np.ascontiguousarray(packed.view(np.int8))]
        fields = trt.PluginFieldCollection([
            trt.PluginField("bits", held[0], trt.PluginFieldType.INT32),
            trt.PluginField("dims", held[1], trt.PluginFieldType.INT32),
            trt.PluginField("axis", held[2], trt.PluginFieldType.INT32),
            trt.PluginField("scales", held[3], trt.PluginFieldType.FLOAT32),
            trt.PluginField("packed", held[4], trt.PluginFieldType.INT8),
        ])
        plugin = creator.create_plugin(f"{case}_{variant}", fields)
        wt = net.add_plugin_v2([xin], plugin).get_output(0)
        if variant == "W4A8":
            wt = qdq(wt, s, axis)
        meta["w_ref"] = dequant(q, s, axis)
        meta["packed_bytes"] = int(packed.size)
        meta["plugin_buffer_bytes"] = int((s.size * 4 + 255) // 256 * 256 + packed.size)
    if kind == "conv":
        layer = net.add_convolution_nd(xdq, w.shape[0], (3, 3), trt.Weights(), trt.Weights())
        layer.padding_nd = (1, 1)
        layer.set_input(1, wt)
        out = layer.get_output(0)
    else:  # tokens [1, T, K] -> [1, K, T, 1] -> 1x1 conv -> [1, T, N]
        n_out, k_in, t = w.shape[0], w.shape[1], x.shape[1]
        sh = net.add_shuffle(xdq)
        sh.reshape_dims = (1, t, k_in, 1)
        sh.second_transpose = (0, 2, 1, 3)
        layer = net.add_convolution_nd(sh.get_output(0), n_out, (1, 1), trt.Weights(),
                                       trt.Weights())
        layer.set_input(1, wt)
        back = net.add_shuffle(layer.get_output(0))
        back.first_transpose = (0, 2, 1, 3)
        back.reshape_dims = (1, t, n_out)
        out = back.get_output(0)
    net.mark_output(out)
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.clear_flag(trt.BuilderFlag.TF32)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError(f"{case} {variant}: build failed")
    return bytes(blob), meta


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ctypes.CDLL(str(PLUGIN), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    cuda = CudaRuntime()
    rng = np.random.default_rng(7)
    report = {}
    for case, (model, init_name, kind) in LAYERS.items():
        g = onnx.load(f"/data1/luojx/g7_models/{model}/model.onnx")
        w = numpy_helper.to_array({i.name: i for i in g.graph.initializer}[init_name]
                                  ).astype(np.float32)
        if kind == "conv":
            x = np.maximum(rng.standard_normal((1, w.shape[1], 56, 56)), 0).astype(np.float32)
        else:
            x = rng.standard_normal((1, 197, w.shape[0])).astype(np.float32)
            fp32_w = w
            w = np.ascontiguousarray(w.T.reshape(w.shape[1], w.shape[0], 1, 1))
        sx = float(np.abs(x).max() / 127)
        xq = np.clip(np.round(x / sx), -128, 127) * sx
        fp32_ref = conv2d(x.astype(np.float64), w.astype(np.float64), 1) if kind == "conv" \
            else x.astype(np.float64) @ fp32_w.astype(np.float64)
        for variant in ("W8A8", "W4A8", "W16A8"):
            key = f"{case}_{variant}"
            entry = {"model": model, "weight": init_name, "shape": list(w.shape)}
            try:
                blob, meta = build(case, kind, variant, x, w, sx, logger)
            except Exception as exc:  # report, keep going
                entry["status"] = f"BUILD_FAILED: {exc}"
                report[key] = entry
                print(key, entry["status"], flush=True)
                continue
            alloc = TrackingAllocator(cuda)
            runtime = trt.Runtime(logger)
            runtime.gpu_allocator = alloc
            engine = runtime.deserialize_cuda_engine(blob)
            inspector = engine.create_engine_inspector()
            info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
            (OUT / f"{key}_inspector.json").write_text(json.dumps(info, indent=1))
            ctx = engine.create_execution_context()
            out_shape = tuple(ctx.get_tensor_shape(engine.get_tensor_name(1)))
            dx, dy = cuda.malloc(x.nbytes), cuda.malloc(int(np.prod(out_shape)) * 4)
            cuda.copy_host_to_device(dx, x)
            y = np.zeros(out_shape, np.float32)
            ctx.execute_v2([dx, dy])
            cuda.copy_device_to_host(y, dy)
            ctx.execute_v2([dx, dy])
            y2 = np.zeros_like(y)
            cuda.copy_device_to_host(y2, dy)
            cuda.free(dx)
            cuda.free(dy)
            ref = conv2d(xq, meta["w_ref"], 1) if kind == "conv" \
                else xq @ meta["w_ref"][:, :, 0, 0].T
            scale = float(np.abs(ref).max())
            layers = [{"name": l.get("Name"), "type": l.get("LayerType"),
                       "tactic": l.get("TacticName"),
                       "inputs": [i.get("Format/Datatype") for i in l.get("Inputs", [])],
                       "outputs": [o.get("Format/Datatype") for o in l.get("Outputs", [])]}
                      for l in info.get("Layers", [])]
            entry.update({
                "status": "PASS",
                "layers": layers,
                "allocations_via_allocator": alloc.log,
                "plugin_buffer_bytes": meta.get("plugin_buffer_bytes"),
                "plugin_buffer_seen": (meta.get("plugin_buffer_bytes") in alloc.log
                                       if "plugin_buffer_bytes" in meta else None),
                "deterministic": bool(np.array_equal(y, y2)),
                "max_rel_err_vs_same_quantized_math": float(np.abs(y - ref).max() / scale),
                "quant_err_vs_fp32_rel": float(np.abs(ref - fp32_ref).max() /
                                               np.abs(fp32_ref).max()),
                "finite": bool(np.isfinite(y).all()),
            })
            del ctx, engine
            report[key] = entry
            print(f"{key:22s} {entry['status']}  rel.err {entry['max_rel_err_vs_same_quantized_math']:.2e}"
                  f"  quant.err vs FP32 {entry['quant_err_vs_fp32_rel']:.2e}"
                  f"  plugin buffer via allocator: {entry['plugin_buffer_seen']}"
                  f"  deterministic {entry['deterministic']}", flush=True)
            for l in layers:
                print(f"     {str(l['type']):18s} in {l['inputs']} -> out {l['outputs']}  {l['tactic']}")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
