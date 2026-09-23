#!/usr/bin/env python3
"""GPU_M2D G7: export the six ImageNet-1k timm models to ONNX for the
TensorRT INT8 PTQ build.

Methodology inherited from REMU tests/stage13/export_stage13_onnx.py
(same three-binding host contract the G5 runner consumes):

  data  (1, 3, 224, 224) float32
  prob  (1, 1) float32   -- softmax top-1 probability
  index (1, 1) int32     -- top-1 class

Per-model facts (num_classes, input size, mean/std, interpolation) come
from the model_meta.json written by download_models.py -- the single
source of truth; nothing is hardcoded per model here except the names.

Run under the torch/timm env (vit_fault python), one model or all:

  /data1/luojx/miniforge3/envs/vit_fault/bin/python \
      tools/g7_prep/export_g7_onnx.py [--models resnet50 ...]

Outputs per model into <weights-root>/<timm-name>/:
  model.onnx        the exported graph (opset 17, fixed batch 1)
  onnx_summary.json identity + preprocessing + sha256 provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnx
import timm
import torch
from torch import nn
from onnx import numpy_helper

WEIGHTS_ROOT = Path("/data1/luojx/g7_models")
MODELS = [
    "resnet50",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "vit_base_patch16_224",
    "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
]
OPSET_VERSION = 17


class Top1Classifier(nn.Module):
    """Preserve the G5/stage13 three-binding host contract."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(data)
        probabilities = torch.softmax(logits, dim=1)
        probability, index = torch.topk(probabilities, k=1, dim=1)
        return probability, index.to(torch.int32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def normalize_slice_end_sentinels(model: onnx.ModelProto) -> int:
    """Replace INT64_MAX Slice-end sentinels with a TRT-safe value."""
    slice_end_inputs = {
        node.input[2]
        for node in model.graph.node
        if node.op_type == "Slice" and len(node.input) >= 3
    }
    rewritten = 0
    for node in model.graph.node:
        if node.op_type != "Constant" or not set(node.output) & slice_end_inputs:
            continue
        for attribute in node.attribute:
            if (
                attribute.type != onnx.AttributeProto.TENSOR
                or attribute.t.data_type != onnx.TensorProto.INT64
            ):
                continue
            values = numpy_helper.to_array(attribute.t)
            if values.size and np.all(values == np.iinfo(np.int64).max):
                replacement = np.full(
                    values.shape, np.iinfo(np.int32).max, dtype=np.int64
                )
                attribute.t.CopyFrom(numpy_helper.from_array(replacement))
                rewritten += 1
    return rewritten


def export_one(name: str) -> Path:
    model_dir = WEIGHTS_ROOT / name
    weights_path = model_dir / "weights.pth"
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    if int(meta["num_classes"]) != 1000:
        raise RuntimeError(f"{name}: expected 1000 classes, got {meta['num_classes']}")
    channels, height, width = (int(v) for v in meta["input_size"])
    if (channels, height, width) != (3, 224, 224):
        raise RuntimeError(f"{name}: unexpected input size {meta['input_size']}")
    weights_sha256 = sha256_file(weights_path)

    classifier = timm.create_model(name, pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()
    wrapped = Top1Classifier(classifier).eval()

    temporary = model_dir / f".model.onnx.tmp.{os.getpid()}"
    if temporary.exists():
        temporary.unlink()

    torch.manual_seed(130013)
    example = torch.zeros((1, 3, height, width), dtype=torch.float32)
    started = time.time()
    with torch.inference_mode():
        reference_probability, reference_index = wrapped(example)
        torch.onnx.export(
            wrapped,
            example,
            temporary,
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["data"],
            output_names=["prob", "index"],
            dynamic_axes=None,
            dynamo=False,
        )

    model = onnx.load(temporary, load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    slice_end_sentinel_rewrites = normalize_slice_end_sentinels(model)
    onnx.checker.check_model(model, full_check=True)

    input_info = model.graph.input[0]
    input_tensor = input_info.type.tensor_type
    outputs = {value.name: value for value in model.graph.output}
    if (
        input_info.name != "data"
        or input_tensor.elem_type != onnx.TensorProto.FLOAT
        or [d.dim_value for d in input_tensor.shape.dim] != [1, 3, height, width]
        or set(outputs) != {"prob", "index"}
        or outputs["prob"].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
        or outputs["index"].type.tensor_type.elem_type != onnx.TensorProto.INT32
    ):
        raise RuntimeError(f"{name}: unexpected ONNX interface contract")

    del model.metadata_props[:]
    metadata = {
        "stage": "g7_imagenet1k",
        "model_key": name,
        "weights_sha256": weights_sha256,
        "opset_version": str(OPSET_VERSION),
        "batch_policy": "fixed_explicit_batch_1",
        "slice_end_sentinel_rewrites": str(slice_end_sentinel_rewrites),
        "preprocessing": json.dumps(
            {
                "mean": meta["mean"],
                "std": meta["std"],
                "interpolation": meta["interpolation"],
                "input_size": meta["input_size"],
            },
            sort_keys=True,
        ),
    }
    for key, value in metadata.items():
        property_value = model.metadata_props.add()
        property_value.key = key
        property_value.value = value
    onnx.save(model, temporary, save_as_external_data=False)
    onnx.checker.check_model(onnx.load(temporary), full_check=True)
    output_path = model_dir / "model.onnx"
    temporary.replace(output_path)

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "g7_imagenet1k",
        "model_key": name,
        "weights_path": str(weights_path),
        "weights_sha256": weights_sha256,
        "onnx_path": str(output_path),
        "onnx_sha256": sha256_file(output_path),
        "onnx_size_bytes": output_path.stat().st_size,
        "opset_version": OPSET_VERSION,
        "tensorrt_compatibility_rewrites": {
            "int64_max_slice_end_to_int32_max_same_int64_type": (
                slice_end_sentinel_rewrites
            )
        },
        "interface_contract": {
            "data": [1, 3, height, width],
            "prob": [1, 1],
            "index": [1, 1],
        },
        "preprocessing": {
            "mean": meta["mean"],
            "std": meta["std"],
            "interpolation": meta["interpolation"],
            "input_size": meta["input_size"],
        },
        "reference_zero_input": {
            "top1_probability": float(reference_probability.item()),
            "top1_index": int(reference_index.item()),
        },
        "export_elapsed_seconds": time.time() - started,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "timm": timm.__version__,
            "onnx": onnx.__version__,
        },
    }
    summary_path = model_dir / "onnx_summary.json"
    atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"{name}: onnx_export=PASS size_bytes={summary['onnx_size_bytes']} "
        f"elapsed_seconds={summary['export_elapsed_seconds']:.1f}",
        flush=True,
    )
    return summary_path


def main() -> int:
    global WEIGHTS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    WEIGHTS_ROOT = args.weights_root
    for name in args.models:
        export_one(name)
    print(f"g7_onnx_export=PASS models={len(args.models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
