#!/usr/bin/env python3
"""GPU_M2D G7: clean (no fault) evaluation of all six ImageNet-1k models
on the 10 000-image eval split -- FP32 (torch/timm source of truth) and
INT8 (TensorRT PTQ engine), same preprocessing, same images.

Quantization loss = fp32_top1 - int8_top1 is therefore isolated from
everything else (identical pixels, identical labels, identical top-1
semantics). INT8 acceptance follows the REMU stage13 clean contract:
two full passes whose per-image (prediction, probability) pairs must be
identical.

Preprocessing and the CUDA runtime helpers are imported from
build_g7_int8_engine.py (single source of truth).

Run under the vit_fault python with the TRT wiring (or via
eval_g7_clean.sh which sets it):

  /data1/luojx/miniforge3/envs/vit_fault/bin/python \
      tools/g7_prep/eval_g7_clean.py [--device 0] [--models resnet50 ...]

Outputs into <weights-root>/<name>/eval/:
  fp32_predictions.csv, int8_predictions_run1.csv, int8_predictions_run2.csv,
  clean_summary.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import timm
import torch

from build_g7_int8_engine import (
    CudaRuntime,
    INTERPOLATION_CV,
    INPUT_SHAPE,
    atomic_text,
    preprocess_image,
    sha256_file,
)

try:
    import tensorrt as trt
except ModuleNotFoundError:
    import tensorrt_bindings as trt

WEIGHTS_ROOT = Path("/data1/luojx/g7_models")
EVAL_CSV = Path(
    "/data1/luojx/datasets/imagenet1k/splits/g7_eval_10000_perclass10.csv"
)
MODELS = [
    "resnet50",
    "mobilenetv3_large_100",
    "efficientnet_b0",
    "vit_base_patch16_224",
    "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
]


def evaluation_rows() -> list[tuple[str, int]]:
    with EVAL_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10000 or set(rows[0]) != {"path", "label"}:
        raise RuntimeError("G7 evaluation split contract changed")
    result = [(row["path"], int(row["label"])) for row in rows]
    if any(
        not Path(path).is_file() or label < 0 or label >= 1000
        for path, label in result
    ):
        raise RuntimeError("evaluation input is missing or has an invalid label")
    return result


def valid_output(probability: float, prediction: int) -> bool:
    return bool(
        math.isfinite(probability)
        and 0.0 <= probability <= 1.0
        and 0 <= prediction < 1000
    )


def prediction_csv(rows: list[dict[str, object]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=("sample_id", "image", "target", "prediction",
                    "probability", "valid", "correct"),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        formatted = dict(row)
        formatted["probability"] = f"{float(row['probability']):.9f}"
        formatted["valid"] = int(bool(row["valid"]))
        formatted["correct"] = int(bool(row["correct"]))
        writer.writerow(formatted)
    return output.getvalue()


def fp32_pass(
    name: str,
    rows_in: list[tuple[str, int]],
    mean: np.ndarray,
    std: np.ndarray,
    interpolation: int,
    resize_scale: int,
) -> tuple[list[dict[str, object]], float]:
    classifier = timm.create_model(name, pretrained=False)
    state_dict = torch.load(
        WEIGHTS_ROOT / name / "weights.pth", map_location="cpu", weights_only=True
    )
    classifier.load_state_dict(state_dict, strict=True)
    classifier.cuda().eval()

    rows: list[dict[str, object]] = []
    started = time.time()
    with torch.inference_mode():
        for sample_id, (image_path, target) in enumerate(rows_in):
            host = preprocess_image(image_path, mean, std, interpolation, resize_scale)
            data = torch.from_numpy(host).cuda()
            logits = classifier(data)
            probabilities = torch.softmax(logits, dim=1)
            probability, index = torch.topk(probabilities, k=1, dim=1)
            probability = float(probability.item())
            prediction = int(index.item())
            valid = valid_output(probability, prediction)
            rows.append(
                {
                    "sample_id": sample_id,
                    "image": image_path,
                    "target": target,
                    "prediction": prediction,
                    "probability": probability,
                    "valid": valid,
                    "correct": bool(valid and prediction == target),
                }
            )
    return rows, time.time() - started


def int8_pass_once(
    context,
    bindings: list[int],
    cuda: CudaRuntime,
    rows_in: list[tuple[str, int]],
    mean: np.ndarray,
    std: np.ndarray,
    interpolation: int,
    resize_scale: int = 224,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    probability_host = np.zeros((1,), dtype=np.float32)
    index_host = np.full((1,), -1, dtype=np.int32)
    for sample_id, (image_path, target) in enumerate(rows_in):
        host = preprocess_image(image_path, mean, std, interpolation, resize_scale)
        cuda.copy_host_to_device(bindings[0], host)
        if not context.execute_v2(bindings):
            raise RuntimeError(f"TensorRT execute_v2 returned false at sample {sample_id}")
        cuda.copy_device_to_host(probability_host, bindings[1])
        cuda.copy_device_to_host(index_host, bindings[2])
        probability = float(probability_host[0])
        prediction = int(index_host[0])
        valid = valid_output(probability, prediction)
        rows.append(
            {
                "sample_id": sample_id,
                "image": image_path,
                "target": target,
                "prediction": prediction,
                "probability": probability,
                "valid": valid,
                "correct": bool(valid and prediction == target),
            }
        )
    return rows


def int8_double_pass(
    name: str,
    rows_in: list[tuple[str, int]],
    mean: np.ndarray,
    std: np.ndarray,
    interpolation: int,
    resize_scale: int,
) -> tuple[list[dict[str, object]], float, float]:
    model_dir = WEIGHTS_ROOT / name
    build_summary = json.loads(
        (model_dir / "engine_summary.json").read_text(encoding="utf-8")
    )
    engine_path = Path(build_summary["engine_path"])
    if (
        build_summary.get("status") != "PASS"
        or build_summary.get("model_key") != name
        or sha256_file(engine_path) != build_summary.get("engine_sha256")
    ):
        raise RuntimeError(f"{name}: INT8 engine identity/provenance mismatch")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None or engine.num_bindings != 3:
        raise RuntimeError(f"{name}: INT8 engine does not match the three-binding contract")
    if [engine.get_binding_name(index) for index in range(3)] != ["data", "prob", "index"]:
        raise RuntimeError(f"{name}: INT8 engine binding order changed")
    expected_shapes = [INPUT_SHAPE, (1, 1), (1, 1)]
    if [tuple(engine.get_binding_shape(index)) for index in range(3)] != expected_shapes:
        raise RuntimeError(f"{name}: INT8 engine binding shapes changed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"{name}: INT8 engine context creation failed")

    cuda = CudaRuntime()
    bindings = [cuda.malloc(3 * 224 * 224 * 4), cuda.malloc(4), cuda.malloc(4)]
    try:
        started1 = time.time()
        run1 = int8_pass_once(
            context, bindings, cuda, rows_in, mean, std, interpolation, resize_scale
        )
        elapsed1 = time.time() - started1
        started2 = time.time()
        run2 = int8_pass_once(
            context, bindings, cuda, rows_in, mean, std, interpolation, resize_scale
        )
        elapsed2 = time.time() - started2
    finally:
        for pointer in bindings:
            cuda.free(pointer)

    identical = all(
        left["prediction"] == right["prediction"]
        and left["probability"] == right["probability"]
        for left, right in zip(run1, run2, strict=True)
    )
    if not identical:
        raise RuntimeError(f"{name}: INT8 repeated-evaluation acceptance failed")
    valid1 = sum(bool(row["valid"]) for row in run1)
    if valid1 != len(run1):
        raise RuntimeError(f"{name}: INT8 pass produced {len(run1) - valid1} invalid outputs")
    return run1, elapsed1, elapsed2


def evaluate_one(name: str) -> dict[str, object]:
    model_dir = WEIGHTS_ROOT / name
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    mean = np.asarray(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(meta["std"], dtype=np.float32).reshape(1, 1, 3)
    interpolation = INTERPOLATION_CV[str(meta["interpolation"])]
    resize_scale = int(INPUT_SHAPE[2] // float(meta["crop_pct"]))
    rows_in = evaluation_rows()

    fp32_rows, fp32_elapsed = fp32_pass(
        name, rows_in, mean, std, interpolation, resize_scale
    )
    fp32_top1 = sum(bool(row["correct"]) for row in fp32_rows) / len(fp32_rows)
    fp32_valid = sum(bool(row["valid"]) for row in fp32_rows)
    if fp32_valid != len(fp32_rows):
        raise RuntimeError(f"{name}: FP32 pass produced invalid outputs")

    int8_rows, int8_elapsed1, int8_elapsed2 = int8_double_pass(
        name, rows_in, mean, std, interpolation, resize_scale
    )
    int8_top1 = sum(bool(row["correct"]) for row in int8_rows) / len(int8_rows)
    agreement = sum(
        bool(left["correct"]) == bool(right["correct"])
        for left, right in zip(fp32_rows, int8_rows, strict=True)
    ) / len(fp32_rows)
    prediction_agreement = sum(
        left["prediction"] == right["prediction"]
        for left, right in zip(fp32_rows, int8_rows, strict=True)
    ) / len(fp32_rows)

    eval_dir = model_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fp32_predictions.csv": prediction_csv(fp32_rows),
        "int8_predictions_run1.csv": prediction_csv(int8_rows),
    }
    for filename, content in outputs.items():
        atomic_text(eval_dir / filename, content)

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "g7_imagenet1k",
        "model_key": name,
        "evaluation_manifest": str(EVAL_CSV),
        "evaluation_manifest_sha256": sha256_file(EVAL_CSV),
        "images": len(rows_in),
        "preprocessing": {
            "policy": "canonical_aspect_resize_center_crop_v2, RGB, /255, mean/std "
                      "(identical FP32/INT8)",
            "mean": meta["mean"],
            "std": meta["std"],
            "interpolation": meta["interpolation"],
            "input_size": meta["input_size"],
            "crop_pct": meta["crop_pct"],
            "resize_scale": resize_scale,
        },
        "fp32": {
            "implementation": "torch/timm",
            "top1_accuracy": fp32_top1,
            "elapsed_seconds": fp32_elapsed,
            "predictions_sha256": sha256_file(eval_dir / "fp32_predictions.csv"),
        },
        "int8": {
            "implementation": json.loads(
                (model_dir / "engine_summary.json").read_text(encoding="utf-8")
            ).get("precision", "tensorrt-8.6.1"),
            "top1_accuracy": int8_top1,
            "elapsed_seconds_run1": int8_elapsed1,
            "elapsed_seconds_run2": int8_elapsed2,
            "repeated_identical": True,
            "engine_sha256": json.loads(
                (model_dir / "engine_summary.json").read_text(encoding="utf-8")
            )["engine_sha256"],
            "predictions_sha256": sha256_file(eval_dir / "int8_predictions_run1.csv"),
        },
        "quantization_loss_pp": (fp32_top1 - int8_top1) * 100.0,
        "correctness_agreement": agreement,
        "prediction_agreement": prediction_agreement,
        "software": {
            "torch": torch.__version__,
            "timm": timm.__version__,
            "tensorrt": trt.__version__,
            "opencv": cv2.__version__,
        },
    }
    atomic_text(
        eval_dir / "clean_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"{name}: fp32_top1={fp32_top1:.4f} int8_top1={int8_top1:.4f} "
        f"loss_pp={summary['quantization_loss_pp']:.2f} "
        f"pred_agree={prediction_agreement:.4f} "
        f"fp32_s={fp32_elapsed:.0f} int8_s={int8_elapsed1:.0f}",
        flush=True,
    )
    return summary


def main() -> int:
    global WEIGHTS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--device", type=int, default=0, choices=(0, 1, 2))
    args = parser.parse_args()
    WEIGHTS_ROOT = args.weights_root
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.device):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain exactly the requested physical GPU"
        )
    for name in args.models:
        evaluate_one(name)
    print(f"g7_clean_evaluation=PASS models={len(args.models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
