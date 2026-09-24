#!/usr/bin/env python3
"""GPU_M2D G7: dump the canonical-preprocessed calibration tensor for the
explicit Q/DQ (ModelOpt ONNX PTQ) path.

Feeds the SAME 1000-image calibration set through the SAME
preprocess_image() (single source of truth in build_g7_int8_engine.py) and
stores it as one float32 array (1000, 3, 224, 224) -- the input contract of
modelopt.onnx.quantization's --calibration_data.  Per model: mean/std/
interpolation/resize_scale differ, so does the array.

Run under the vit_fault python (has cv2 + numpy); no GPU needed:

  /data1/luojx/miniforge3/envs/vit_fault/bin/python \
      tools/g7_prep/dump_g7_calib_npy.py [--models resnet50 ...]

Output: <weights-root>/<name>/calib_canonical.npy (+ sha256 in
calib_canonical.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_g7_int8_engine import (
    CALIBRATION_CSV,
    INPUT_SHAPE,
    INTERPOLATION_CV,
    MODELS,
    WEIGHTS_ROOT,
    atomic_text,
    preprocess_image,
    read_calibration_paths,
    sha256_file,
)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    """np.save into a temp file handle, then rename (no .npy auto-suffix)."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    global WEIGHTS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=WEIGHTS_ROOT)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    WEIGHTS_ROOT = args.weights_root

    image_paths = read_calibration_paths()
    for name in args.models:
        model_dir = WEIGHTS_ROOT / name
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        mean = np.asarray(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(meta["std"], dtype=np.float32).reshape(1, 1, 3)
        interpolation = INTERPOLATION_CV[str(meta["interpolation"])]
        resize_scale = int(INPUT_SHAPE[2] // float(meta["crop_pct"]))

        started = time.time()
        batch = np.empty((len(image_paths), *INPUT_SHAPE[1:]), dtype=np.float32)
        for index, image_path in enumerate(image_paths):
            batch[index] = preprocess_image(
                image_path, mean, std, interpolation, resize_scale
            )[0]
        array_path = model_dir / "calib_canonical.npy"
        atomic_npy(array_path, batch)
        summary = {
            "model_key": name,
            "calibration_manifest": str(CALIBRATION_CSV),
            "images": len(image_paths),
            "shape": list(batch.shape),
            "dtype": "float32",
            "resize_scale": resize_scale,
            "interpolation": str(meta["interpolation"]),
            "mean": meta["mean"],
            "std": meta["std"],
            "npy_sha256": sha256_file(array_path),
            "npy_bytes": array_path.stat().st_size,
            "elapsed_seconds": time.time() - started,
        }
        atomic_text(
            model_dir / "calib_canonical.json",
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
        finite = bool(np.isfinite(batch).all())
        print(
            f"{name}: calib_npy=PASS shape={batch.shape} resize_scale={resize_scale} "
            f"finite={finite} bytes={summary['npy_bytes']} "
            f"elapsed_s={summary['elapsed_seconds']:.0f}",
            flush=True,
        )
        if not finite:
            raise RuntimeError(f"{name}: calibration tensor contains non-finite values")
    print(f"g7_calib_npy=PASS models={len(args.models)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
