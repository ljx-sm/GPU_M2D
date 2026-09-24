#!/usr/bin/env python3
"""GPU_M2D G7: bit-exact preprocessing parity check -- C++ runner vs the
python canonical contract.

The G5/G7 campaign runner (apps/resnet50_int8_g1_5.cpp --preprocess
canonical) must reproduce build_g7_int8_engine.preprocess_image
BIT-IDENTICALLY: both call the same OpenCV C++ routines, so any bit
difference means the C++ port drifted from the contract (wrong rounding
mode, dtype promotion, arithmetic order, or interpolation flag). A
bit-exact clean pass is what makes the campaign baseline equal the
recorded INT8 clean eval (78.50% for resnet50).

For each tested image the driver runs the runner in --dump-preprocessed
mode (pure CPU: exits after preprocessing, before any engine load or
CUDA call) and compares the float32 tensor against the python contract.

Index selection: --spread N evenly spaced rows PLUS --smallest N rows
with the smallest min-side (exercises INTER_AREA downscale, the model's
own upscale path, and the odd-difference round() crop offsets).

Run under the vit_fault python with the TRT LD_LIBRARY_PATH wiring.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from build_g7_int8_engine import (  # noqa: E402
    INTERPOLATION_CV,
    INPUT_SHAPE,
    WEIGHTS_ROOT,
    preprocess_image,
)


def smallest_side_indices(csv_path: Path, count: int) -> list[int]:
    """Rows whose image has the smallest min(width, height) -- header-only
    PIL decode (no pixel load)."""
    from PIL import Image  # torchvision env dependency

    rows = [line.split(",", 1) for line in
            csv_path.read_text(encoding="utf-8").splitlines()[1:] if line]
    sides: list[tuple[int, int]] = []
    for index, (path, _label) in enumerate(rows):
        with Image.open(path) as image:
            sides.append((min(image.size), index))
    sides.sort()
    picked = sorted(index for _side, index in sides[:count])
    print(f"smallest min-side images: "
          f"{[(side, index) for side, index in sides[:count]]}")
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--runner", type=Path,
                        default=PROJECT / "build-g1.5/gpu_m2d_resnet50_int8_g1_5")
    parser.add_argument("--sample-csv", type=Path,
                        default=Path("/data1/luojx/datasets/imagenet1k/splits/"
                                     "g7_eval_10000_perclass10.csv"))
    parser.add_argument("--spread", type=int, default=28,
                        help="evenly spaced row indices to test")
    parser.add_argument("--smallest", type=int, default=4,
                        help="extra smallest-min-side rows to test")
    parser.add_argument("--extra", default="",
                        help="comma-separated extra row indices")
    args = parser.parse_args()

    meta = json.loads((WEIGHTS_ROOT / args.model / "model_meta.json")
                      .read_text(encoding="utf-8"))
    mean = np.asarray(meta["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(meta["std"], dtype=np.float32).reshape(1, 1, 3)
    interpolation = INTERPOLATION_CV[meta["interpolation"]]
    resize_scale = int(INPUT_SHAPE[2] // float(meta["crop_pct"]))

    rows = [line.split(",", 1) for line in
            args.sample_csv.read_text(encoding="utf-8").splitlines()[1:]
            if line]
    indices = list(range(0, len(rows), max(1, len(rows) // max(1, args.spread))))
    if args.smallest:
        for index in smallest_side_indices(args.sample_csv, args.smallest):
            if index not in indices:
                indices.append(index)
    for token in filter(None, args.extra.split(",")):
        index = int(token)
        if index not in indices:
            indices.append(index)
    indices.sort()

    remu_root = Path("/data1/luojx/REMU")
    library_path = ":".join(str(part) for part in (
        remu_root / ".local/deps/tensorrt-8.6.1/tensorrt_libs",
        remu_root / ".local/deps/conda/lib",
        "") if part)
    import os
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = library_path

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for index in indices:
            path, _label = rows[index]
            dump = Path(tmp) / f"cpp_{index}.bin"
            command = [
                str(args.runner),
                "--engine", "/dev/null",  # dump mode exits before the read
                "--sample-csv", str(args.sample_csv),
                "--sample-index", str(index),
                "--output-prefix", str(Path(tmp) / "unused"),
                "--class-count", "1000",
                "--preprocess", "canonical",
                "--resize-scale", str(resize_scale),
                "--interp", meta["interpolation"],
                "--mean", ",".join(str(value) for value in meta["mean"]),
                "--std", ",".join(str(value) for value in meta["std"]),
                "--dump-preprocessed", str(dump),
            ]
            result = subprocess.run(command, capture_output=True, text=True,
                                    env=environment, check=False)
            if result.returncode != 0 or not dump.is_file():
                print(f"row {index}: runner failed: {result.stdout.strip()} "
                      f"{result.stderr.strip()}")
                failures += 1
                continue
            cpp = np.fromfile(dump, dtype=np.float32)
            reference = preprocess_image(path, mean, std, interpolation,
                                         resize_scale)
            expected = reference.reshape(-1)
            if cpp.shape != expected.shape:
                print(f"row {index}: shape {cpp.shape} != {expected.shape}")
                failures += 1
                continue
            if np.array_equal(cpp, expected):
                print(f"row {index}: BITWISE_MATCH {path}")
            else:
                differing = int(np.count_nonzero(cpp != expected))
                max_abs = float(np.max(np.abs(cpp - expected)))
                print(f"row {index}: MISMATCH {path} differing={differing} "
                      f"max_abs={max_abs:.3e}")
                failures += 1

    if failures:
        print(f"preprocess parity: FAIL ({failures}/{len(indices)} images)")
        return 1
    print(f"preprocess parity: PASS ({len(indices)} images bit-identical, "
          f"resize_scale={resize_scale}, interp={meta['interpolation']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
