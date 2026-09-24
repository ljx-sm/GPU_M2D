#!/usr/bin/env python3
"""G7 diagnostic: FP32 top-1 under timm-CANONICAL preprocessing
(Resize(crop_pct) + CenterCrop, PIL antialias, per-model interpolation)
on the same 10K eval split -- quantifies how much of the FP32 gap to
paper numbers is explained by our square-resize policy.

Pure reference measurement (torch/timm); not the G7 pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import timm
import torch

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    from torchvision import transforms as tvt
    from PIL import Image

    with EVAL_CSV.open(newline="", encoding="utf-8") as handle:
        rows = [(r["path"], int(r["label"])) for r in csv.DictReader(handle)]
    if len(rows) != 10000:
        raise RuntimeError("eval split contract changed")

    for name in args.models:
        model_dir = WEIGHTS_ROOT / name
        meta = json.loads((model_dir / "model_meta.json").read_text())
        # timm canonical val transform, exactly as timm.create_transform(224)
        interpolation = {
            "bicubic": tvt.InterpolationMode.BICUBIC,
            "bilinear": tvt.InterpolationMode.BILINEAR,
        }[str(meta["interpolation"])]
        _, height, width = (int(v) for v in meta["input_size"])
        scale = int(round(height / float(meta["crop_pct"])))
        transform = tvt.Compose(
            [
                tvt.Resize(scale, interpolation=interpolation),
                tvt.CenterCrop((height, width)),
                tvt.PILToTensor(),
                tvt.ConvertImageDtype(torch.float32),
                tvt.Normalize(
                    mean=[float(v) for v in meta["mean"]],
                    std=[float(v) for v in meta["std"]],
                ),
            ]
        )
        classifier = timm.create_model(name, pretrained=False)
        classifier.load_state_dict(
            torch.load(model_dir / "weights.pth", map_location="cpu",
                       weights_only=True),
            strict=True,
        )
        classifier.cuda().eval()

        correct = 0
        started = time.time()
        with torch.inference_mode():
            for image_path, target in rows:
                image = Image.open(image_path).convert("RGB")
                data = transform(image).unsqueeze(0).cuda()
                prediction = int(classifier(data).argmax(dim=1).item())
                correct += int(prediction == target)
        top1 = correct / len(rows)
        print(
            f"{name}: canonical_fp32_top1={top1:.4f} "
            f"resize_scale={scale} interp={meta['interpolation']} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
