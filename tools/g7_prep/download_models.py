#!/usr/bin/env python3
"""GPU_M2D G7: download the six ImageNet-1k models (timm) + preprocessing
metadata.

User decisions 2026-09-23: variants = ResNet-50, MobileNetV3-Large,
EfficientNet-B0, ViT-B/16, DeiT-S, Swin-T; download everything once up
front (base work), run the fault-injection campaigns one model at a
time starting with ResNet-50.

For each model this stores under --out/<timm-name>/:
  weights.pth        state_dict (timm ImageNet-1k pretrained)
  model_meta.json    default_cfg (top-1, input_size, mean/std,
                     crop_pct, interpolation) -- the runner's CPU
                     preprocessing will be parameterized from this
                     file, so quantization/inference never hardcode a
                     second source of truth.

Run under an env with torch+timm, e.g.
  /data1/luojx/miniforge3/envs/vit_fault/bin/python \
      tools/g7_prep/download_models.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--out", type=Path,
                        default=Path("/data1/luojx/g7_models"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    import torch
    import timm

    for name in args.models:
        target = args.out / name
        if (target / "weights.pth").is_file() \
                and (target / "model_meta.json").is_file():
            print(f"{name}: already present, skipping")
            continue
        print(f"{name}: creating + downloading pretrained weights ...")
        model = timm.create_model(name, pretrained=True)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target / "weights.pth")
        cfg = model.default_cfg
        meta = {key: (list(value) if isinstance(value, (tuple, list))
                      else str(value))
                for key, value in cfg.items()
                if key in ("url", "input_size", "interpolation",
                           "mean", "std", "crop_pct", "first_conv",
                           "classifier", "num_classes",
                           "license", "original_img_size")}
        (target / "model_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params/1e6:.1f}M params, "
              f"input {meta['input_size']}, "
              f"top1 {meta.get('license') and ''}"
              f"saved to {target}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
