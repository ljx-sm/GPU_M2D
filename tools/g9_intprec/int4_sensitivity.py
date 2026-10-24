#!/usr/bin/env python3
"""G9 integer-precision study: choose which ResNet-50 layers stay INT8 in
the INT4 engine (user decision 2026-10-23: mixed precision, "INT4
weights, sensitive layers in INT8").

Uniform INT4 (group 128, MSE clipping) collapses this checkpoint (timm
resnet50 A1). On the 1,000-image CALIBRATION split (disjoint from the 10K
evaluation split, so the choice is not tuned on test images) and the
BN-folded PyTorch model (the folding the ONNX export carries), each
Conv/Linear is quantized to INT4 alone -- with the engine builder's own
quantizer (build_intprec_engine.quantize) -- and ranked by the top-1 drop.
Then the k most sensitive layers stay INT8 (per-channel) and the rest go
INT4; the chosen k is the smallest whose calibration top-1 is within
--max-drop pp of the all-INT8 model.

Writes artifacts/g9_intprec/int4_keep8_<model>.json: the ranking, the
k sweep, and the kept layers under their TensorRT layer names (the names
build_intprec_engine.py --keep8-json matches).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import timm

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools/g7_prep"))
sys.path.insert(0, str(PROJECT / "tools/g9_intprec"))
from build_g7_int8_engine import INPUT_SHAPE, INTERPOLATION_CV, preprocess_image  # noqa: E402
from build_intprec_engine import dequantize, quantize  # noqa: E402

CALIB_CSV = Path("/data1/luojx/datasets/imagenet1k/splits/g7_calib_1000_perclass1.csv")


def fold_bn(model):
    model = copy.deepcopy(model)
    mods = dict(model.named_modules())
    for name, bn in mods.items():
        if not isinstance(bn, torch.nn.BatchNorm2d):
            continue
        conv = mods.get(name.replace("bn", "conv") if "bn" in name else name[:-1] + "0")
        if not isinstance(conv, torch.nn.Conv2d):
            raise RuntimeError(f"{name}: no conv to fold into")
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.data = conv.weight.data * s[:, None, None, None]
        bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.bias)
        conv.bias = torch.nn.Parameter(bn.bias + (bias - bn.running_mean) * s)
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1)
        bn.eps = 0.0
    return model


def trt_name(module: str) -> str:
    """timm module path -> the TensorRT layer name of its Conv/Gemm."""
    if module == "fc":
        return "/classifier/fc/Gemm"
    segments = []
    for token in module.split("."):
        if token.isdigit():
            segments[-1] = f"{segments[-1]}.{token}"
            segments.append(segments[-1])
            segments[-2] = segments[-2].rsplit(".", 1)[0]
        else:
            segments.append(token)
    return "/classifier/" + "/".join(segments) + "/Conv"


def int4(w: torch.Tensor, group: int) -> torch.Tensor:
    arr = w.detach().cpu().numpy().astype(np.float32)
    q, s, g = quantize(arr, 4, group, "mse")
    return torch.from_numpy(dequantize(q, s, g)).to(w.device)


def int8(w: torch.Tensor) -> torch.Tensor:
    arr = w.detach().cpu().numpy().astype(np.float32)
    q, s, g = quantize(arr, 8, 0, "max")
    return torch.from_numpy(dequantize(q, s, g)).to(w.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["resnet50"])
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--max-drop", type=float, default=3.0)
    args = parser.parse_args()
    md = Path("/data1/luojx/g7_models") / args.model
    meta = json.loads((md / "model_meta.json").read_text())
    model = timm.create_model(args.model, pretrained=False, num_classes=1000)
    model.load_state_dict(torch.load(md / "weights.pth", map_location="cpu", weights_only=True))
    model = fold_bn(model.eval()).cuda()
    mean = np.asarray(meta["mean"], np.float32).reshape(1, 1, 3)
    std = np.asarray(meta["std"], np.float32).reshape(1, 1, 3)
    interp = INTERPOLATION_CV[str(meta["interpolation"])]
    resize = int(INPUT_SHAPE[2] // float(meta["crop_pct"]))
    rows = [(r["path"], int(r["label"])) for r in csv.DictReader(CALIB_CSV.open())]
    x = torch.from_numpy(np.concatenate(
        [preprocess_image(p, mean, std, interp, resize).reshape(1, 3, 224, 224) for p, _ in rows]))
    y = torch.tensor([t for _, t in rows])

    @torch.no_grad()
    def top1(m) -> float:
        hits = 0
        for i in range(0, len(x), 100):
            hits += int((m(x[i:i + 100].cuda()).argmax(1).cpu() == y[i:i + 100]).sum())
        return 100.0 * hits / len(x)

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    q4 = {n: int4(m.weight, args.group) for n, m in layers}
    q8 = {n: int8(m.weight) for n, m in layers}
    fp32 = top1(model)
    orig = {n: m.weight.data.clone() for n, m in layers}
    for n, m in layers:
        m.weight.data = q8[n]
    all8 = top1(model)
    ranking = []
    for n, m in layers:
        m.weight.data = q4[n]
        ranking.append({"layer": n, "trt_name": trt_name(n), "params": m.weight.numel(),
                        "drop_pp": all8 - top1(model)})
        m.weight.data = q8[n]
    ranking.sort(key=lambda r: -r["drop_pp"])
    total = sum(r["params"] for r in ranking)
    sweep, chosen = [], None
    for k in range(0, len(ranking) + 1):
        keep = {r["layer"] for r in ranking[:k]}
        for n, m in layers:
            m.weight.data = q8[n] if n in keep else q4[n]
        acc = top1(model)
        share = sum(r["params"] for r in ranking[:k]) / total
        sweep.append({"k": k, "top1": acc, "int8_weight_share": share})
        print(f"k={k:2d} keep INT8 {100 * share:5.1f}% of weights: top-1 {acc:5.1f}% "
              f"(all INT8 {all8:.1f}%)", flush=True)
        if acc >= all8 - args.max_drop:
            chosen = k
            break
    for n, m in layers:
        m.weight.data = orig[n]
    if chosen is None:
        raise SystemExit("no k reaches the accuracy target")
    out = {"model": args.model, "split": str(CALIB_CSV), "images": len(rows),
           "group": args.group, "clip": "mse", "fp32_top1": fp32, "all_int8_top1": all8,
           "max_drop_pp": args.max_drop, "k": chosen,
           "keep8_trt_names": [r["trt_name"] for r in ranking[:chosen]],
           "keep8_layers": [r["layer"] for r in ranking[:chosen]],
           "ranking": ranking, "sweep": sweep}
    path = PROJECT / f"artifacts/g9_intprec/int4_keep8_{args.model}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"chosen k={chosen}: {out['keep8_layers']} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
