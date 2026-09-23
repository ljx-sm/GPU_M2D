#!/usr/bin/env python3
"""GPU_M2D G7: build the ImageNet-1K calibration and evaluation splits.

User decisions 2026-09-23 (docs: plan doc G7-T0):

  - calibration set: 1000 classes x 1 image  = 1000 images, drawn from
    the prepared val tree (PTQ only -- no train-set access needed);
  - evaluation set:   1000 classes x 10 images = 10 000 images;
  - the two sets MUST be disjoint (checked, fail-closed);
  - both are deterministic under --seed (recorded in the manifest).

Label integrity: the prepared tree /data1/luojx/datasets/imagenet1k
already sorts val images into 1000 wnid class directories with
class_to_idx.json and the official ILSVRC val_map.txt ground truth.
The builder derives each image's label from the class_to_idx mapping of
its directory wnid and CROSS-CHECKS it against val_map.txt -- any
mismatch refuses (fail-closed), so the split never silently inherits a
preparation error.

Output (RESISC45 split convention, runner contract: "path,label" CSV
with absolute paths and integer labels):
  <out>/g7_calib_1000_perclass1.csv
  <out>/g7_eval_10000_perclass10.csv
  <out>/g7_splits_manifest.json

Run: python3 tools/g7_prep/build_imagenet_splits.py [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path("/data1/luojx/datasets/imagenet1k")
CALIB_PER_CLASS = 1
EVAL_PER_CLASS = 10
VAL_PER_CLASS_EXPECTED = 50  # ILSVRC2012 val: exactly 50 per class


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    val_dir = args.root / "val"
    class_to_idx = json.loads((args.root / "class_to_idx.json").read_text())
    val_map: dict[str, int] = {}
    for line in (args.root / "val_map.txt").read_text().splitlines():
        name, label = line.split("\t")
        val_map[name] = int(label)

    wnids = sorted(class_to_idx)
    if len(wnids) != 1000:
        raise SystemExit(f"expected 1000 classes, got {len(wnids)}")
    labels = [class_to_idx[w] for w in wnids]
    if sorted(labels) != list(range(1000)):
        raise SystemExit("class_to_idx is not a bijection onto 0..999")

    rng = random.Random(args.seed)
    calib: list[tuple[Path, int]] = []
    eval_: list[tuple[Path, int]] = []
    checked = 0
    for wnid in wnids:
        label = class_to_idx[wnid]
        images = sorted((val_dir / wnid).glob("*"))
        if len(images) != VAL_PER_CLASS_EXPECTED:
            raise SystemExit(f"{wnid}: {len(images)} images, expected "
                             f"{VAL_PER_CLASS_EXPECTED}")
        for image in images:
            official = val_map.get(image.name)
            if official != label:
                raise SystemExit(f"label mismatch: {image} dir-label "
                                 f"{label} != val_map {official}")
            checked += 1
        picked = rng.sample(images, CALIB_PER_CLASS + EVAL_PER_CLASS)
        calib.extend((p, label) for p in picked[:CALIB_PER_CLASS])
        eval_.extend((p, label) for p in picked[CALIB_PER_CLASS:])

    # disjointness is structural (rng.sample without replacement within a
    # class) but verify it globally anyway -- fail-closed, not trusted
    calib_names = {p.name for p, _ in calib}
    eval_names = {p.name for p, _ in eval_}
    if calib_names & eval_names:
        raise SystemExit("calibration/evaluation sets intersect")

    out_dir = args.root / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("g7_calib_1000_perclass1.csv", calib),
                       ("g7_eval_10000_perclass10.csv", eval_)):
        with (out_dir / name).open("w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["path", "label"])
            writer.writerows((str(p), label) for p, label in rows)

    calib_hist = Counter(label for _, label in calib)
    eval_hist = Counter(label for _, label in eval_)
    manifest = {
        "seed": args.seed,
        "source_root": str(args.root),
        "label_cross_check": "every image's directory wnid label "
                             "cross-checked against val_map.txt "
                             f"({checked} images, 0 mismatches)",
        "calib": {"file": "g7_calib_1000_perclass1.csv", "images": len(calib),
                  "per_class": dict(Counter(calib_hist.values()))},
        "eval": {"file": "g7_eval_10000_perclass10.csv", "images": len(eval_),
                 "per_class": dict(Counter(eval_hist.values()))},
        "disjoint": True,
    }
    (out_dir / "g7_splits_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"cross-checked {checked} val images against val_map.txt: 0 "
          "mismatches")
    print(f"calib: {len(calib)} images "
          f"({dict(Counter(calib_hist.values()))} per class), "
          f"eval: {len(eval_)} images "
          f"({dict(Counter(eval_hist.values()))} per class), disjoint: True")
    print(f"wrote {out_dir}/g7_calib_1000_perclass1.csv, "
          f"g7_eval_10000_perclass10.csv, g7_splits_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
