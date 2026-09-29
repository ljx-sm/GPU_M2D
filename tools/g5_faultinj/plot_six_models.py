#!/usr/bin/env python3
"""GPU_M2D G7-v2 unified figure: top-1 accuracy vs BER, six models.

One paper-ready panel, deliberately minimal (user spec 2026-09-29):
45-85 % linear y axis, 1e-7 .. 1e-5 log x axis at the seven frozen BERs,
one curve per model (five-model extension ladder L1-L7 plus the
ResNet-50 v2 baseline at the SAME BERs -- its nine-level ladder shares
these seven as L3-L9, matched by BER value, never by level name).
No title, no annotations, no value labels, no baseline rules, no error
bars (CIs live in the per-model points CSVs and RESULTS.md) -- only the
axes, the six curves, and the legend.

Every plotted number is DERIVED at run time by the same collect() used
by plot_accuracy_curve.py (nothing hand-typed); the figure's plotted
rows are also written to a points CSV for the paper's data-availability
statement.

Run under an env with matplotlib, e.g.
  /data1/luojx/miniforge3/envs/vit_fault/bin/python \\
      tools/g5_faultinj/plot_six_models.py
Writes artifacts/g7/campaign/fig_six_models/fig_six_models_accuracy_
vs_ber.{png,pdf} + fig_six_models_points.csv.  Style-only re-renders
can pass --points-csv <that csv> to skip the ~18 min of collect()
passes over the campaign runs.

Colors: Okabe-Ito (colorblind-safe). Warm family = CNNs, cool family =
attention models, so the two architectures separate at a glance;
markers double-encode the model for grayscale printing.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plot_accuracy_curve import collect  # noqa: E402  (matplotlib-free)

import fault_model  # noqa: E402

INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ROOT = HERE.parents[1] / "artifacts/g7/campaign"

# (workload, legend label, Okabe-Ito color, marker)
# warm = CNN, cool = attention
SERIES = [
    ("g7v2_imagenet1k_resnet50", "ResNet-50", "#0072B2", "o"),
    ("g7v2_imagenet1k_mobilenetv3_large_100", "MobileNetV3-L", "#E69F00", "s"),
    ("g7v2_imagenet1k_efficientnet_b0", "EfficientNet-B0", "#D55E00", "^"),
    ("g7v2_imagenet1k_deit_small_patch16_224", "DeiT-S", "#009E73", "D"),
    ("g7v2_imagenet1k_swin_tiny_patch4_window7_224", "Swin-T", "#56B4E9", "v"),
    ("g7v2_imagenet1k_vit_base_patch16_224", "ViT-B", "#CC79A7", "P"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--min-trials", type=int, default=100)
    parser.add_argument("--points-csv", type=Path, default=None,
                        help="re-render path: read the plotted rows back "
                             "from a previously written points CSV instead "
                             "of re-deriving them from the campaign runs "
                             "(seconds instead of minutes; for style "
                             "tweaks only, data provenance unchanged)")
    args = parser.parse_args()
    outdir = args.outdir or args.root / "fig_six_models"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the seven BERs of the five-model ladder are the figure's x grid;
    # the ResNet-50 nine-level ladder contributes its points AT THESE
    # BERs (matched by value), keeping one x axis for all six models
    ladder = fault_model.WORKLOADS[
        "g7v2_imagenet1k_efficientnet_b0"]["levels"]
    bers = [entry["ber"] for entry in ladder]

    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rows: list[tuple] = []
    if args.points_csv is not None:
        with args.points_csv.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                rows.append((row["model"], row["level"], float(row["ber"]),
                             int(row["trials_averaged"]),
                             float(row["acc_pct"]),
                             float(row["clean_acc_pct"])))
    else:
        for workload, label, color, marker in SERIES:
            data = collect(args.root, args.min_trials, workload)
            points = sorted(((data[n]["ber"], data[n]["acc"] * 100, n)
                             for n in data), key=lambda p: p[0])
            keep = [p for p in points if any(p[0] == b for b in bers)]
            if len(keep) != len(bers):
                missing = [f"{b:g}" for b in bers
                           if not any(p[0] == b for p in keep)]
                raise SystemExit(f"{label}: no VERIFIED point at BER "
                                 f"{missing}")
            for ber, acc, name in keep:
                rows.append((label, name, ber, data[name]["trials"], acc,
                             data[name]["clean"] * 100))

    for _, label, color, marker in SERIES:
        pts = sorted((r[2], r[4]) for r in rows if r[0] == label)
        if len(pts) != len(bers):
            missing = [f"{b:g}" for b in bers
                       if not any(p[0] == b for p in pts)]
            raise SystemExit(f"{label}: no point at BER {missing}")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color=color, linewidth=1.8, zorder=2)
        ax.plot(xs, ys, marker, color=color, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3,
                linestyle="none", label=label)

    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e-5)
    ax.set_ylim(45, 85)
    ax.set_yticks(range(45, 86, 5))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3.5)
    def tick_label(ber: float) -> str:
        exponent = math.floor(math.log10(ber))
        coeff = ber / 10 ** exponent
        if abs(coeff - 1.0) < 1e-9:
            return rf"$10^{{{exponent}}}$"
        return rf"${coeff:.0f}\times 10^{{{exponent}}}$"

    ax.set_xticks(bers)
    # rotate: 5e-6 / 7e-6 / 1e-5 sit only ~0.15 decades apart, so their
    # coefficient labels overlap when set horizontally
    ax.set_xticklabels([tick_label(b) for b in bers], rotation=28,
                       rotation_mode="anchor", ha="right")
    ax.set_xlabel("Bit error rate (BER)", fontsize=10.5, color="#0b0b0b")
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=10.5, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8.6, loc="lower left",
              labelcolor="#52514e", handletextpad=0.4, borderaxespad=0.4)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_six_models_accuracy_vs_ber.{ext}",
                    dpi=300, facecolor="white", bbox_inches="tight")
    with (outdir / "fig_six_models_points.csv").open(
            "w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["model", "level", "ber", "trials_averaged",
                         "acc_pct", "clean_acc_pct"])
        for label, name, ber, trials, acc, clean in rows:
            writer.writerow([label, name, f"{ber:g}", trials,
                             f"{acc:.4f}", f"{clean:.4f}"])
    print(f"wrote fig_six_models_accuracy_vs_ber.{{png,pdf}} + points csv "
          f"to {outdir}")
    for label, name, ber, trials, acc, clean in rows:
        if name in ("L1", "L7") or name in ("L3", "L9"):
            print(f"  {label:16s} {name:3s} BER {ber:g}  acc {acc:.2f}%"
                  f"  ({trials} trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
