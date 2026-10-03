#!/usr/bin/env python3
"""G8-T4 three-way figure for one model: top-1 accuracy vs BER for
DRAM-only (the G7-v2 runs), SRAM-only and DRAM + SRAM (user spec
2026-10-03): 50-80 % y axis, log x axis at the seven BERs 1e-7 ... 1e-5,
the clean accuracy as a dashed baseline, the model name as the only
title, no other annotation.

Every plotted number is derived at run time by the same collect() as the
G7-v2 figures (accuracy over normally completed trials; DUE / crash
trials excluded). The plotted rows are also written to a points CSV,
and --points-csv re-renders from it without re-reading the runs.

  /data1/luojx/miniforge3/envs/vit_fault/bin/python \\
      tools/g8_cache/plot_threeway.py resnet50 --label ResNet-50
Writes artifacts/g8/t4/fig/<model>_threeway.{png,pdf} + _points.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "tools/g5_faultinj"))

import fault_model  # noqa: E402
from plot_accuracy_curve import collect  # noqa: E402

INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# (mode, legend label, campaign root, Okabe-Ito color, marker)
SERIES = [
    ("dram_only", "DRAM-only", PROJECT / "artifacts/g7/campaign",
     "#0072B2", "o"),
    ("sram_only", "SRAM-only", PROJECT / "artifacts/g8/t4/sram_only/campaign",
     "#E69F00", "s"),
    ("dram_sram", "DRAM+SRAM", PROJECT / "artifacts/g8/t4/campaign",
     "#D55E00", "^"),
]


def ladder(workload: str) -> list[dict]:
    """The model's frozen levels inside 1e-7 ... 1e-5 (seven BERs)."""
    return [row for row in fault_model.WORKLOADS[workload]["levels"]
            if 1e-7 * (1 - 1e-9) <= row["ber"] <= 1e-5 * (1 + 1e-9)]


def collect_rows(workload: str, min_trials: int) -> list[dict]:
    levels = ladder(workload)
    rows = []
    for mode, label, root, _, _ in SERIES:
        data = collect(root, min_trials, workload,
                       levels=[row["level"] for row in levels])
        missing = [row["level"] for row in levels if row["level"] not in data]
        if missing:
            raise SystemExit(f"{label}: no VERIFIED run for {missing} "
                             f"under {root}")
        for row in levels:
            point = data[row["level"]]
            rows.append({"mode": mode, "level": row["level"],
                         "ber": row["ber"], "trials_averaged": point["trials"],
                         "acc_pct": point["acc"] * 100,
                         "clean_acc_pct": point["clean"] * 100})
    return rows


def tick_label(ber: float) -> str:
    exponent = math.floor(math.log10(ber))
    coeff = ber / 10 ** exponent
    if abs(coeff - 1.0) < 1e-9:
        return rf"$10^{{{exponent}}}$"
    return rf"${coeff:.0f}\times 10^{{{exponent}}}$"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="e.g. resnet50 (g7v2_imagenet1k_ prefix optional)")
    parser.add_argument("--label", help="title, e.g. ResNet-50")
    parser.add_argument("--min-trials", type=int, default=100)
    parser.add_argument("--points-csv", type=Path,
                        help="re-render from an earlier points CSV")
    parser.add_argument("--outdir", type=Path,
                        default=PROJECT / "artifacts/g8/t4/fig")
    args = parser.parse_args()
    model = args.model.removeprefix("g7v2_imagenet1k_")
    workload = f"g7v2_imagenet1k_{model}"
    if workload not in fault_model.WORKLOADS:
        raise SystemExit(f"unknown workload {workload}")

    if args.points_csv is not None:
        with args.points_csv.open(encoding="utf-8", newline="") as source:
            rows = [dict(row, ber=float(row["ber"]),
                         acc_pct=float(row["acc_pct"]),
                         clean_acc_pct=float(row["clean_acc_pct"]))
                    for row in csv.DictReader(source)]
    else:
        rows = collect_rows(workload, args.min_trials)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bers = [row["ber"] for row in ladder(workload)]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    clean = rows[0]["clean_acc_pct"]
    ax.axhline(clean, color=INK_MUTED, linestyle="--", linewidth=1.1,
               zorder=1, label="Clean")
    for mode, label, _, color, marker in SERIES:
        pts = sorted((r["ber"], r["acc_pct"]) for r in rows
                     if r["mode"] == mode)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, "-", marker=marker, color=color, linewidth=1.8,
                markersize=5.5, markeredgecolor="white", markeredgewidth=0.7,
                zorder=3, label=label)

    ax.set_xscale("log")
    ax.set_xlim(bers[0] / 1.25, bers[-1] * 1.25)
    ax.set_ylim(50, 80)
    ax.set_yticks(range(50, 81, 5))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3.5)
    ax.set_xticks(bers)
    ax.minorticks_off()
    # 5e-6 / 7e-6 / 1e-5 sit ~0.15 decades apart: rotate so they don't collide
    ax.set_xticklabels([tick_label(b) for b in bers], rotation=28,
                       rotation_mode="anchor", ha="right")
    ax.set_xlabel("Bit error rate (BER)", fontsize=10.5, color=INK)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=10.5, color=INK)
    ax.set_title(args.label or model, fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.8, loc="lower left",
              labelcolor="#52514e", handletextpad=0.4, borderaxespad=0.4)

    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / f"{model}_threeway"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white",
                    bbox_inches="tight")
    with open(f"{stem}_points.csv", "w", encoding="utf-8",
              newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["mode", "level", "ber", "trials_averaged",
                         "acc_pct", "clean_acc_pct"])
        for row in rows:
            writer.writerow([row["mode"], row["level"], f"{row['ber']:g}",
                             row["trials_averaged"], f"{row['acc_pct']:.4f}",
                             f"{row['clean_acc_pct']:.4f}"])
    print(f"wrote {stem}.{{png,pdf}} + {stem.name}_points.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
