#!/usr/bin/env python3
"""G8-T4 rho = 10 crash figure (user spec 2026-10-22, after REMU Fig. 6):
crash probability vs BER for the six models in the DRAM + SRAM mode, one
line per model, log x axis at the seven BERs 1e-7 ... 1e-5.

Crash probability = PROCESS_FATAL trials / trials requested (100) of the
level, read from the consolidated results CSV that
build_t4_results.py --rho 10 derives from the VERIFIED runs
(docs/G8_T4_RHO10_RESULTS.csv). DUE trials are not crashes and are not
counted. The plotted rows are written to a points CSV next to the figure.

  /data1/luojx/miniforge3/envs/vit_fault/bin/python \\
      tools/g8_cache/plot_crash_rate.py
Writes artifacts/g8/rho10/fig/crash_vs_ber_dram_sram.{png,pdf} + _points.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from plot_threeway import AXIS, GRID, INK, INK_MUTED, tick_label  # noqa: E402

# (model label in the CSV, categorical slot color, marker, line style):
# CNNs solid, Transformers dashed; the reference categorical order
MODELS = [
    ("ResNet-50", "#2a78d6", "o", "-"),
    ("MobileNetV3-L", "#eb6834", "s", "-"),
    ("EfficientNet-B0", "#1baf7a", "D", "-"),
    ("DeiT-S", "#eda100", "^", "--"),
    ("Swin-T", "#e87ba4", "v", "--"),
    ("ViT-B", "#008300", "P", "--"),
]
MODE = "dram_sram"


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = [r for r in csv.DictReader(source) if r["mode"] == MODE]
    out = []
    for r in rows:
        requested = int(r["trials_requested"])
        out.append({"model": r["model"], "level": r["level"],
                    "ber": float(r["ber"]), "crash_trials": int(r["crash_trials"]),
                    "trials_requested": requested,
                    "crash_pct": 100.0 * int(r["crash_trials"]) / requested})
    for model, *_ in MODELS:
        if len([r for r in out if r["model"] == model]) != 7:
            raise SystemExit(f"{model}: expected 7 {MODE} levels in {path}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path,
                        default=PROJECT / "docs/G8_T4_RHO10_RESULTS.csv")
    parser.add_argument("--outdir", type=Path,
                        default=PROJECT / "artifacts/g8/rho10/fig")
    args = parser.parse_args()
    rows = load(args.results_csv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bers = sorted({r["ber"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for model, color, marker, style in MODELS:
        pts = sorted((r["ber"], r["crash_pct"]) for r in rows
                     if r["model"] == model)
        # ResNet-50 and MobileNetV3-L both sit at 0 %: MobileNetV3-L is
        # drawn as a larger hollow marker so the filled one stays visible
        hollow = model == "MobileNetV3-L"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                marker=marker, color=color, linewidth=1.8,
                markersize=8 if hollow else 5.5,
                markerfacecolor="white" if hollow else color,
                markeredgecolor=color if hollow else "white",
                markeredgewidth=1.4 if hollow else 0.7,
                zorder=2 if hollow else 3, label=model)

    ax.set_xscale("log")
    ax.set_xlim(bers[0] / 1.25, bers[-1] * 1.25)
    ax.set_ylim(-3, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3.5)
    ax.set_xticks(bers)
    ax.minorticks_off()
    ax.set_xticklabels([tick_label(b) for b in bers], rotation=40,
                       rotation_mode="anchor", ha="right")
    ax.set_xlabel("DRAM BER (SRAM BER = 10 × DRAM BER)", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("Crash probability (%)", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left", ncol=2,
              labelcolor="#52514e", handletextpad=0.4, borderaxespad=0.4,
              columnspacing=1.2)

    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / "crash_vs_ber_dram_sram"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white",
                    bbox_inches="tight")
    with open(f"{stem}_points.csv", "w", encoding="utf-8",
              newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["model", "level", "ber", "crash_trials",
                         "trials_requested", "crash_pct"])
        for model, *_ in MODELS:
            for r in sorted((r for r in rows if r["model"] == model),
                            key=lambda r: r["ber"]):
                writer.writerow([model, r["level"], f"{r['ber']:g}",
                                 r["crash_trials"], r["trials_requested"],
                                 f"{r['crash_pct']:.1f}"])
    print(f"wrote {stem}.png / .pdf / _points.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
