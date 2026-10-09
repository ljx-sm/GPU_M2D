#!/usr/bin/env python3
"""G8-T4 per-mode figures: one figure per fault mode (DRAM-only,
SRAM-only, DRAM + SRAM), six curves each (the six G7-v2 models), top-1
accuracy vs BER at the seven BERs 1e-7 ... 1e-5.

Style: model colours and markers are imported from the G7-v2 six-model
figure (tools/g5_faultinj/plot_six_models.py SERIES: Okabe-Ito, warm =
CNN, cool = attention); axes, ticks and the mode name as the only title
follow the per-model three-way figures (plot_threeway.py). The y axis is
shared by all three figures (computed from all plotted points) so the
modes can be compared directly.

Data: the exact points behind the six per-model figures
(artifacts/g8/t4/fig/<model>_threeway_points.csv). Every point is
cross-checked against docs/G8_T4_RESULTS.csv and, for DRAM-only, against
the G7-v2 six-model figure's points; any disagreement > 0.005 pp aborts.

  /data1/luojx/miniforge3/envs/vit_fault/bin/python \\
      tools/g8_cache/plot_mode_figures.py
Writes artifacts/g8/t4/fig/fig_mode_{dram_only,sram_only,dram_sram}.
{png,pdf} + fig_modes_points.csv.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "tools/g5_faultinj"))
sys.path.insert(0, str(HERE))

from build_t4_results import MODELS  # noqa: E402
from plot_six_models import AXIS, GRID, INK_MUTED, SERIES  # noqa: E402

FIG = PROJECT / "artifacts/g8/t4/fig"
RESULTS_CSV = PROJECT / "docs/G8_T4_RESULTS.csv"
G7_POINTS = PROJECT / ("artifacts/g7/campaign/fig_six_models/"
                       "fig_six_models_points.csv")
MODES = [("dram_only", "DRAM-only"), ("sram_only", "SRAM-only"),
         ("dram_sram", "DRAM+SRAM")]
TOLERANCE_PP = 0.005
INK = "#0b0b0b"


def load_points() -> dict:
    """(mode, model label) -> sorted [(ber, acc %)] with cross-checks."""
    style = {label: (color, marker) for _, label, color, marker in SERIES}
    if set(style) != {label for _, label in MODELS}:
        raise SystemExit("plot_six_models SERIES and the model list differ")
    with RESULTS_CSV.open(encoding="utf-8") as f:
        table = {(r["model"], r["mode"], float(r["ber"])): float(r["top1_pct"])
                 for r in csv.DictReader(f)}
    with G7_POINTS.open(encoding="utf-8") as f:
        g7 = {(r["model"], float(r["ber"])): float(r["acc_pct"])
              for r in csv.DictReader(f)}
    points: dict = {}
    for model, label in MODELS:
        with (FIG / f"{model}_threeway_points.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ber, acc = float(r["ber"]), float(r["acc_pct"])
                ref = table.get((label, r["mode"], ber))
                if ref is None or abs(ref - acc) > TOLERANCE_PP:
                    raise SystemExit(f"{label} {r['mode']} {ber:g}: figure "
                                     f"{acc} vs results table {ref}")
                if r["mode"] == "dram_only" and \
                        abs(g7[(label, ber)] - acc) > TOLERANCE_PP:
                    raise SystemExit(f"{label} {ber:g}: figure {acc} vs G7-v2 "
                                     f"six-model {g7[(label, ber)]}")
                points.setdefault((r["mode"], label), []).append((ber, acc))
    for key, rows in points.items():
        rows.sort()
        if len(rows) != 7:
            raise SystemExit(f"{key}: {len(rows)} points, expected 7")
    if len(points) != len(MODES) * len(MODELS):
        raise SystemExit(f"{len(points)} curves, expected 18")
    return points


def tick_label(ber: float) -> str:
    exponent = math.floor(math.log10(ber))
    coeff = ber / 10 ** exponent
    if abs(coeff - 1.0) < 1e-9:
        return rf"$10^{{{exponent}}}$"
    return rf"${coeff:.0f}\times 10^{{{exponent}}}$"


def main() -> int:
    points = load_points()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    every = [acc for rows in points.values() for _, acc in rows]
    bottom = min(50, 5 * math.floor((min(every) - 1.0) / 5))
    top = max(80, 5 * math.ceil((max(every) + 0.5) / 5))
    bers = sorted({ber for rows in points.values() for ber, _ in rows})

    for mode, title in MODES:
        fig, ax = plt.subplots(figsize=(5.8, 4.1))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for _, label, color, marker in SERIES:
            xs, ys = zip(*points[(mode, label)])
            ax.plot(xs, ys, "-", marker=marker, color=color, linewidth=1.8,
                    markersize=5.5, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=3, label=label)
        ax.set_xscale("log")
        ax.set_xlim(bers[0] / 1.25, bers[-1] * 1.25)
        ax.set_ylim(bottom, top)
        ax.set_yticks(range(bottom, top + 1, 5))
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.9)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3.5)
        ax.set_xticks(bers)
        ax.minorticks_off()
        ax.set_xticklabels([tick_label(b) for b in bers], rotation=28,
                           rotation_mode="anchor", ha="right")
        ax.set_xlabel("Bit error rate (BER)", fontsize=10.5, color=INK)
        ax.set_ylabel("Top-1 accuracy (%)", fontsize=10.5, color=INK)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8.6, loc="lower left",
                  labelcolor="#52514e", handletextpad=0.4, borderaxespad=0.4)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"fig_mode_{mode}.{ext}", dpi=300,
                        facecolor="white", bbox_inches="tight")
        plt.close(fig)

    with (FIG / "fig_modes_points.csv").open("w", encoding="utf-8",
                                              newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["mode", "model", "ber", "acc_pct"])
        for mode, _ in MODES:
            for _, label, _, _ in SERIES:
                for ber, acc in points[(mode, label)]:
                    writer.writerow([mode, label, f"{ber:g}", f"{acc:.4f}"])
    print(f"wrote fig_mode_{{dram_only,sram_only,dram_sram}}.{{png,pdf}} and "
          f"fig_modes_points.csv in {FIG.relative_to(PROJECT)}; y axis "
          f"{bottom}-{top} % shared; all 126 points cross-checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
