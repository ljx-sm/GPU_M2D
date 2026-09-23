#!/usr/bin/env python3
"""GPU_M2D G5 figure: top-1 accuracy vs BER over the nine frozen levels.

Derives every plotted number from the campaign CSVs at run time (same
discipline as analyze_campaign.py: nothing is hand-typed):

  - per-trial accuracy = correct/1000 from g1_5_g5_image_detail.csv joined
    against g1_5_g5_clean_pass.csv ground truth (DUE images count wrong);
  - per-level point = mean over trials pooled across that level's VERIFIED
    runs (L1-L5: 3 cards x 100 trials; L6-L9: single card x 100 trials);
  - error bars = 95% CI of the trial mean (1.96 * sd/sqrt(n_trials)) -- the
    honest unit here is the trial: B and (s,d,t) are frozen within a level,
    only positions randomize, so the spread across trials is exactly the
    spatial-randomness uncertainty the 100 repetitions exist to average out;
  - BERs and level order come from fault_model.LEVELS (single source of
    truth; assert_frozen_levels runs first).

Needs matplotlib (NOT stdlib-only, unlike the rest of g5_faultinj): run
under an env that has it, e.g.
  /data1/luojx/miniforge3/envs/vit_fault/bin/python \\
      tools/g5_faultinj/plot_accuracy_curve.py
Writes artifacts/g5/campaign/fig_accuracy_vs_ber.{png,pdf} plus
fig_accuracy_vs_ber_points.csv (the plotted data, for the paper's data
availability statement).

Colors follow the project viz palette: one series blue #2a78d6, clean
baseline dashed muted #898781, hairline grid #e1e0d9, text in ink grays
(values/labels never colored).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fault_model  # noqa: E402  (stdlib; authoritative BER ladder)
from analyze_campaign import analyze_run  # noqa: E402  (stdlib)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"

SUBTITLE = ("ResNet-50 (INT8 PTQ, TensorRT) on the RESISC45 1000-image "
            "eval split\n"
            "NVIDIA GeForce RTX 4090 — GDDR6X device-memory bit flips "
            "(SBU 60% / MCU 40% event mix)\n"
            "100 trials × 1000 images per level; error bars: 95% CI "
            "over trials; DUE = 0 at all levels")


def per_trial_accuracies(run_dir: Path) -> tuple[list[float], float]:
    """(per-trial accuracy, clean accuracy) for one VERIFIED run."""
    clean_rows = (run_dir / "g1_5_g5_clean_pass.csv")
    labels = {}
    clean_correct = 0
    with clean_rows.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            labels[row["image_index"]] = int(row["label"])
            if int(row["clean_class"]) == int(row["label"]):
                clean_correct += 1
    images = len(labels)

    correct_per_trial: dict[int, int] = defaultdict(int)
    with (run_dir / "g1_5_g5_image_detail.csv").open(encoding="utf-8",
                                                     newline="") as source:
        for row in csv.DictReader(source):
            trial = int(row["trial_index"])
            injected = row["injected_class"]
            # DUE ("NA") and every wrong top-1 count as incorrect; the
            # primary accuracy number is DUE-counted-wrong, matching
            # analyze_campaign.py's inj_acc_due_wrong.
            if injected != "NA" and int(injected) == labels[row["image_index"]]:
                correct_per_trial[trial] += 1
    trials = sorted(correct_per_trial)
    return ([correct_per_trial[t] / images for t in trials],
            clean_correct / images)


def collect(root: Path, min_trials: int) -> dict[str, dict]:
    """level -> {ber, trials, acc, ci, clean} pooled over VERIFIED runs."""
    levels = fault_model.assert_frozen_levels() or fault_model.LEVELS
    order = [entry["level"] for entry in levels]
    by_level: dict[str, list[float]] = defaultdict(list)
    clean_accs: dict[str, list[float]] = defaultdict(list)
    for run_dir in sorted(root.glob("run_*")):
        if not (run_dir / "summary.json").is_file():
            continue
        run = analyze_run(run_dir)
        if not run or run.get("status") != "G5_CAMPAIGN_VERIFIED":
            continue
        if run.get("trials", 0) < min_trials:
            continue
        accs, clean = per_trial_accuracies(run_dir)
        by_level[run["level"]].extend(accs)
        clean_accs[run["level"]].append(clean)

    out: dict[str, dict] = {}
    for name in order:
        accs = by_level.get(name, [])
        if not accs:
            raise SystemError(f"no VERIFIED runs for {name} under {root}")
        mean = sum(accs) / len(accs)
        sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1))
        entry = fault_model.level_by_name(name)
        out[name] = {
            "ber": entry["ber"],
            "trials": len(accs),
            "acc": mean,
            "ci": 1.96 * sd / math.sqrt(len(accs)),
            "clean": sum(clean_accs[name]) / len(clean_accs[name]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=HERE.parents[1] / "artifacts/g5/campaign")
    parser.add_argument("--min-trials", type=int, default=100,
                        help="formal campaigns only (drops smoke runs)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="default: --root")
    args = parser.parse_args()
    outdir = args.outdir or args.root

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    data = collect(args.root, args.min_trials)
    names = list(data)
    bers = [data[n]["ber"] for n in names]
    accs = [data[n]["acc"] * 100 for n in names]
    cis = [data[n]["ci"] * 100 for n in names]
    clean = sum(data[n]["clean"] for n in names) / len(names) * 100

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # grid + spines: recessive chrome
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK_MUTED, labelsize=9.5, length=3.5)

    # clean baseline: neutral reference, not a series
    ax.axhline(clean, color=INK_MUTED, linewidth=1.1, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate(f"clean baseline: {clean:.2f}%", xy=(0.985, clean),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=8.5,
                color=INK_SECONDARY)

    # the measured series (line under markers; CI bars under markers too)
    ax.errorbar(bers, accs, yerr=cis, color=SERIES, linewidth=2.0,
                elinewidth=1.0, capsize=2.5, capthick=1.0,
                ecolor=INK_MUTED, zorder=2)
    ax.plot(bers, accs, "o", color=SERIES, markersize=6.5,
            markeredgecolor="white", markeredgewidth=0.9, zorder=3,
            linestyle="none")

    # knee marker between L5 and L6 (geometric midpoint of 1e-6 / 5e-6)
    knee = math.sqrt(1e-6 * 5e-6)
    ax.axvline(knee, color=AXIS, linewidth=1.0, linestyle=(0, (2, 3)),
               zorder=1)
    ax.annotate("knee", xy=(knee, 0.985), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", ha="left",
                va="top", fontsize=8.5, color=INK_MUTED, style="italic")

    # selective direct labels (values in ink, never in the series color)
    labels_at = {  # level -> (dx, dy) in points, ha
        "L6": (-4, 9), "L7": (0, -14), "L8": (-1, 9), "L9": (10, 0),
    }
    for name, (dx, dy) in labels_at.items():
        i = names.index(name)
        ax.annotate(f"{accs[i]:.2f}", xy=(bers[i], accs[i]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
                    fontsize=8.5, color=INK_SECONDARY)

    ax.set_xscale("log")
    ax.set_xlim(0.62e-8, 2.6e-4)
    ax.set_ylim(28, 102)
    ax.yaxis.set_major_locator(FixedLocator(list(range(30, 101, 10))))
    ax.set_xlabel("Bit error rate (BER, fraction of resident bits flipped "
                  "per trial)", fontsize=11, color=INK_PRIMARY, labelpad=8)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=11, color=INK_PRIMARY)

    def tick_label(ber: float) -> str:
        if abs(math.log10(ber) - round(math.log10(ber))) < 1e-9:
            return rf"$10^{{{round(math.log10(ber)):.0f}}}$"
        coeff = ber / 10 ** math.floor(math.log10(ber))
        return rf"${coeff:.0f}\times 10^{{{math.floor(math.log10(ber)):.0f}}}$"

    ax.xaxis.set_major_locator(FixedLocator(bers))
    ax.set_xticklabels([tick_label(b) for b in bers])
    ax.minorticks_off()
    # level names as a second, muted row under the tick labels
    for name, ber in zip(names, bers):
        ax.annotate(name, xy=(ber, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_MUTED)

    fig.suptitle("Top-1 accuracy vs. injected GDDR6X bit-error rate",
                 fontsize=13, fontweight="bold", color=INK_PRIMARY, y=0.975)
    ax.set_title(SUBTITLE, fontsize=8.3, color=INK_SECONDARY, pad=12)

    fig.tight_layout(rect=(0, 0.01, 1, 0.99))

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_accuracy_vs_ber.{ext}", dpi=300,
                    facecolor="white", bbox_inches="tight")
    with (outdir / "fig_accuracy_vs_ber_points.csv").open(
            "w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["level", "ber", "trials", "acc_pct", "ci95_pct",
                         "clean_acc_pct"])
        for name in names:
            d = data[name]
            writer.writerow([name, f"{d['ber']:g}", d["trials"],
                             f"{d['acc']*100:.4f}", f"{d['ci']*100:.4f}",
                             f"{d['clean']*100:.4f}"])
    print(f"wrote fig_accuracy_vs_ber.{{png,pdf}} + points csv to {outdir}")
    for name in names:
        d = data[name]
        print(f"  {name}: BER {d['ber']:g}  acc {d['acc']*100:.2f}%"
              f"  +/- {d['ci']*100:.2f} pp  ({d['trials']} trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
