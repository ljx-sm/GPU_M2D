#!/usr/bin/env python3
"""GPU_M2D G5 figure: top-1 accuracy vs BER over the nine frozen levels.

Derives every plotted number from the campaign CSVs at run time (same
discipline as analyze_campaign.py: nothing is hand-typed):

  - per-trial accuracy = correct/N_images from g1_5_g5_image_detail.csv
    joined against g1_5_g5_clean_pass.csv ground truth; trials aborted by
    DUE are EXCLUDED from the accuracy mean and reported separately (user
    decision 2026-09-25) -- DUE is a reliability metric of its own, not an
    accuracy penalty; a kept trial completed every image, so its accuracy
    is convention-free;
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

WORKLOAD_SUBTITLES = {
    "g5_resisc45_resnet50": (
        "ResNet-50 (INT8 PTQ, TensorRT) on the RESISC45 1000-image "
        "eval split\n"
        "NVIDIA GeForce RTX 4090 — GDDR6X device-memory bit flips "
        "(SBU 60% / MCU 40% event mix)\n"
        "100 trials × 1000 images per level; error bars: 95% CI "
        "over trials; DUE = 0 at all levels"),
    "g7_imagenet1k_resnet50": (
        "ResNet-50 (INT8 PTQ, explicit Q/DQ, TensorRT) on the ImageNet-1K "
        "10,000-image eval split\n"
        "NVIDIA GeForce RTX 4090 — GDDR6X device-memory bit flips "
        "(SBU 60% / MCU 40% event mix)\n"
        "100 trials × 10,000 images per level; DUE-aborted trials excluded "
        "from the mean (DUE rate reported separately);\n"
        "error bars: 95% CI over the averaged trials"),
    "g7v2_imagenet1k_resnet50": (
        "ResNet-50 (INT8 PTQ, explicit Q/DQ, head quantized, TensorRT) "
        "on the ImageNet-1K 10,000-image eval split\n"
        "NVIDIA GeForce RTX 4090 — GDDR6X device-memory bit flips "
        "(SBU 60% / MCU 40% event mix)\n"
        "100 trials × 10,000 images per level; DUE-aborted trials excluded "
        "from the mean (DUE rate reported separately);\n"
        "error bars: 95% CI over the averaged trials"),
}


def per_trial_accuracies(run_dir: Path) -> tuple[list[float], float, int]:
    """(accuracies of the non-DUE trials, clean accuracy, #DUE trials) for
    one VERIFIED run. Convention (user decision 2026-09-25): a trial with
    any DUE ("NA") image is dropped from the accuracy mean entirely -- DUE
    is reported as its own rate, never folded into accuracy; a kept trial
    completed every image, so its accuracy is convention-free."""
    clean_rows = (run_dir / "g1_5_g5_clean_pass.csv")
    labels = {}
    clean_correct = 0
    with clean_rows.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            labels[row["image_index"]] = int(row["label"])
            if int(row["clean_class"]) == int(row["label"]):
                clean_correct += 1
    images = len(labels)

    all_trials: set[int] = set()
    due_trials: set[int] = set()
    correct_per_trial: dict[int, int] = defaultdict(int)
    with (run_dir / "g1_5_g5_image_detail.csv").open(encoding="utf-8",
                                                     newline="") as source:
        for row in csv.DictReader(source):
            trial = int(row["trial_index"])
            injected = row["injected_class"]
            all_trials.add(trial)
            if injected == "NA":
                due_trials.add(trial)
                continue
            if int(injected) == labels[row["image_index"]]:
                correct_per_trial[trial] += 1
    kept = sorted(all_trials - due_trials)
    return ([correct_per_trial[t] / images for t in kept],
            clean_correct / images, len(due_trials))


def collect(root: Path, min_trials: int,
            workload: str = fault_model.DEFAULT_WORKLOAD,
            levels: list[str] | None = None) -> dict[str, dict]:
    """level -> {ber, trials, acc, ci, clean, due_trials} pooled over
    VERIFIED runs of THIS workload only -- a root may hold several
    workloads' runs (the g7 root holds the v1-engine and v2-engine
    campaigns, whose L1..L5 names collide at DIFFERENT BERs), and pooling
    across them would silently mix engines and BER points. `levels`
    optionally restricts the plotted ladder to a subset (kept in frozen
    order). Accuracy pools only non-DUE trials; `trials` is the count
    actually averaged and `due_trials` the excluded abort count."""
    table = (fault_model.assert_frozen_levels(workload)
             or fault_model.WORKLOADS[workload]["levels"])
    order = [entry["level"] for entry in table]
    if levels is not None:
        unknown = [name for name in levels if name not in order]
        if unknown:
            raise SystemError(f"levels {unknown} are not in the {workload} "
                              "frozen ladder")
        keep = set(levels)
        order = [name for name in order if name in keep]
    by_level: dict[str, list[float]] = defaultdict(list)
    clean_accs: dict[str, list[float]] = defaultdict(list)
    due_by_level: dict[str, int] = defaultdict(int)
    for run_dir in sorted(root.glob("run_*")):
        if not (run_dir / "summary.json").is_file():
            continue
        run = analyze_run(run_dir)
        if not run or run.get("status") != "G5_CAMPAIGN_VERIFIED":
            continue
        if run.get("workload") != workload:
            continue
        if run.get("trials", 0) < min_trials:
            continue
        accs, clean, n_due = per_trial_accuracies(run_dir)
        by_level[run["level"]].extend(accs)
        clean_accs[run["level"]].append(clean)
        due_by_level[run["level"]] += n_due

    out: dict[str, dict] = {}
    for name in order:
        accs = by_level.get(name, [])
        n_due = due_by_level.get(name, 0)
        if not accs:
            if n_due:
                raise SystemError(f"{name}: all {n_due} pooled trials "
                                  "aborted with DUE -- no non-DUE accuracy "
                                  "mean exists")
            raise SystemError(f"no VERIFIED runs for {name} under {root}")
        mean = sum(accs) / len(accs)
        sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1))
        entry = fault_model.level_by_name(name, workload)
        out[name] = {
            "ber": entry["ber"],
            "trials": len(accs),
            "acc": mean,
            "ci": 1.96 * sd / math.sqrt(len(accs)),
            "clean": sum(clean_accs[name]) / len(clean_accs[name]),
            "due_trials": n_due,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=HERE.parents[1] / "artifacts/g5/campaign")
    parser.add_argument("--workload", default=fault_model.DEFAULT_WORKLOAD,
                        choices=sorted(fault_model.WORKLOADS),
                        help="whose frozen level ladder to plot")
    parser.add_argument("--min-trials", type=int, default=100,
                        help="formal campaigns only (drops smoke runs)")
    parser.add_argument("--levels", default=None,
                        help="comma-separated subset of the frozen ladder "
                             "to plot, in frozen order (default: all "
                             "levels, e.g. L3,L4,L5,L6,L7,L8,L9)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="default: --root")
    args = parser.parse_args()
    outdir = args.outdir or args.root
    wanted = args.levels.split(",") if args.levels else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    data = collect(args.root, args.min_trials, args.workload, wanted)
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

    # knee marker between L5 and L6 (geometric midpoint of 1e-6 / 5e-6) --
    # only for ladders that actually straddle it with points on both sides
    # (the G5 nine-level ladder; a G7 five-level ladder must not inherit
    # the G5 knee as an assumption)
    knee = math.sqrt(1e-6 * 5e-6)
    if min(bers) < knee < max(bers) and \
            sum(1 for b in bers if b < knee) >= 2 and \
            sum(1 for b in bers if b > knee) >= 2:
        ax.axvline(knee, color=AXIS, linewidth=1.0, linestyle=(0, (2, 3)),
                   zorder=1)
        ax.annotate("knee", xy=(knee, 0.985),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, 0), textcoords="offset points", ha="left",
                    va="top", fontsize=8.5, color=INK_MUTED, style="italic")

    # selective direct labels (values in ink, never in the series color):
    # the nine-level G5 ladder labels its knee region L6-L9 (unchanged --
    # keeps the G5 rendering byte-identical); a short ladder labels its
    # last two points, BOTH on the right of their markers: near-equal
    # neighbors (g7v2 L7/L8) put the second-to-last LEFT slot inside the
    # previous point's CI bar and on the connecting segment, while the
    # right side at marker height clears bars, segments, and gridlines
    labels_at = ({"L6": (-4, 9), "L7": (0, -14), "L8": (-1, 9), "L9": (10, 0)}
                 if len(names) >= 8 else
                 {names[-2]: (11, 0), names[-1]: (11, 0)})
    for name, (dx, dy) in labels_at.items():
        if name not in names:
            continue
        i = names.index(name)
        ax.annotate(f"{accs[i]:.2f}", xy=(bers[i], accs[i]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
                    fontsize=8.5, color=INK_SECONDARY)

    # DUE bookkeeping on the record: the mean excludes DUE-aborted trials,
    # so their count belongs on the figure -- as ONE footnote line under
    # the level-name row, never as per-point notes: the sky around the
    # L7-L9 markers is crowded (tall bars, connecting segments, gridlines
    # at this data height) and every fixed-offset placement there either
    # crossed ink or needed an opaque patch that erased neighboring text;
    # the margin below the axis names is empty at every rendering
    due_levels = [(name, data[name]["due_trials"]) for name in names
                  if data[name]["due_trials"]]
    if due_levels:
        note = ", ".join(f"{name} {n}/{data[name]['trials'] + n}"
                         for name, n in due_levels)
        mid = math.sqrt(data[due_levels[0][0]]["ber"]
                        * data[due_levels[-1][0]]["ber"])
        ax.annotate(f"DUE (excluded from the mean): {note}",
                    xy=(mid, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -36), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_MUTED)

    ax.set_xscale("log")
    # data-driven window (the G5 hardcode assumed the nine-level ladder's
    # 1e-8..1e-4 span and the L9 ~36% floor); the floor follows the lowest
    # CI bound so a near-flat five-level ladder gets a readable zoom
    ax.set_xlim(min(bers) * 0.62, max(bers) * 2.6)
    y_floor = max(0.0, min(a - c for a, c in zip(accs, cis)) - 4.0)
    ax.set_ylim(y_floor, 102)
    span = 102 - y_floor
    step = 10 if span > 45 else 5 if span > 18 else 2 if span > 7 else 1
    ticks = list(range(int(math.ceil(y_floor / step)) * step, 101, step))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
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
    ax.set_title(WORKLOAD_SUBTITLES[args.workload], fontsize=8.3,
                 color=INK_SECONDARY, pad=12)

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
        due = d["due_trials"]
        print(f"  {name}: BER {d['ber']:g}  acc {d['acc']*100:.2f}%"
              f"  +/- {d['ci']*100:.2f} pp  ({d['trials']} trials"
              + (f" averaged, {due} DUE-excluded)" if due else ")"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
