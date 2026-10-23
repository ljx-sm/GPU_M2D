#!/usr/bin/env python3
"""G8-T4 rho = 10 crash attribution figure (user spec 2026-10-22), DRAM +
SRAM mode, six models:

  (a) size of each model's crash-critical TensorRT control block: the
      first create_execution_context allocation (kernel control state),
      bytes on a log axis; MobileNetV3-L's is excluded from its injection
      surface (fault_model surface_excludes) and drawn hatched;
  (b) crash probability vs mean hits on that block per trial, one point
      per model and level, with the pooled fit P = 1 - (1 - q)^k.

A hit is one independent fault location in the block, as in
t4_discussion_numbers.hit_observations: a DRAM fault event (1-3 bits), or
an applied cache flip (for a crashed trial, only flips that started by
the crash image). Every run file is read once per segment; the crash
image is read from the tail of the segment's harness.log (the crashed
trial is always the segment's last). The per-level numbers are cached in
artifacts/g8/rho10/analysis/crash_attribution.json (--reuse replots).

  /data1/luojx/miniforge3/envs/vit_fault/bin/python \\
      tools/g8_cache/plot_crash_attribution.py [--reuse]
Writes artifacts/g8/rho10/fig/crash_attribution_dram_sram.{png,pdf} +
_points.csv.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "tools/g5_faultinj"))

import t4_discussion_numbers as t4  # noqa: E402
from plot_crash_rate import MODELS as STYLE  # noqa: E402
from plot_threeway import AXIS, GRID, INK, INK_MUTED  # noqa: E402

ROOT = PROJECT / "artifacts/g8/rho10/campaign"
CACHE = PROJECT / "artifacts/g8/rho10/analysis/crash_attribution.json"
FLIP = re.compile(rb"event=CACHE_FLIPPED,trial_index=(\d+),cache_index=\d+,"
                  rb"image=(\d+)")
# the crash-critical control block = the first create_execution_context
# allocation of each engine (g1_5_allocations.csv order)
CONTROL_PHASE = "create_execution_context"


def crash_image(seg_dir: Path, dying: int) -> int | None:
    """Image of the dying trial's last applied cache flip, read backwards
    from the end of harness.log."""
    with (seg_dir / "harness.log").open("rb") as source:
        size = source.seek(0, os.SEEK_END)
        step = 1 << 22
        pos = size
        while pos > 0:
            pos = max(0, pos - step)
            source.seek(pos)
            chunk = source.read(min(size - pos, 2 * step))
            matches = list(FLIP.finditer(chunk))
            mine = [m for m in matches if int(m.group(1)) == dying]
            if mine:
                return int(mine[-1].group(2))
            if any(int(m.group(1)) < dying for m in matches):
                return None
    return None


def control_block(seg_dir: Path) -> tuple[str, int]:
    with (seg_dir / "g1_5_allocations.csv").open() as source:
        for row in csv.DictReader(source):
            if row["allocation_phase"] == CONTROL_PHASE:
                return row["allocation_id"], int(row["size_bytes"])
    raise SystemExit(f"{seg_dir}: no {CONTROL_PHASE} allocation")


def level_obs(run: Path, block: str) -> list[tuple[int, bool]]:
    """(hits in block, crashed) per trial of one dram_sram run."""
    obs = []
    for seg_dir, seg in t4.segments(run):
        dying = seg["completed"] if seg.get("crashed_trial") is not None \
            else None
        trials = seg["completed"] + (dying is not None)
        hits: collections.Counter = collections.Counter()
        events = collections.defaultdict(set)
        for row in t4.work_rows(seg_dir, run):
            if row["allocation_id"] == block:
                events[int(row["trial_index"])].add(row["event_index"])
        for trial, ev in events.items():
            hits[trial] += len(ev)
        result = seg_dir / "g1_5_g8_cache_site_result.csv"
        if result.exists():
            with result.open() as source:
                for row in csv.DictReader(source):
                    if row["applied"] == "1" and row["allocation_id"] == block:
                        hits[int(row["trial_index"])] += 1
        if dying is not None:
            image = crash_image(seg_dir, dying)
            if image is not None:
                with (seg_dir / "cache_work.csv").open() as source:
                    for row in csv.DictReader(source):
                        if int(row["trial_index"]) == dying and \
                                row["allocation_id"] == block and \
                                int(row["start_image"]) <= image:
                            hits[dying] += 1
        obs += [(hits[trial], trial == dying) for trial in range(trials)]
    return obs


def compute() -> dict:
    out = {}
    for model, label in t4.MODELS:
        first = t4.find_run(ROOT, t4.workload(model),
                            t4.levels(model)[0]["level"])
        block, size = control_block(t4.segments(first)[0][0])
        excluded = block in fault_model_excludes(model)
        rows = []
        for level in t4.levels(model):
            run = t4.find_run(ROOT, t4.workload(model), level["level"])
            obs = level_obs(run, block)
            rows.append({"level": level["level"], "ber": level["ber"],
                         "trials": len(obs),
                         "hits": sum(k for k, _ in obs),
                         "crashes": sum(c for _, c in obs),
                         "crashes_with_hit": sum(c and k > 0 for k, c in obs),
                         "obs": obs})
            print(f"  {label} {level['level']}: {rows[-1]['hits'] / len(obs):.3f} "
                  f"hits/trial, {rows[-1]['crashes']} crashes", flush=True)
        out[label] = {"block": block, "size": size, "excluded": excluded,
                      "levels": rows}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


def fault_model_excludes(model: str) -> tuple[str, ...]:
    import fault_model
    return tuple(fault_model.WORKLOADS[t4.workload(model)]
                 .get("surface_excludes", ()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true",
                        help=f"replot from {CACHE.relative_to(PROJECT)}")
    parser.add_argument("--outdir", type=Path,
                        default=PROJECT / "artifacts/g8/rho10/fig")
    args = parser.parse_args()
    data = json.loads(CACHE.read_text()) if args.reuse and CACHE.exists() \
        else compute()

    # pooled per-hit fatality over the models whose block is ever fatal
    fatal_models = [m for m, d in data.items()
                    if any(r["crashes"] for r in d["levels"])]
    pooled = [o for m in fatal_models for r in data[m]["levels"]
              for o in r["obs"]]
    q = t4.fatality([tuple(o) for o in pooled])
    for label, d in data.items():
        crashes = sum(r["crashes"] for r in d["levels"])
        with_hit = sum(r["crashes_with_hit"] for r in d["levels"])
        if crashes != with_hit:
            raise SystemExit(f"{label}: {crashes - with_hit} crash(es) without a "
                             f"hit in {d['block']} -- attribution does not hold")
    print(f"pooled q = {q:.3f} over {fatal_models}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {label: (color, marker, ls) for label, color, marker, ls in STYLE}
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1, 1.55]})
    for ax in (ax_a, ax_b):
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(0.9)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3.5)
    fig.patch.set_facecolor("white")

    # (a) control-block size
    labels = [label for label, *_ in STYLE]
    for i, label in enumerate(labels):
        d = data[label]
        color = style[label][0]
        ax_a.bar(i, d["size"], width=0.62, color="white" if d["excluded"]
                 else color, edgecolor=color, linewidth=1.4,
                 hatch="///" if d["excluded"] else None, zorder=3)
        text = f"{d['size'] / 1024:.1f} KB" + ("\nexcluded" if d["excluded"]
                                               else "")
        ax_a.text(i, d["size"] * 1.25, text, ha="center", va="bottom",
                  fontsize=7.8, color="#52514e")
    ax_a.set_yscale("log")
    ax_a.set_ylim(800, 2.5e6)
    ax_a.set_xticks(range(len(labels)))
    ax_a.set_xticklabels(labels, rotation=35, ha="right",
                         rotation_mode="anchor")
    ax_a.set_ylabel("Control-block size (bytes)", fontsize=10.5, color=INK)
    ax_a.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax_a.set_title("(a)", fontsize=10.5, color=INK, loc="left")

    # (b) crash probability vs hits on the block
    xs_all = []
    for label in labels:
        d = data[label]
        if d["excluded"]:
            continue
        color, marker, ls = style[label]
        pts = [(r["hits"] / r["trials"], 100 * r["crashes"] / r["trials"])
               for r in d["levels"]]
        xs_all += [x for x, _ in pts if x > 0]
        ax_b.plot([x for x, _ in pts], [y for _, y in pts], ls, marker=marker,
                  color=color, linewidth=1.4, markersize=5.5,
                  markeredgecolor="white", markeredgewidth=0.7, zorder=3,
                  label=label)
    lo, hi = min(xs_all) / 1.5, max(xs_all) * 1.5
    grid = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    ax_b.plot(grid, [100 * (1 - (1 - q) ** k) for k in grid], color=INK_MUTED,
              linestyle=":", linewidth=1.3, zorder=2,
              label=f"$1-(1-q)^k$, $q$ = {q:.2f}")
    ax_b.set_xscale("log")
    ax_b.set_xlim(lo, hi)
    ax_b.set_ylim(-3, 100)
    ax_b.set_yticks(range(0, 101, 20))
    ax_b.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax_b.set_xlabel("Mean hits on the control block per trial ($k$)",
                    fontsize=10.5, color=INK)
    ax_b.set_ylabel("Crash probability (%)", fontsize=10.5, color=INK)
    ax_b.set_title("(b)", fontsize=10.5, color=INK, loc="left")
    ax_b.legend(frameon=False, fontsize=8.4, loc="upper left",
                labelcolor="#52514e", handletextpad=0.4, borderaxespad=0.4)

    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.outdir / "crash_attribution_dram_sram"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white",
                    bbox_inches="tight")
    with open(f"{stem}_points.csv", "w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["model", "control_block", "block_bytes", "excluded",
                         "level", "ber", "trials", "hits_per_trial",
                         "crashes", "crashes_with_hit", "crash_pct",
                         "pooled_q"])
        for label in labels:
            d = data[label]
            for r in d["levels"]:
                writer.writerow([label, d["block"], d["size"], int(d["excluded"]),
                                 r["level"], f"{r['ber']:g}", r["trials"],
                                 f"{r['hits'] / r['trials']:.4f}", r["crashes"],
                                 r["crashes_with_hit"],
                                 f"{100 * r['crashes'] / r['trials']:.1f}",
                                 f"{q:.4f}"])
    print(f"wrote {stem}.png / .pdf / _points.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
