#!/usr/bin/env python3
"""Build the consolidated G8-T4 results table: six G7-v2 models x seven
BER levels (1e-7 ... 1e-5) x three modes (DRAM-only = the G7-v2 runs,
SRAM-only, DRAM + SRAM) -> top-1 accuracy, DUE trials, crash
(PROCESS_FATAL) trials, and the DRAM / SRAM bits per trial.

Every number is derived from the VERIFIED run directories with the same
helpers as threeway_table.py (accuracy over normally completed trials;
completed incl. DUE + crash = 100). Each accuracy is cross-checked
against the points CSV behind that model's three-way figure
(artifacts/g8/t4/fig/<model>_threeway_points.csv); any disagreement
> 0.005 pp aborts, so the document and the figures cannot diverge.

  python3 tools/g8_cache/build_t4_results.py [--from-csv]
Writes docs/G8_T4_RESULTS.md and docs/G8_T4_RESULTS.csv. --from-csv
re-renders the Markdown from the existing CSV (formatting only; the
run-derived numbers and the figure cross-check come from a full run).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "tools/g5_faultinj"))

import fault_model  # noqa: E402
from threeway_table import MODES, find_run, run_stats  # noqa: E402

# same order and names as docs/G7V2_RESULTS.md
MODELS = [("resnet50", "ResNet-50"),
          ("mobilenetv3_large_100", "MobileNetV3-L"),
          ("efficientnet_b0", "EfficientNet-B0"),
          ("deit_small_patch16_224", "DeiT-S"),
          ("swin_tiny_patch4_window7_224", "Swin-T"),
          ("vit_base_patch16_224", "ViT-B")]
FIG = PROJECT / "artifacts/g8/t4/fig"
OUT_MD = PROJECT / "docs/G8_T4_RESULTS.md"
OUT_CSV = PROJECT / "docs/G8_T4_RESULTS.csv"
TOLERANCE_PP = 0.005


def ber_label(ber: float) -> str:
    mantissa, exponent = f"{ber:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def collect() -> list[dict]:
    rows = []
    for model, label in MODELS:
        workload = f"g7v2_imagenet1k_{model}"
        levels = [r for r in fault_model.WORKLOADS[workload]["levels"]
                  if 1e-7 * (1 - 1e-9) <= r["ber"] <= 1e-5 * (1 + 1e-9)]
        with (FIG / f"{model}_threeway_points.csv").open(encoding="utf-8") as f:
            plotted = {(r["mode"], r["level"]): float(r["acc_pct"])
                       for r in csv.DictReader(f)}
        for level in levels:
            for mode, _, root in MODES:
                run = find_run(root, workload, level["level"])
                st = run_stats(run)
                if st["completed"] + st["crash"] != 100:
                    raise SystemExit(f"{label} {mode} {level['level']}: "
                                     "completed + crash != 100")
                fig_acc = plotted.get((mode, level["level"]))
                if fig_acc is None or abs(fig_acc - st["acc"]) > TOLERANCE_PP:
                    raise SystemExit(f"{label} {mode} {level['level']}: table "
                                     f"{st['acc']:.4f} != figure {fig_acc}")
                sram = "/".join(str(n) for n in st["n_cache"]) \
                    if mode != "dram_only" else "0"
                rows.append({
                    "model": label, "workload": workload,
                    "level": level["level"], "ber": level["ber"],
                    "mode": mode,
                    "dram_bits_per_trial": level["bits"]
                    if mode != "sram_only" else 0,
                    "sram_bits_per_trial": sram,
                    "top1_pct": round(st["acc"], 2),
                    "clean_top1_pct": round(st["clean"], 2),
                    "trials_requested": 100,
                    "trials_completed": st["completed"],
                    "due_trials": st["due"],
                    "crash_trials": st["crash"],
                    "trials_averaged": len(st["accs"]),
                    "run_dir": str(run.relative_to(PROJECT)),
                })
                print(f"{label:16} {level['level']} {mode:9} "
                      f"{st['acc']:6.2f}  DUE {st['due']}  crash {st['crash']}",
                      flush=True)
    return rows


def write_csv(rows: list[dict]) -> None:
    with OUT_CSV.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, ber=f"{row['ber']:g}",
                                 top1_pct=f"{row['top1_pct']:.2f}",
                                 clean_top1_pct=f"{row['clean_top1_pct']:.2f}"))


def load_csv() -> list[dict]:
    with OUT_CSV.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["ber"] = float(row["ber"])
        for key in ("top1_pct", "clean_top1_pct"):
            row[key] = float(row[key])
        for key in ("dram_bits_per_trial", "trials_requested",
                    "trials_completed", "due_trials", "crash_trials",
                    "trials_averaged"):
            row[key] = int(row[key])
    return rows


def bits(value) -> str:
    """'1153' -> '1,153'; several per-segment values 'a/b' kept."""
    return "/".join(f"{int(part):,}" for part in str(value).split("/"))


def write_markdown(rows: list[dict]) -> None:
    get = {(r["model"], r["level"], r["mode"]): r for r in rows}
    lines = [
        "# G8-T4: DRAM-only / SRAM-only / DRAM + SRAM results, six models",
        "",
        "Generated by `tools/g8_cache/build_t4_results.py` from the VERIFIED "
        "campaign runs; do not edit by hand (re-run the script). The same "
        "numbers are in [G8_T4_RESULTS.csv](G8_T4_RESULTS.csv), with each "
        "row's run directory.",
        "",
        "- **Modes.** DRAM-only = the G7-v2 runs (`artifacts/g7/campaign`); "
        "SRAM-only = L2 cache faults only (`artifacts/g8/t4/sram_only/"
        "campaign`); DRAM + SRAM = both (`artifacts/g8/t4/campaign`).",
        "- **Levels.** The seven BERs 1e-7 … 1e-5. Cache BER = the level's "
        "DRAM BER (ρ = 1, G8 plan §3.3). ResNet-50 runs a nine-level "
        "ladder; its L3–L9 are these seven BERs.",
        "- **Trials.** 100 per level and mode, 10,000 images each. "
        "Completed trials (including DUE) + crash trials = 100.",
        "- **Top-1** is the mean over normally completed trials: DUE "
        "(first invalid output aborts the trial) and crash (PROCESS_FATAL, "
        "the runner process died) trials are excluded and counted "
        "separately, as in G7-v2.",
        "- **Bits per trial.** DRAM bits are the frozen B of the level "
        "(DRAM-only and DRAM + SRAM). SRAM bits are n_cache = round(BER × "
        "R_eff_bits), measured per process on an idle GPU 0 (SRAM-only "
        "and DRAM + SRAM). They equal the DRAM bits except for ViT-B, "
        "which is larger than L2 (R_eff ≈ 79 % of R).",
        "- **Checks.** Every accuracy matches the points CSV of the "
        "model's figure (`artifacts/g8/t4/fig/<model>_threeway.png`) to "
        f"{TOLERANCE_PP} pp. Every cache-mode run passes the idle-GPU "
        "audit (`tools/g8_cache/audit_idle.py`, G8 plan §15.5).",
        "- **DRAM-only vs `G7V2_RESULTS.md` §3.** The DRAM-only DUE / "
        "crash counts here are recomputed from the runs. They match that "
        "table in 40 of 42 cells. The two exceptions are Swin-T L4 and "
        "L6: that table shows 0 DUE, but the runs record 1 DUE trial "
        "each (trial 41 at L4, trial 0 at L6) in their summary, trial "
        "results and image rows. The accuracies there (74.20, 62.25) are "
        "identical in both documents, because those DUE trials were "
        "already excluded.",
        "",
    ]
    order = ["dram_only", "sram_only", "dram_sram"]
    for model, label in MODELS:
        levels = sorted({r["level"] for r in rows if r["model"] == label},
                        key=lambda name: int(name[1:]))
        clean = get[(label, levels[0], "dram_only")]["clean_top1_pct"]
        lines += [
            f"## {label}",
            "",
            f"Clean top-1 {clean:.2f} %. Figure: "
            f"`artifacts/g8/t4/fig/{model}_threeway.png`.",
            "",
            "| Level | BER | DRAM bits / trial | SRAM bits / trial "
            "| Top-1 %: DRAM-only | SRAM-only | DRAM + SRAM "
            "| DUE: DRAM-only | SRAM-only | DRAM + SRAM "
            "| Crash: DRAM-only | SRAM-only | DRAM + SRAM |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: "
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for level in levels:
            d, s, b = (get[(label, level, m)] for m in order)
            lines.append(
                f"| {level} | {ber_label(d['ber'])} "
                f"| {d['dram_bits_per_trial']:,} | {bits(b['sram_bits_per_trial'])} "
                f"| {d['top1_pct']:.2f} | {s['top1_pct']:.2f} "
                f"| {b['top1_pct']:.2f} "
                f"| {d['due_trials']} | {s['due_trials']} | {b['due_trials']} "
                f"| {d['crash_trials']} | {s['crash_trials']} "
                f"| {b['crash_trials']} |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-csv", action="store_true",
                        help="re-render the Markdown from the existing CSV")
    args = parser.parse_args()
    if args.from_csv:
        rows = load_csv()
    else:
        rows = collect()
        write_csv(rows)
    write_markdown(rows)
    written = OUT_MD.relative_to(PROJECT) if args.from_csv else \
        f"{OUT_MD.relative_to(PROJECT)} and {OUT_CSV.relative_to(PROJECT)}"
    print(f"wrote {written} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
