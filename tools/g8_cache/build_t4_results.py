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

  python3 tools/g8_cache/build_t4_results.py [--rho 1|10] [--from-csv]
Writes docs/G8_T4_RESULTS.{md,csv} (rho 1, the default) or
docs/G8_T4_RHO10_RESULTS.{md,csv} (rho 10: cache BER = 10 x DRAM BER,
figures from artifacts/g8/rho10/fig). --from-csv re-renders the
Markdown from the existing CSV (formatting only; the run-derived
numbers and the figure cross-check come from a full run).
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
from threeway_table import find_run, modes_for, run_stats  # noqa: E402

# same order and names as docs/G7V2_RESULTS.md
MODELS = [("resnet50", "ResNet-50"),
          ("mobilenetv3_large_100", "MobileNetV3-L"),
          ("efficientnet_b0", "EfficientNet-B0"),
          ("deit_small_patch16_224", "DeiT-S"),
          ("swin_tiny_patch4_window7_224", "Swin-T"),
          ("vit_base_patch16_224", "ViT-B")]
# per frozen rho: the figures cross-checked against, and the outputs
OUTPUTS = {
    "1": {"fig": PROJECT / "artifacts/g8/t4/fig",
          "md": PROJECT / "docs/G8_T4_RESULTS.md",
          "csv": PROJECT / "docs/G8_T4_RESULTS.csv"},
    "10": {"fig": PROJECT / "artifacts/g8/rho10/fig",
           "md": PROJECT / "docs/G8_T4_RHO10_RESULTS.md",
           "csv": PROJECT / "docs/G8_T4_RHO10_RESULTS.csv"},
}
TOLERANCE_PP = 0.005


def ber_label(ber: float) -> str:
    mantissa, exponent = f"{ber:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def collect(rho: str) -> list[dict]:
    fig = OUTPUTS[rho]["fig"]
    rows = []
    for model, label in MODELS:
        workload = f"g7v2_imagenet1k_{model}"
        levels = [r for r in fault_model.WORKLOADS[workload]["levels"]
                  if 1e-7 * (1 - 1e-9) <= r["ber"] <= 1e-5 * (1 + 1e-9)]
        with (fig / f"{model}_threeway_points.csv").open(encoding="utf-8") as f:
            plotted = {(r["mode"], r["level"]): float(r["acc_pct"])
                       for r in csv.DictReader(f)}
        for level in levels:
            for mode, _, root in modes_for(rho):
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


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, ber=f"{row['ber']:g}",
                                 top1_pct=f"{row['top1_pct']:.2f}",
                                 clean_top1_pct=f"{row['clean_top1_pct']:.2f}"))


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
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


def header(rho: str) -> list[str]:
    if rho == "10":
        return header_rho10()
    return [
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
        "crash counts here are recomputed from the runs and match that "
        "table in all 42 cells. Two cells there (Swin-T L4 and L6) "
        "originally showed 0 DUE and were corrected on 2026-10-08 to 1 "
        "DUE each (trial 41 at L4, trial 0 at L6), as the runs record. "
        "The accuracies (74.20, 62.25) were unaffected, because those DUE "
        "trials had already been excluded.",
        "",
    ]


def header_rho10() -> list[str]:
    return [
        "# G8-T4, ρ = 10: DRAM-only / SRAM-only / DRAM + SRAM results, "
        "six models",
        "",
        "Generated by `tools/g8_cache/build_t4_results.py --rho 10` from "
        "the VERIFIED campaign runs; do not edit by hand (re-run the "
        "script). The same numbers are in "
        "[G8_T4_RHO10_RESULTS.csv](G8_T4_RHO10_RESULTS.csv), with each "
        "row's run directory. The ρ = 1 results are in "
        "[G8_T4_RESULTS.md](G8_T4_RESULTS.md).",
        "",
        "- **Modes.** DRAM-only = the G7-v2 runs (`artifacts/g7/campaign`), "
        "the same runs as in the ρ = 1 report; SRAM-only = L2 cache faults "
        "only (`artifacts/g8/rho10/sram_only/campaign`); DRAM + SRAM = "
        "both (`artifacts/g8/rho10/campaign`).",
        "- **Levels.** The seven DRAM BERs 1e-7 … 1e-5. Cache BER = 10 × "
        "the level's DRAM BER = 1e-6 … 1e-4 (frozen ρ = 10, "
        "[G8_T3_SRAM_BER_DERIVATION.md](G8_T3_SRAM_BER_DERIVATION.md)). "
        "ResNet-50 runs a nine-level ladder; its L3–L9 are these seven "
        "BERs.",
        "- **Trials.** 100 per level and mode, 10,000 images each. "
        "Completed trials (including DUE) + crash trials = 100.",
        "- **Top-1** is the mean over normally completed trials: DUE "
        "(first invalid output aborts the trial) and crash (PROCESS_FATAL, "
        "the runner process died) trials are excluded and counted "
        "separately, as in G7-v2. The mean is over 100 − DUE − crash "
        "trials (`trials_averaged` in the CSV). At ρ = 10 that count gets "
        "small: EfficientNet-B0 and Swin-T crash in 69–85 trials at "
        "7e-6 and 1e-5 in both cache modes, so those cells average 14–22 "
        "trials, most of them collapsed.",
        "- **Bits per trial.** DRAM bits are the frozen B of the level "
        "(DRAM-only and DRAM + SRAM). SRAM bits are n_cache = round(10 × "
        "BER × R_eff_bits), measured per process on an idle GPU 0 "
        "(SRAM-only and DRAM + SRAM). They are ≈ 10 × the DRAM bits "
        "except for ViT-B (≈ 7.9 ×), which is larger than L2 (R_eff ≈ "
        "79 % of R).",
        "- **Injection surface.** Every model injects over its full "
        "frozen R in all three modes except MobileNetV3-L, whose R is "
        "exclusion-scoped: its `trt-internal-5/6` ctx-phase pools "
        "(248,320 B, 2.66 % of the measured 9,336,232 B) are process-fatal "
        "on any hit and are excluded from the DRAM faults, the cache "
        "faults and the residency probe ([G7V2_RESULTS.md](G7V2_RESULTS.md) "
        "§4, user decision 2026-09-27).",
        "- **Checks.** Every accuracy matches the points CSV of the "
        "model's figure (`artifacts/g8/rho10/fig/<model>_threeway.png`) "
        f"to {TOLERANCE_PP} pp. Every cache-mode run passes the idle-GPU "
        "audit (`tools/g8_cache/audit_idle.py --rho 10`, G8 plan §15.5).",
        "- **DRAM-only.** These columns are identical to the ρ = 1 report "
        "and match `G7V2_RESULTS.md` §3 (see the ρ = 1 report for the "
        "Swin-T L4 / L6 DUE correction of 2026-10-08).",
        "",
    ]


def write_markdown(rows: list[dict], rho: str) -> None:
    get = {(r["model"], r["level"], r["mode"]): r for r in rows}
    fig = OUTPUTS[rho]["fig"].relative_to(PROJECT)
    lines = header(rho)
    order = ["dram_only", "sram_only", "dram_sram"]
    for model, label in MODELS:
        levels = sorted({r["level"] for r in rows if r["model"] == label},
                        key=lambda name: int(name[1:]))
        clean = get[(label, levels[0], "dram_only")]["clean_top1_pct"]
        lines += [
            f"## {label}",
            "",
            f"Clean top-1 {clean:.2f} %. Figure: "
            f"`{fig}/{model}_threeway.png`.",
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
    OUTPUTS[rho]["md"].write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rho", default="1", choices=sorted(OUTPUTS),
                        help="cache-mode runs of this frozen rho")
    parser.add_argument("--from-csv", action="store_true",
                        help="re-render the Markdown from the existing CSV")
    args = parser.parse_args()
    out = OUTPUTS[args.rho]
    if args.from_csv:
        rows = load_csv(out["csv"])
    else:
        rows = collect(args.rho)
        write_csv(rows, out["csv"])
    write_markdown(rows, args.rho)
    md, csv_path = (out[k].relative_to(PROJECT) for k in ("md", "csv"))
    written = md if args.from_csv else f"{md} and {csv_path}"
    print(f"wrote {written} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
