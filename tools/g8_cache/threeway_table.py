#!/usr/bin/env python3
"""G8-T4 three-way table for one model: per BER level (1e-7 ... 1e-5) and
mode (DRAM-only = the G7-v2 runs, SRAM-only, DRAM + SRAM): DRAM and SRAM
bits per trial, DUE trials, crash (PROCESS_FATAL) trials and top-1
accuracy over the normally completed trials -- the G7-v2 convention:
completed (incl. DUE) + crash = 100, accuracy excludes DUE and crash.

Additivity at the level mean: loss(D+S) - loss(D) - loss(S) with a 95 %
CI from the three independent per-trial samples (normal approximation).
Unlike a per-trial paired test it does not need identical fault sites,
which only hold when a process lands on the same physical pages.

Runs are taken only if VERIFIED with 100 requested trials (a level with
crashes has fewer completed trials and is still complete).

  python3 tools/g8_cache/threeway_table.py deit_small_patch16_224
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tools/g5_faultinj"))
import fault_model  # noqa: E402

MODES = [("dram_only", "DRAM-only", PROJECT / "artifacts/g7/campaign"),
         ("sram_only", "SRAM-only", PROJECT / "artifacts/g8/t4/sram_only/campaign"),
         ("dram_sram", "DRAM+SRAM", PROJECT / "artifacts/g8/t4/campaign")]


def find_run(root: Path, workload: str, level: str) -> Path:
    found = []
    for path in root.glob(f"run_{level}_gpu*/summary.json"):
        s = json.loads(path.read_text())
        if s.get("workload") == workload and s.get("level") == level and \
                s.get("status") == "G5_CAMPAIGN_VERIFIED" and \
                s.get("trials_requested") == 100:
            found.append(path.parent)
    if len(found) != 1:
        raise SystemExit(f"{root.name}: {len(found)} complete runs for "
                         f"{workload} {level} (need exactly 1)")
    return found[0]


def run_stats(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    with (run / "g1_5_g5_clean_pass.csv").open(encoding="utf-8") as f:
        label = {r["image_index"]: int(r["label"]) for r in csv.DictReader(f)}
    correct: dict[int, int] = {}
    due: set[int] = set()
    with (run / "g1_5_g5_image_detail.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = int(r["trial_index"])
            correct.setdefault(t, 0)
            if r["injected_class"] == "NA":
                due.add(t)
            elif int(r["injected_class"]) == label[r["image_index"]]:
                correct[t] += 1
    images = len(label)
    accs = [100 * correct[t] / images for t in correct if t not in due]
    clean = 100 * sum(1 for r in label.values()) and None
    with (run / "g1_5_g5_clean_pass.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clean = 100 * sum(int(r["clean_class"]) == int(r["label"])
                      for r in rows) / len(rows)
    n_cache = [ (seg.get("g8_cache") or {}).get("n_cache")
                for seg in s.get("segment_details") or []]
    return {"accs": accs, "acc": statistics.mean(accs),
            "sd": statistics.stdev(accs), "due": len(due),
            "crash": s.get("process_fatal_count", 0),
            "completed": s.get("trials_completed"), "clean": clean,
            "n_cache": sorted({n for n in n_cache if n is not None}),
            "mode": s.get("fault_mode", "dram_only")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    args = parser.parse_args()
    model = args.model.removeprefix("g7v2_imagenet1k_")
    workload = f"g7v2_imagenet1k_{model}"
    levels = [r for r in fault_model.WORKLOADS[workload]["levels"]
              if 1e-7 * (1 - 1e-9) <= r["ber"] <= 1e-5 * (1 + 1e-9)]
    table = {}
    for key, _, root in MODES:
        for row in levels:
            st = run_stats(find_run(root, workload, row["level"]))
            if st["completed"] + st["crash"] != 100:
                raise SystemExit(f"{key} {row['level']}: completed "
                                 f"{st['completed']} + crash {st['crash']} != 100")
            table[key, row["level"]] = st
    clean = table["dram_only", levels[0]["level"]]["clean"]
    print(f"{workload}  clean top-1 {clean:.2f} %  (accuracy over normally "
          "completed trials; DUE / crash out of 100)\n")
    print("level  BER   | DRAM bits SRAM bits | DUE/crash  D-only S-only D+S "
          "| top-1 %  D-only  S-only   D+S")
    for row in levels:
        L = row["level"]
        d, s, b = (table[k, L] for k in ("dram_only", "sram_only", "dram_sram"))
        sram_bits = "/".join(str(n) for n in s["n_cache"])
        print(f"{L:5} {row['ber']:.0e} | {row['bits']:9} {sram_bits:>9} | "
              f"       {d['due']}/{d['crash']}    {s['due']}/{s['crash']}"
              f"    {b['due']}/{b['crash']} | {d['acc']:14.2f} {s['acc']:7.2f}"
              f" {b['acc']:7.2f}")
    print("\nadditivity at the level mean (loss = clean - accuracy, pp)")
    print("level | loss D  loss S   sum  | loss D+S | D+S - D - S [95% CI]")
    for row in levels:
        L = row["level"]
        d, s, b = (table[k, L] for k in ("dram_only", "sram_only", "dram_sram"))
        ld, ls, lb = clean - d["acc"], clean - s["acc"], clean - b["acc"]
        se = math.sqrt(sum(x["sd"] ** 2 / len(x["accs"]) for x in (d, s, b)))
        inter = lb - ld - ls
        print(f"{L:5} | {ld:6.2f} {ls:6.2f} {ld + ls:6.2f} | {lb:8.2f} | "
              f"{inter:+.2f} [{inter - 1.96 * se:+.2f}, {inter + 1.96 * se:+.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
