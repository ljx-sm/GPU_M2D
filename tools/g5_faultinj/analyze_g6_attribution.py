#!/usr/bin/env python3
"""GPU_M2D G6-T0 attribution analysis (pure stdlib, offline).

Answers, from the campaign CSVs alone (nothing hand-typed), the four
questions of the first G6 pass over the nine-level campaign:

  A. flip integrity -- did every level really flip B x trials bits?
     per level: VERIFIED run count, total sites, expected sites, any
     guard/reverse-map violations (the runner already verified
     after == before ^ mask per site and the orchestrator re-verified
     every row; this recounts from the CSVs);
  B. where the flips land -- per allocation: residency share vs site
     share (sampling unbiasedness) and the restore_check tally that
     proves persistence: the INT8 weight allocation restores byte-exact
     after every pass (the engine never rewrites it), while the
     activation/scratch allocation is always rewritten (soft upsets);
  C. weight corrosion per trial -- distinct weight-allocation bytes hit
     per trial (min/mean/max) as a fraction of the weight blob: the
     quantity that actually crosses the knee;
  D. perturbation vs decision margin -- clean top-1 probability
     percentiles from the clean pass, and per-level |dProbability|
     percentiles over valid (trial, image) rows: the two distributions
     whose crossing explains the plateau and the collapse.

Run: python3 tools/g5_faultinj/analyze_g6_attribution.py
     [--root artifacts/g5/campaign] [--min-trials 100]
     [--weight-alloc trt-internal-0]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import fault_model


def normalize_alloc(alloc: str) -> str:
    """Card-independent allocation key: the TRT binding ids carry the
    device index (trt-binding-data-gpu-0/1/2 are the same binding)."""
    return re.sub(r"-gpu-\d+$", "", alloc)


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    i = (len(values) - 1) * p / 100.0
    lo = int(i)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def formal_runs(root: Path, min_trials: int) -> list[Path]:
    out = []
    for run_dir in sorted(root.glob("run_*")):
        summary = run_dir / "summary.json"
        if not summary.is_file():
            continue
        data = json.loads(summary.read_text())
        if data.get("status") != "G5_CAMPAIGN_VERIFIED":
            continue
        if int(data.get("trials_completed", data.get("trials", 0))) < min_trials:
            continue
        out.append(run_dir)
    return out


def residency(run_dir: Path) -> tuple[dict[str, int], dict[str, str]]:
    """allocation -> resident bytes (snapshot), allocation -> label."""
    sizes: dict[str, int] = defaultdict(int)
    labels: dict[str, str] = {}
    with (run_dir / "snapshot/snapshot_pages.csv").open(encoding="utf-8",
                                                        newline="") as src:
        for row in csv.DictReader(line for line in src
                                  if not line.startswith("#")):
            if row["allocation_id"]:
                key = normalize_alloc(row["allocation_id"])
                sizes[key] += (int(row["byte_end_in_page"], 16)
                               - int(row["byte_start_in_page"], 16))
                labels[key] = row["semantic_label"]
    return sizes, labels


def analyze(root: Path, min_trials: int, weight_alloc: str) -> int:
    fault_model.assert_frozen_levels()
    runs = formal_runs(root, min_trials)
    if not runs:
        raise SystemError(f"no VERIFIED runs with >= {min_trials} trials "
                          f"under {root}")

    by_level: dict[str, list[Path]] = defaultdict(list)
    sizes, labels = None, None
    sites_by_alloc: Counter = Counter()
    restore_by_alloc: dict[str, Counter] = defaultdict(Counter)
    weight_bytes: dict[str, list[int]] = defaultdict(list)
    weight_bits: dict[str, list[int]] = defaultdict(list)
    integrity: dict[str, dict] = {}

    for run_dir in runs:
        summary = json.loads((run_dir / "summary.json").read_text())
        level = summary["level"]
        by_level[level].append(run_dir)
        run_sizes, run_labels = residency(run_dir)
        if sizes is None:
            sizes, labels = run_sizes, run_labels
        elif dict(run_sizes) != dict(sizes):
            raise SystemError(f"residency differs across runs: {run_dir}")

        bad = 0
        n_sites = 0
        trial_bytes: dict[int, set[int]] = defaultdict(set)
        trial_bits: dict[int, int] = defaultdict(int)
        with (run_dir / "g1_5_g5_site_result.csv").open(encoding="utf-8",
                                                        newline="") as src:
            for row in csv.DictReader(src):
                n_sites += 1
                if row["guard_bytes_unchanged"] != "1" \
                        or row["reverse_map_ok"] != "1":
                    bad += 1
                alloc = normalize_alloc(row["allocation_id"])
                sites_by_alloc[alloc] += 1
                restore_by_alloc[alloc][row["restore_check"].split(":")[0]] += 1
                if alloc == weight_alloc:
                    trial_bytes[int(row["trial_index"])].add(
                        int(row["byte_offset"]))
                    trial_bits[int(row["trial_index"])] += 1
        row_ = integrity.setdefault(level, {"sites": 0, "bad": 0,
                                            "trials": 0})
        row_["sites"] += n_sites
        row_["bad"] += bad
        row_["trials"] += len(trial_bytes)
        weight_bytes[level].extend(len(v) for v in trial_bytes.values())
        weight_bits[level].extend(trial_bits.values())

    weight_size = sizes.get(weight_alloc, 0)
    if not weight_size:
        raise SystemError(f"weight allocation {weight_alloc!r} not in snapshot")

    print(f"G6-T0 attribution over {len(runs)} VERIFIED runs "
          f"({', '.join(sorted(by_level))})")
    print()

    print("[A] flip integrity (sites counted from g1_5_g5_site_result.csv)")
    print(f"{'lvl':3} {'runs':4} {'trials':6} {'sites':>9} {'expected':>9} "
          f"{'match':>5} {'viol':>4}")
    for level in sorted(by_level):
        entry = fault_model.level_by_name(level)
        row = integrity[level]
        expected = entry["bits"] * row["trials"]
        print(f"{level:3} {len(by_level[level]):4} {row['trials']:6} "
              f"{row['sites']:9} {expected:9} "
              f"{'OK' if row['sites'] == expected else 'DIFF':>5} "
              f"{row['bad']:4}")
    print()

    print("[B] where the flips land (pooled over all runs; residency from "
          "the run snapshot)")
    total_sites = sum(sites_by_alloc.values())
    total_res = sum(sizes.values())
    print(f"{'allocation':28} {'semantic_label':26} {'bytes':>9} "
          f"{'res%':>6} {'site%':>6}  restore_check")
    for alloc in sorted(sizes, key=lambda a: -sizes[a]):
        tally = ", ".join(f"{k}={v}" for k, v in
                          sorted(restore_by_alloc[alloc].items()))
        print(f"{alloc:28} {labels[alloc]:26} {sizes[alloc]:9} "
              f"{100 * sizes[alloc] / total_res:6.2f} "
              f"{100 * sites_by_alloc[alloc] / total_sites:6.2f}  {tally}")
    print()

    print(f"[C] corrosion of the weight allocation ({weight_alloc}, "
          f"{weight_size} B) per trial")
    print(f"{'lvl':3} {'BER':>8} {'wBytes mean':>11} {'(min..max)':>14} "
          f"{'%ofW':>7} {'wBits mean':>10}")
    for level in sorted(by_level):
        entry = fault_model.level_by_name(level)
        vals = weight_bytes[level]
        mean = sum(vals) / len(vals)
        bits = sum(weight_bits[level]) / len(weight_bits[level])
        print(f"{level:3} {entry['ber']:8g} {mean:11.0f} "
              f"({min(vals)}..{max(vals)})  {100 * mean / weight_size:7.4f} "
              f"{bits:10.0f}")
    print()

    print("[D] perturbation vs decision margin")
    any_run = runs[0]
    clean_p = []
    with (any_run / "g1_5_g5_clean_pass.csv").open(encoding="utf-8",
                                                   newline="") as src:
        for row in csv.DictReader(src):
            clean_p.append(float(row["clean_probability"]))
    below = sum(1 for x in clean_p if x < 0.5) / len(clean_p)
    print(f"clean P(top1): p10={pct(clean_p, 10):.3f} "
          f"p50={pct(clean_p, 50):.3f} p90={pct(clean_p, 90):.3f} "
          f"frac<0.5={below:.3f}")
    print(f"{'lvl':3} {'|dP| p50':>9} {'p90':>7} {'p99':>7} {'max':>6} "
          f"{'frac>1e-6':>10}")
    for level in sorted(by_level):
        deltas = []
        for run_dir in by_level[level]:
            clean = {}
            with (run_dir / "g1_5_g5_clean_pass.csv").open(
                    encoding="utf-8", newline="") as src:
                for row in csv.DictReader(src):
                    clean[row["image_index"]] = float(row["clean_probability"])
            with (run_dir / "g1_5_g5_image_detail.csv").open(
                    encoding="utf-8", newline="") as src:
                for row in csv.DictReader(src):
                    if row["injected_class"] != "NA":
                        deltas.append(abs(float(row["injected_probability"])
                                          - clean[row["image_index"]]))
        nonzero = sum(1 for x in deltas if x > 1e-6)
        print(f"{level:3} {pct(deltas, 50):9.4f} {pct(deltas, 90):7.4f} "
              f"{pct(deltas, 99):7.4f} {max(deltas):6.3f} "
              f"{nonzero / len(deltas):10.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "artifacts/g5/campaign")
    parser.add_argument("--min-trials", type=int, default=100,
                        help="formal campaigns only (drops smoke runs)")
    parser.add_argument("--weight-alloc", default="trt-internal-0",
                        help="allocation treated as the INT8 weight blob "
                             "(largest deserialize_engine internal)")
    args = parser.parse_args()
    return analyze(args.root, args.min_trials, args.weight_alloc)


if __name__ == "__main__":
    raise SystemExit(main())
