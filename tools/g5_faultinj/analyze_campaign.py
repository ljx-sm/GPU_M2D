#!/usr/bin/env python3
"""GPU_M2D G5 campaign aggregation (pure stdlib, offline).

Walks artifacts/g5/campaign/run_*/ (one dir per campaign = level x card),
recomputes every metric from the four result CSVs independently of the
runner and the orchestrator, and prints per-run and pooled-per-level
tables:

  - per-image: top-1 change rate / numeric SDC rate / DUE rate
    (binomial 95% CI on the pooled top-1 rate), vs the clean pass;
  - accuracy vs ground truth (clean vs injected; the PRIMARY injected
    accuracy averages only normally-completed trials -- trials aborted
    by DUE are excluded from the mean entirely and reported through the
    separate DUE columns (user decision 2026-09-25); DUE-counted-wrong
    and evaluated-only stay in the per-run dict as brackets);
  - right->wrong vs wrong->right decomposition of top-1 changes;
  - mean |dProbability| over valid outputs;
  - trial-level: P(trial has >=1 top-1 change), P(trial DUE);
  - restore/sanity echoes from the verified summaries.

Run: python3 tools/g5_faultinj/analyze_campaign.py [--root artifacts/g5/campaign]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import fault_model

PROB_TOL = 1.0e-6


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def run_workload(summary: dict) -> str:
    """The run's workload; summaries predating the workload parameter
    (the G5 core campaigns) are g5_resisc45_resnet50 by construction."""
    return summary.get("workload") or fault_model.DEFAULT_WORKLOAD


def analyze_run(run_dir: Path) -> dict | None:
    summary = json.loads((run_dir / "summary.json").read_text())
    level = summary["level"]
    device = summary["device"]
    workload = run_workload(summary)
    if summary["status"] != "G5_CAMPAIGN_VERIFIED":
        return {"level": level, "device": device, "status": summary["status"],
                "workload": workload,
                "failures": summary.get("failures", [])}

    clean_rows = load_rows(run_dir / "g1_5_g5_clean_pass.csv")
    label = {r["image_index"]: int(r["label"]) for r in clean_rows}
    clean_cls = {r["image_index"]: int(r["clean_class"]) for r in clean_rows}
    images = len(clean_rows)
    clean_correct = sum(1 for i in range(images)
                        if clean_cls[str(i)] == label[str(i)])

    img_rows = load_rows(run_dir / "g1_5_g5_image_detail.csv")
    trials = sorted({int(r["trial_index"]) for r in img_rows})

    n_eval = n_valid = n_top1 = n_numeric = n_invalid = 0
    n_inj_correct = n_r2w = n_w2r = n_swap = 0
    prob_delta_sum = 0.0
    prob_delta_max = 0.0
    correct_per_trial: dict[int, int] = defaultdict(int)
    trials_top1 = set()
    trials_due = set()
    for row in img_rows:
        trial = int(row["trial_index"])
        n_eval += int(row["evaluated"] == "1")
        if row["injected_class"] == "NA":
            n_invalid += 1
            trials_due.add(trial)
            continue
        n_valid += 1
        inj = int(row["injected_class"])
        image = row["image_index"]
        clean = clean_cls[image]
        delta = abs(float(row["injected_probability"]) -
                    float(row["clean_probability"]))
        prob_delta_sum += delta
        prob_delta_max = max(prob_delta_max, delta)
        if inj == label[image]:
            n_inj_correct += 1
            correct_per_trial[trial] += 1
        if inj != clean:
            n_top1 += 1
            trials_top1.add(trial)
            if clean == label[image] and inj != label[image]:
                n_r2w += 1
            elif clean != label[image] and inj == label[image]:
                n_w2r += 1
            else:
                n_swap += 1
        elif delta > PROB_TOL:
            n_numeric += 1

    n_trials = len(trials)
    per_trial_images = images
    # primary accuracy: mean over the normally-completed trials only (a
    # DUE trial contributes to the DUE columns, never to the accuracy);
    # the two bracket conventions below stay available in the dict
    n_nondue_trials = n_trials - len(trials_due)
    nondue_correct = sum(count for trial, count in correct_per_trial.items()
                         if trial not in trials_due)
    return {
        "level": level, "device": device,
        "status": summary["status"], "workload": workload,
        "trials": n_trials, "images": images,
        "total_images": n_trials * per_trial_images,
        "sites": summary.get("total_sites"),
        "clean_acc": clean_correct / images,
        "inj_acc_nondue": (nondue_correct / (n_nondue_trials * per_trial_images)
                           if n_nondue_trials else float("nan")),
        "inj_acc_due_wrong": n_inj_correct / (n_trials * per_trial_images),
        "inj_acc_eval_only": n_inj_correct / n_valid if n_valid else float("nan"),
        "top1": n_top1, "numeric": n_numeric, "invalid": n_invalid,
        "valid": n_valid,
        "top1_rate": n_top1 / n_valid if n_valid else float("nan"),
        "numeric_rate": (n_top1 + n_numeric) / n_valid if n_valid else float("nan"),
        "invalid_rate": n_invalid / (n_trials * per_trial_images),
        "r2w": n_r2w, "w2r": n_w2r, "swap": n_swap,
        "mean_dprob": prob_delta_sum / n_valid if n_valid else float("nan"),
        "max_dprob": prob_delta_max,
        "trials_top1": len(trials_top1), "trials_due": len(trials_due),
        "restore": summary.get("restore_totals", {}),
        "outcome_hist": summary.get("trial_outcome_histogram", {}),
    }


def pooled(runs: list[dict]) -> dict:
    n_valid = sum(r["valid"] for r in runs)
    n_top1 = sum(r["top1"] for r in runs)
    n_invalid = sum(r["invalid"] for r in runs)
    total_images = sum(r["total_images"] for r in runs)
    # an all-DUE level has n_valid == 0 (first invalid aborts the trial);
    # report 0/NaN rather than divide by zero
    p = n_top1 / n_valid if n_valid else 0.0
    ci = 1.96 * math.sqrt(p * (1 - p) / n_valid) if n_valid else 0.0
    return {
        "trials": sum(r["trials"] for r in runs),
        "total_images": total_images,
        "top1_rate": p, "top1_ci": ci,
        "numeric_rate": ((n_top1 + sum(r["numeric"] for r in runs)) / n_valid
                         if n_valid else float("nan")),
        "invalid_rate": n_invalid / total_images,
        "acc_clean": sum(r["clean_acc"] for r in runs) / len(runs),
        "acc_inj": sum(r["inj_acc_nondue"] for r in runs) / len(runs),
        "r2w_per_1000": sum(r["r2w"] for r in runs) / sum(r["trials"] for r in runs),
        "w2r_per_1000": sum(r["w2r"] for r in runs) / sum(r["trials"] for r in runs),
        "trials_top1": sum(r["trials_top1"] for r in runs),
        "trials_due": sum(r["trials_due"] for r in runs),
        "mean_dprob": (sum(r["mean_dprob"] * r["valid"] for r in runs) / n_valid
                       if n_valid else float("nan")),
        "max_dprob": max(r["max_dprob"] for r in runs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "artifacts/g5/campaign")
    parser.add_argument("--min-trials", type=int, default=1,
                        help="skip runs with fewer trials (e.g. 100 keeps "
                             "formal campaigns and drops smoke runs)")
    parser.add_argument("--workload", default=None,
                        choices=sorted(fault_model.WORKLOADS),
                        help="keep only runs of this workload -- REQUIRED "
                             "(enforced) when a campaign root holds more "
                             "than one workload's runs (e.g. "
                             "artifacts/g7/campaign: the v1 and v2 ladders "
                             "share level names L1..L5 with DIFFERENT BERs, "
                             "so pooling them silently mixes engines and "
                             "BER points)")
    args = parser.parse_args()

    runs = []
    skipped_workload = 0
    seen_workloads: dict[str, int] = {}
    for run_dir in sorted(args.root.glob("run_*")):
        if (run_dir / "summary.json").is_file():
            result = analyze_run(run_dir)
            if not result:
                continue
            wl = result.get("workload", "?")
            seen_workloads[wl] = seen_workloads.get(wl, 0) + 1
            if args.workload and wl != args.workload:
                skipped_workload += 1
                continue
            if result.get("trials", 0) >= args.min_trials:
                result["dir"] = run_dir.name
                runs.append(result)
    if args.workload and skipped_workload:
        print(f"runs of other workloads skipped: {skipped_workload}")
    if args.workload is None and len(seen_workloads) > 1:
        # fail-closed: level names collide at DIFFERENT BERs across
        # workloads, so pooling a whole root silently mixes engines and
        # BER points (the help text always required --workload here; the
        # code now refuses instead of trusting the caller)
        detail = ", ".join(f"{w} ({n} runs)"
                           for w, n in sorted(seen_workloads.items()))
        print(f"REFUSING to pool {args.root}: runs of multiple workloads "
              f"({detail}); their level names collide at different BERs. "
              "Pass --workload explicitly.", file=sys.stderr)
        return 2

    by_level: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        if "trials" in run:
            # bootstrap / refused runs carry level=None and no trials;
            # they are reported above but never pooled
            by_level[run["level"]].append(run)

    print(f"campaign runs analyzed: {len(runs)}")
    bad = [r for r in runs if r["status"] != "G5_CAMPAIGN_VERIFIED"]
    print(f"non-verified runs: {len(bad)}"
          + (f" -> {[r['dir'] for r in bad]}" if bad else ""))
    print()

    hdr = (f"{'lvl':3} {'dev':3} {'trials':6} {'top1/valid':>11} "
           f"{'numeric%':>9} {'DUE%':>6} {'cleanAcc':>8} {'injAcc':>7} "
           f"{'r2w/w2r':>9} {'mean|dP|':>8} {'trialTop1':>9} {'trialDUE':>8}")
    print(hdr)
    for level in sorted(by_level):
        for run in sorted(by_level[level], key=lambda r: r["device"]):
            if run["status"] != "G5_CAMPAIGN_VERIFIED":
                print(f"{level:3} {run['device']:3}  STATUS={run['status']}")
                continue
            print(f"{level:3} {run['device']:3} {run['trials']:6} "
                  f"{run['top1']:7}/{run['valid']:<7}"
                  f" {run['numeric_rate']*100:8.3f}%"
                  f" {run['invalid_rate']*100:5.3f}%"
                  f" {run['clean_acc']*100:7.2f}%"
                  f" {run['inj_acc_nondue']*100:6.2f}%"
                  f" {run['r2w']:3}/{run['w2r']:<3}"
                  f" {run['mean_dprob']:8.5f}"
                  f" {run['trials_top1']:5}/{run['trials']:3}"
                  f" {run['trials_due']:4}/{run['trials']:3}")

    devices = sorted({r["device"] for r in runs})
    print()
    print(f"pooled per level ({len(devices)} card"
          f"{'s' if len(devices) != 1 else ''}):")
    print(f"{'lvl':3} {'trials':6} {'top1 rate [95% CI]':>24} "
          f"{'numeric%':>9} {'DUE%':>6} {'cleanAcc':>8} {'injAcc':>7} "
          f"{'mean|dP|':>8} {'P(trial top1)':>13} {'P(trial DUE)':>12}")
    for level in sorted(by_level):
        good = [r for r in by_level[level]
                if r["status"] == "G5_CAMPAIGN_VERIFIED"]
        if not good:
            continue
        p = pooled(good)
        print(f"{level:3} {p['trials']:6} "
              f"{p['top1_rate']*100:10.4f}% +/-{p['top1_ci']*100:6.4f}"
              f" {p['numeric_rate']*100:8.3f}%"
              f" {p['invalid_rate']*100:5.3f}%"
              f" {p['acc_clean']*100:7.2f}%"
              f" {p['acc_inj']*100:6.2f}%"
              f" {p['mean_dprob']:8.5f}"
              f" {p['trials_top1']/p['trials']*100:12.1f}%"
              f" {p['trials_due']/p['trials']*100:11.1f}%")

    spread_lines = []
    for level in sorted(by_level):
        good = [r for r in by_level[level]
                if r["status"] == "G5_CAMPAIGN_VERIFIED"]
        if len(good) == 3:
            rates = [r["top1_rate"] * 100 for r in good]
            spread_lines.append(f"  {level}: {max(rates) - min(rates):.4f} pp "
                                f"({', '.join(f'{x:.4f}' for x in rates)})")
    if spread_lines:
        print()
        print("card spread (max-min across cards, top-1 rate in pp):")
        for line in spread_lines:
            print(line)

    print()
    print("restore echoes (allocation-level, summed over runs):")
    for level in sorted(by_level):
        good = [r for r in by_level[level]
                if r["status"] == "G5_CAMPAIGN_VERIFIED"]
        totals = defaultdict(int)
        for run in good:
            for key, value in run["restore"].items():
                totals[key] += value
        print(f"  {level}: exact={totals['restore_alloc_exact']} "
              f"mismatch={totals['restore_alloc_mismatch']} "
              f"skipped={totals['restore_alloc_skipped']} "
              f"mismatch_bytes={totals['restore_mismatch_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
