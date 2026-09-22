#!/usr/bin/env python3
"""GPU_M2D G5 campaign aggregation (pure stdlib, offline).

Walks artifacts/g5/campaign/run_*/ (one dir per campaign = level x card),
recomputes every metric from the four result CSVs independently of the
runner and the orchestrator, and prints per-run and pooled-per-level
tables:

  - per-image: top-1 change rate / numeric SDC rate / DUE rate
    (binomial 95% CI on the pooled top-1 rate), vs the clean pass;
  - accuracy vs ground truth (clean vs injected; DUE counted as wrong
    in the primary number, evaluated-only as the bracket);
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
from collections import defaultdict
from pathlib import Path

PROB_TOL = 1.0e-6


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def analyze_run(run_dir: Path) -> dict | None:
    summary = json.loads((run_dir / "summary.json").read_text())
    level = summary["level"]
    device = summary["device"]
    if summary["status"] != "G5_CAMPAIGN_VERIFIED":
        return {"level": level, "device": device, "status": summary["status"],
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
    return {
        "level": level, "device": device,
        "status": summary["status"],
        "trials": n_trials, "images": images,
        "total_images": n_trials * per_trial_images,
        "sites": summary.get("total_sites"),
        "clean_acc": clean_correct / images,
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
    p = n_top1 / n_valid
    ci = 1.96 * math.sqrt(p * (1 - p) / n_valid) if n_valid else 0.0
    return {
        "trials": sum(r["trials"] for r in runs),
        "total_images": total_images,
        "top1_rate": p, "top1_ci": ci,
        "numeric_rate": (n_top1 + sum(r["numeric"] for r in runs)) / n_valid,
        "invalid_rate": n_invalid / total_images,
        "acc_clean": sum(r["clean_acc"] for r in runs) / len(runs),
        "acc_inj": sum(r["inj_acc_due_wrong"] for r in runs) / len(runs),
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
    args = parser.parse_args()

    runs = []
    for run_dir in sorted(args.root.glob("run_*")):
        if (run_dir / "summary.json").is_file():
            result = analyze_run(run_dir)
            if result and result.get("trials", 0) >= args.min_trials:
                result["dir"] = run_dir.name
                runs.append(result)

    by_level: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
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
                  f" {run['inj_acc_due_wrong']*100:6.2f}%"
                  f" {run['r2w']:3}/{run['w2r']:<3}"
                  f" {run['mean_dprob']:8.5f}"
                  f" {run['trials_top1']:5}/{run['trials']:3}"
                  f" {run['trials_due']:4}/{run['trials']:3}")

    print()
    print("pooled per level (3 cards):")
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

    print()
    print("card spread (max-min across cards, top-1 rate in pp):")
    for level in sorted(by_level):
        good = [r for r in by_level[level]
                if r["status"] == "G5_CAMPAIGN_VERIFIED"]
        if len(good) == 3:
            rates = [r["top1_rate"] * 100 for r in good]
            print(f"  {level}: {max(rates) - min(rates):.4f} pp "
                  f"({', '.join(f'{x:.4f}' for x in rates)})")

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
