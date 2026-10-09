#!/usr/bin/env python3
"""Recompute every number quoted in docs/G8_T4_DISCUSSION.md from the
VERIFIED G8-T4 / G7-v2 run directories (nothing is hand-typed).

Sections (same order as the document):
  Q1  accuracy rises between adjacent BER levels: significance, collapsed
      trials, medians;
  Q2  crashes: the culprit cache flip of every mid-trial crash, hits in
      the crash-critical allocation and the per-hit fatality per mode,
      expected hits from the fault model, EfficientNet-B0 DRAM-only
      hits per level;
  Q3  ViT-B: per-level mean / median / collapsed / non-collapsed mean,
      DRAM-fault pairing (physical pages), cache flips landing on a
      DRAM-flipped bit, the zero-fault determinism run;
  Q4  accuracy per attempted trial at 1e-5, engine-written share of
      cache flips, borderline images, DRAM pairing for all models.

Per-trial accuracies are computed once from the image rows and cached in
artifacts/g8/t4/analysis/t4_trials.json (--reuse reads that cache).

  python3 tools/g8_cache/t4_discussion_numbers.py [--reuse]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "tools/g5_faultinj"))

import fault_model  # noqa: E402
from build_t4_results import MODELS  # noqa: E402
from threeway_table import MODES, find_run  # noqa: E402

CACHE = PROJECT / "artifacts/g8/t4/analysis/t4_trials.json"
ZERO_FAULT = PROJECT / "artifacts/g8/t4_zerofault/campaign"
CRIT = "trt-internal-2"        # crash-critical TensorRT context allocation
COLLAPSE = 40.0                # a trial below this top-1 % has collapsed
ROOT = {mode: root for mode, _, root in MODES}
FLIP = re.compile(r"event=CACHE_FLIPPED,trial_index=(\d+),cache_index=\d+,"
                  r"image=(\d+)")


def workload(model: str) -> str:
    return f"g7v2_imagenet1k_{model}"


def levels(model: str) -> list[dict]:
    return [r for r in fault_model.WORKLOADS[workload(model)]["levels"]
            if 1e-7 * (1 - 1e-9) <= r["ber"] <= 1e-5 * (1 + 1e-9)]


def segments(run: Path) -> list[tuple[Path, dict]]:
    """(segment dir, record) per runner process; old single-process runs
    have no segment_details and keep their files at the run root."""
    s = json.loads((run / "summary.json").read_text())
    segs = s.get("segment_details") or [
        {"dir": ".", "completed": s["trials_completed"], "crashed_trial": None}]
    return [(run / Path(seg["dir"]).name if seg["dir"] != "." else run, seg)
            for seg in segs]


def work_rows(seg_dir: Path, run: Path) -> list[dict]:
    path = seg_dir / "work.csv"
    if not path.exists():
        path = run / "work.csv"
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(l for l in source if not l.startswith("#")))


# ---------------------------------------------------------------- trials
def trial_table(reuse: bool) -> dict:
    if reuse and CACHE.exists():
        return json.loads(CACHE.read_text())
    out = {}
    for model, label in MODELS:
        for level in levels(model):
            for mode, _, root in MODES:
                run = find_run(root, workload(model), level["level"])
                with (run / "g1_5_g5_clean_pass.csv").open() as f:
                    truth = {r["image_index"]: int(r["label"])
                             for r in csv.DictReader(f)}
                correct: dict[int, int] = {}
                due: set[int] = set()
                with (run / "g1_5_g5_image_detail.csv").open() as f:
                    for r in csv.DictReader(f):
                        t = int(r["trial_index"])
                        correct.setdefault(t, 0)
                        if r["injected_class"] == "NA":
                            due.add(t)
                        elif int(r["injected_class"]) == truth[r["image_index"]]:
                            correct[t] += 1
                s = json.loads((run / "summary.json").read_text())
                out[f"{label}|{level['level']}|{mode}"] = {
                    "run": str(run.relative_to(PROJECT)),
                    "acc": {str(t): 100 * c / len(truth)
                            for t, c in correct.items() if t not in due},
                    "due": sorted(due),
                    "crash": s.get("process_fatal_count", 0) or 0}
                print(f"  read {label} {level['level']} {mode}",
                      file=sys.stderr, flush=True)
    CACHE.write_text(json.dumps(out))
    return out


def accs(d: dict, label: str, level: str, mode: str) -> list[float]:
    return list(d[f"{label}|{level}|{mode}"]["acc"].values())


# -------------------------------------------------------------------- Q1
def q1(d: dict) -> None:
    print("== Q1  accuracy rises between adjacent BER levels")
    rises = significant = median_rises = steps = 0
    for model, label in MODELS:
        names = [lv["level"] for lv in levels(model)]
        for mode, _, _ in MODES:
            for a, b in zip(names, names[1:]):
                steps += 1
                xa, xb = accs(d, label, a, mode), accs(d, label, b, mode)
                ma, mb = statistics.mean(xa), statistics.mean(xb)
                sa = statistics.stdev(xa) / math.sqrt(len(xa))
                sb = statistics.stdev(xb) / math.sqrt(len(xb))
                da, db = statistics.median(xa), statistics.median(xb)
                if db > da + 0.05:
                    median_rises += 1
                if mb > ma:
                    rises += 1
                    z = (mb - ma) / math.hypot(sa, sb)
                    significant += z >= 1.96
                    print(f"  {label:15} {mode:9} {a}->{b}: mean {ma:.2f}->"
                          f"{mb:.2f} (+{mb - ma:.2f}, z={z:.2f}); collapsed "
                          f"{sum(x < COLLAPSE for x in xa)}->"
                          f"{sum(x < COLLAPSE for x in xb)}; median "
                          f"{da:.2f}->{db:.2f}")
    print(f"  {rises} rises in {steps} steps; significant at 95 %: "
          f"{significant}; median rises > 0.05 pp: {median_rises}\n")


# -------------------------------------------------------------------- Q2
def crash_image(seg_dir: Path, local: int) -> int | None:
    last = None
    with (seg_dir / "harness.log").open(errors="replace") as source:
        for line in source:
            match = FLIP.search(line)
            if match and int(match.group(1)) == local:
                last = int(match.group(2))
    return last


def q2() -> None:
    print("== Q2  crashes")
    kinds: collections.Counter = collections.Counter()
    culprits: collections.Counter = collections.Counter()
    crit_any = collections.Counter()
    for model, label in MODELS:
        for level in levels(model):
            for mode in ("sram_only", "dram_sram"):
                run = find_run(ROOT[mode], workload(model), level["level"])
                for seg_dir, seg in segments(run):
                    if seg.get("crashed_trial") is None:
                        continue
                    local = seg["completed"]
                    image = crash_image(seg_dir, local)
                    if image is None:
                        kinds[(label, mode, "first inference (DRAM)")] += 1
                        continue
                    kinds[(label, mode, "mid-trial (cache)")] += 1
                    with (seg_dir / "cache_work.csv").open() as f:
                        flips = [r for r in csv.DictReader(f)
                                 if int(r["trial_index"]) == local
                                 and int(r["start_image"]) == image]
                    for r in flips:
                        culprits[(label, r["allocation_id"],
                                  r["cache_class"])] += 1
                    crit_any[label] += any(r["allocation_id"] == CRIT
                                           for r in flips)
    print("  crashes in cache modes by kind:")
    for key, n in sorted(kinds.items()):
        print(f"    {key}: {n}")
    print("  flips starting at the crash image, by allocation (candidates):")
    for key, n in sorted(culprits.items()):
        print(f"    {key}: {n}")
    for label in sorted(crit_any):
        n = sum(v for k, v in kinds.items()
                if k[0] == label and k[2].startswith("mid"))
        print(f"  {label}: mid-trial crashes whose crash-image flips include "
              f"{CRIT}: {crit_any[label]} of {n}")
    print()
    for model, label in MODELS:
        if label not in ("Swin-T", "EfficientNet-B0"):
            continue
        size, r_bytes = crit_size(model), \
            fault_model.WORKLOADS[workload(model)]["resident_bytes_nominal"]
        lv = levels(model)
        events = sum(r["s"] + r["d"] + r["t"] for r in lv)
        bits = sum(r["bits"] for r in lv)
        print(f"  {label}: {CRIT} = {size} B = {100 * size / r_bytes:.3f} % "
              f"of R; bits/event {bits / events:.2f}; expected hits/trial: "
              f"DRAM events {events / 7 * size / r_bytes:.2f}, SRAM flips "
              f"{bits / 7 * size / r_bytes:.2f}")
        for mode, name, _ in MODES:
            obs = hit_observations(model, mode)
            crashes = sum(c for _, c in obs)
            no_hit = sum(1 for k, c in obs if c and k == 0)
            print(f"    {name:9} trials {len(obs)}, crashes {crashes} "
                  f"(without a {CRIT} hit: {no_hit}), mean hits/trial "
                  f"{sum(k for k, _ in obs) / len(obs):.2f}, fatality per "
                  f"hit q = {fatality(obs):.3f}")
    print(f"  EfficientNet-B0 DRAM-only per level ({CRIT} DRAM events):")
    for level in levels("efficientnet_b0"):
        obs = hit_observations("efficientnet_b0", "dram_only", level["level"])
        print(f"    {level['level']}: trials with >=1 hit "
              f"{sum(k > 0 for k, _ in obs)}, events {sum(k for k, _ in obs)},"
              f" crashes {sum(c for _, c in obs)} (with a hit: "
              f"{sum(1 for k, c in obs if c and k)})")
    print()


def crit_size(model: str) -> int:
    run = find_run(ROOT["sram_only"], workload(model), levels(model)[0]["level"])
    seg_dir = segments(run)[0][0]
    with (seg_dir / "g1_5_allocations.csv").open() as f:
        return {r["allocation_id"]: int(r["size_bytes"])
                for r in csv.DictReader(f)}[CRIT]


def hit_observations(model: str, mode: str,
                     only_level: str | None = None) -> list[tuple[int, bool]]:
    """(independent hits in CRIT, crashed) per trial: DRAM hits are fault
    events (1-3 bits each), SRAM hits are applied cache flips (up to the
    crash image for the dying trial)."""
    obs = []
    for level in levels(model):
        if only_level and level["level"] != only_level:
            continue
        run = find_run(ROOT[mode], workload(model), level["level"])
        for seg_dir, seg in segments(run):
            dying = seg["completed"] if seg.get("crashed_trial") is not None \
                else None
            trials = seg["completed"] + (dying is not None)
            hits: collections.Counter = collections.Counter()
            if mode != "sram_only":
                events = collections.defaultdict(set)
                for r in work_rows(seg_dir, run):
                    if r["allocation_id"] == CRIT:
                        events[int(r["trial_index"])].add(r["event_index"])
                for t, e in events.items():
                    hits[t] += len(e)
            if mode != "dram_only":
                result = seg_dir / "g1_5_g8_cache_site_result.csv"
                if result.exists():
                    with result.open() as f:
                        for r in csv.DictReader(f):
                            if r["applied"] == "1" and r["allocation_id"] == CRIT:
                                hits[int(r["trial_index"])] += 1
                if dying is not None:
                    image = crash_image(seg_dir, dying)
                    if image is not None:
                        with (seg_dir / "cache_work.csv").open() as f:
                            for r in csv.DictReader(f):
                                if int(r["trial_index"]) == dying and \
                                        r["allocation_id"] == CRIT and \
                                        int(r["start_image"]) <= image:
                                    hits[dying] += 1
            obs += [(hits[t], t == dying) for t in range(trials)]
    return obs


def fatality(obs: list[tuple[int, bool]]) -> float:
    """MLE of q in P(no crash | k hits) = (1 - q)^k (ternary search)."""
    def loglik(q: float) -> float:
        return sum(math.log(1 - (1 - q) ** k) if c else k * math.log(1 - q)
                   for k, c in obs if k > 0)
    lo, hi = 1e-6, 0.999
    for _ in range(100):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (a, hi) if loglik(a) < loglik(b) else (lo, b)
    return (lo + hi) / 2


# -------------------------------------------------------------------- Q3
def pages(run: Path) -> dict:
    path = run / "snapshot/snapshot_pages.csv"
    if not path.exists():
        path = run / "segment_000/snapshot/snapshot_pages.csv"
    with path.open() as f:
        return {(r["allocation_id"], int(r["va_page_base"], 16)
                 - int(r["allocation_va_base"], 16)): r["fb_pa_page_base"]
                for r in csv.DictReader(f)
                if not r["allocation_id"].startswith("g8-")}


def q3(d: dict) -> None:
    print("== Q3  ViT-B: mean / median / collapsed / non-collapsed mean")
    for level in levels("vit_base_patch16_224"):
        cells = []
        for mode, name, _ in MODES:
            x = accs(d, "ViT-B", level["level"], mode)
            kept = [v for v in x if v >= COLLAPSE]
            cells.append(f"{name} {statistics.mean(x):.2f}/"
                         f"{statistics.median(x):.2f}/"
                         f"{sum(v < COLLAPSE for v in x)}/"
                         f"{statistics.mean(kept):.2f}")
        print(f"  {level['level']}: " + " | ".join(cells))
    same_bit = same_byte = total = 0
    for level in levels("vit_base_patch16_224"):
        run = find_run(ROOT["dram_sram"], workload("vit_base_patch16_224"),
                       level["level"])
        for seg_dir, _ in segments(run):
            bit_set = collections.defaultdict(set)
            byte_set = collections.defaultdict(set)
            for r in work_rows(seg_dir, run):
                bit_set[r["trial_index"]].add(
                    (r["allocation_id"], r["byte_offset"], r["bit_in_byte"]))
                byte_set[r["trial_index"]].add(
                    (r["allocation_id"], r["byte_offset"]))
            with (seg_dir / "g1_5_g8_cache_site_result.csv").open() as f:
                for r in csv.DictReader(f):
                    if r["applied"] != "1":
                        continue
                    total += 1
                    t = r["trial_index"]
                    same_bit += (r["allocation_id"], r["byte_offset"],
                                 r["bit_in_byte"]) in bit_set[t]
                    same_byte += (r["allocation_id"],
                                  r["byte_offset"]) in byte_set[t]
    print(f"  ViT-B DRAM+SRAM: applied cache flips {total}; on a DRAM-flipped "
          f"bit {same_bit}; on a DRAM-flipped byte {same_byte}")
    for run in sorted(ZERO_FAULT.glob("run_L1_*")):
        with (run / "g1_5_g5_clean_pass.csv").open() as f:
            clean = {r["image_index"]: (r["clean_class"], r["clean_probability"])
                     for r in csv.DictReader(f)}
        per_trial = collections.defaultdict(lambda: [0, 0])
        with (run / "g1_5_g5_image_detail.csv").open() as f:
            for r in csv.DictReader(f):
                same = (r["injected_class"], r["injected_probability"]) == \
                    clean[r["image_index"]]
                per_trial[r["trial_index"]][0] += same
                per_trial[r["trial_index"]][1] += 1
        print(f"  zero-fault run {run.name}: images identical to the clean "
              f"pass (class and probability) per trial: "
              f"{ {t: f'{a}/{n}' for t, (a, n) in sorted(per_trial.items())} }")
    print()


# -------------------------------------------------------------------- Q4
def q4(d: dict) -> None:
    print("== Q4  at 1e-5: top-1 over completed trials -> per attempted "
          "trial (DUE / crash counted as 0)")
    for model, label in MODELS:
        top = levels(model)[-1]["level"]
        cells = []
        for mode, name, _ in MODES:
            x = accs(d, label, top, mode)
            cells.append(f"{name} {statistics.mean(x):.2f} -> "
                         f"{sum(x) / 100:.2f}")
        print(f"  {label:15} " + " | ".join(cells))
    print("  engine-written share of applied SRAM-only cache flips at 1e-5 "
          "(segments that finished):")
    for model, label in MODELS:
        run = find_run(ROOT["sram_only"], workload(model), levels(model)[-1]["level"])
        counts: collections.Counter = collections.Counter()
        for _, seg in segments(run):
            counts.update((seg.get("g8_cache") or {}).get("removal_counts") or {})
        written = counts["restored"] + counts["overwritten"]
        print(f"    {label:15} {100 * written / (written + counts['re_xor']):.1f} %"
              f" (read-only {counts['re_xor']}, engine-written {written})")
    print("  borderline images (SRAM-only, lowest BER): changed in >= 90 % of "
          "trials; median clean top-1 probability changed vs never changed")
    for model, label in MODELS:
        run = find_run(ROOT["sram_only"], workload(model), levels(model)[0]["level"])
        with (run / "g1_5_g5_clean_pass.csv").open() as f:
            clean = {r["image_index"]: (int(r["clean_class"]), int(r["label"]),
                                        float(r["clean_probability"]))
                     for r in csv.DictReader(f)}
        changed: collections.Counter = collections.Counter()
        to_truth = 0
        trials = set()
        with (run / "g1_5_g5_image_detail.csv").open() as f:
            for r in csv.DictReader(f):
                trials.add(r["trial_index"])
                if r["injected_class"] != "NA" and \
                        int(r["injected_class"]) != clean[r["image_index"]][0]:
                    changed[r["image_index"]] += 1
        always = [i for i, c in changed.items() if c >= 0.9 * len(trials)]
        never = [clean[i][2] for i in clean if i not in changed]
        prob = statistics.median(clean[i][2] for i in always) if always else float("nan")
        wrong = sum(clean[i][0] != clean[i][1] for i in always)
        print(f"    {label:15} {levels(model)[0]['level']}: {len(always):3d} images"
              f" (clean-wrong {wrong}); median prob {prob:.3f} vs "
              f"{statistics.median(never):.3f}")
    print("  DRAM fault pairing: physical pages equal, DRAM-only vs DRAM+SRAM "
          "(lowest BER level)")
    for model, label in MODELS:
        name = levels(model)[0]["level"]
        a = pages(find_run(ROOT["dram_only"], workload(model), name))
        b = pages(find_run(ROOT["dram_sram"], workload(model), name))
        print(f"    {label:15} {sum(a[k] == b.get(k) for k in a)} of {len(a)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true",
                        help=f"read per-trial accuracies from {CACHE.name}")
    args = parser.parse_args()
    d = trial_table(args.reuse)
    q1(d)
    q2()
    q3(d)
    q4(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
