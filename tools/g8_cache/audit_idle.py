#!/usr/bin/env python3
"""G8-T4 idle-GPU audit: was every cache-mode campaign run of a model
measured and executed on an otherwise idle GPU 0? (User rule 2026-10-04:
a co-tenant shrinks the measured R_eff and hence n_cache, so a level run
next to one is not comparable and must be re-run.)

Per run (and per restart segment) three independent checks:

1. start   -- the orchestrator's cotenancy_at_start: no other compute app
              on the run's GPU (apps on other GPUs are fine) and GPU-0
              memory at the idle floor (<= 64 MiB);
2. residency -- every segment's map: R_eff == the full surface, auto
              stride 1, and n_cache == round(BER_cache x surface bits),
              i.e. the idle value;
3. trials  -- per-trial pass time (TRIAL_BEGIN -> TRIAL_INJECTED_END),
              (DUE trials excluded) corrected for its one known
              deterministic driver: every
              DRAM site in the input binding is re-applied after each of
              the 10K per-image input copies (~150 ms per site per trial).
              A co-tenant shows as a sustained block of slow residuals:
              flagged when >= 4 consecutive trials exceed max(0.3 s,
              4 x MAD) and that streak has shuffled p < 0.01.

Only checks 1-2 decide the verdict (OK / RERUN): a run's numbers depend
on its residency pass alone -- the pass fixes R_eff, n_cache and every
cache flip's start/last image, the runner then XORs on that schedule and
inference is deterministic, so a co-tenant arriving mid-trial changes
trial durations but no output. Check 3 is reported as a NOTE (it also
cannot tell GPU from CPU contention).

  python3 tools/g8_cache/audit_idle.py deit_small_patch16_224
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOTS = [("dram_sram", PROJECT / "artifacts/g8/t4/campaign"),
         ("sram_only", PROJECT / "artifacts/g8/t4/sram_only/campaign")]
IDLE_MEMORY_MIB = 64
EVENT = re.compile(r"GPU_M2D_EVENT,event=(TRIAL_BEGIN|TRIAL_INJECTED_END),"
                   r"trial_index=(\d+),.*?wall_time_ns=(\d+)")
DUE_PASS_SECONDS = 2.0     # DUE trials are excluded by outcome; this also
                           # drops any other truncated pass
MIN_STREAK = 4
STREAK_P = 0.01


def segment_rows(segment: dict, run_dir: Path) -> list[tuple[float, int]]:
    """(pass seconds, input-binding DRAM sites) per full trial pass."""
    seg_dir = Path(segment["harness_log"]).parent
    work = seg_dir / "work.csv"
    if not work.is_file():
        work = run_dir / "work.csv"
    binding: dict[int, int] = {}
    with work.open(encoding="utf-8") as source:
        for row in csv.DictReader(l for l in source if not l.startswith("#")):
            trial = int(row["trial_index"])
            binding[trial] = binding.get(trial, 0) + \
                row["allocation_id"].startswith("trt-binding-data")
    trial_file = seg_dir / "g1_5_g5_trial_result.csv"
    due = set()
    if trial_file.is_file():
        with trial_file.open(encoding="utf-8") as source:
            due = {row["trial_index"] for row in csv.DictReader(source)
                   if row["injected_outcome"] == "DUE_INVALID_OUTPUT"}
    begin: dict[str, int] = {}
    rows = []
    with open(segment["harness_log"], errors="replace") as source:
        for line in source:
            match = EVENT.search(line)
            if not match:
                continue
            kind, trial, ns = match.group(1), match.group(2), int(match.group(3))
            if kind == "TRIAL_BEGIN":
                begin[trial] = ns
            elif trial in begin:
                seconds = (ns - begin[trial]) / 1e9
                if trial not in due and seconds > DUE_PASS_SECONDS:
                    rows.append((seconds, binding.get(int(trial), 0)))
    return rows


def longest_streak(values: list[float], threshold: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value > threshold else 0
        best = max(best, current)
    return best


def timing_check(segments: list[list[tuple[float, int]]]) -> tuple[bool, str]:
    sites = [n for rows in segments for _, n in rows]
    seconds = [s for rows in segments for s, _ in rows]
    slope = 0.0
    if len(set(sites)) > 1:
        mean_n, mean_s = statistics.mean(sites), statistics.mean(seconds)
        slope = (sum((n - mean_n) * (s - mean_s) for n, s in zip(sites, seconds))
                 / sum((n - mean_n) ** 2 for n in sites))
    residuals = []
    for rows in segments:            # per-segment baseline (fresh process)
        res = [s - slope * n for s, n in rows]
        base = statistics.median(res)
        residuals += [r - base for r in res]
    mad = statistics.median(abs(r) for r in residuals)
    threshold = max(0.3, 4 * mad)
    streak = longest_streak(residuals, threshold)
    rng = random.Random(0)
    shuffled = [longest_streak(rng.sample(residuals, len(residuals)), threshold)
                for _ in range(2000)]
    p = sum(x >= streak for x in shuffled) / len(shuffled)
    flagged = streak >= MIN_STREAK and p < STREAK_P
    return flagged, (f"slow block {streak} trials > {threshold:.2f}s "
                     f"(p={p:.3f}, {slope * 1000:.0f} ms/binding site)")


def audit_run(mode: str, summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text())
    run_dir = summary_path.parent
    reasons = []
    cot = s.get("cotenancy_at_start") or {}
    uuid = s.get("device_uuid", "")
    on_gpu = [app for app in cot.get("compute_apps_all_gpus") or []
              if uuid and app.startswith(uuid)]
    if on_gpu:
        reasons.append(f"co-tenant on GPU at start: {on_gpu}")
    memory = cot.get("device_memory", "")
    try:
        used = int(memory.split(",")[1].split()[0])
    except (IndexError, ValueError):
        used = None
    if used is None or used > IDLE_MEMORY_MIB:
        reasons.append(f"GPU memory at start {used} MiB")
    timing = []
    for segment in s["segment_details"]:
        g8 = segment.get("g8_cache") or {}
        checks = g8.get("self_checks") or {}
        idle_n = round(g8["cache_ber"] * g8["surface_bits"])
        tag = f"seg{segment.get('segment')}"
        if abs(g8["r_eff_bits"] - g8["surface_bits"]) > 1e-3 * g8["surface_bits"] \
                and g8["surface_bits"] <= 72 * 2**20 * 8:
            reasons.append(f"{tag}: R_eff {g8['r_eff_bits'] / g8['surface_bits']:.4f} of surface")
        if checks.get("stride") != 1 and g8["surface_bits"] <= 72 * 2**20 * 8:
            reasons.append(f"{tag}: auto stride {checks.get('stride')}")
        if g8["n_cache"] != idle_n and g8["surface_bits"] <= 72 * 2**20 * 8:
            reasons.append(f"{tag}: n_cache {g8['n_cache']} != idle {idle_n}")
        timing.append(segment_rows(segment, run_dir))
    flagged, note = timing_check(timing)
    return {"mode": mode, "level": s["level"], "run": run_dir.name,
            "segments": len(s["segment_details"]), "reasons": reasons,
            "timing": ("NOTE mid-run " + note) if flagged else ""}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    args = parser.parse_args()
    workload = "g7v2_imagenet1k_" + args.model.removeprefix("g7v2_imagenet1k_")
    results = []
    for mode, root in ROOTS:
        for path in sorted(root.glob("run_L*/summary.json")):
            s = json.loads(path.read_text())
            if s.get("workload") != workload or \
                    s.get("status") != "G5_CAMPAIGN_VERIFIED" or \
                    s.get("trials_requested", 0) < 100:
                continue
            results.append(audit_run(mode, path))
    if not results:
        print(f"no verified cache-mode runs for {workload}")
        return 1
    for r in sorted(results, key=lambda r: (r["mode"], r["level"])):
        verdict = "RERUN" if r["reasons"] else "OK"
        print(f"{r['mode']:9} {r['level']}  {verdict:5}  {r['run']}  "
              f"segs={r['segments']}  " + "; ".join(r["reasons"])
              + ("  " + r["timing"] if r["timing"] else ""))
    return 2 if any(r["reasons"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
