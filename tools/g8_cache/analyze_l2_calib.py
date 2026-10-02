#!/usr/bin/env python3
"""G8-T0 analyzer for g8_l2_calib CSVs (docs/G8_CACHE_FAULT_PLAN.md §4, §7 V6).

Per parallelism level (threads per SM) it pools the latency histograms of
the known-hit states (warm, reprobe) and the known-miss state (cold) and
reports:

  - the hit and miss latency ranges and their gap;
  - a hit/miss threshold (midpoint of the empty gap when one exists,
    otherwise the minimum-error cut) and the misclassification rates;
  - probe throughput (lines per ms, median over rounds) and the projected
    full-surface probe time for the six G7-v2 surfaces, per 128-B line
    and per 32-B sector.

At the probe parallelism level it then applies that threshold to the
capacity sweep (hit fraction vs working set; must collapse near the L2
size if the probe really measures L2) and the sector test (does touching
sector 0 bring sectors 1..3 into L2?).

Gate V6 (per input file, at --probe-tps): an EMPTY gap between every hit
and every miss sample, cold classified miss >= 99.9 %, warm and reprobe
classified hit >= 99.9 %. Across files the threshold must be stable
(every file's threshold lies inside every other file's gap).

Pure stdlib. Exit codes: 0 gate pass, 2 gate fail, 1 usage/input error.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
import tempfile
from pathlib import Path

HIT_STATES = ("warm", "reprobe")
MISS_STATES = ("cold",)
MIN_CLASS_RATE = 0.999
# G7-v2 frozen injection surfaces (bytes), fault_model.WORKLOADS.
SURFACES = {
    "mobilenetv3": 9_087_912,
    "efficientnet_b0": 17_144_232,
    "deit_s": 26_010_832,
    "resnet50_v2": 28_832_268,
    "swin_t": 43_799_616,
    "vit_b": 93_867_728,
}


class InputError(RuntimeError):
    pass


def load(path: Path) -> dict:
    """hist[(mode, tps, state, param)] -> Counter(bin_lo -> count);
    ms[(...)] -> [kernel ms per round]; lines[(...)] -> lines per probe."""
    hist = collections.defaultdict(collections.Counter)
    ms = collections.defaultdict(list)
    lines = {}
    meta = None
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0] == "meta":
                meta = {"gpu": row[1], "sms": int(row[2]),
                        "l2_bytes": int(row[3]), "sm_mhz": float(row[6])}
            elif row[0] == "probe":
                key = (row[1], int(row[2]), row[3], int(row[5]))
                ms[key].append(float(row[7]))
                lines[key] = int(row[6])
            elif row[0] == "hist":
                key = (row[1], int(row[2]), row[3], int(row[5]))
                hist[key][int(row[6])] += int(row[7])
            else:
                raise InputError(f"{path}: unknown row kind {row[0]!r}")
    if meta is None or not hist:
        raise InputError(f"{path}: no meta row or no histograms")
    return {"meta": meta, "hist": hist, "ms": ms, "lines": lines}


def pooled(data: dict, mode: str, tps: int, states) -> collections.Counter:
    out = collections.Counter()
    for (m, t, s, _p), counter in data["hist"].items():
        if m == mode and t == tps and s in states:
            out.update(counter)
    return out


def bin_range(counter: collections.Counter, bin_cycles: int) -> tuple[int, int]:
    """(lowest bin start, highest bin END) of a histogram."""
    return min(counter), max(counter) + bin_cycles


def fraction_below(counter: collections.Counter, threshold: int) -> float:
    total = sum(counter.values())
    return sum(c for b, c in counter.items() if b < threshold) / total


def choose_threshold(hits, misses, bin_cycles):
    hit_lo, hit_hi = bin_range(hits, bin_cycles)
    miss_lo, miss_hi = bin_range(misses, bin_cycles)
    gap = miss_lo - hit_hi
    if gap >= 0:
        return (hit_hi + miss_lo) // 2, gap, (hit_hi, miss_lo)
    best = None
    for cut in sorted(set(hits) | set(misses)):
        err = (1 - fraction_below(hits, cut)) + fraction_below(misses, cut)
        if best is None or err < best[0]:
            best = (err, cut)
    return best[1], gap, None


def analyze_file(data: dict, probe_tps: int, bin_cycles: int) -> dict:
    report = {"meta": data["meta"], "levels": {}, "capacity": {},
              "sector": {}}
    levels = sorted({t for (m, t, _s, _p) in data["hist"] if m == "calib"})
    for tps in levels:
        hits = pooled(data, "calib", tps, HIT_STATES)
        misses = pooled(data, "calib", tps, MISS_STATES)
        threshold, gap, gap_bounds = choose_threshold(hits, misses, bin_cycles)
        warm = pooled(data, "calib", tps, ("warm",))
        reprobe = pooled(data, "calib", tps, ("reprobe",))
        lines_per_ms = []
        for key, values in data["ms"].items():
            if key[0] == "calib" and key[1] == tps:
                lines_per_ms.extend(data["lines"][key] / v for v in values
                                    if v > 0)
        rate = statistics.median(lines_per_ms) if lines_per_ms else 0.0
        projected = {name: {"lines_ms": (b / 128) / rate if rate else None,
                            "sectors_ms": (b / 32) / rate if rate else None}
                     for name, b in SURFACES.items()}
        report["levels"][tps] = {
            "hit_range": list(bin_range(hits, bin_cycles)),
            "miss_range": list(bin_range(misses, bin_cycles)),
            "gap_cycles": gap,
            "gap_bounds": list(gap_bounds) if gap_bounds else None,
            "threshold": threshold,
            "warm_hit_rate": fraction_below(warm, threshold),
            "reprobe_hit_rate": fraction_below(reprobe, threshold),
            "cold_miss_rate": 1 - fraction_below(misses, threshold),
            "lines_per_ms": rate,
            "projected_probe_ms": projected,
        }
    if probe_tps not in report["levels"]:
        raise InputError(f"no calib data at --probe-tps {probe_tps}")
    threshold = report["levels"][probe_tps]["threshold"]
    for (m, t, s, p), counter in sorted(data["hist"].items()):
        if t != probe_tps:
            continue
        if m == "capacity":
            report["capacity"].setdefault(p, collections.Counter()).update(
                counter)
        elif m == "sector":
            report["sector"].setdefault(p, collections.Counter()).update(
                counter)
    report["capacity"] = {w: fraction_below(c, threshold)
                          for w, c in sorted(report["capacity"].items())}
    report["sector"] = {s: fraction_below(c, threshold)
                        for s, c in sorted(report["sector"].items())}
    return report


def gate(reports: list[dict], probe_tps: int) -> list[str]:
    failures = []
    gaps = []
    for index, report in enumerate(reports):
        level = report["levels"][probe_tps]
        tag = f"file {index}"
        if level["gap_cycles"] < 0:
            failures.append(f"{tag}: hit and miss latencies overlap "
                            f"({level['gap_cycles']} cycles)")
        else:
            gaps.append(level["gap_bounds"])
        for name in ("warm_hit_rate", "reprobe_hit_rate", "cold_miss_rate"):
            if level[name] < MIN_CLASS_RATE:
                failures.append(f"{tag}: {name} {level[name]:.5f} < "
                                f"{MIN_CLASS_RATE}")
    if gaps and not failures:
        lo = max(g[0] for g in gaps)
        hi = min(g[1] for g in gaps)
        for index, report in enumerate(reports):
            t = report["levels"][probe_tps]["threshold"]
            if not lo <= t <= hi:
                failures.append(f"file {index}: threshold {t} outside the "
                                f"common gap [{lo}, {hi}]")
    return failures


def print_report(paths, reports, probe_tps, failures) -> None:
    for path, report in zip(paths, reports):
        meta = report["meta"]
        print(f"== {path}  ({meta['gpu']}, {meta['sms']} SMs, "
              f"L2 {meta['l2_bytes'] >> 20} MiB, SM {meta['sm_mhz']:.0f} MHz)")
        print("  tps  hit[min,max)   miss[min,max)   gap  thr  warm%   "
              "reprobe%  cold-miss%  lines/ms")
        for tps, lv in report["levels"].items():
            print(f"  {tps:4d} {lv['hit_range'][0]:5d},{lv['hit_range'][1]:5d}"
                  f"   {lv['miss_range'][0]:5d},{lv['miss_range'][1]:5d}"
                  f"  {lv['gap_cycles']:5d} {lv['threshold']:4d}"
                  f"  {100 * lv['warm_hit_rate']:7.3f}"
                  f"  {100 * lv['reprobe_hit_rate']:7.3f}"
                  f"   {100 * lv['cold_miss_rate']:8.3f}"
                  f"  {lv['lines_per_ms']:10.0f}")
        lv = report["levels"][probe_tps]
        print(f"  projected full-surface probe at {probe_tps} threads/SM "
              "(ms; 128-B lines / 32-B sectors):")
        for name, p in lv["projected_probe_ms"].items():
            print(f"    {name:16s} {p['lines_ms']:.3f} / {p['sectors_ms']:.3f}")
        print(f"  capacity (hit fraction after touching W twice, "
              f"{probe_tps} threads/SM):")
        print("    " + "  ".join(f"{w}MiB:{f:.3f}"
                                 for w, f in report["capacity"].items()))
        print("  sector (hit fraction of sector s after touching sector 0):")
        print("    " + "  ".join(f"s{s}:{f:.3f}"
                                 for s, f in report["sector"].items()))
    if failures:
        print("G8_T0_PROBE_CALIBRATION_FAIL")
        for f in failures:
            print(f"  - {f}")
    else:
        thresholds = [r["levels"][probe_tps]["threshold"] for r in reports]
        print(f"G8_T0_PROBE_CALIBRATION_PASS probe_tps={probe_tps} "
              f"thresholds={thresholds}")


def self_test() -> int:
    def write(path: Path, hit_max: int, miss_min: int) -> None:
        rows = [["meta", "TestGPU", "128", str(72 << 20), "32", "1",
                 "2500.0", "512"]]
        for state, lo, hi in (("warm", 288, hit_max), ("reprobe", 288, hit_max),
                              ("cold", miss_min, miss_min + 800)):
            rows.append(["probe", "calib", "32", state, "0", "32", "262144",
                         "0.02"])
            for b in range(lo - lo % 16, hi, 16):
                rows.append(["hist", "calib", "32", state, "0", "32", str(b),
                             "100"])
        for w, hit in ((16, True), (128, False)):
            rows.append(["probe", "capacity", "32", "touched2x", "0", str(w),
                         "1000", "0.01"])
            rows.append(["hist", "capacity", "32", "touched2x", "0", str(w),
                         "304" if hit else str(miss_min), "1000"])
        for s in range(4):
            rows.append(["probe", "sector", "32", "touch_s0", "0", str(s),
                         "1000", "0.01"])
            rows.append(["hist", "sector", "32", "touch_s0", "0", str(s),
                         "304" if s == 0 else str(miss_min), "1000"])
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.csv"
        write(good, 416, 608)
        rep = analyze_file(load(good), 32, 16)
        lv = rep["levels"][32]
        assert lv["gap_cycles"] > 0, lv
        assert 416 <= lv["threshold"] <= 608, lv
        assert lv["warm_hit_rate"] == 1.0 and lv["cold_miss_rate"] == 1.0
        assert rep["capacity"][16] == 1.0 and rep["capacity"][128] == 0.0
        assert rep["sector"][0] == 1.0 and rep["sector"][3] == 0.0
        assert gate([rep], 32) == []
        # two files with compatible gaps pass the stability check
        good2 = Path(tmp) / "good2.csv"
        write(good2, 400, 640)
        assert gate([rep, analyze_file(load(good2), 32, 16)], 32) == []
        # overlapping distributions fail the gate
        bad = Path(tmp) / "bad.csv"
        write(bad, 900, 608)
        failures = gate([analyze_file(load(bad), 32, 16)], 32)
        assert any("overlap" in f for f in failures), failures
        # a missing probe level is an input error
        try:
            analyze_file(load(good), 64, 16)
        except InputError:
            pass
        else:
            raise AssertionError("missing probe level must raise")
    print("analyze_l2_calib self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="*", type=Path)
    parser.add_argument("--probe-tps", type=int, default=32,
                        help="probe parallelism (threads per SM) to gate")
    parser.add_argument("--bin-cycles", type=int, default=16)
    parser.add_argument("--json", type=Path, help="write the report as JSON")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.csv:
        parser.error("give one or more calibration CSVs")
    try:
        reports = [analyze_file(load(p), args.probe_tps, args.bin_cycles)
                   for p in args.csv]
    except (InputError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    failures = gate(reports, args.probe_tps)
    print_report(args.csv, reports, args.probe_tps, failures)
    if args.json:
        args.json.write_text(json.dumps(
            {"probe_tps": args.probe_tps, "failures": failures,
             "files": [str(p) for p in args.csv], "reports": reports},
            indent=1, default=str))
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
