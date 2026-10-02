#!/usr/bin/env python3
"""G8-T0 analyzer for one runner L2 probe pass (--l2-probe-out PREFIX).

Reads PREFIX_{ranges,images,boundaries,hist}.csv written by
apps/resnet50_int8_g1_5.cpp (run_l2_probe_pass) and reports:

  - neutrality (gate V7, docs/G8_CACHE_FAULT_PLAN.md §7): every image of
    the probed pass must equal the clean pass bit-for-bit;
  - cost: mean inference time per image vs mean probe-sweep time;
  - in-situ threshold sanity: the share of probe latencies that fall
    inside the T0 calibration gap (--gap LO HI); on real workload memory
    the hit/miss split should stay bimodal with an (almost) empty gap;
  - per-allocation residency over the pass: mean / min / max fraction of
    units classified L2-resident across the probe sweeps;
  - a coarse R_eff preview: the inference-time-weighted average of the
    resident bytes seen at the sweeps (unit counts x unit size). This is
    the allocation-level aggregate of plan §3.2's R_eff_bits, not the
    per-line value (the T0 hook records counts, not per-unit timelines).

Pure stdlib. Exit codes: 0 gate pass, 2 gate fail, 1 input error.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
import tempfile
from pathlib import Path


class InputError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise InputError(f"missing {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def per_alloc_fraction(sweeps, image, alloc, per_alloc) -> float:
    """Resident fraction of `alloc` at the sweep of `image`; an allocation
    not sampled at that sweep (sub-sampling) uses its pass mean."""
    if alloc in sweeps[image]:
        return sweeps[image][alloc]
    return per_alloc[alloc]["mean"]


def analyze(prefix: Path, gap: tuple[int, int]) -> dict:
    base = str(prefix)
    ranges = read_csv(Path(base + "_ranges.csv"))
    images = read_csv(Path(base + "_images.csv"))
    boundaries = read_csv(Path(base + "_boundaries.csv"))
    hist_rows = read_csv(Path(base + "_hist.csv"))
    if not ranges or not images or not boundaries:
        raise InputError(f"{prefix}: empty probe-pass outputs")

    unit_bytes = int(ranges[0]["unit_bytes"])
    units = {r["allocation_id"]: int(r["units"]) for r in ranges}
    labels = {r["allocation_id"]: (r["allocation_phase"], r["semantic_label"],
                                   int(r["size_bytes"])) for r in ranges}

    mismatches = sum(1 for r in images
                     if r["probability_bit_identical"] != "1"
                     or r["class_identical"] != "1")
    infer_ms = {int(r["image_index"]): float(r["infer_ms"]) for r in images}

    # image -> alloc -> resident fraction among the units probed at that
    # sweep (sub-sampled sweeps probe a staggered 1/stride of the units;
    # a range with no probed unit at a sweep carries no sample there)
    sweeps = collections.defaultdict(dict)
    probe_ms = {}
    for r in boundaries:
        image = int(r["image_index"])
        probed = int(r.get("probed", r["units"]))
        if probed:
            sweeps[image][r["allocation_id"]] = int(r["l2_hits"]) / probed
        probe_ms[image] = float(r["probe_ms"])
    sweep_images = sorted(sweeps)

    per_alloc = {}
    for alloc, n in units.items():
        fractions = [sweeps[i][alloc] for i in sweep_images
                     if alloc in sweeps[i]]
        if not fractions:
            raise InputError(f"{prefix}: allocation {alloc} never probed")
        per_alloc[alloc] = {
            "units": n, "mean": statistics.fmean(fractions),
            "min": min(fractions), "max": max(fractions),
            "first": fractions[0], "last": fractions[-1],
        }

    # Inference-time-weighted resident bytes: sweep at image i covers the
    # images [i, next sweep), weighted by their inference times.
    weighted = 0.0
    total_time = sum(infer_ms.values())
    for idx, start in enumerate(sweep_images):
        end = (sweep_images[idx + 1] if idx + 1 < len(sweep_images)
               else max(infer_ms) + 1)
        span = sum(infer_ms.get(i, 0.0) for i in range(start, end))
        resident_units = sum(units[a] * per_alloc_fraction(sweeps, start, a,
                                                           per_alloc)
                             for a in units)
        weighted += resident_units * unit_bytes * span
    r_eff_bytes = weighted / total_time if total_time else 0.0
    surface_bytes = sum(n for (_p, _l, n) in labels.values())

    hist = collections.Counter()
    for r in hist_rows:
        hist[int(r["bin_lo_cycles"])] += int(r["count"])
    total_samples = sum(hist.values())
    in_gap = sum(c for b, c in hist.items() if gap[0] <= b < gap[1])

    return {
        "prefix": str(prefix), "unit_bytes": unit_bytes,
        "images": len(images), "mismatches": mismatches,
        "sweeps": len(sweep_images),
        "infer_ms_mean": statistics.fmean(infer_ms.values()),
        "infer_ms_p50": statistics.median(infer_ms.values()),
        "probe_ms_mean": statistics.fmean(probe_ms.values()),
        "samples": total_samples,
        "in_gap_fraction": in_gap / total_samples if total_samples else 0.0,
        "per_alloc": per_alloc, "labels": labels,
        "r_eff_bytes": r_eff_bytes, "surface_bytes": surface_bytes,
    }


def report(result: dict, gap) -> list[str]:
    failures = []
    if result["mismatches"]:
        failures.append(f"{result['mismatches']} images differ from the "
                        "clean pass (probe pass not neutral)")
    print(f"== {result['prefix']}  (unit {result['unit_bytes']} B)")
    print(f"  images {result['images']}  mismatches {result['mismatches']}  "
          f"sweeps {result['sweeps']}")
    print(f"  inference ms/image mean {result['infer_ms_mean']:.3f} "
          f"p50 {result['infer_ms_p50']:.3f};  probe ms/sweep mean "
          f"{result['probe_ms_mean']:.4f}  "
          f"({100 * result['probe_ms_mean'] / result['infer_ms_mean']:.2f} % "
          "of one inference)")
    print(f"  latency samples {result['samples']}; inside calibration gap "
          f"[{gap[0]},{gap[1]}): {100 * result['in_gap_fraction']:.4f} %")
    print("  allocation          phase                    bytes      units  "
          "resident mean/min/max  first/last")
    for alloc, a in sorted(result["per_alloc"].items(),
                           key=lambda kv: -kv[1]["units"]):
        phase, _label, size = result["labels"][alloc]
        print(f"  {alloc:18s}  {phase:22s} {size:10d} {a['units']:9d}  "
              f"{a['mean']:.3f}/{a['min']:.3f}/{a['max']:.3f}       "
              f"{a['first']:.3f}/{a['last']:.3f}")
    print(f"  time-weighted resident bytes (R_eff preview) "
          f"{result['r_eff_bytes']:.0f} of surface {result['surface_bytes']} "
          f"({100 * result['r_eff_bytes'] / result['surface_bytes']:.1f} %)")
    print("G8_T0_PROBE_PASS_NEUTRAL" if not failures
          else "G8_T0_PROBE_PASS_FAIL")
    for f in failures:
        print(f"  - {f}")
    return failures


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "t"

        def write(name, header, rows):
            with open(f"{prefix}_{name}.csv", "w", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(header)
                w.writerows(rows)

        write("ranges", ["allocation_id", "gpu_va", "size_bytes", "units",
                         "unit_bytes", "allocation_phase", "semantic_label"],
              [["a", "0x1000", "1280", "10", "128", "deserialize_engine", "X"],
               ["b", "0x2000", "640", "5", "128", "create_execution_context",
                "Y"]])
        write("images", ["image_index", "infer_ms", "probability_bit_identical",
                         "class_identical"],
              [[i, "2.0", "1", "1"] for i in range(4)])
        # a fully resident, b half resident at both sweeps (every 2 images)
        write("boundaries", ["image_index", "probe_ms", "allocation_id",
                             "units", "l2_hits"],
              [[0, "0.01", "a", 10, 10], [0, "0.01", "b", 5, 2],
               [2, "0.01", "a", 10, 10], [2, "0.01", "b", 5, 3]])
        write("hist", ["allocation_id", "bin_lo_cycles", "count"],
              [["a", 304, 20], ["b", 304, 5], ["b", 704, 5]])
        res = analyze(prefix, (448, 592))
        assert res["mismatches"] == 0 and res["sweeps"] == 2
        assert res["per_alloc"]["b"]["min"] == 0.4
        assert res["per_alloc"]["b"]["max"] == 0.6
        # (12 + 13) units over equal spans -> mean 12.5 units x 128 B
        assert abs(res["r_eff_bytes"] - 12.5 * 128) < 1e-9, res["r_eff_bytes"]
        assert res["in_gap_fraction"] == 0.0
        assert report(res, (448, 592)) == []
        # one mismatched image fails the neutrality gate
        write("images", ["image_index", "infer_ms", "probability_bit_identical",
                         "class_identical"],
              [[0, "2.0", "1", "1"], [1, "2.0", "0", "1"],
               [2, "2.0", "1", "1"], [3, "2.0", "1", "1"]])
        res = analyze(prefix, (448, 592))
        assert res["mismatches"] == 1
        assert report(res, (448, 592)), "mismatch must fail"
        # sub-sampled sweeps: fractions come from probed units only, and a
        # range unprobed at a sweep falls back to its pass mean
        write("images", ["image_index", "infer_ms", "probability_bit_identical",
                         "class_identical"],
              [[i, "2.0", "1", "1"] for i in range(4)])
        write("boundaries", ["image_index", "probe_ms", "allocation_id",
                             "units", "probed", "l2_hits"],
              [[0, "0.01", "a", 10, 5, 5], [0, "0.01", "b", 5, 0, 0],
               [2, "0.01", "a", 10, 5, 4], [2, "0.01", "b", 5, 4, 2]])
        res = analyze(prefix, (448, 592))
        assert res["per_alloc"]["a"]["min"] == 0.8
        assert res["per_alloc"]["b"]["mean"] == 0.5  # one sample only
        # sweep 0: a 1.0*10 + b mean 0.5*5 = 12.5; sweep 2: 8 + 2.5 = 10.5
        assert abs(res["r_eff_bytes"] - 11.5 * 128) < 1e-9, res["r_eff_bytes"]
    print("analyze_l2_probe_pass self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix", nargs="*", type=Path)
    parser.add_argument("--gap", nargs=2, type=int, default=(400, 480),
                        metavar=("LO", "HI"),
                        help="T0 calibration gap in cycles (default: the "
                             "common gap of the three GPU 0 single-lane "
                             "runs at 16 probes/SM, threshold 440)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.prefix:
        parser.error("give one or more --l2-probe-out prefixes")
    failures = []
    try:
        for prefix in args.prefix:
            failures += report(analyze(prefix, tuple(args.gap)), args.gap)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
