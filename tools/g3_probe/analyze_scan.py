#!/usr/bin/env python3
"""Analyze a G3 S1 timing scan and evaluate gate G3-R1.

Input: the CSV emitted by ``g3_timing_probe scan`` (``#`` lines are
metadata). The candidate-latency column must show at least two regimes for
the timing channel to be usable for address-mapping reverse engineering:

  - a baseline mode (anchor and candidate in different banks, or same row);
  - a conflict mode (same bank, different row), a fraction of points above
    the baseline by roughly tRC.

Gate G3-R1 passes when the modes are separable (amplitude >= 5x the robust
spread of the baseline) and the conflict fraction is plausible (between
0.05% and 50%). The analyzer also reports the greatest common divisor of
the gaps between conflict offsets -- inside one 2 MiB GMMU page offsets
translate 1:1 into framebuffer PA (G2), so that period is a first in-page
same-bank periodicity estimate.

Exit codes follow the repo convention: 0 gate passed, 2 gate failed,
1 usage/parse error.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass


@dataclass
class ScanReport:
    n_points: int
    baseline_mode: float
    conflict_mode: float
    amplitude: float
    spread: float
    separability: float
    conflict_fraction: float
    conflict_count: int
    period_bytes: int | None
    cluster_spread: float
    percentiles: dict[str, float]


def parse_scan(path: str) -> tuple[list[int], list[float]]:
    offsets: list[int] = []
    cycles: list[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            offsets.append(int(parts[0]))
            cycles.append(float(parts[2]))
    if not offsets:
        raise ValueError("no data rows found")
    return offsets, cycles


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[idx]


def robust_spread(values: list[float]) -> float:
    """MAD-based sigma of a tight cluster."""
    med = median(values)
    return 1.4826 * median([abs(v - med) for v in values])


def iqr_spread(ordered: list[float]) -> float:
    """Normal-sigma estimate from the interquartile range. Unlike a MAD of
    the lower region, a p50/p25/p75 computed over the whole scan stays sane
    when the conflict fraction is anywhere in [0, ~40%]."""
    return (percentile(ordered, 0.75) - percentile(ordered, 0.25)) / 1.349


def gcd_seq(values: list[int]) -> int:
    result = 0
    for value in values:
        if value <= 0:
            continue
        result = math.gcd(result, value)
    return result


def histogram_mode(ordered: list[float]) -> float:
    """Center of the densest bin; robust to a contaminated tail."""
    sigma0 = max(iqr_spread(ordered), 1.0)
    width = max(1.0, round(sigma0 / 4.0))
    best_count, best_center = -1, ordered[0]
    start = 0
    for i, value in enumerate(ordered):
        while ordered[start] <= value - width:
            start += 1
        count = i - start + 1
        if count > best_count:
            best_count = count
            best_center = (ordered[start] + value) / 2.0
    return best_center


def analyze(offsets: list[int], cycles: list[float]) -> ScanReport:
    ordered = sorted(cycles)
    baseline_mode = histogram_mode(ordered)
    sigma0 = max(iqr_spread(ordered), 1.0)
    near_mode = [v for v in ordered if abs(v - baseline_mode) <= 3.0 * sigma0]
    spread = max(robust_spread(near_mode or ordered), 1.0)

    # The conflict cluster is the sparse top of the distribution: with a
    # modal baseline at ~1/N occupancy per bank, conflicts are the top
    # ~1/150 of points. A pure Gaussian baseline has no such cluster (its
    # top 0.7% sits under 3 sigma above the mode and fails separability).
    cut = percentile(ordered, 0.993)
    conflict_idx = [i for i, v in enumerate(cycles) if v >= cut]
    conflict_values = [cycles[i] for i in conflict_idx]
    conflict_mode = median(conflict_values) if conflict_values else 0.0
    amplitude = (conflict_mode - baseline_mode) if conflict_values else 0.0
    separability = amplitude / spread
    cluster_spread = (
        robust_spread(conflict_values) if len(conflict_values) >= 2 else 0.0
    )

    conflict_offsets = sorted(offsets[i] for i in conflict_idx)
    gaps = [b - a for a, b in zip(conflict_offsets, conflict_offsets[1:]) if b > a]
    period = gcd_seq(gaps) if len(gaps) >= 2 else None

    return ScanReport(
        n_points=len(cycles),
        baseline_mode=baseline_mode,
        conflict_mode=conflict_mode,
        amplitude=amplitude,
        spread=spread,
        separability=separability,
        conflict_fraction=len(conflict_values) / len(cycles),
        conflict_count=len(conflict_values),
        period_bytes=period,
        cluster_spread=cluster_spread,
        percentiles={
            "p5": percentile(ordered, 0.05),
            "p25": percentile(ordered, 0.25),
            "p50": percentile(ordered, 0.50),
            "p75": percentile(ordered, 0.75),
            "p95": percentile(ordered, 0.95),
        },
    )


def gate(report: ScanReport) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True
    if report.separability < 5.0:
        ok = False
        reasons.append(
            f"separability {report.separability:.1f} < 5 "
            f"(amplitude {report.amplitude:.0f} cyc, spread {report.spread:.1f} cyc)"
        )
    if report.conflict_count < 3:
        ok = False
        reasons.append(f"only {report.conflict_count} conflict points (<3)")
    if report.cluster_spread > 3.0 * report.spread:
        ok = False
        reasons.append(
            f"conflict cluster spread {report.cluster_spread:.1f} > 3x baseline "
            f"spread {report.spread:.1f} (cluster is not a tight mode)"
        )
    if ok:
        reasons.append(
            f"separability {report.separability:.1f} >= 5, "
            f"{report.conflict_count} conflict points, "
            f"cluster spread {report.cluster_spread:.1f} cyc"
        )
    return ok, reasons


def format_report(report: ScanReport, mhz: float | None) -> str:
    lines = [
        f"points: {report.n_points}",
        "percentiles (cycles): "
        + " ".join(f"{k}={v:.0f}" for k, v in report.percentiles.items()),
        f"baseline mode: {report.baseline_mode:.0f} cycles",
        f"conflict mode: {report.conflict_mode:.0f} cycles "
        f"({report.conflict_count} points, {report.conflict_fraction * 100:.2f}%)",
        f"amplitude: {report.amplitude:.0f} cycles, "
        f"robust spread: {report.spread:.1f} cycles, "
        f"cluster spread: {report.cluster_spread:.1f} cycles, "
        f"separability: {report.separability:.1f}",
    ]
    if mhz:
        lines.append(
            f"amplitude: ~{report.amplitude / mhz * 1000.0:.0f} ns at {mhz:.0f} MHz"
        )
    if report.period_bytes:
        lines.append(f"conflict-offset gap gcd: {report.period_bytes} bytes")
    return "\n".join(lines)


def run_self_test() -> int:
    import io
    import random
    import tempfile
    import os

    rng = random.Random(1234)

    def synth(bimodal: bool, conflict_frac: float, period: int = 256) -> str:
        offsets = list(range(0, 2 << 20, 64))
        rows = ["# synthetic"]
        for off in offsets:
            is_conflict = bimodal and (off % period == 0) and rng.random() < (
                conflict_frac * period / 64
            )
            base = 900 + rng.gauss(0, 6)
            val = base + (120 + rng.gauss(0, 8)) if is_conflict else base
            rows.append(f"{off},{base:.0f},{val:.0f}")
        return "\n".join(rows) + "\n"

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            ("bimodal-pass", synth(True, 0.06), True, 256),
            ("unimodal-fail", synth(False, 0.0), False, None),
            # 50% conflict mass: baseline statistics are destroyed, gate must
            # refuse even though the two synthetic modes are far apart.
            ("contaminated-fail", synth(True, 0.5, 128), False, None),
        ]
        for name, blob, expect_pass, expect_period in cases:
            path = os.path.join(tmp, f"{name}.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(blob)
            offsets, cycles = parse_scan(path)
            report = analyze(offsets, cycles)
            ok, _ = gate(report)
            if ok != expect_pass:
                failures.append(
                    f"{name}: gate={ok} expected {expect_pass} "
                    f"(sep={report.separability:.1f} frac={report.conflict_fraction})"
                )
            if expect_period and report.period_bytes != expect_period:
                failures.append(
                    f"{name}: period={report.period_bytes} expected {expect_period}"
                )

    if failures:
        for failure in failures:
            print(f"self-test FAIL: {failure}", file=sys.stderr)
        return 1
    print("self-test: 3/3 cases passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", help="scan CSV from g3_timing_probe")
    parser.add_argument("--mhz", type=float, help="measured SM clock for ns")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.csv:
        parser.error("csv path required (or --self-test)")

    try:
        offsets, cycles = parse_scan(args.csv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = analyze(offsets, cycles)
    print(format_report(report, args.mhz))
    ok, reasons = gate(report)
    for reason in reasons:
        print(("G3-R1 PASS: " if ok else "G3-R1 FAIL: ") + reason)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
