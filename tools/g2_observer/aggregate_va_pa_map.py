#!/usr/bin/python3
"""Aggregate passing G2 TensorRT probe runs into artifacts/g2/gpu_va_pa_map.csv.

Scans the run directories under artifacts/g2/observer/trt/, keeps only the
latest run per device whose summary status is
G2_TENSORRT_LOCAL_PA_FULL_COVERAGE_OBSERVED, and concatenates their page-level
gpu_va_pa_map.csv files (schema gpu-m2d.g2-observer.tensorrt-probe.v1) into
the canonical G2 map. Mappings are never reused across runs: the output is
regenerated from the newest passing run of each device on every invocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

PASS_STATUS = "G2_TENSORRT_LOCAL_PA_FULL_COVERAGE_OBSERVED"


def latest_passing_runs(trt_root: Path) -> dict[int, Path]:
    latest: dict[int, Path] = {}
    for summary_path in trt_root.glob("run_*/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if summary.get("status") != PASS_STATUS:
            continue
        run_dir = summary_path.parent
        if not (run_dir / "gpu_va_pa_map.csv").is_file():
            continue
        device = int(summary["device"])
        current = latest.get(device)
        if current is None or run_dir.name > current.name:
            latest[device] = run_dir
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trt-root", type=Path,
                        default=PROJECT / "artifacts/g2/observer/trt")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "artifacts/g2/gpu_va_pa_map.csv")
    args = parser.parse_args()

    runs = latest_passing_runs(args.trt_root)
    if not runs:
        print("G2_MAP_AGGREGATE_FAIL: no passing TensorRT probe runs found", file=sys.stderr)
        return 2
    device_count = len(subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, check=True).stdout.split())
    missing = sorted(set(range(device_count)) - set(runs))
    if missing:
        print(f"G2_MAP_AGGREGATE_FAIL: devices {missing} have no passing run", file=sys.stderr)
        return 2

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for device in sorted(runs):
        map_path = runs[device] / "gpu_va_pa_map.csv"
        with map_path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            block = list(reader)
        if not block:
            print(f"G2_MAP_AGGREGATE_FAIL: empty map for device {device}", file=sys.stderr)
            return 2
        rows.extend(block)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    devices = sorted({int(row["device"]) for row in rows})
    allocations = len({row["allocation_id"] for row in rows})
    print(f"G2_MAP_AGGREGATE_PASS output={args.output} devices={devices} "
          f"allocations={allocations} rows={len(rows)} "
          f"runs={[runs[d].name for d in devices]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
