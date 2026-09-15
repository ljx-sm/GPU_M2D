#!/usr/bin/python3
"""Offline replay of the TensorRT probe validation against captured runs.

Re-runs the complete post-run validation of ``run_g2_tensorrt_probe.py``
(ledger build, lifecycle/registry consistency, hold-interval, UUID, and
address-space checks) using only the ``events.csv`` / ``harness.log`` /
``g1_5_allocations.csv`` files a previous probe run already wrote. No GPU,
probe attachment, or sudo is involved, so orchestrator changes can be
verified against real captured evidence before a live re-run.

Exit codes: 0 all replayed runs are clean, 2 any validation failure, 1
missing or malformed inputs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from run_g2_tensorrt_probe import (
    LIFECYCLE_SKELETON,
    VARIABLE_EVENTS,
    build_allocation_ledger,
    normalize_gpu_uuid,
    parse_harness_output,
    read_csv,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_ROOT = PROJECT / "artifacts/g2/observer/trt"


class ReplayObserver:
    """Exposes the two attributes of G2Observer the validation reads."""

    def __init__(self, events_csv: Path):
        with events_csv.open(encoding="utf-8", newline="") as source:
            self.rows = list(csv.DictReader(source))
        self.lost_event_count = 0
        self.target_tgid = 0


def newest_run_per_device(root: Path) -> list[Path]:
    runs: dict[int, Path] = {}
    for run_dir in root.glob("run_trt_gpu*"):
        if not (run_dir / "events.csv").is_file():
            continue
        try:
            device = int(run_dir.name.split("gpu")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        current = runs.get(device)
        if current is None or run_dir.stat().st_mtime > current.stat().st_mtime:
            runs[device] = run_dir
    return [runs[device] for device in sorted(runs)]


def replay(run_dir: Path, hold_seconds: int) -> list[str]:
    required = [run_dir / name for name in
                ("events.csv", "harness.log", "g1_5_allocations.csv")]
    for path in required:
        if not path.is_file():
            return [f"missing {path.name} in {run_dir.name}"]

    output = (run_dir / "harness.log").read_text(encoding="utf-8")
    registry = read_csv(run_dir / "g1_5_allocations.csv")
    allocated, lifecycle, times = parse_harness_output(output)
    observer = ReplayObserver(run_dir / "events.csv")

    device_uuid = ""
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        import json
        device_uuid = json.loads(summary_path.read_text()).get("device_uuid", "")

    failures: list[str] = []
    if "GPU_M2D_G1_5_PASS" not in output:
        failures.append("runner did not report GPU_M2D_G1_5_PASS")
    expected_skeleton = [name for name in LIFECYCLE_SKELETON
                         if not (name in ("HOLD_BEGIN", "HOLD_END")
                                 and hold_seconds <= 0)]
    skeleton = [name for name in lifecycle if name not in VARIABLE_EVENTS]
    if skeleton != expected_skeleton:
        failures.append(f"lifecycle skeleton mismatch: {skeleton}")

    allocated_by_id = {record.get("allocation_id", ""): record
                       for record in allocated}
    if len(allocated_by_id) != len(allocated):
        failures.append("duplicate ALLOCATED records for one allocation_id")
    for row in registry:
        record = allocated_by_id.get(row["allocation_id"])
        if record is None:
            failures.append(f"{row['allocation_id']}: no ALLOCATED event")
        elif record.get("base_va", "").lower() != row["gpu_va"].lower() or \
                record.get("size_bytes") != row["size_bytes"]:
            failures.append(f"{row['allocation_id']}: ALLOCATED event disagrees with registry")
    freed_records = [dict(item.partition("=")[::2] for item in line.split(",")[1:])
                     for line in output.splitlines()
                     if line.startswith("GPU_M2D_EVENT,event=FREE,")]
    if len(freed_records) != len(allocated):
        failures.append(f"expected {len(allocated)} FREE events, got {len(freed_records)}")
    freed_by_id = {record.get("allocation_id", "") for record in freed_records}
    for allocation_id in allocated_by_id:
        if allocation_id not in freed_by_id:
            failures.append(f"{allocation_id}: no FREE event before PROCESS_END")

    teardown_end_ns = times.get("PROCESS_END", 0)
    snapshot_ns = times.get("SNAPSHOT_READY", 0)
    hold_begin_ns = times.get("HOLD_BEGIN", 0)
    hold_end_ns = times.get("HOLD_END", 0)

    segments_total = 0
    for row in registry:
        segments, ledger_failures = build_allocation_ledger(
            observer.rows, row, allocated_by_id.get(row["allocation_id"], {}),
            teardown_end_ns)
        failures.extend(ledger_failures)
        if row["active_at_injection"] == "1" and hold_seconds > 0 and segments:
            unmapped = int(segments[0]["unmapped_at_ns"])
            if not (snapshot_ns and hold_begin_ns and hold_end_ns
                    and segments[0]["mapped_at_ns"] < snapshot_ns
                    and hold_end_ns < unmapped):
                failures.append(
                    f"{row['allocation_id']}: mapping not alive across the hold interval")
        segments_total += len(segments)

    kernel_uuids = sorted({str(row["gpu_uuid"]) for row in observer.rows
                           if row["event_type"] == "PTE_HEADER" and row["gpu_uuid"]})
    if len(kernel_uuids) > 1:
        failures.append(f"multiple kernel GPU UUIDs observed: {kernel_uuids}")
    elif kernel_uuids and device_uuid and \
            normalize_gpu_uuid(kernel_uuids[0]) != normalize_gpu_uuid(device_uuid):
        failures.append(f"kernel GPU UUID {kernel_uuids[0]} != device UUID {device_uuid}")
    if len({str(row["address_space_id"]) for row in observer.rows
            if row["event_type"] in ("MAP_RETURN", "PTE_HEADER")}) > 1:
        failures.append("multiple UVM address-space IDs observed")

    print(f"{run_dir.name}: allocations={len(registry)} segments={segments_total} "
          f"events={len(observer.rows)} "
          f"{'CLEAN' if not failures else 'FAIL_CLOSED'}")
    for failure in failures:
        print(f"  FAIL: {failure}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="*",
                        help="run directories to replay (default: newest per device)")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hold-seconds", type=int, default=2,
                        help="hold interval the original run used")
    args = parser.parse_args()

    run_dirs = args.run_dirs or newest_run_per_device(args.root)
    if not run_dirs:
        print("no completed runs found to replay", file=sys.stderr)
        return 1
    failures = 0
    for run_dir in run_dirs:
        if replay(run_dir, args.hold_seconds):
            failures += 1
    print("REPLAY:", "ALL_CLEAN" if failures == 0
          else f"{failures}/{len(run_dirs)} runs FAIL_CLOSED")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
