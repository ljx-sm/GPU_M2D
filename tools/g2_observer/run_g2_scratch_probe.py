#!/usr/bin/python3
"""GPU_M2D G2 scratch probe orchestrator.

Runs one pre-allocation-gated observation of a single scratch allocation
(``device`` = plain cudaMalloc, ``vmm`` = CUDA VMM) under the read-only
G2 PTE observer. Must be started through ``sudo`` from the research
account: root is used only for the kprobe attachment, while the CUDA
child is dropped back to the invoking user. Nothing is modified on the
driver side and no other GPU process is touched.

Sequence:
  1. validate the pinned driver/PTE contract (module hashes, symbols);
  2. attach the read-only probes;
  3. start the harness, which blocks before creating any CUDA context;
  4. set the target TGID and open the gate;
  5. watch MAP/PTE/FREE events, decode every captured PTE;
  6. validate coverage, lifetime enclosure, and the optional one-bit XOR
     closeout;
  7. write events.csv / samples.csv / summary.json under artifacts/g2/.

Ported from the REMU gpu_va_pa_mapping orchestrators (frozen 2026-09-08)
and adapted to the GPU_M2D full-PTE observer and markers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pwd
import select
import subprocess
import sys
import time
from pathlib import Path

from g2_observer import (
    DEFAULT_CONTRACT,
    DEFAULT_OPEN_SOURCE_REPO,
    G2Observer,
    load_and_validate_contract,
    sha256,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

EXPECTED_LIFECYCLE = [
    "PROCESS_READY",
    "WAIT_PRE_ALLOC_GATE",
    "PRE_ALLOC_GATE_OPEN",
    "CONTEXT_BEGIN",
    "CONTEXT_READY",
    "ALLOCATION_BEGIN",
    "ALLOCATED",
    "ACCESS_BEGIN",
    "ACCESS_END",
    "HOLD_BEGIN",
    "HOLD_END",
    "FREE_BEGIN",
    "FREE_END",
]


def drop_to_invoking_user() -> tuple[int, int]:
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if not sudo_uid or not sudo_gid or int(sudo_uid) == 0:
        raise RuntimeError("run through sudo from a non-root research account so the CUDA child is unprivileged")
    return int(sudo_uid), int(sudo_gid)


def child_preexec(uid: int, gid: int):
    username = pwd.getpwuid(uid).pw_name

    def drop() -> None:
        os.initgroups(username, gid)
        os.setgid(gid)
        os.setuid(uid)

    return drop


def chown_outputs(paths: list[Path], uid: int, gid: int) -> None:
    for path in paths:
        if path.exists():
            os.chown(path, uid, gid)


def read_until_marker(process: subprocess.Popen[bytes], marker: bytes,
                      timeout_seconds: float) -> tuple[bytes, bool]:
    """Read an unbuffered child pipe until a complete marker line arrives.

    TextIOWrapper.readline() with select() can prefetch later lines into
    the userspace buffer, after which select() waits forever on an empty
    kernel pipe. Reading raw bytes keeps readiness and data in sync.
    """
    if process.stdout is None:
        raise RuntimeError("harness stdout pipe is unavailable")
    fd = process.stdout.fileno()
    captured = bytearray()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        wait_seconds = min(0.1, max(0.0, deadline - time.monotonic()))
        readable, _, _ = select.select([fd], [], [], wait_seconds)
        if readable:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            captured.extend(chunk)
            if any(line.startswith(marker) for line in captured.splitlines()):
                return bytes(captured), True
        if process.poll() is not None:
            break
    return bytes(captured), False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", choices=("device", "vmm"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size-mib", type=int, default=8)
    parser.add_argument("--hold-seconds", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=25.0)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "artifacts/g2/observer")
    parser.add_argument("--no-xor-closeout", action="store_true")
    return parser.parse_args()


def parse_harness_output(text: str) -> tuple[dict[str, str], list[str], dict[int, dict[str, str]], dict[str, str]]:
    allocation: dict[str, str] = {}
    lifecycle: list[str] = []
    samples: dict[int, dict[str, str]] = {}
    xor_closeout: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("GPU_M2D_EVENT,event=ALLOCATED,"):
            for item in line.split(","):
                key, _, value = item.partition("=")
                allocation[key] = value
        elif line.startswith("GPU_M2D_EVENT,event=") and ",wall_time_ns=" in line:
            lifecycle.append(line.split("event=", 1)[1].split(",", 1)[0])
        elif line.startswith("GPU_M2D_SAMPLE,"):
            record: dict[str, str] = {}
            for item in line.split(",")[1:]:
                key, _, value = item.partition("=")
                record[key] = value
            if "index" in record:
                samples[int(record["index"])] = record
        elif line.startswith("GPU_M2D_XOR,"):
            for item in line.split(",")[1:]:
                key, _, value = item.partition("=")
                xor_closeout[key] = value
    return allocation, lifecycle, samples, xor_closeout


def event_times(text: str) -> dict[str, int]:
    times: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("GPU_M2D_EVENT,event="):
            continue
        name = ""
        monotonic = None
        for item in line.split(","):
            key, _, value = item.partition("=")
            if key == "event":
                name = value
            elif key == "monotonic_ns":
                monotonic = int(value)
        if name and monotonic is not None and name not in times:
            times[name] = monotonic
    return times


def phase_for(timestamp_ns: int, times: dict[str, int]) -> str:
    order = [
        ("allocation", "ALLOCATION_BEGIN", "ALLOCATED"),
        ("free", "FREE_BEGIN", "FREE_END"),
        ("context", "CONTEXT_BEGIN", "ALLOCATION_BEGIN"),
        ("access", "ALLOCATED", "FREE_BEGIN"),
    ]
    for phase, begin, end in order:
        if begin in times and end in times and times[begin] <= timestamp_ns <= times[end]:
            return phase
    return "other"


def write_samples(path: Path, samples: list[dict[str, object]]) -> None:
    fields = [
        "allocation_id", "gpu_uuid", "address_space_id", "rm_va_space_id",
        "requested_sample_index", "allocation_offset", "sample_va", "pte_index",
        "page_offset", "raw_pte_lo", "raw_pte_hi", "raw_entry", "entry_size",
        "high_word_zero", "valid", "aperture", "physical_page_base",
        "physical_address", "page_size", "gpu_local_pa_observed",
        "mapping_alive_during_hold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def main() -> int:
    args = parse_args()
    if os.geteuid() != 0:
        raise PermissionError("run through sudo; the CUDA child is dropped back to the invoking user")
    uid, gid = drop_to_invoking_user()
    if args.size_mib <= 0 or args.size_mib > 64:
        raise RuntimeError("size-mib must be in [1, 64]")
    contract, contract_hash = load_and_validate_contract(args.contract.resolve(),
                                                          DEFAULT_OPEN_SOURCE_REPO)
    test_binary = HERE / "g2_scratch_harness"
    if not test_binary.is_file() or not os.access(test_binary, os.X_OK):
        raise RuntimeError(f"build the harness first: {test_binary}")

    run_dir = args.output_root / f"run_{args.api}_gpu{args.device}_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    event_output = run_dir / "events.csv"
    sample_output = run_dir / "samples.csv"
    summary_output = run_dir / "summary.json"
    test_log = run_dir / "harness.log"
    gate = run_dir / f".gate_{os.getpid()}_{time.time_ns()}"

    observer = G2Observer(contract, contract_hash, 0, event_output)
    process: subprocess.Popen[bytes] | None = None
    started_ns = time.time_ns()
    try:
        process = subprocess.Popen(
            [
                str(test_binary), "--api", args.api, "--gate-file", str(gate),
                "--device", str(args.device), "--size-mib", str(args.size_mib),
                "--hold-seconds", str(args.hold_seconds),
                "--gate-timeout-seconds", str(max(5, int(args.timeout_seconds))),
            ] + ([] if args.no_xor_closeout else ["--xor-closeout"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            preexec_fn=child_preexec(uid, gid),
        )
        observer.set_target_tgid(process.pid)
        prelude, gate_ready = read_until_marker(
            process, b"GPU_M2D_EVENT,event=WAIT_PRE_ALLOC_GATE,", min(args.timeout_seconds, 10.0)
        )
        if not gate_ready:
            process.terminate()
            remainder, _ = process.communicate(timeout=2)
            raise RuntimeError("harness did not reach the pre-allocation gate:\n"
                               + (prelude + remainder).decode(errors="replace"))
        gate.write_text(f"observer_ready target_tgid={process.pid}\n", encoding="utf-8")
        os.chmod(gate, 0o644)

        deadline = time.monotonic() + args.timeout_seconds
        while process.poll() is None and time.monotonic() < deadline:
            observer.poll(50)
        if process.poll() is None:
            process.terminate()
            raise RuntimeError("harness exceeded timeout")
        for _ in range(10):
            observer.poll(50)
        output_tail, _ = process.communicate(timeout=2)
        output = (prelude + output_tail).decode("utf-8", errors="replace")
    finally:
        gate.unlink(missing_ok=True)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        observer.close()

    test_log.write_text(output, encoding="utf-8")
    allocation, lifecycle, requested_samples, xor_closeout = parse_harness_output(output)
    times = event_times(output)
    failures: list[str] = []
    if process is None or process.returncode != 0:
        failures.append(f"harness exit code {None if process is None else process.returncode}")
    if lifecycle != EXPECTED_LIFECYCLE:
        failures.append(f"lifecycle mismatch: {lifecycle}")

    phase_rows: dict[str, list[dict[str, object]]] = {name: [] for name in
                                                      ("context", "allocation", "access", "free", "other")}
    for row in observer.rows:
        phase_rows[phase_for(int(row["timestamp_ns"]), times)].append(row)
    allocation_maps = [row for row in phase_rows["allocation"] if row["event_type"] == "MAP_RETURN"]
    allocation_pte_headers = [row for row in phase_rows["allocation"] if row["event_type"] == "PTE_HEADER"]
    free_rows = [row for row in phase_rows["free"] if row["event_type"] == "FREE_RETURN"]
    if len(allocation_maps) != 1:
        failures.append(f"expected one allocation-window MAP_RETURN, got {len(allocation_maps)}")
    if len(allocation_pte_headers) != 1:
        failures.append(f"expected one allocation-window PTE_HEADER, got {len(allocation_pte_headers)}")
    if len(free_rows) != 1:
        failures.append(f"expected one free-window FREE_RETURN, got {len(free_rows)}")

    map_row = allocation_maps[0] if len(allocation_maps) == 1 else None
    pte_row = allocation_pte_headers[0] if len(allocation_pte_headers) == 1 else None
    free_row = free_rows[0] if len(free_rows) == 1 else None

    if not args.no_xor_closeout:
        expected_xor = {
            "status": "PASS", "target_offset": "4096", "bit_index": "3", "mask": "0x08",
        }
        if not xor_closeout:
            failures.append("scratch XOR closeout record missing")
        else:
            for key, expected in expected_xor.items():
                if xor_closeout.get(key) != expected:
                    failures.append(f"scratch XOR closeout {key}: expected {expected}, "
                                    f"got {xor_closeout.get(key)}")

    expected_base = allocation.get("base_va", "").lower()
    expected_length = int(allocation.get("size_bytes", "0"))
    expected_uuid = allocation.get("gpu_uuid", "").lower()
    if not expected_base or not expected_length or not expected_uuid:
        failures.append("harness ALLOCATED record incomplete")
    if map_row:
        if int(map_row["status"]) != 0:
            failures.append(f"map returned status {map_row['status']}")
        if str(map_row["map_base"]).lower() != expected_base or int(map_row["map_length"]) != expected_length:
            failures.append("map range does not match allocation")
        if str(map_row["gpu_uuid"]).lower() != expected_uuid:
            failures.append("kernel mapping GPU UUID does not match CUDA allocation UUID")
        if not map_row["address_space_id"]:
            failures.append("map address-space ID missing")
    if free_row:
        if int(free_row["status"]) != 0:
            failures.append(f"free returned status {free_row['status']}")
        if str(free_row["map_base"]).lower() != expected_base or int(free_row["map_length"]) != expected_length:
            failures.append("free range does not match allocation")
        if map_row and free_row["address_space_id"] != map_row["address_space_id"]:
            failures.append("map/free address-space IDs differ")

    samples: list[dict[str, object]] = []
    complete_local_coverage = False
    required_times = ("ALLOCATED", "ACCESS_END", "FREE_BEGIN", "FREE_END")
    hold_duration_ns = times.get("FREE_BEGIN", 0) - times.get("ACCESS_END", 0)
    mapping_alive_during_hold = bool(
        map_row and pte_row and free_row
        and all(name in times for name in required_times)
        and int(map_row["timestamp_ns"]) < times["ALLOCATED"]
        and int(pte_row["timestamp_ns"]) < times["ALLOCATED"]
        and times["ALLOCATED"] < times["ACCESS_END"] < times["FREE_BEGIN"]
        and times["FREE_BEGIN"] <= int(free_row["timestamp_ns"]) <= times["FREE_END"]
        and hold_duration_ns >= args.hold_seconds * 1_000_000_000
    )

    captured_entries: dict[int, dict[str, object]] = {}
    for row in phase_rows["allocation"]:
        if row["event_type"] != "PTE_ENTRY":
            continue
        pte_index = int(row["pte_index"])
        entry = {
            "raw_pte_lo": row["raw_pte_lo"],
            "raw_pte_hi": row["raw_pte_hi"],
            "raw_entry": row["raw_entry"],
            "entry_size": int(row["pte_size"]),
            "high_word_zero": row["high_word_zero"],
            "valid": row["valid"],
            "aperture": row["aperture"],
            "physical_page_base": str(row["physical_page_base"]),
        }
        if pte_index in captured_entries and captured_entries[pte_index]["raw_entry"] != entry["raw_entry"]:
            failures.append(f"conflicting duplicate PTE entry for index {pte_index}")
        else:
            captured_entries[pte_index] = entry

    if pte_row:
        if int(pte_row["status"]) != 0:
            failures.append(f"PTE query returned status {pte_row['status']}")
        if str(pte_row["map_base"]).lower() != expected_base:
            failures.append("PTE event map range does not match allocation")
        if str(pte_row["gpu_uuid"]).lower() != expected_uuid:
            failures.append("PTE event GPU UUID does not match CUDA allocation UUID")
        if not pte_row["address_space_id"] or not pte_row["rm_va_space_id"]:
            failures.append("PTE address-space identity missing")
        if int(pte_row["pte_size"]) not in set(contract["pte_format"]["supported_entry_sizes_bytes"]):
            failures.append(f"unexpected PTE size {pte_row['pte_size']}")
        if str(pte_row["payload_complete"]) != "true":
            failures.append("PTE payload incomplete (num_written exceeded one query batch)")
        page_size = int(pte_row["mapping_page_size"])
        if page_size <= 0 or page_size & (page_size - 1):
            failures.append(f"invalid mapping page size {page_size}")
        num_written = int(pte_row["num_written"])
        if int(pte_row["num_remaining"]) != 0:
            failures.append(f"PTE query left {pte_row['num_remaining']} entries unreported")
        for index in range(num_written):
            if index not in captured_entries:
                failures.append(f"PTE entry {index} of {num_written} was not captured")
                break

        sample_local: list[bool] = []
        if page_size > 0:
            for requested_index in sorted(requested_samples):
                requested = requested_samples[requested_index]
                allocation_offset = int(requested["offset_bytes"])
                query_relative = int(pte_row["map_offset"]) + allocation_offset - int(pte_row["query_offset"])
                if query_relative < 0 or query_relative >= int(pte_row["query_size"]):
                    failures.append(f"requested VA offset {allocation_offset} is outside the PTE query")
                    continue
                pte_index, page_offset = divmod(query_relative, page_size)
                entry = captured_entries.get(pte_index)
                if entry is None:
                    failures.append(f"PTE index {pte_index} for requested VA was not captured")
                    continue
                physical_base_text = str(entry["physical_page_base"])
                physical_address = (f"0x{int(physical_base_text, 16) + page_offset:x}"
                                    if physical_base_text else "")
                is_local = (
                    entry["valid"] == "true"
                    and entry["aperture"] == "VIDEO"
                    and entry["high_word_zero"] == "true"
                    and bool(physical_base_text)
                )
                sample_local.append(is_local)
                samples.append({
                    "allocation_id": allocation.get("allocation_id", ""),
                    "gpu_uuid": pte_row["gpu_uuid"],
                    "address_space_id": pte_row["address_space_id"],
                    "rm_va_space_id": pte_row["rm_va_space_id"],
                    "requested_sample_index": requested_index,
                    "allocation_offset": allocation_offset,
                    "sample_va": requested["va"],
                    "pte_index": pte_index,
                    "page_offset": page_offset,
                    **entry,
                    "physical_address": physical_address,
                    "page_size": page_size,
                    "gpu_local_pa_observed": str(is_local).lower(),
                    "mapping_alive_during_hold": str(mapping_alive_during_hold).lower(),
                })
        complete_local_coverage = (
            len(samples) == len(requested_samples)
            and all(sample_local)
            and all(index in captured_entries for index in range(num_written))
            and all(str(entry["valid"]) == "true" and str(entry["aperture"]) == "VIDEO"
                    for entry in captured_entries.values())
        )

    if pte_row and len(samples) != len(requested_samples):
        failures.append(f"expected {len(requested_samples)} VA-to-PTE correlations, got {len(samples)}")
    if map_row and free_row and not mapping_alive_during_hold:
        failures.append("map/free timestamps do not enclose the allocation hold interval")
    if observer.lost_event_count:
        failures.append(f"lost BPF events: {observer.lost_event_count}")

    write_samples(sample_output, samples)
    if failures:
        status = "FAIL_CLOSED"
    elif complete_local_coverage:
        status = "GPU_LOCAL_PTE_FULL_COVERAGE_OBSERVED"
    else:
        status = "PTE_PAYLOAD_OBSERVED_NONLOCAL_OR_UNRESOLVED"
    summary = {
        "schema_version": "gpu-m2d.g2-observer.scratch-probe.v1",
        "status": status,
        "meaning": "GPU-local PA is claimed only for valid captured PTEs whose decoded aperture is VIDEO",
        "failures": failures,
        "allocation_api": args.api,
        "target_tgid": observer.target_tgid,
        "device": args.device,
        "allocation": allocation,
        "lifecycle": lifecycle,
        "event_count": len(observer.rows),
        "phase_event_counts": {
            phase: {name: sum(row["event_type"] == name for row in rows)
                    for name in ("MAP_RETURN", "PTE_HEADER", "PTE_ENTRY", "FREE_RETURN")}
            for phase, rows in phase_rows.items()
        },
        "captured_pte_entry_count": len(captured_entries),
        "expected_pte_entry_count": int(pte_row["num_written"]) if pte_row else None,
        "requested_va_sample_count": len(requested_samples),
        "sample_count": len(samples),
        "gpu_local_sample_count": sum(row["gpu_local_pa_observed"] == "true" for row in samples),
        "mapping_page_size": pte_row["mapping_page_size"] if pte_row else None,
        "pte_size": pte_row["pte_size"] if pte_row else None,
        "kernel_gpu_uuid": pte_row["gpu_uuid"] if pte_row else None,
        "address_space_id": map_row["address_space_id"] if map_row else None,
        "rm_va_space_id": pte_row["rm_va_space_id"] if pte_row else None,
        "mapping_alive_during_hold": mapping_alive_during_hold,
        "hold_duration_monotonic_ns": hold_duration_ns,
        "gpu_local_pa_observed": complete_local_coverage and not failures,
        "scratch_xor_closeout": xor_closeout or None,
        "scratch_xor_closeout_passed": bool(xor_closeout) and xor_closeout.get("status") == "PASS",
        "lost_event_count": observer.lost_event_count,
        "contract_sha256": contract_hash,
        "nvidia_module_sha256": contract["nvidia_module_sha256"],
        "nvidia_uvm_module_sha256": contract["nvidia_uvm_module_sha256"],
        "run_directory": str(run_dir),
        "event_output": str(event_output),
        "event_output_sha256": sha256(event_output),
        "sample_output": str(sample_output),
        "sample_output_sha256": sha256(sample_output),
        "test_log": str(test_log),
        "test_log_sha256": sha256(test_log),
        "started_wall_time_ns": started_ns,
        "ended_wall_time_ns": time.time_ns(),
        "driver_modified": False,
        "uvm_state_modified": False,
        "pte_modified": False,
        "other_gpu_processes_stopped": False,
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    chown_outputs([event_output, sample_output, summary_output, test_log, run_dir], uid, gid)
    print(f"api={args.api} device={args.device} status={status} "
          f"events={len(observer.rows)} ptes={len(captured_entries)} samples={len(samples)} "
          f"local={summary['gpu_local_sample_count']}")
    print(f"summary={summary_output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 2 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"GPU_M2D_G2_OBSERVER_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
