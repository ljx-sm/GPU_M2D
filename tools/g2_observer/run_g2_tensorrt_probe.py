#!/usr/bin/python3
"""GPU_M2D G2 TensorRT probe orchestrator.

Runs one pre-allocation-gated observation of the complete G1.5 ResNet-50
INT8 TensorRT workload under the read-only G2 PTE observer. The G1.5
runner is executed in observer mode (``--observer-gate``): it blocks
before any CUDA context exists, then reports every AllocationRegistry
lifetime transition as GPU_M2D_EVENT lines. This orchestrator attaches
the probes, opens the gate, and correlates every registered allocation
with the kernel MAP/PTE/FREE events of its exact VA range.

For each allocation the observer must deliver a complete, contiguous,
valid, local-VIDEO (framebuffer) PTE payload covering the whole
allocation; the page-level result is written to gpu_va_pa_map.csv.
Any gap, non-local page, payload truncation, ordering violation, or
lost event fails the whole run closed.

Must be started through ``sudo`` from the research account: root is used
only for the kprobe attachment, while the TensorRT child is dropped back
to the invoking user. Nothing is modified on the driver side and no
other GPU process is touched.
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
from typing import Any

from g2_observer import (
    DEFAULT_CONTRACT,
    DEFAULT_OPEN_SOURCE_REPO,
    G2Observer,
    load_and_validate_contract,
    sha256,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]

LIFECYCLE_SKELETON = [
    "PROCESS_READY",
    "WAIT_PRE_ALLOC_GATE",
    "PRE_ALLOC_GATE_OPEN",
    "CONTEXT_BEGIN",
    "CONTEXT_READY",
    "RUNTIME_BEGIN",
    "BINDINGS_READY",
    "CLEAN_INFERENCE_BEGIN",
    "CLEAN_INFERENCE_END",
    "INJECTED_INFERENCE_BEGIN",
    "INJECTED_INFERENCE_END",
    "SNAPSHOT_READY",
    "HOLD_BEGIN",
    "HOLD_END",
    "TEARDOWN_BEGIN",
    "PROCESS_END",
]
VARIABLE_EVENTS = {"ALLOCATED", "FREE"}

MAP_SOURCE = "ebpf:uvm_api_map_external_allocation+nvUvmInterfaceGetExternalAllocPtes"
MAP_CONFIDENCE = "definition_level_gmmu_pte_payload"

MAP_FIELDS = [
    "run_id", "g1_5_run_id", "device", "gpu_uuid", "allocation_id",
    "allocation_api", "cuda_buffer_id", "allocation_size_bytes",
    "semantic_label", "allocation_phase", "active_at_snapshot",
    "va_page_base", "va_page_end_exclusive", "fb_pa_page_base", "page_size",
    "aperture", "pte_valid", "pte_size", "raw_pte_lo", "raw_pte_hi",
    "mapping_generation", "address_space_id", "rm_va_space_id",
    "mapped_at_ns", "unmapped_at_ns", "covered_allocation_bytes",
    "physical_coverage_status", "source", "confidence",
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
    if process.stdout is None:
        raise RuntimeError("runner stdout pipe is unavailable")
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
    remu_root = Path(os.environ.get("GPU_M2D_REMU_ROOT", "/data1/luojx/REMU"))
    dataset_root = Path(os.environ.get(
        "GPU_M2D_DATASET_ROOT", "/data1/luojx/datasets/REMU_stage8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True,
                        help="prebuilt gpu_m2d_resnet50_int8_g1_5 binary")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--engine", type=Path,
                        default=remu_root / "artifacts/stage8/engines/paper_priority/resnet50_resisc45_int8_ptq.engine")
    parser.add_argument("--sample-csv", type=Path,
                        default=dataset_root / "RESISC45/splits/original_repo_1000_eval.csv")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--element", type=int, default=0)
    parser.add_argument("--bit", type=int, default=0)
    parser.add_argument("--hold-seconds", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "artifacts/g2/observer/trt")
    parser.add_argument("--remu-root", type=Path, default=remu_root)
    return parser.parse_args()


def parse_harness_output(text: str) -> tuple[list[dict[str, str]], list[str], dict[str, int]]:
    allocated: list[dict[str, str]] = []
    lifecycle: list[str] = []
    times: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("GPU_M2D_EVENT,event="):
            continue
        fields: dict[str, str] = {}
        for item in line.split(","):
            key, _, value = item.partition("=")
            fields[key] = value
        name = fields.get("event", "")
        if not name:
            continue
        lifecycle.append(name)
        if name == "ALLOCATED":
            allocated.append(fields)
        if "monotonic_ns" in fields and name not in times:
            times[name] = int(fields["monotonic_ns"])
    return allocated, lifecycle, times


def phase_for(timestamp_ns: int, times: dict[str, int]) -> str:
    order = [
        ("context", "CONTEXT_BEGIN", "RUNTIME_BEGIN"),
        ("allocation", "RUNTIME_BEGIN", "SNAPSHOT_READY"),
        ("inference", "CLEAN_INFERENCE_BEGIN", "INJECTED_INFERENCE_END"),
        ("teardown", "TEARDOWN_BEGIN", "PROCESS_END"),
    ]
    for phase, begin, end in order:
        if begin in times and end in times and times[begin] <= timestamp_ns <= times[end]:
            return phase
    return "other"


def pte_signature(row: dict[str, object]) -> tuple[object, ...]:
    return (row["timestamp_ns"], row["map_base"], row["query_offset"],
            row["query_size"])


def build_allocation_ledger(rows: list[dict[str, object]],
                            allocation: dict[str, str],
                            allocated_event: dict[str, str],
                            teardown_end_ns: int) -> tuple[list[dict[str, object]], list[str]]:
    """Port of the validated REMU build_ledger, generalized per allocation.

    Every PTE segment covering the allocation must be valid, local VIDEO,
    and high-word-zero; the clipped segments must tile the allocation with
    no gap and no overlap.
    """
    failures: list[str] = []
    allocation_id = allocation["allocation_id"]
    base = int(allocation["gpu_va"], 16)
    size = int(allocation["size_bytes"])
    end = base + size

    maps = [row for row in rows if row["event_type"] == "MAP_RETURN"
            and row["map_base"]
            and int(str(row["map_base"]), 16) == base]
    headers = [row for row in rows if row["event_type"] == "PTE_HEADER"
               and row["map_base"]
               and int(str(row["map_base"]), 16) == base]
    frees = [row for row in rows if row["event_type"] == "FREE_RETURN"
             and row["map_base"]
             and int(str(row["map_base"]), 16) == base]
    if len(maps) != 1:
        failures.append(f"{allocation_id}: expected one MAP_RETURN, got {len(maps)}")
    if not headers:
        failures.append(f"{allocation_id}: PTE_HEADER missing")
        return [], failures
    if any(int(str(row["status"])) != 0 for row in maps + headers):
        failures.append(f"{allocation_id}: nonzero map/PTE status")

    mapped_at = min(int(str(row["timestamp_ns"])) for row in maps) if maps else 0
    later_frees = [int(str(row["timestamp_ns"])) for row in frees
                   if int(str(row["timestamp_ns"])) > mapped_at]
    unmapped_at = min(later_frees) if later_frees else None
    if unmapped_at is None:
        failures.append(f"{allocation_id}: no FREE_RETURN after mapping")
    elif unmapped_at > teardown_end_ns:
        failures.append(f"{allocation_id}: mapping outlived the teardown window")

    segments: list[dict[str, object]] = []
    for header in headers:
        if str(header["payload_complete"]) != "true":
            failures.append(f"{allocation_id}: PTE query exceeded the bounded complete-capture contract")
            continue
        entries = [row for row in rows if row["event_type"] == "PTE_ENTRY"
                   and pte_signature(row) == pte_signature(header)]
        if len(entries) != int(header["num_written"]):
            failures.append(f"{allocation_id}: PTE payload count {len(entries)} != num_written {header['num_written']}")
            continue
        if int(header["num_remaining"]) != 0:
            failures.append(f"{allocation_id}: PTE query left {header['num_remaining']} entries unreported")
            continue
        page_size = int(header["mapping_page_size"])
        if page_size <= 0 or page_size & (page_size - 1):
            failures.append(f"{allocation_id}: invalid page size {page_size}")
            continue
        for entry in entries:
            page_va = int(str(header["map_base"]), 16) + int(header["query_offset"]) \
                + int(entry["pte_index"]) * page_size - int(header["map_offset"])
            segment_start = max(base, page_va)
            segment_end = min(end, page_va + page_size)
            if segment_start >= segment_end:
                continue
            physical_text = str(entry["physical_page_base"])
            if not (entry["valid"] == "true" and entry["aperture"] == "VIDEO"
                    and entry["high_word_zero"] == "true" and physical_text):
                failures.append(f"{allocation_id}: non-local or unresolved PTE at va 0x{page_va:x}")
                continue
            segments.append({
                "allocation_id": allocation_id,
                "address_space_id": header["address_space_id"],
                "rm_va_space_id": header["rm_va_space_id"],
                "page_size": page_size,
                "pte_size": int(header["pte_size"]),
                "va_page_base": page_va,
                "covered_va_start": segment_start,
                "covered_va_end_exclusive": segment_end,
                "physical_page_base": int(physical_text, 16),
                "aperture": entry["aperture"],
                "pte_valid": entry["valid"],
                "raw_pte_lo": entry["raw_pte_lo"],
                "raw_pte_hi": entry["raw_pte_hi"],
                "mapped_at_ns": mapped_at,
                "unmapped_at_ns": unmapped_at,
                "header": header,
            })

    segments.sort(key=lambda row: int(row["covered_va_start"]))
    cursor = base
    for segment in segments:
        start = int(segment["covered_va_start"])
        segment_end = int(segment["covered_va_end_exclusive"])
        if start != cursor:
            failures.append(f"{allocation_id}: coverage gap or overlap at 0x{cursor:x}")
            break
        cursor = segment_end
    if cursor != end:
        failures.append(f"{allocation_id}: coverage ends at 0x{cursor:x}, expected 0x{end:x}")
    if segments and "monotonic_ns" in allocated_event and \
            mapped_at >= int(allocated_event["monotonic_ns"]):
        failures.append(f"{allocation_id}: kernel mapping does not precede the ALLOCATED event")
    return segments, failures


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def main() -> int:
    args = parse_args()
    if os.geteuid() != 0:
        raise PermissionError("run through sudo; the CUDA child is dropped to the invoking user")
    uid, gid = drop_to_invoking_user()
    contract, contract_hash = load_and_validate_contract(args.contract.resolve(),
                                                         DEFAULT_OPEN_SOURCE_REPO)
    if not args.runner.is_file() or not os.access(args.runner, os.X_OK):
        raise RuntimeError(f"runner missing or not executable: {args.runner}")

    trt_runtime_dir = args.remu_root / ".local/deps/tensorrt-8.6.1/tensorrt_libs"
    opencv_lib_dir = args.remu_root / ".local/deps/conda/lib"
    for required in (args.engine, args.sample_csv, trt_runtime_dir / "libnvinfer.so.8"):
        if not required.is_file():
            raise RuntimeError(f"missing required asset: {required}")

    run_dir = args.output_root / f"run_trt_gpu{args.device}_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # The runner child is dropped to the invoking user and writes its
    # output-prefix CSVs directly into this directory.
    os.chown(run_dir, uid, gid)
    event_output = run_dir / "events.csv"
    map_output = run_dir / "gpu_va_pa_map.csv"
    summary_output = run_dir / "summary.json"
    test_log = run_dir / "harness.log"
    gate = run_dir / f".gate_{os.getpid()}_{time.time_ns()}"
    run_id = run_dir.name

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(trt_runtime_dir), str(opencv_lib_dir),
                          os.environ.get("LD_LIBRARY_PATH", "")) if part)

    device_uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(args.device)],
        text=True, stdout=subprocess.PIPE, check=True).stdout.strip()

    observer = G2Observer(contract, contract_hash, 0, event_output)
    process: subprocess.Popen[bytes] | None = None
    started_ns = time.time_ns()
    try:
        process = subprocess.Popen(
            [
                str(args.runner.resolve()),
                "--engine", str(args.engine.resolve()),
                "--sample-csv", str(args.sample_csv.resolve()),
                "--sample-index", str(args.sample_index),
                "--device", str(args.device),
                "--element", str(args.element),
                "--bit", str(args.bit),
                "--output-prefix", str((run_dir / "g1_5").resolve()),
                "--observer-gate", str(gate),
                "--hold-seconds", str(args.hold_seconds),
                "--gate-timeout-seconds", str(max(5, int(args.timeout_seconds))),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=environment,
            preexec_fn=child_preexec(uid, gid),
        )
        observer.set_target_tgid(process.pid)
        prelude, gate_ready = read_until_marker(
            process, b"GPU_M2D_EVENT,event=WAIT_PRE_ALLOC_GATE,", min(args.timeout_seconds, 10.0)
        )
        if not gate_ready:
            process.terminate()
            remainder, _ = process.communicate(timeout=2)
            raise RuntimeError("runner did not reach the pre-allocation gate:\n"
                               + (prelude + remainder).decode(errors="replace"))
        gate.write_text(f"observer_ready target_tgid={process.pid}\n", encoding="utf-8")
        os.chmod(gate, 0o644)

        deadline = time.monotonic() + args.timeout_seconds
        while process.poll() is None and time.monotonic() < deadline:
            observer.poll(50)
        if process.poll() is None:
            process.terminate()
            raise RuntimeError("runner exceeded timeout")
        for _ in range(20):
            observer.poll(50)
        output_tail, _ = process.communicate(timeout=5)
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
    allocated, lifecycle, times = parse_harness_output(output)
    failures: list[str] = []
    if process is None or process.returncode != 0:
        failures.append(f"runner exit code {None if process is None else process.returncode}")
    if "GPU_M2D_G1_5_PASS" not in output:
        failures.append("runner did not report GPU_M2D_G1_5_PASS")

    expected_skeleton = [name for name in LIFECYCLE_SKELETON
                         if not (name in ("HOLD_BEGIN", "HOLD_END") and args.hold_seconds <= 0)]
    skeleton = [name for name in lifecycle if name not in VARIABLE_EVENTS]
    if skeleton != expected_skeleton:
        failures.append(f"lifecycle skeleton mismatch: {skeleton}")

    registry_path = run_dir / "g1_5_allocations.csv"
    if not registry_path.is_file():
        failures.append("g1_5_allocations.csv was not written")
        raise SystemExit(finalize(failures, run_dir, summary_output, map_output,
                                  test_log, event_output, observer, contract,
                                  contract_hash, args, run_id, device_uuid,
                                  started_ns, [], allocated, lifecycle, times,
                                  uid, gid, None))
    registry = read_csv(registry_path)
    if not registry:
        failures.append("allocation registry is empty")

    # Every registry row must have exactly one matching ALLOCATED record.
    allocated_by_id = {record.get("allocation_id", ""): record for record in allocated}
    if len(allocated_by_id) != len(allocated):
        failures.append("duplicate ALLOCATED records for one allocation_id")
    for row in registry:
        record = allocated_by_id.get(row["allocation_id"])
        if record is None:
            failures.append(f"{row['allocation_id']}: no ALLOCATED event")
            continue
        if record.get("base_va", "").lower() != row["gpu_va"].lower() or \
                record.get("size_bytes") != row["size_bytes"]:
            failures.append(f"{row['allocation_id']}: ALLOCATED event disagrees with registry")
    freed_records = [dict(item.partition("=")[::2] for item in line.split(",")[1:])
                     for line in output.splitlines()
                     if line.startswith("GPU_M2D_EVENT,event=FREE,")]
    freed_by_id = {record.get("allocation_id", ""): record for record in freed_records}
    if len(freed_records) != len(allocated):
        failures.append(f"expected {len(allocated)} FREE events, got {len(freed_records)}")
    for allocation_id in allocated_by_id:
        if allocation_id not in freed_by_id:
            failures.append(f"{allocation_id}: no FREE event before PROCESS_END")

    teardown_end_ns = times.get("PROCESS_END", 0)
    snapshot_ns = times.get("SNAPSHOT_READY", 0)
    hold_begin_ns = times.get("HOLD_BEGIN", 0)
    hold_end_ns = times.get("HOLD_END", 0)

    map_rows: list[dict[str, object]] = []
    per_allocation_status: dict[str, str] = {}
    active_at_snapshot: dict[str, bool] = {row["allocation_id"]: row["active"] == "1"
                                           for row in registry}
    g1_5_run_id = registry[0]["run_id"] if registry else ""
    for row in registry:
        record = allocated_by_id.get(row["allocation_id"], {})
        segments, ledger_failures = build_allocation_ledger(
            observer.rows, row, record, teardown_end_ns)
        failures.extend(ledger_failures)
        if row["active"] == "1" and args.hold_seconds > 0 and segments:
            unmapped = int(segments[0]["unmapped_at_ns"])
            if not (snapshot_ns and hold_begin_ns and hold_end_ns
                    and segments[0]["mapped_at_ns"] < snapshot_ns
                    and hold_end_ns < unmapped):
                failures.append(f"{row['allocation_id']}: mapping not alive across the hold interval")
        per_allocation_status[row["allocation_id"]] = (
            "LOCAL_VIDEO_COMPLETE" if segments and not ledger_failures else "FAIL_CLOSED")
        for segment in segments:
            map_rows.append({
                "run_id": run_id,
                "g1_5_run_id": g1_5_run_id,
                "device": args.device,
                "gpu_uuid": segment["header"]["gpu_uuid"],
                "allocation_id": segment["allocation_id"],
                "allocation_api": record.get("allocation_api", ""),
                "cuda_buffer_id": record.get("cuda_buffer_id", ""),
                "allocation_size_bytes": row["size_bytes"],
                "semantic_label": row["semantic_label"],
                "allocation_phase": row["allocation_phase"],
                "active_at_snapshot": str(active_at_snapshot.get(row["allocation_id"], False)).lower(),
                "va_page_base": f"0x{int(segment['va_page_base']):x}",
                "va_page_end_exclusive": f"0x{int(segment['covered_va_end_exclusive']):x}",
                "fb_pa_page_base": f"0x{int(segment['physical_page_base']):x}",
                "page_size": segment["page_size"],
                "aperture": segment["aperture"],
                "pte_valid": segment["pte_valid"],
                "pte_size": segment["pte_size"],
                "raw_pte_lo": segment["raw_pte_lo"],
                "raw_pte_hi": segment["raw_pte_hi"],
                "mapping_generation": 1,
                "address_space_id": segment["address_space_id"],
                "rm_va_space_id": segment["rm_va_space_id"],
                "mapped_at_ns": segment["mapped_at_ns"],
                "unmapped_at_ns": segment["unmapped_at_ns"],
                "covered_allocation_bytes": int(segment["covered_va_end_exclusive"])
                - int(segment["covered_va_start"]),
                "physical_coverage_status": "LOCAL_VIDEO_COMPLETE",
                "source": MAP_SOURCE,
                "confidence": MAP_CONFIDENCE,
            })

    kernel_uuids = sorted({str(row["gpu_uuid"]) for row in observer.rows
                           if row["event_type"] == "PTE_HEADER" and row["gpu_uuid"]})
    if len(kernel_uuids) > 1:
        failures.append(f"multiple kernel GPU UUIDs observed: {kernel_uuids}")
    elif kernel_uuids and kernel_uuids[0] != device_uuid:
        failures.append(f"kernel GPU UUID {kernel_uuids[0]} != device UUID {device_uuid}")
    if len({str(row["address_space_id"]) for row in observer.rows
            if row["event_type"] in ("MAP_RETURN", "PTE_HEADER")}) > 1:
        failures.append("multiple UVM address-space IDs observed")
    if observer.lost_event_count:
        failures.append(f"lost BPF events: {observer.lost_event_count}")

    shared_pages: dict[str, set[str]] = {}
    for row in map_rows:
        shared_pages.setdefault(str(row["fb_pa_page_base"]), set()).add(str(row["allocation_id"]))
    shared_backing_groups = sorted(
        sorted(ids) for ids in shared_pages.values() if len(ids) > 1)

    return finalize(failures, run_dir, summary_output, map_output, test_log,
                    event_output, observer, contract, contract_hash, args,
                    run_id, device_uuid, started_ns, map_rows, allocated,
                    lifecycle, times, uid, gid, {
                        "per_allocation_coverage": per_allocation_status,
                        "kernel_gpu_uuids_observed": kernel_uuids,
                        "shared_backing_page_allocation_groups": shared_backing_groups,
                        "registry_allocation_count": len(registry),
                        "active_at_snapshot_count": sum(active_at_snapshot.values()),
                        "g1_5_run_id": g1_5_run_id,
                    })


def finalize(failures: list[str], run_dir: Path, summary_output: Path,
             map_output: Path, test_log: Path, event_output: Path,
             observer: G2Observer, contract: dict[str, Any], contract_hash: str,
             args: argparse.Namespace, run_id: str, device_uuid: str,
             started_ns: int, map_rows: list[dict[str, object]],
             allocated: list[dict[str, str]], lifecycle: list[str],
             times: dict[str, int], uid: int, gid: int,
             extra: dict[str, Any] | None) -> int:
    map_output.parent.mkdir(parents=True, exist_ok=True)
    with map_output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=MAP_FIELDS)
        writer.writeheader()
        writer.writerows(map_rows)

    status = "FAIL_CLOSED" if failures else "G2_TENSORRT_LOCAL_PA_FULL_COVERAGE_OBSERVED"
    summary = {
        "schema_version": "gpu-m2d.g2-observer.tensorrt-probe.v1",
        "status": status,
        "meaning": "GPU-local PA is claimed only for valid captured PTEs whose decoded aperture is VIDEO",
        "failures": failures,
        "run_id": run_id,
        "device": args.device,
        "device_uuid": device_uuid,
        "target_tgid": observer.target_tgid,
        "engine_path": str(args.engine),
        "engine_sha256": sha256(args.engine) if args.engine.is_file() else "",
        "lifecycle_skeleton": [name for name in lifecycle if name not in VARIABLE_EVENTS],
        "lifecycle_full": lifecycle,
        "event_count": len(observer.rows),
        "allocated_event_count": len(allocated),
        "map_row_count": len(map_rows),
        "gpu_local_pa_observed": not failures and bool(map_rows),
        "map_output": str(map_output),
        "map_output_sha256": sha256(map_output) if map_rows else "",
        "event_output": str(event_output),
        "event_output_sha256": sha256(event_output),
        "test_log": str(test_log),
        "test_log_sha256": sha256(test_log),
        "lost_event_count": observer.lost_event_count,
        "contract_sha256": contract_hash,
        "nvidia_module_sha256": contract["nvidia_module_sha256"],
        "nvidia_uvm_module_sha256": contract["nvidia_uvm_module_sha256"],
        "started_wall_time_ns": started_ns,
        "ended_wall_time_ns": time.time_ns(),
        "driver_modified": False,
        "uvm_state_modified": False,
        "pte_modified": False,
        "other_gpu_processes_stopped": False,
    }
    if extra is not None:
        summary.update(extra)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    chown_outputs([event_output, map_output, summary_output, test_log, run_dir], uid, gid)
    print(f"device={args.device} status={status} allocations={summary.get('registry_allocation_count')} "
          f"map_rows={len(map_rows)} events={len(observer.rows)}")
    print(f"summary={summary_output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 2 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(f"GPU_M2D_G2_TENSORRT_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
