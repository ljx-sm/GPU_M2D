#!/usr/bin/python3
"""GPU_M2D G2 VMM alias probe orchestrator.

Runs one pre-allocation-gated observation of a CUDA VMM alias mapping:
ONE physical allocation mapped at TWO distinct VA ranges. The observer
must capture complete PTE payloads for BOTH mappings, every page must
be valid and local VIDEO (framebuffer), each physical page must appear
under both VA ranges (the double-mapping required by the G2 acceptance
plan), and the reverse mapping PA page -> VA pages must be one-to-many
(two VA pages per physical page).

The harness additionally proves the alias semantically at runtime
(pattern write through the primary VA read back through the secondary
VA, one-bit XOR closeout observed through the secondary VA).

Must be started through ``sudo`` from the research account: root is
used only for the kprobe attachment, while the CUDA child is dropped
back to the invoking user.
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

EXPECTED_LIFECYCLE = [
    "PROCESS_READY",
    "WAIT_PRE_ALLOC_GATE",
    "PRE_ALLOC_GATE_OPEN",
    "CONTEXT_BEGIN",
    "CONTEXT_READY",
    "ALLOCATION_BEGIN",
    "ALLOCATED",
    "ALLOCATED",
    "ACCESS_BEGIN",
    "ACCESS_END",
    "HOLD_BEGIN",
    "HOLD_END",
    "FREE_BEGIN",
    "FREE_END",
]

PAGE_FIELDS = [
    "allocation_id", "alias_role", "gpu_uuid", "address_space_id",
    "rm_va_space_id", "page_offset", "page_va", "pte_index",
    "raw_pte_lo", "raw_pte_hi", "raw_entry", "pte_size", "page_size",
    "valid", "aperture", "high_word_zero", "fb_pa_page_base",
    "gpu_local_pa_observed",
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size-mib", type=int, default=8)
    parser.add_argument("--hold-seconds", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=25.0)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "artifacts/g2/observer/alias")
    return parser.parse_args()


def parse_harness_output(text: str) -> tuple[list[dict[str, str]], list[str],
                                             dict[str, int], dict[str, str]]:
    allocated: list[dict[str, str]] = []
    lifecycle: list[str] = []
    times: dict[str, int] = {}
    alias_record: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("GPU_M2D_EVENT,event="):
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
        elif line.startswith("GPU_M2D_ALIAS,"):
            for item in line.split(",")[1:]:
                key, _, value = item.partition("=")
                alias_record[key] = value
    return allocated, lifecycle, times, alias_record


def pte_signature(row: dict[str, object]) -> tuple[object, ...]:
    return (row["timestamp_ns"], row["map_base"], row["query_offset"],
            row["query_size"])


def collect_pages(rows: list[dict[str, object]], base: int) -> tuple[dict[int, dict[str, object]], list[str]]:
    """Map allocation-relative page offset -> decoded page record for one VA range."""
    failures: list[str] = []
    headers = [row for row in rows if row["event_type"] == "PTE_HEADER"
               and row["map_base"] and int(str(row["map_base"]), 16) == base]
    maps = [row for row in rows if row["event_type"] == "MAP_RETURN"
            and row["map_base"] and int(str(row["map_base"]), 16) == base]
    frees = [row for row in rows if row["event_type"] == "FREE_RETURN"
             and row["map_base"] and int(str(row["map_base"]), 16) == base]
    if len(maps) != 1:
        failures.append(f"expected one MAP_RETURN for base 0x{base:x}, got {len(maps)}")
    if not headers:
        failures.append(f"PTE_HEADER missing for base 0x{base:x}")
        return {}, failures
    if len(frees) != 1:
        failures.append(f"expected one FREE_RETURN for base 0x{base:x}, got {len(frees)}")
    if any(int(str(row["status"])) != 0 for row in maps + headers + frees):
        failures.append(f"nonzero map/PTE/free status for base 0x{base:x}")

    pages: dict[int, dict[str, object]] = {}
    for header in headers:
        if str(header["payload_complete"]) != "true":
            failures.append(f"PTE query exceeded the bounded complete-capture contract for base 0x{base:x}")
            continue
        if int(header["num_remaining"]) != 0:
            failures.append(f"PTE query left entries unreported for base 0x{base:x}")
            continue
        page_size = int(header["mapping_page_size"])
        if page_size <= 0 or page_size & (page_size - 1):
            failures.append(f"invalid page size {page_size} for base 0x{base:x}")
            continue
        entries = [row for row in rows if row["event_type"] == "PTE_ENTRY"
                   and pte_signature(row) == pte_signature(header)]
        if len(entries) != int(header["num_written"]):
            failures.append(f"PTE payload count mismatch for base 0x{base:x}")
            continue
        for entry in entries:
            page_va = int(str(header["map_base"]), 16) + int(header["query_offset"]) \
                + int(entry["pte_index"]) * page_size - int(header["map_offset"])
            offset = page_va - base
            physical_text = str(entry["physical_page_base"])
            if not (entry["valid"] == "true" and entry["aperture"] == "VIDEO"
                    and entry["high_word_zero"] == "true" and physical_text):
                failures.append(f"non-local or unresolved PTE at offset {offset} of base 0x{base:x}")
                continue
            record = {
                "pte_index": entry["pte_index"],
                "raw_pte_lo": entry["raw_pte_lo"],
                "raw_pte_hi": entry["raw_pte_hi"],
                "raw_entry": entry["raw_entry"],
                "pte_size": int(header["pte_size"]),
                "page_size": page_size,
                "valid": entry["valid"],
                "aperture": entry["aperture"],
                "high_word_zero": entry["high_word_zero"],
                "fb_pa_page_base": physical_text,
                "gpu_uuid": header["gpu_uuid"],
                "address_space_id": header["address_space_id"],
                "rm_va_space_id": header["rm_va_space_id"],
                "page_va": page_va,
                "mapped_at_ns": int(str(maps[0]["timestamp_ns"])) if len(maps) == 1 else "",
                "unmapped_at_ns": int(str(frees[0]["timestamp_ns"])) if len(frees) == 1 else "",
            }
            if offset in pages and pages[offset]["fb_pa_page_base"] != physical_text:
                failures.append(f"conflicting duplicate PTE at offset {offset} of base 0x{base:x}")
            else:
                pages[offset] = record
    return pages, failures


def write_pages(path: Path, pages: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PAGE_FIELDS)
        writer.writeheader()
        writer.writerows(pages)


def main() -> int:
    args = parse_args()
    if os.geteuid() != 0:
        raise PermissionError("run through sudo; the CUDA child is dropped to the invoking user")
    uid, gid = drop_to_invoking_user()
    if args.size_mib <= 0 or args.size_mib > 64:
        raise RuntimeError("size-mib must be in [1, 64]")
    contract, contract_hash = load_and_validate_contract(args.contract.resolve(),
                                                         DEFAULT_OPEN_SOURCE_REPO)
    test_binary = HERE / "g2_alias_harness"
    if not test_binary.is_file() or not os.access(test_binary, os.X_OK):
        raise RuntimeError(f"build the harness first: {test_binary}")

    run_dir = args.output_root / f"run_alias_gpu{args.device}_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    event_output = run_dir / "events.csv"
    pages_output = run_dir / "alias_pages.csv"
    summary_output = run_dir / "summary.json"
    test_log = run_dir / "harness.log"
    gate = run_dir / f".gate_{os.getpid()}_{time.time_ns()}"

    observer = G2Observer(contract, contract_hash, 0, event_output)
    process: subprocess.Popen[bytes] | None = None
    started_ns = time.time_ns()
    try:
        process = subprocess.Popen(
            [
                str(test_binary), "--gate-file", str(gate),
                "--device", str(args.device), "--size-mib", str(args.size_mib),
                "--hold-seconds", str(args.hold_seconds),
                "--gate-timeout-seconds", str(max(5, int(args.timeout_seconds))),
            ],
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
    allocated, lifecycle, times, alias_record = parse_harness_output(output)
    failures: list[str] = []
    if process is None or process.returncode != 0:
        failures.append(f"harness exit code {None if process is None else process.returncode}")
    if lifecycle != EXPECTED_LIFECYCLE:
        failures.append(f"lifecycle mismatch: {lifecycle}")
    if alias_record.get("status") != "PASS":
        failures.append(f"runtime alias verification failed: {alias_record.get('status')}")

    primary = next((r for r in allocated if r.get("allocation_api") == "vmm-alias-primary"), None)
    secondary = next((r for r in allocated if r.get("allocation_api") == "vmm-alias-secondary"), None)
    if primary is None or secondary is None:
        failures.append("primary/secondary ALLOCATED records missing")
    expected_size = int(primary["size_bytes"]) if primary else 0
    expected_uuid = primary.get("gpu_uuid", "").lower() if primary else ""

    primary_pages: dict[int, dict[str, object]] = {}
    secondary_pages: dict[int, dict[str, object]] = {}
    if primary and secondary:
        if primary["base_va"] == secondary["base_va"]:
            failures.append("primary and secondary VA ranges are identical")
        if primary["size_bytes"] != secondary["size_bytes"] or expected_size != args.size_mib * 1024 * 1024:
            failures.append("alias VA sizes disagree with the requested allocation")
        primary_base = int(primary["base_va"], 16)
        secondary_base = int(secondary["base_va"], 16)
        primary_pages, primary_failures = collect_pages(observer.rows, primary_base)
        secondary_pages, secondary_failures = collect_pages(observer.rows, secondary_base)
        failures.extend(primary_failures)
        failures.extend(secondary_failures)
        for label, pages, base in (("primary", primary_pages, primary_base),
                                   ("secondary", secondary_pages, secondary_base)):
            for offset in sorted(pages):
                row = pages[offset]
                if str(row["gpu_uuid"]).lower() != expected_uuid:
                    failures.append(f"{label} page UUID does not match the CUDA allocation UUID")
                    break
        # The double-mapping check: every page offset must exist under both
        # VA ranges and decode to the SAME framebuffer physical page.
        if set(primary_pages) != set(secondary_pages):
            failures.append("primary and secondary page-offset sets differ")
        else:
            for offset in sorted(primary_pages):
                if primary_pages[offset]["fb_pa_page_base"] != secondary_pages[offset]["fb_pa_page_base"]:
                    failures.append(
                        f"alias double-mapping broken at page offset {offset}: "
                        f"{primary_pages[offset]['fb_pa_page_base']} != "
                        f"{secondary_pages[offset]['fb_pa_page_base']}")
        # Reverse mapping: each physical page must back exactly two VA pages.
        reverse: dict[str, set[tuple[str, int]]] = {}
        for role, pages in (("primary", primary_pages), ("secondary", secondary_pages)):
            for offset, row in pages.items():
                reverse.setdefault(str(row["fb_pa_page_base"]), set()).add((role, offset))
        non_two = {pa: targets for pa, targets in reverse.items() if len(targets) != 2}
        if non_two:
            failures.append(f"reverse mapping is not one-to-many (2 VA pages per PA page): {sorted(non_two)}")
    else:
        reverse = {}

    hold_duration_ns = times.get("FREE_BEGIN", 0) - times.get("ACCESS_END", 0)
    mapping_alive_during_hold = bool(
        primary_pages and secondary_pages
        and all(name in times for name in ("ALLOCATED", "ACCESS_END", "FREE_BEGIN", "FREE_END"))
        and max(int(str(row["mapped_at_ns"])) for row in primary_pages.values()) < times["ALLOCATED"]
        and times["ALLOCATED"] < times["ACCESS_END"] < times["FREE_BEGIN"]
        and times["FREE_BEGIN"] <= min(
            int(str(row["unmapped_at_ns"])) for row in list(primary_pages.values())
            + list(secondary_pages.values())) <= times["FREE_END"]
        and hold_duration_ns >= args.hold_seconds * 1_000_000_000
    )
    if primary_pages and secondary_pages and not mapping_alive_during_hold:
        failures.append("map/free timestamps do not enclose the alias hold interval")
    if observer.lost_event_count:
        failures.append(f"lost BPF events: {observer.lost_event_count}")

    page_rows: list[dict[str, object]] = []
    for role, allocation, pages in (("primary", primary, primary_pages),
                                    ("secondary", secondary, secondary_pages)):
        for offset in sorted(pages):
            row = pages[offset]
            page_rows.append({
                "allocation_id": allocation.get("allocation_id", "") if allocation else "",
                "alias_role": role,
                "gpu_uuid": row["gpu_uuid"],
                "address_space_id": row["address_space_id"],
                "rm_va_space_id": row["rm_va_space_id"],
                "page_offset": offset,
                "page_va": f"0x{int(row['page_va']):x}",
                "pte_index": row["pte_index"],
                "raw_pte_lo": row["raw_pte_lo"],
                "raw_pte_hi": row["raw_pte_hi"],
                "raw_entry": row["raw_entry"],
                "pte_size": row["pte_size"],
                "page_size": row["page_size"],
                "valid": row["valid"],
                "aperture": row["aperture"],
                "high_word_zero": row["high_word_zero"],
                "fb_pa_page_base": row["fb_pa_page_base"],
                "gpu_local_pa_observed": "true",
            })
    write_pages(pages_output, page_rows)

    alias_confirmed = bool(
        primary_pages and secondary_pages
        and set(primary_pages) == set(secondary_pages)
        and all(primary_pages[offset]["fb_pa_page_base"] ==
                secondary_pages[offset]["fb_pa_page_base"] for offset in primary_pages)
        and alias_record.get("status") == "PASS"
    )
    if failures:
        status = "FAIL_CLOSED"
    elif alias_confirmed:
        status = "G2_VMM_ALIAS_DOUBLE_MAPPING_OBSERVED"
    else:
        status = "ALIAS_PTE_OBSERVED_BUT_NOT_CONFIRMED"
    summary: dict[str, Any] = {
        "schema_version": "gpu-m2d.g2-observer.alias-probe.v1",
        "status": status,
        "meaning": "one VMM physical allocation mapped at two VAs must decode to the same local VIDEO PA pages",
        "failures": failures,
        "device": args.device,
        "target_tgid": observer.target_tgid,
        "primary_allocation": primary,
        "secondary_allocation": secondary,
        "lifecycle": lifecycle,
        "event_count": len(observer.rows),
        "primary_page_count": len(primary_pages),
        "secondary_page_count": len(secondary_pages),
        "alias_pages_output": str(pages_output),
        "alias_pages_sha256": sha256(pages_output) if page_rows else "",
        "distinct_fb_pa_pages": len({str(row["fb_pa_page_base"]) for row in page_rows}),
        "reverse_mapping_one_to_many": all(len(t) == 2 for t in reverse.values()) if reverse else False,
        "alias_confirmed_at_page_level": alias_confirmed,
        "runtime_alias_verification": alias_record or None,
        "runtime_alias_passed": alias_record.get("status") == "PASS",
        "mapping_alive_during_hold": mapping_alive_during_hold,
        "hold_duration_monotonic_ns": hold_duration_ns,
        "kernel_gpu_uuid": (primary_pages[0]["gpu_uuid"] if primary_pages else None),
        "lost_event_count": observer.lost_event_count,
        "contract_sha256": contract_hash,
        "nvidia_module_sha256": contract["nvidia_module_sha256"],
        "nvidia_uvm_module_sha256": contract["nvidia_uvm_module_sha256"],
        "run_directory": str(run_dir),
        "event_output": str(event_output),
        "event_output_sha256": sha256(event_output),
        "test_log": str(test_log),
        "test_log_sha256": sha256(test_log),
        "started_wall_time_ns": started_ns,
        "ended_wall_time_ns": time.time_ns(),
        "driver_modified": False,
        "uvm_state_modified": False,
        "pte_modified": False,
        "other_gpu_processes_stopped": False,
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    chown_outputs([event_output, pages_output, summary_output, test_log, run_dir], uid, gid)
    print(f"device={args.device} status={status} pages={len(page_rows)} "
          f"distinct_pa={summary['distinct_fb_pa_pages']} "
          f"runtime_alias={summary['runtime_alias_passed']}")
    print(f"summary={summary_output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 2 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"GPU_M2D_G2_ALIAS_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
