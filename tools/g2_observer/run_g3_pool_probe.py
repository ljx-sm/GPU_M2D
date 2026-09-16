#!/usr/bin/env python3
"""GPU_M2D G3 pool probe orchestrator (S2).

Runs one pre-allocation-gated observation of the G3 timing pool under the
read-only G2 PTE observer, then executes a pair-timing workload inside the
observed pool without ever tearing it down:

  1. validate the pinned driver/PTE contract and attach the probes;
  2. start the pool harness, which blocks before any CUDA context exists;
  3. open the gate; the harness allocates N chunks (cudaMalloc) and blocks
     again at the release gate while warming up the timing kernel;
  4. build the PA ledger for every chunk with the validated G2 ledger
     code (containment matching, complete valid local-VIDEO PTE payloads,
     gapless tiling) *before* releasing any work -- a query address is
     only accepted once its page's framebuffer PA is observed;
  5. select the work CSV from the observed page table (g3_pool.py) and
     release the harness, which times the pairs and writes result rows;
  6. after teardown, re-run the strict ledger (kernel FREE events must
     land inside the teardown window) and write pool_map.csv: one row per
     GMMU page of every chunk with its observed framebuffer PA base.

Fail-closed rules: any ledger failure other than the expected
pending-free marker before release, payload truncation, non-VIDEO page,
coverage gap, lost BPF event, lifecycle mismatch, missing PASS marker, or
incomplete result rows fails the whole run. The pre-release ledger pass
reuses build_allocation_ledger with an open teardown window and accepts
exactly the per-allocation "no FREE_RETURN after mapping" notice; every
other diagnostic is fatal. Timing numbers are logged informationally --
S2 proves the pool map and the gated timing path, not mapping rules.

Must be started through ``sudo`` from the research account: root is used
only for the kprobe attachment, while the CUDA child is dropped back to
the invoking user. Nothing is modified on the driver side and no other
GPU process is touched. The timing phase additionally requires the target
GPU to have no co-tenant compute process (the orchestrator refuses to
start otherwise).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from run_g2_tensorrt_probe import (
    MAP_CONFIDENCE,
    MAP_SOURCE,
    build_allocation_ledger,
    chown_outputs,
    child_preexec,
    drop_to_invoking_user,
    normalize_gpu_uuid,
    parse_harness_output,
    read_until_marker,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
G3_PROBE_DIR = PROJECT / "tools/g3_probe"
sys.path.insert(0, str(G3_PROBE_DIR))
from g3_pool import (  # noqa: E402  (path-based import of a sibling tool)
    POOL_MAP_FIELDS,
    PoolMap,
    Page,
    select_bit_scan_queries,
    select_sanity_queries,
    write_work_csv,
)

LIFECYCLE_SKELETON = [
    "PROCESS_READY",
    "WAIT_PRE_ALLOC_GATE",
    "PRE_ALLOC_GATE_OPEN",
    "CONTEXT_BEGIN",
    "CONTEXT_READY",
    "ALLOCATION_BEGIN",
    "ALLOCATION_END",
    "POOL_READY",
    "WARMUP_BEGIN",
    "WARMUP_END",
    "WORK_RELEASED",
    "WORK_BEGIN",
    "WORK_DONE",
    "FREE_BEGIN",
    "FREE_END",
    "TEARDOWN_BEGIN",
    "PROCESS_END",
]
VARIABLE_EVENTS = {"ALLOCATED", "FREE"}
PENDING_FREE_NOTICE = ": no FREE_RETURN after mapping"
OPEN_TEARDOWN_NS = 2**63 - 1

POOL_SCHEMA = "gpu-m2d.g3-pool-probe.v1"
RESULT_FIELDS = ["query_id", "chunk_a", "ofs_a", "chunk_b", "ofs_b",
                 "cycles_a", "cycles_b"]


def drain_pipe(process: subprocess.Popen[bytes], captured: bytearray) -> None:
    """Drains whatever the child has left in its stdout pipe."""
    if process.stdout is None:
        return
    fd = process.stdout.fileno()
    os.set_blocking(fd, False)
    try:
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            captured.extend(chunk)
    except BlockingIOError:
        pass


def drain_until_marker(process: subprocess.Popen[bytes], observer: G2Observer,
                       marker: bytes, captured: bytearray,
                       timeout_seconds: float) -> bool:
    """Reads child stdout while continuously draining the observer ring.

    Unlike read_until_marker (used around the pre-allocation gate, where no
    events flow yet), this keeps observer.poll running so a long allocation
    burst cannot overflow the perf buffer while we wait for POOL_READY or
    process exit.
    """
    if process.stdout is None:
        raise RuntimeError("harness stdout pipe is unavailable")
    fd = process.stdout.fileno()
    os.set_blocking(fd, False)
    deadline = time.monotonic() + timeout_seconds
    found = any(line.startswith(marker) for line in captured.splitlines())
    while not found and time.monotonic() < deadline:
        observer.poll(50)
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            chunk = b""
        if chunk:
            captured.extend(chunk)
            found = any(line.startswith(marker) for line in captured.splitlines())
            continue
        if process.poll() is not None:
            drain_pipe(process, captured)
            found = any(line.startswith(marker) for line in captured.splitlines())
            break
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="run the offline ledger/result-parsing tests and exit")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=100000)
    parser.add_argument("--modifier", type=int, default=5)
    parser.add_argument("--cross-page", type=int, default=8,
                        help="sanity mode: cross-page queries over the observed PA range")
    parser.add_argument("--work-mode", choices=("sanity", "bit-scan"), default="sanity",
                        help="query selection: S2 sanity triple, or the S3 bit-scan matrix")
    parser.add_argument("--in-page-bases", type=int, default=4,
                        help="bit-scan mode: base pages voting per in-page bit")
    parser.add_argument("--pairs-per-bit", type=int, default=64,
                        help="bit-scan mode: cap on page-level pairs per PA bit")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "artifacts/g3/pool")
    return parser.parse_args()


def query_device_uuid(device: int) -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(device)],
        text=True, stdout=subprocess.PIPE, check=True).stdout.strip()


def refuse_if_occupied(device_uuid: str) -> None:
    """The timing phase needs an otherwise-idle GPU; co-tenant CUDA
    traffic corrupts the measurements (never touched, just detected)."""
    occupied = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, check=True).stdout.split()
    if normalize_gpu_uuid(device_uuid) in {normalize_gpu_uuid(u) for u in occupied}:
        raise RuntimeError(f"device {device_uuid} has co-tenant compute processes;"
                           " the G3 timing pool requires an idle GPU")


def segments_to_pages(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (chunk, GMMU page); the ledger emits clipped segments,
    consecutive ones inside one page collapse into the page extent."""
    pages: dict[tuple[str, int], dict[str, Any]] = {}
    for segment in segments:
        key = (str(segment["allocation_id"]), int(segment["va_page_base"]))
        existing = pages.get(key)
        row = {
            "allocation_id": segment["allocation_id"],
            "va_page_base": int(segment["va_page_base"]),
            "va_page_end_exclusive": int(segment["va_page_base"])
            + int(segment["page_size"]),
            "fb_pa_page_base": int(segment["physical_page_base"]),
            "page_size": int(segment["page_size"]),
            "aperture": segment["aperture"],
            "pte_valid": segment["pte_valid"],
            "pte_size": int(segment["pte_size"]),
            "raw_pte_lo": segment["raw_pte_lo"],
            "raw_pte_hi": segment["raw_pte_hi"],
            "mapped_at_ns": int(segment["mapped_at_ns"]),
            "unmapped_at_ns": int(segment["unmapped_at_ns"]) if segment["unmapped_at_ns"] is not None else "",
            "gpu_uuid": segment["header"]["gpu_uuid"],
        }
        if existing is not None and existing != row:
            raise RuntimeError(f"conflicting page rows for {key}")
        pages[key] = row
    return sorted(pages.values(), key=lambda row: (row["allocation_id"],
                                                   row["va_page_base"]))


def ledger_rows(allocated: list[dict[str, str]]) -> list[dict[str, str]]:
    """Registry-style rows the G2 ledger code expects, in chunk order."""
    return [{"allocation_id": record["allocation_id"],
             "gpu_va": record["base_va"],
             "size_bytes": record["size_bytes"]}
            for record in sorted(allocated, key=lambda record: int(record["chunk_index"]))]


def build_pool_ledger(observer: G2Observer, allocated: list[dict[str, str]],
                      teardown_end_ns: int,
                      allow_pending_free: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Runs the validated G2 ledger per chunk.

    Before the release gate no kernel FREE has happened yet, so with an
    open teardown window each allocation must produce exactly the pending
    notice and nothing else; after teardown the strict pass must be clean.
    """
    failures: list[str] = []
    all_pages: list[dict[str, Any]] = []
    for row in ledger_rows(allocated):
        record = next(item for item in allocated
                      if item["allocation_id"] == row["allocation_id"])
        segments, ledger_failures = build_allocation_ledger(
            observer.rows, row, record, teardown_end_ns)
        if allow_pending_free:
            pending = [failure for failure in ledger_failures
                       if failure.endswith(PENDING_FREE_NOTICE)]
            other = [failure for failure in ledger_failures
                     if not failure.endswith(PENDING_FREE_NOTICE)]
            if len(pending) != 1:
                other.append(f"{row['allocation_id']}: expected the pending-free"
                             f" notice before release, got {ledger_failures}")
            failures.extend(other)
        else:
            failures.extend(ledger_failures)
        all_pages.extend(segments_to_pages(segments))
    return all_pages, failures


def read_result_csv(path: Path, expected_queries: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"result CSV missing: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        # The harness prefixes a '# ...' provenance comment line.
        lines = [line for line in source if not line.startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames != RESULT_FIELDS:
        raise RuntimeError(f"unexpected result header: {reader.fieldnames}")
    rows = list(reader)
    if len(rows) != expected_queries:
        raise RuntimeError(f"expected {expected_queries} result rows, got {len(rows)}")
    if {int(row["query_id"]) for row in rows} != set(range(expected_queries)):
        raise RuntimeError("result query ids are not exactly 0..N-1")
    for row in rows:
        int(row["cycles_a"])
        int(row["cycles_b"])
    return rows


def timing_summary(rows: list[dict[str, str]], work: list[tuple[int, int, int, int, int]]) \
        -> list[dict[str, Any]]:
    """Informational per-query view with the observed PA of both sides."""
    work_by_id = {query[0]: query for query in work}
    summary: list[dict[str, Any]] = []
    for row in rows:
        _, chunk_a, ofs_a, chunk_b, ofs_b = work_by_id[int(row["query_id"])]
        summary.append({
            "query_id": int(row["query_id"]),
            "chunk_a": chunk_a, "ofs_a": ofs_a,
            "chunk_b": chunk_b, "ofs_b": ofs_b,
            "cycles_a": int(row["cycles_a"]),
            "cycles_b": int(row["cycles_b"]),
        })
    return summary


def finalize(failures: list[str], run_dir: Path, summary_output: Path,
             pool_map_output: Path, work_output: Path, result_output: Path,
             test_log: Path, event_output: Path, observer: G2Observer,
             contract: dict[str, Any], contract_hash: str, args: argparse.Namespace,
             run_id: str, device_uuid: str, started_ns: int,
             map_rows: list[dict[str, object]], result_rows: list[dict[str, Any]],
             query_count: int, rate_mhz: float | None, uid: int, gid: int,
             extra: dict[str, Any] | None) -> int:
    pool_map_output.parent.mkdir(parents=True, exist_ok=True)
    with pool_map_output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=POOL_MAP_FIELDS)
        writer.writeheader()
        writer.writerows(map_rows)

    status = "FAIL_CLOSED" if failures else "G3_POOL_PA_MAP_COMPLETE_OBSERVED"
    summary = {
        "schema_version": POOL_SCHEMA,
        "status": status,
        "meaning": "framebuffer PA of every pool page is claimed only from valid"
                   " captured PTEs whose decoded aperture is VIDEO (G2 observer)",
        "failures": failures,
        "run_id": run_id,
        "device": args.device,
        "device_uuid": device_uuid,
        "target_tgid": observer.target_tgid,
        "chunks": args.chunks,
        "chunk_mib": args.chunk_mib,
        "pool_bytes": args.chunks * args.chunk_mib * 1024 * 1024,
        "iters": args.iters,
        "modifier": args.modifier,
        "query_count": query_count,
        "result_row_count": len(result_rows),
        "rate_mhz_during_work": rate_mhz,
        "map_row_count": len(map_rows),
        "distinct_fb_pages": len({str(row["fb_pa_page_base"]) for row in map_rows}),
        "gpu_local_pa_observed": not failures and bool(map_rows),
        "pool_map_output": str(pool_map_output),
        "pool_map_output_sha256": sha256(pool_map_output) if map_rows else "",
        "work_output": str(work_output),
        "result_output": str(result_output),
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
    chown_outputs([event_output, pool_map_output, work_output, result_output,
                   summary_output, test_log, run_dir], uid, gid)
    print(f"device={args.device} status={status} chunks={args.chunks} "
          f"pages={len(map_rows)} queries={query_count} "
          f"results={len(result_rows)} events={len(observer.rows)}")
    print(f"summary={summary_output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 2 if failures else 0


def _synthetic_rows(map_base: int, n_pages: int, free_at: int | None = None) \
        -> list[dict[str, object]]:
    """Observer-schema rows for one allocation covering n_pages x 2 MiB."""
    page = 2 * 1024 * 1024

    def row(**over):
        base = dict(timestamp_ns=0, pid=1, tgid=1, event_type="", status=0,
                    address_space_id="uvmfile-x", rm_va_space_id="rmvas-x",
                    map_base="", map_length=0, map_offset=0, gpu_uuid="GPU-abc",
                    query_offset=0, query_size=0, mapping_page_size=0,
                    num_written=0, num_remaining=0, pte_size=8, pte_index="",
                    payload_complete="false", raw_pte_lo="", raw_pte_hi="",
                    raw_entry="", high_word_zero="", valid="", aperture="",
                    physical_page_base="", gpu_local_pa="false",
                    need_l2_invalidate="false", contract_sha256="x")
        base.update(over)
        return base

    rows = [row(event_type="MAP_RETURN", map_base=f"0x{map_base:x}",
                map_length=n_pages * page, timestamp_ns=1000),
            row(event_type="PTE_HEADER", map_base=f"0x{map_base:x}",
                map_length=n_pages * page, map_offset=0, query_offset=0,
                query_size=n_pages * page, mapping_page_size=page,
                num_written=n_pages, num_remaining=0,
                payload_complete="true", timestamp_ns=1001)]
    for index in range(n_pages):
        rows.append(row(event_type="PTE_ENTRY", map_base=f"0x{map_base:x}",
                        map_length=n_pages * page, map_offset=0, query_offset=0,
                        query_size=n_pages * page, pte_index=index,
                        valid="true", aperture="VIDEO", high_word_zero="true",
                        physical_page_base=f"0x{0x120000000 + index * page:x}",
                        raw_pte_lo="0x1", raw_pte_hi="0x0", timestamp_ns=1001))
    if free_at is not None:
        rows.append(row(event_type="FREE_RETURN", map_base=f"0x{map_base:x}",
                        map_length=n_pages * page, timestamp_ns=free_at))
    return rows


class _FakeObserver:
    def __init__(self, rows: list[dict[str, object]]):
        self.rows = rows
        self.lost_event_count = 0


def _synthetic_allocated(base: int, size: int, chunk: int = 0) -> list[dict[str, str]]:
    return [{"allocation_id": f"chunk{chunk}", "base_va": hex(base),
             "size_bytes": str(size), "chunk_index": str(chunk),
             "monotonic_ns": "1002"}]


def self_test() -> int:
    """Offline tests of the ledger wrappers and result parsing (no root,
    no GPU, no live observer): what `make -C tools/g2_observer check` runs."""
    page = 2 * 1024 * 1024
    alloc_base = 0x100000000

    # Pre-pass (no kernel FREE yet): only the expected pending-free notice.
    pages, failures = build_pool_ledger(
        _FakeObserver(_synthetic_rows(alloc_base, 4)), _synthetic_allocated(alloc_base, 4 * page),
        OPEN_TEARDOWN_NS, allow_pending_free=True)
    assert failures == [], failures
    assert len(pages) == 4 and pages[0]["fb_pa_page_base"] == 0x120000000

    # Strict pass with the FREE inside the teardown window: clean, and the
    # unmapped timestamp is recorded.
    pages, failures = build_pool_ledger(
        _FakeObserver(_synthetic_rows(alloc_base, 4, free_at=5000)),
        _synthetic_allocated(alloc_base, 4 * page), 9000, allow_pending_free=False)
    assert failures == [] and pages[0]["unmapped_at_ns"] == 5000, (pages, failures)

    # FREE landing after the teardown window fails closed.
    _, failures = build_pool_ledger(
        _FakeObserver(_synthetic_rows(alloc_base, 4, free_at=9500)),
        _synthetic_allocated(alloc_base, 4 * page), 9000, allow_pending_free=False)
    assert any("outlived the teardown window" in failure for failure in failures)

    # A non-local page is fatal even during the pre-pass.
    rows = _synthetic_rows(alloc_base, 4)
    for row in rows:
        if row["event_type"] == "PTE_ENTRY" and row["pte_index"] == 1:
            row["aperture"] = "SYSMEM"
    _, failures = build_pool_ledger(
        _FakeObserver(rows), _synthetic_allocated(alloc_base, 4 * page),
        OPEN_TEARDOWN_NS, allow_pending_free=True)
    assert any("non-local" in failure for failure in failures)

    # An allocation interior to a bigger map: clipped segments collapse
    # into full-page rows (RM packing).
    map_base = 0x200000000
    interior = map_base + 2 * page
    pages, failures = build_pool_ledger(
        _FakeObserver(_synthetic_rows(map_base, 4)),
        _synthetic_allocated(interior, 2 * page, chunk=7),
        OPEN_TEARDOWN_NS, allow_pending_free=True)
    assert failures == [] and len(pages) == 2
    assert pages[0]["va_page_base"] == interior
    assert pages[0]["fb_pa_page_base"] == 0x120000000 + 2 * page

    # Result parsing tolerates the harness's '# ...' provenance comment and
    # rejects wrong counts / ids (this is the bug the first sudo run hit).
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
        handle.write("# g3 pool query results: cycles are the minimum over 10 launches, modifier=.volatile\n")
        handle.write("query_id,chunk_a,ofs_a,chunk_b,ofs_b,cycles_a,cycles_b\n")
        handle.write("0,0,0,0,0,1010,1010\n")
        handle.write("1,0,0,0,8192,1023,1023\n")
        path = Path(handle.name)
    try:
        rows = read_result_csv(path, 2)
        assert rows[1]["cycles_b"] == "1023"
        try:
            read_result_csv(path, 3)
            raise AssertionError("wrong row count must fail")
        except RuntimeError:
            pass
    finally:
        path.unlink(missing_ok=True)

    print("g3 pool probe self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if os.geteuid() != 0:
        raise PermissionError("run through sudo; the CUDA child is dropped back to the invoking user")
    uid, gid = drop_to_invoking_user()
    if not (1 <= args.chunks <= 4096) or not (1 <= args.chunk_mib <= 512):
        raise RuntimeError("chunks must be in [1, 4096] and chunk-mib in [1, 512]")
    contract, contract_hash = load_and_validate_contract(args.contract.resolve(),
                                                          DEFAULT_OPEN_SOURCE_REPO)
    harness = G3_PROBE_DIR / "g3_pool_harness"
    if not harness.is_file() or not os.access(harness, os.X_OK):
        raise RuntimeError(f"build the harness first: make -C tools/g3_probe all ({harness})")

    device_uuid = query_device_uuid(args.device)
    refuse_if_occupied(device_uuid)

    run_dir = args.output_root / f"run_pool_gpu{args.device}_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chown(run_dir, uid, gid)
    event_output = run_dir / "events.csv"
    pool_map_output = run_dir / "pool_map.csv"
    work_output = run_dir / "work.csv"
    result_output = run_dir / "result.csv"
    summary_output = run_dir / "summary.json"
    test_log = run_dir / "harness.log"
    gate = run_dir / f".gate_{os.getpid()}_{time.time_ns()}"
    release = run_dir / f".release_{os.getpid()}_{time.time_ns()}"
    run_id = run_dir.name

    observer = G2Observer(contract, contract_hash, 0, event_output)
    process: subprocess.Popen[bytes] | None = None
    started_ns = time.time_ns()
    captured = bytearray()
    pool_ready = False
    work: list[tuple[int, int, int, int, int]] = []
    result_rows: list[dict[str, Any]] = []
    rate_mhz: float | None = None
    map_rows: list[dict[str, object]] = []
    failures: list[str] = []
    extra: dict[str, Any] = {}
    try:
        process = subprocess.Popen(
            [
                str(harness.resolve()),
                "--gate-file", str(gate),
                "--release-file", str(release),
                "--work-file", str(work_output),
                "--result-file", str(result_output),
                "--device", str(args.device),
                "--chunks", str(args.chunks),
                "--chunk-mib", str(args.chunk_mib),
                "--iters", str(args.iters),
                "--warmup", str(args.warmup),
                "--modifier", str(args.modifier),
                "--gate-timeout-seconds", str(max(5, int(args.timeout_seconds))),
                "--release-timeout-seconds", str(max(120, int(args.timeout_seconds))),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            preexec_fn=child_preexec(uid, gid),
        )
        observer.set_target_tgid(process.pid)

        prelude, gate_ready = read_until_marker(
            process, b"GPU_M2D_EVENT,event=WAIT_PRE_ALLOC_GATE,",
            min(args.timeout_seconds, 10.0))
        captured.extend(prelude)
        if not gate_ready:
            process.terminate()
            raise RuntimeError("harness did not reach the pre-allocation gate")
        gate.write_text(f"observer_ready target_tgid={process.pid}\n", encoding="utf-8")
        os.chmod(gate, 0o644)

        pool_ready = drain_until_marker(
            process, observer, b"GPU_M2D_EVENT,event=POOL_READY,", captured,
            args.timeout_seconds)
        if not pool_ready:
            raise RuntimeError("harness did not reach POOL_READY")
        # Let trailing PTE events of the last chunks land before the ledger.
        for _ in range(20):
            observer.poll(50)

        allocated, _, _ = parse_harness_output(captured.decode("utf-8", errors="replace"))
        if len(allocated) != args.chunks:
            raise RuntimeError(f"expected {args.chunks} ALLOCATED events, got {len(allocated)}")
        chunk_of = {record["allocation_id"]: int(record["chunk_index"])
                    for record in allocated}

        pre_pass_pages, pre_failures = build_pool_ledger(
            observer, allocated, OPEN_TEARDOWN_NS, allow_pending_free=True)
        if pre_failures or not pre_pass_pages:
            raise RuntimeError("pre-release pool ledger failed: " + "; ".join(pre_failures))

        pages = [Page(chunk_index=chunk_of[page["allocation_id"]],
                      va_page_base=int(page["va_page_base"]),
                      va_page_end=int(page["va_page_end_exclusive"]),
                      fb_pa_page_base=int(page["fb_pa_page_base"]),
                      page_size=int(page["page_size"]))
                 for page in pre_pass_pages]
        pool = PoolMap(pages)
        if args.work_mode == "bit-scan":
            queries = select_bit_scan_queries(pool, in_page_bases=args.in_page_bases,
                                              pairs_per_bit=args.pairs_per_bit)
        else:
            queries = select_sanity_queries(pool, cross_page=args.cross_page)
        work = [(index, query.chunk_a, query.ofs_a, query.chunk_b, query.ofs_b)
                for index, query in enumerate(queries)]
        write_work_csv(work_output, queries, mode=args.work_mode)
        os.chmod(work_output, 0o644)
        extra["query_selection"] = {
            "mode": args.work_mode,
            "in_page_candidates": [0, 8192, 852224],
            "cross_page": args.cross_page,
            "in_page_bases": args.in_page_bases,
            "pairs_per_bit": args.pairs_per_bit,
            "distinct_fb_pages": len(pool.pa_pages),
        }
        release.write_text(f"pool_map_complete target_tgid={process.pid}\n",
                           encoding="utf-8")
        os.chmod(release, 0o644)

        finished = drain_until_marker(process, observer, b"GPU_M2D_G3_POOL_PASS",
                                      captured, args.timeout_seconds)
        if not finished:
            if process.poll() is None:
                process.terminate()
            raise RuntimeError("harness did not finish the work phase")
        exit_deadline = time.monotonic() + args.timeout_seconds
        while process.poll() is None and time.monotonic() < exit_deadline:
            observer.poll(50)
        if process.poll() is None:
            process.terminate()
            raise RuntimeError("harness exceeded timeout")
        for _ in range(20):
            observer.poll(50)
        drain_pipe(process, captured)

        output = captured.decode("utf-8", errors="replace")

        allocated, lifecycle, times = parse_harness_output(output)
        if process.returncode != 0:
            failures.append(f"harness exit code {process.returncode}")
        if "GPU_M2D_G3_POOL_PASS" not in output:
            failures.append("harness did not report GPU_M2D_G3_POOL_PASS")
        skeleton = [name for name in lifecycle if name not in VARIABLE_EVENTS]
        if skeleton != LIFECYCLE_SKELETON:
            failures.append(f"lifecycle skeleton mismatch: {skeleton}")
        if len(allocated) != args.chunks:
            failures.append(f"expected {args.chunks} ALLOCATED events, got {len(allocated)}")
        freed = [line for line in output.splitlines()
                 if line.startswith("GPU_M2D_EVENT,event=FREE,")]
        if len(freed) != args.chunks:
            failures.append(f"expected {args.chunks} FREE events, got {len(freed)}")

        final_pages, final_failures = build_pool_ledger(
            observer, allocated, times.get("PROCESS_END", 0), allow_pending_free=False)
        failures.extend(final_failures)

        map_rows = [{
            "run_id": run_id,
            "device": args.device,
            "gpu_uuid": page["gpu_uuid"],
            "chunk_index": chunk_of[page["allocation_id"]],
            "allocation_id": page["allocation_id"],
            "va_page_base": f"0x{int(page['va_page_base']):x}",
            "va_page_end_exclusive": f"0x{int(page['va_page_end_exclusive']):x}",
            "fb_pa_page_base": f"0x{int(page['fb_pa_page_base']):x}",
            "page_size": page["page_size"],
            "aperture": page["aperture"],
            "pte_valid": page["pte_valid"],
            "raw_pte_lo": page["raw_pte_lo"],
            "raw_pte_hi": page["raw_pte_hi"],
            "mapped_at_ns": page["mapped_at_ns"],
            "unmapped_at_ns": page["unmapped_at_ns"],
            "source": MAP_SOURCE,
            "confidence": MAP_CONFIDENCE,
        } for page in sorted(final_pages,
                             key=lambda page: (chunk_of[page["allocation_id"]],
                                               int(page["va_page_base"])))]

        kernel_uuids = sorted({str(row["gpu_uuid"]) for row in observer.rows
                               if row["event_type"] == "PTE_HEADER" and row["gpu_uuid"]})
        if len(kernel_uuids) > 1:
            failures.append(f"multiple kernel GPU UUIDs observed: {kernel_uuids}")
        elif kernel_uuids and normalize_gpu_uuid(kernel_uuids[0]) != normalize_gpu_uuid(device_uuid):
            failures.append(f"kernel GPU UUID {kernel_uuids[0]} != device UUID {device_uuid}")
        if len({str(row["address_space_id"]) for row in observer.rows
                if row["event_type"] in ("MAP_RETURN", "PTE_HEADER")}) > 1:
            failures.append("multiple UVM address-space IDs observed")
        if observer.lost_event_count:
            failures.append(f"lost BPF events: {observer.lost_event_count}")

        if not failures:
            result_rows = read_result_csv(result_output, len(work))
            result_rows = timing_summary(result_rows, work)
            work_begin = [line for line in output.splitlines()
                          if line.startswith("GPU_M2D_EVENT,event=WORK_BEGIN,")]
            if work_begin:
                fields = dict(item.partition("=")[::2]
                              for item in work_begin[0].split(",")[1:])
                rate_mhz = float(fields.get("rate_mhz", "nan"))
        shared: dict[str, set[int]] = {}
        for row in map_rows:
            shared.setdefault(str(row["fb_pa_page_base"]), set()).add(int(row["chunk_index"]))
        extra["shared_backing_page_chunk_groups"] = sorted(
            sorted(groups) for groups in shared.values() if len(groups) > 1)
        extra["timing_results_informational"] = result_rows
    finally:
        gate.unlink(missing_ok=True)
        release.unlink(missing_ok=True)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        observer.close()
        test_log.write_text(captured.decode("utf-8", errors="replace"),
                            encoding="utf-8")

    return finalize(failures, run_dir, summary_output, pool_map_output,
                    work_output, result_output, test_log, event_output,
                    observer, contract, contract_hash, args, run_id,
                    device_uuid, started_ns, map_rows, result_rows,
                    len(work), rate_mhz, uid, gid, extra)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(f"GPU_M2D_G3_POOL_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
