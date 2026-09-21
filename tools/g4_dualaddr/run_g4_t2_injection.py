#!/usr/bin/env python3
"""GPU_M2D G4-T2: gated live dual-addressing XOR through the whole chain.

One run joins the three validated legs ONLINE, on a live TensorRT workload:

  1. the G2 eBPF observer attaches before any CUDA context exists (the
     runner blocks at the pre-allocation gate);
  2. the runner allocates the full G1.5 workload (TRT-internal + semantic
     bindings), runs a clean inference, writes its gate-time allocation
     registry and blocks at the injection gate;
  3. the orchestrator builds the per-allocation PTE ledger from the live
     observer (pre-pass: complete valid local-VIDEO coverage, exactly one
     pending-free notice per allocation), writes the run's
     gpu_va_pa_map_gate.csv, and folds it with the registry and the G3
     table into the G4 snapshot (build_snapshot.run_build -- the same
     fail-closed join G4-T1 validated offline);
  4. reverse-chain sites are selected FROM THE SNAPSHOT (GDDR PA page ->
     VA page -> resident allocation byte): one in a semantic binding
     region, one in a TRT-internal region, deterministic policy, bank-
     linked pages preferred;
  5. the work file is written and the release gate opens; the runner
     verifies each target against its LIVE registry (expected VA), XORs
     the bit on device, verifies after == before ^ mask, verifies every
     other byte of the allocation is unchanged, reverse-maps, runs the
     injected inference (DUE tolerated as an outcome), restores every
     fault (byte and whole-allocation checks), and proves no residue with
     a sanity inference that must reproduce the clean output;
  6. after teardown the strict ledger re-runs (frees inside the teardown
     window, mappings alive across the injection window), the gate and
     final VA-PA maps must be identical (mid-run remap detector), the
     gate and final registries must be identical, and every result row is
     re-verified independently of the runner's own checks.

Fail-closed: any ledger failure, snapshot problem, empty target category,
result disagreement, remap, lost BPF event, lifecycle mismatch, or
missing PASS marker fails the run (exit 2). Co-tenant processes are never
touched and never refuse the run (post-timing policy); their state at
start is recorded in the summary.

Must be started through ``sudo`` from the research account (root only
attaches the probes; the CUDA child is dropped back to the invoking
user). Exit codes: 0 ok, 2 fail-closed, 1 error.
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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
G2_DIR = PROJECT / "tools/g2_observer"
sys.path.insert(0, str(G2_DIR))
sys.path.insert(0, str(G2_DIR.parent / "g3_probe"))

from g2_observer import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_OPEN_SOURCE_REPO,
    G2Observer,
    load_and_validate_contract,
    sha256,
)
from run_g2_tensorrt_probe import (  # noqa: E402
    MAP_CONFIDENCE,
    MAP_FIELDS,
    MAP_SOURCE,
    build_allocation_ledger,
    chown_outputs,
    child_preexec,
    drop_to_invoking_user,
    normalize_gpu_uuid,
    parse_harness_output,
    read_until_marker,
)
from run_g3_pool_probe import (  # noqa: E402
    OPEN_TEARDOWN_NS,
    PENDING_FREE_NOTICE,
    drain_pipe,
    drain_until_marker,
)
from build_snapshot import run_build  # noqa: E402

T2_SCHEMA = "gpu-m2d.g4-t2.injection.v1"
PASS_MARKER = "GPU_M2D_G4_T2_PASS"

ALLOC_FIELDS = ["run_id", "allocation_id", "device", "gpu_va", "size_bytes",
                "alignment_bytes", "owner", "allocation_phase", "lifetime",
                "active_at_injection", "semantic_label"]

T2_SKELETON = [
    "PROCESS_READY",
    "WAIT_PRE_ALLOC_GATE",
    "PRE_ALLOC_GATE_OPEN",
    "CONTEXT_BEGIN",
    "CONTEXT_READY",
    "RUNTIME_BEGIN",
    "BINDINGS_READY",
    "CLEAN_INFERENCE_BEGIN",
    "CLEAN_INFERENCE_END",
    "ALLOCATION_REGISTRY_GATE_WRITTEN",
    "INJECTION_GATE_WAIT",
    "INJECTION_WORK_BEGIN",
    "INJECTED_INFERENCE_BEGIN",
    "INJECTED_INFERENCE_END",
    "RESTORE_BEGIN",
    "RESTORE_END",
    "SANITY_INFERENCE_BEGIN",
    "SANITY_INFERENCE_END",
    "INJECTION_WORK_END",
    "SNAPSHOT_READY",
    "HOLD_BEGIN",
    "HOLD_END",
    "TEARDOWN_BEGIN",
    "PROCESS_END",
]
T2_VARIABLE_EVENTS = {"ALLOCATED", "FREE", "TARGET_BEGIN", "TARGET_FLIPPED",
                      "TARGET_RESTORED", "INJECTED_OUTCOME"}

WORK_FIELDS = ["target_id", "allocation_id", "byte_offset", "bit_in_byte",
               "expected_gpu_va"]
RESULT_FIELDS = ["run_id", "device", "image", "target", "target_id",
                 "allocation_id", "semantic_label", "byte_offset",
                 "bit_in_byte", "xor_mask", "gpu_va", "expected_gpu_va",
                 "before", "after", "guard_bytes_unchanged", "reverse_map_ok",
                 "restored_byte_ok", "restore_guard_ok", "clean_class",
                 "clean_probability", "injected_class", "injected_probability",
                 "injected_outcome", "sanity_class", "sanity_probability",
                 "sanity_matches_clean"]
MAP_STABLE_FIELDS = ["fb_pa_page_base", "pte_valid", "aperture",
                     "physical_coverage_status", "page_size"]


class SelectionError(RuntimeError):
    """No injectable site of a requested kind in this run's snapshot."""


# --------------------------------------------------------------------------
# pure helpers (offline-testable)
# --------------------------------------------------------------------------

def _int(row: dict, key: str) -> int:
    return int(str(row[key]), 16 if str(row[key]).startswith("0x") else 10)


def select_targets(rows: list[dict], binding_bit: int, internal_bit: int,
                   ) -> tuple[list[dict], list[str]]:
    """Reverse-chain site selection from one run's snapshot rows.

    Policy (deterministic, every step recorded in the notes):
      t1-binding  -- semantic binding region, preferring bank-linked pages
                    (the GDDR leg is the scarce half of the chain), then
                    the ``data`` input binding (a fault that feeds the
                    network), then lowest PA page;
      t2-internal -- TRT-internal owned-unknown region, preferring bank-
                    linked pages, then the LARGEST allocation (the biggest
                    physical footprint), then lowest PA page.
    Byte: midpoint of the allocation's resident range on the chosen page;
    bit: the caller-fixed constants. The whole chain (PA page, GDDR class,
    VA page, byte offset) travels with the target.
    """
    notes: list[str] = []

    def build_target(row: dict, target_id: str, bit: int) -> dict:
        va_page = _int(row, "va_page_base")
        start = _int(row, "byte_start_in_page")
        end = _int(row, "byte_end_in_page")
        base = _int(row, "allocation_va_base")
        off_in_page = (start + end) // 2
        byte_offset = va_page + off_in_page - base
        return {
            "target_id": target_id,
            "allocation_id": row["allocation_id"],
            "semantic_label": row["semantic_label"],
            "byte_offset": byte_offset,
            "bit": bit,
            "expected_gpu_va": base + byte_offset,
            "chain": {
                "allocation_va_base": row["allocation_va_base"],
                "allocation_size_bytes":
                    int(str(row["allocation_size_bytes"])),
                "va_page_base": row["va_page_base"],
                "byte_start_in_page": row["byte_start_in_page"],
                "byte_end_in_page": row["byte_end_in_page"],
                "off_in_page": f"{off_in_page:#x}",
                "fb_pa_page_base": row["fb_pa_page_base"],
                "bank_linked": int(str(row["bank_linked"])),
                "channel_root": row["channel_root"],
                "channel_size": row["channel_size"],
                "row_class_count": int(str(row["row_class_count"])),
                "same_row_site_nodes": int(str(row["same_row_site_nodes"])),
                "valid_anchor_masks": row["valid_anchor_masks"],
            },
        }

    def choose(candidates: list[dict], key, target_id: str, bit: int,
               kind: str) -> dict:
        ordered = sorted(candidates, key=key)
        chosen = ordered[0]
        if int(str(chosen["bank_linked"])) != 1:
            notes.append(f"{target_id}: no bank-linked page hosts this kind "
                         f"({kind}); selected an UNLINKED page -- bank stays "
                         "UNKNOWN, site is still a valid SEU location")
        else:
            notes.append(f"{target_id}: selected {kind} "
                         f"{chosen['allocation_id']} on bank-linked PA page "
                         f"{chosen['fb_pa_page_base']} (component "
                         f"{chosen['channel_root'] or '-'})")
        if int(str(chosen["same_row_site_nodes"])) > 0:
            notes.append(f"{target_id}: chosen page also hosts measured "
                         "same-row classes")
        return build_target(chosen, target_id, bit)

    binding_rows = [r for r in rows
                    if str(r["semantic_label"]).startswith("TENSOR:")]
    internal_rows = [r for r in rows
                     if r["semantic_label"] == "TENSORRT_INTERNAL_UNKNOWN"]
    if not binding_rows:
        raise SelectionError("snapshot has no semantic binding region row")
    if not internal_rows:
        raise SelectionError("snapshot has no TRT-internal region row")

    targets = [
        choose(binding_rows,
               key=lambda r: (0 if str(r["bank_linked"]) == "1" else 1,
                              0 if r["semantic_label"] == "TENSOR:data" else 1,
                              _int(r, "fb_pa_page_base"),
                              _int(r, "va_page_base")),
               target_id="t1-binding", bit=binding_bit, kind="binding"),
        choose(internal_rows,
               key=lambda r: (0 if str(r["bank_linked"]) == "1" else 1,
                              -int(str(r["allocation_size_bytes"])),
                              _int(r, "fb_pa_page_base"),
                              _int(r, "va_page_base")),
               target_id="t2-internal", bit=internal_bit, kind="internal"),
    ]
    return targets, notes


def write_work_csv(path: Path, targets: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as sink:
        sink.write("# gpu-m2d g4-t2 reverse-chain XOR targets "
                   "(selected from this run's dual-addressing snapshot)\n")
        writer = csv.DictWriter(sink, fieldnames=WORK_FIELDS)
        writer.writeheader()
        for target in targets:
            writer.writerow({
                "target_id": target["target_id"],
                "allocation_id": target["allocation_id"],
                "byte_offset": target["byte_offset"],
                "bit_in_byte": target["bit"],
                "expected_gpu_va": f"{target['expected_gpu_va']:#x}",
            })


def verify_result_rows(targets: list[dict],
                       rows: list[dict]) -> list[str]:
    """Independent re-verification of the runner's result rows."""
    failures: list[str] = []
    by_id = {row["target_id"]: row for row in rows}
    if len(by_id) != len(rows):
        failures.append("duplicate target_id rows in the G4 result CSV")
    if set(by_id) != {t["target_id"] for t in targets}:
        failures.append("G4 result target_ids do not match the work file")
        return failures
    for target in targets:
        row = by_id[target["target_id"]]
        where = target["target_id"]
        if row["allocation_id"] != target["allocation_id"]:
            failures.append(f"{where}: allocation mismatch")
        if int(row["byte_offset"]) != target["byte_offset"] or \
                int(row["bit_in_byte"]) != target["bit"]:
            failures.append(f"{where}: byte/bit mismatch")
        if int(row["gpu_va"], 16) != target["expected_gpu_va"] or \
                int(row["expected_gpu_va"], 16) != target["expected_gpu_va"]:
            failures.append(f"{where}: gpu_va mismatch")
        mask = int(row["xor_mask"])
        before = int(row["before"])
        after = int(row["after"])
        if mask != (1 << target["bit"]):
            failures.append(f"{where}: xor_mask {mask} != 1<<bit")
        if after != (before ^ mask):
            failures.append(f"{where}: after {after} != before {before} "
                            f"^ mask {mask}")
        for flag in ("guard_bytes_unchanged", "reverse_map_ok",
                     "restored_byte_ok", "restore_guard_ok",
                     "sanity_matches_clean"):
            if row[flag] != "1":
                failures.append(f"{where}: {flag} not set")
        if row["injected_outcome"] not in ("BENIGN", "SDC_TOP1",
                                           "SDC_NUMERIC", "DUE_INVALID_OUTPUT"):
            failures.append(f"{where}: unknown injected_outcome "
                            f"{row['injected_outcome']}")
    return failures


def diff_map_rows(gate_rows: list[dict], final_rows: list[dict],
                  label: str) -> list[str]:
    """The mid-run remap detector: the join-relevant fields of every VA
    page must be identical between the gate-time map (what the snapshot
    was built from) and the strict post-teardown map."""
    def key(row: dict) -> tuple[str, str]:
        return (row["allocation_id"], row["va_page_base"])

    gate = {key(row): row for row in gate_rows}
    final = {key(row): row for row in final_rows}
    failures: list[str] = []
    if set(gate) != set(final):
        missing = sorted(set(gate) - set(final))
        extra = sorted(set(final) - set(gate))
        failures.append(f"{label}: page key sets differ "
                        f"(missing={missing} extra={extra})")
        return failures
    for page_key in sorted(gate):
        for field in MAP_STABLE_FIELDS:
            if str(gate[page_key][field]) != str(final[page_key][field]):
                failures.append(
                    f"{label}: {field} of {page_key[0]}@{page_key[1]} "
                    f"changed {gate[page_key][field]!r} -> "
                    f"{final[page_key][field]!r} between gate and teardown "
                    "(mid-run remap)")
    return failures


def parse_field_events(output: str, name: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    prefix = f"GPU_M2D_EVENT,event={name},"
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        fields: dict[str, str] = {}
        for item in line[len(prefix):].split(","):
            key, _, value = item.partition("=")
            if "=" in item:
                fields[key] = value
        events.append(fields)
    return events


def cross_check_target_events(output: str, result_rows: list[dict],
                              ) -> list[str]:
    failures: list[str] = []
    flipped = parse_field_events(output, "TARGET_FLIPPED")
    by_id = {row["target_id"]: row for row in result_rows}
    if len(flipped) != len(result_rows):
        failures.append(f"expected {len(result_rows)} TARGET_FLIPPED events, "
                        f"got {len(flipped)}")
        return failures
    for event in flipped:
        row = by_id.get(event.get("target_id", ""))
        if row is None:
            failures.append(f"TARGET_FLIPPED for unknown target "
                            f"{event.get('target_id')}")
            continue
        if event.get("before") != row["before"] or \
                event.get("after") != row["after"] or \
                event.get("xor_mask") != row["xor_mask"] or \
                event.get("gpu_va", "").lower() != row["gpu_va"].lower():
            failures.append(f"{event.get('target_id')}: TARGET_FLIPPED event "
                            "disagrees with the result CSV")
    outcomes = parse_field_events(output, "INJECTED_OUTCOME")
    if len(outcomes) != 1 or outcomes[0].get("outcome") != \
            next(iter(result_rows), {}).get("injected_outcome"):
        failures.append("INJECTED_OUTCOME event disagrees with the result CSV")
    return failures


# --------------------------------------------------------------------------
# live helpers
# --------------------------------------------------------------------------

def query_device_uuid(device: int) -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
         "-i", str(device)],
        text=True, stdout=subprocess.PIPE, check=True).stdout.strip()


def cotenancy_snapshot(device: int) -> dict[str, Any]:
    """Co-tenant state at start. Recorded, never a refusal (the standing
    post-timing policy): observation/XOR/inference are insensitive to
    co-tenants; only VRAM pressure can push allocations out of the table
    universe, and the snapshot's universe check refuses that fail-closed."""
    def run(args: list[str]) -> str:
        return subprocess.run(["nvidia-smi"] + args, text=True,
                              stdout=subprocess.PIPE,
                              check=True).stdout.strip()
    apps = [line for line in run(
        ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader"]).splitlines() if line]
    memory = run(["-i", str(device),
                  "--query-gpu=uuid,memory.used,memory.free",
                  "--format=csv,noheader"])
    return {
        "policy": "recorded_not_refused",
        "captured_at_ns": time.time_ns(),
        "compute_apps_all_gpus": apps,
        "device_memory": memory,
    }


def build_preledger(observer_rows: list[dict],
                    allocated: list[dict[str, str]],
                    ) -> tuple[list[dict], list[str]]:
    """G2 ledger at the injection gate: complete local-VIDEO coverage per
    allocation, exactly one pending-free notice each (nothing is freed
    yet), any other diagnostic fatal."""
    failures: list[str] = []
    segments: list[dict] = []
    for record in sorted(allocated, key=lambda r: r["allocation_id"]):
        row = {"allocation_id": record["allocation_id"],
               "gpu_va": record["base_va"],
               "size_bytes": record["size_bytes"]}
        alloc_segments, alloc_failures = build_allocation_ledger(
            observer_rows, row, record, OPEN_TEARDOWN_NS)
        pending = [f for f in alloc_failures if f.endswith(PENDING_FREE_NOTICE)]
        other = [f for f in alloc_failures
                 if not f.endswith(PENDING_FREE_NOTICE)]
        if len(pending) != 1:
            other.append(f"{row['allocation_id']}: expected the pending-free "
                         f"notice at the gate, got {alloc_failures}")
        failures.extend(other)
        segments.extend(alloc_segments)
    return segments, failures


def build_final_ledger(observer_rows: list[dict],
                       allocated: list[dict[str, str]],
                       teardown_end_ns: int,
                       ) -> tuple[list[dict], list[str]]:
    failures: list[str] = []
    segments: list[dict] = []
    for record in sorted(allocated, key=lambda r: r["allocation_id"]):
        row = {"allocation_id": record["allocation_id"],
               "gpu_va": record["base_va"],
               "size_bytes": record["size_bytes"]}
        alloc_segments, alloc_failures = build_allocation_ledger(
            observer_rows, row, record, teardown_end_ns)
        failures.extend(alloc_failures)
        segments.extend(alloc_segments)
    return segments, failures


def map_rows_from_segments(segments: list[dict], run_id: str,
                           g1_5_run_id: str, device: int,
                           registry: list[dict[str, str]],
                           ) -> list[dict[str, object]]:
    """G2-schema map rows (same construction the TRT probe validates)."""
    registry_by_id = {row["allocation_id"]: row for row in registry}
    rows: list[dict[str, object]] = []
    for segment in segments:
        record = registry_by_id.get(segment["allocation_id"], {})
        rows.append({
            "run_id": run_id,
            "g1_5_run_id": g1_5_run_id,
            "device": device,
            "gpu_uuid": segment["header"]["gpu_uuid"],
            "allocation_id": segment["allocation_id"],
            "allocation_api": record.get("owner", ""),
            "cuda_buffer_id": "",
            "allocation_size_bytes": record.get("size_bytes", ""),
            "semantic_label": record.get("semantic_label", ""),
            "allocation_phase": record.get("allocation_phase", ""),
            "active_at_snapshot": "true",
            "va_page_base": f"{int(segment['va_page_base']):x}",
            "va_page_end_exclusive":
                f"{int(segment['va_page_base']) + int(segment['page_size']):x}",
            "fb_pa_page_base": f"{int(segment['physical_page_base']):x}",
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
            "unmapped_at_ns":
                "" if segment["unmapped_at_ns"] is None
                else segment["unmapped_at_ns"],
            "covered_allocation_bytes":
                int(segment["covered_va_end_exclusive"])
                - int(segment["covered_va_start"]),
            "physical_coverage_status": "LOCAL_VIDEO_COMPLETE",
            "source": MAP_SOURCE,
            "confidence": MAP_CONFIDENCE,
        })
    return rows


def write_map_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=MAP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path, expected_fields: list[str]) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        lines = [line for line in source if not line.startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames != expected_fields:
        raise RuntimeError(f"unexpected header in {path}: {reader.fieldnames}")
    return list(reader)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    remu_root = Path(os.environ.get("GPU_M2D_REMU_ROOT", "/data1/luojx/REMU"))
    dataset_root = Path(os.environ.get(
        "GPU_M2D_DATASET_ROOT", "/data1/luojx/datasets/REMU_stage8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runner", type=Path,
                        default=PROJECT / "build-g1.5/gpu_m2d_resnet50_int8_g1_5")
    parser.add_argument("--engine", type=Path,
                        default=remu_root / "artifacts/stage8/engines/paper_priority/resnet50_resisc45_int8_ptq.engine")
    parser.add_argument("--sample-csv", type=Path,
                        default=dataset_root / "RESISC45/splits/original_repo_1000_eval.csv")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--table", type=Path,
                        default=PROJECT / "artifacts/g3/table_v4")
    parser.add_argument("--binding-bit", type=int, default=5,
                        help="bit for the t1 binding-region target (fixed "
                             "policy constant, recorded in the manifest)")
    parser.add_argument("--internal-bit", type=int, default=2,
                        help="bit for the t2 TRT-internal target")
    parser.add_argument("--hold-seconds", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "artifacts/g4/t2")
    return parser.parse_args()


def finalize(failures: list[str], run_dir: Path, summary_output: Path,
             test_log: Path, event_output: Path, observer: G2Observer,
             args: argparse.Namespace, run_id: str, device_uuid: str,
             started_ns: int, extra: dict[str, Any], uid: int, gid: int,
             ) -> int:
    status = ("FAIL_CLOSED" if failures
              else "G4_T2_DUALADDR_CHAIN_XOR_VERIFIED")
    summary = {
        "schema_version": T2_SCHEMA,
        "status": status,
        "meaning": "every target's bit was XOR-flipped on device through a "
                   "PA observed live this run and joined to the G3 table; "
                   "after == before ^ mask, guard bytes unchanged, restore "
                   "verified, post-restore inference reproduces the clean run",
        "failures": failures,
        "run_id": run_id,
        "device": args.device,
        "device_uuid": device_uuid,
        "target_tgid": observer.target_tgid,
        "table_dir": str(args.table),
        "engine_path": str(args.engine),
        "engine_sha256": sha256(args.engine) if args.engine.is_file() else "",
        "runner_path": str(args.runner),
        "runner_sha256": sha256(args.runner) if args.runner.is_file() else "",
        "event_count": len(observer.rows),
        "lost_event_count": observer.lost_event_count,
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
    summary.update(extra)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True)
                              + "\n", encoding="utf-8")
    outputs = [event_output, test_log, summary_output]
    for name in ("gpu_va_pa_map.csv", "gpu_va_pa_map_gate.csv",
                 "work.csv", "work_detail.json", "g1_5_g4_result.csv",
                 "g1_5_allocations.csv", "g1_5_allocations_gate.csv"):
        candidate = run_dir / name
        if candidate.is_file():
            outputs.append(candidate)
    snapshot_dir = run_dir / "snapshot"
    if snapshot_dir.is_dir():
        for name in ("snapshot_pages.csv", "manifest.json"):
            candidate = snapshot_dir / name
            if candidate.is_file():
                outputs.append(candidate)
    chown_outputs(outputs + [run_dir], uid, gid)
    print(f"device={args.device} status={status} run={run_id}")
    print(f"summary={summary_output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 2 if failures else 0


def run_once(args: argparse.Namespace) -> int:
    uid, gid = drop_to_invoking_user()
    contract, contract_hash = load_and_validate_contract(
        args.contract.resolve(), DEFAULT_OPEN_SOURCE_REPO)
    if not args.runner.is_file() or not os.access(args.runner, os.X_OK):
        raise RuntimeError(f"runner missing or not executable: {args.runner}")
    trt_runtime_dir = Path(os.environ.get(
        "GPU_M2D_REMU_ROOT", "/data1/luojx/REMU")) / \
        ".local/deps/tensorrt-8.6.1/tensorrt_libs"
    opencv_lib_dir = Path(os.environ.get(
        "GPU_M2D_REMU_ROOT", "/data1/luojx/REMU")) / ".local/deps/conda/lib"
    for required in (args.engine, args.sample_csv,
                     trt_runtime_dir / "libnvinfer.so.8",
                     args.table / "gddr_seed_table.csv",
                     args.table / "bank_classes.csv"):
        if not required.is_file():
            raise RuntimeError(f"missing required asset: {required}")

    cotenancy = cotenancy_snapshot(args.device)
    free_mib = 0.0
    try:
        free_mib = float(cotenancy["device_memory"].split(",")[-1]
                         .replace("MiB", "").strip())
    except (ValueError, IndexError):
        pass
    if free_mib and free_mib < 1024:
        print(f"GPU_M2D_G4_T2_WARNING: only {free_mib:.0f} MiB free on device "
              f"{args.device}; a run whose allocations land outside the 22 GiB "
              "table universe is refused fail-closed by the snapshot check")

    device_uuid = query_device_uuid(args.device)
    run_dir = args.output_root / f"run_t2_gpu{args.device}_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chown(run_dir, uid, gid)
    event_output = run_dir / "events.csv"
    summary_output = run_dir / "summary.json"
    test_log = run_dir / "harness.log"
    registry_output = run_dir / "g1_5_allocations.csv"
    registry_gate_copy = run_dir / "g1_5_allocations_gate.csv"
    map_output = run_dir / "gpu_va_pa_map.csv"
    map_gate_output = run_dir / "gpu_va_pa_map_gate.csv"
    work_output = run_dir / "work.csv"
    work_detail_output = run_dir / "work_detail.json"
    result_output = run_dir / "g1_5_g4_result.csv"
    gate = run_dir / f".gate_{os.getpid()}_{time.time_ns()}"
    release = run_dir / f".release_{os.getpid()}_{time.time_ns()}"
    run_id = run_dir.name

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(trt_runtime_dir), str(opencv_lib_dir),
                          os.environ.get("LD_LIBRARY_PATH", "")) if part)

    observer = G2Observer(contract, contract_hash, 0, event_output)
    process: subprocess.Popen[bytes] | None = None
    started_ns = time.time_ns()
    captured = bytearray()
    failures: list[str] = []
    extra: dict[str, Any] = {"cotenancy_at_start": cotenancy}
    try:
        process = subprocess.Popen(
            [
                str(args.runner.resolve()),
                "--engine", str(args.engine.resolve()),
                "--sample-csv", str(args.sample_csv.resolve()),
                "--sample-index", str(args.sample_index),
                "--device", str(args.device),
                "--output-prefix", str((run_dir / "g1_5").resolve()),
                "--observer-gate", str(gate),
                "--hold-seconds", str(args.hold_seconds),
                "--gate-timeout-seconds", str(max(5, int(args.timeout_seconds))),
                "--injection-work", str(work_output.resolve()),
                "--injection-release", str(release.resolve()),
                "--injection-gate-timeout-seconds",
                str(max(120, int(args.timeout_seconds))),
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
            process, b"GPU_M2D_EVENT,event=WAIT_PRE_ALLOC_GATE,",
            min(args.timeout_seconds, 10.0))
        captured.extend(prelude)
        if not gate_ready:
            process.terminate()
            raise RuntimeError("runner did not reach the pre-allocation gate")
        gate.write_text(f"observer_ready target_tgid={process.pid}\n",
                        encoding="utf-8")
        os.chmod(gate, 0o644)

        reached = drain_until_marker(
            process, observer, b"GPU_M2D_EVENT,event=INJECTION_GATE_WAIT,",
            captured, args.timeout_seconds)
        if not reached:
            raise RuntimeError("runner did not reach the injection gate")
        for _ in range(20):
            observer.poll(50)

        output_so_far = captured.decode("utf-8", errors="replace")
        allocated, _, _ = parse_harness_output(output_so_far)
        if not allocated:
            raise RuntimeError("no ALLOCATED events before the injection gate")
        if not registry_output.is_file():
            raise RuntimeError("runner did not write the gate-time registry")
        registry = read_csv_rows(registry_output, ALLOC_FIELDS)
        registry_gate_copy.write_text(registry_output.read_text(),
                                      encoding="utf-8")
        # every registry row <-> exactly one ALLOCATED event, values agreeing
        allocated_by_id = {r["allocation_id"]: r for r in allocated}
        if len(allocated_by_id) != len(allocated):
            failures.append("duplicate ALLOCATED records at the gate")
        for row in registry:
            record = allocated_by_id.get(row["allocation_id"])
            if record is None:
                failures.append(f"{row['allocation_id']}: no ALLOCATED event")
                continue
            if record.get("base_va", "").lower() != row["gpu_va"].lower() or \
                    record.get("size_bytes") != row["size_bytes"]:
                failures.append(f"{row['allocation_id']}: ALLOCATED event "
                                "disagrees with the gate registry")
        if failures:
            raise RuntimeError("gate registry/ALLOCATED disagreement: "
                               + "; ".join(failures))

        g1_5_run_id = registry[0]["run_id"] if registry else ""
        gate_segments, gate_failures = build_preledger(observer.rows,
                                                       allocated)
        if gate_failures or not gate_segments:
            raise RuntimeError("injection-gate ledger failed: "
                               + "; ".join(gate_failures))
        gate_map_rows = map_rows_from_segments(gate_segments, run_id,
                                               g1_5_run_id, args.device,
                                               registry)
        write_map_csv(map_gate_output, gate_map_rows)
        os.chmod(map_gate_output, 0o644)

        snapshot_dir = run_dir / "snapshot"
        result = run_build(registry_output, map_gate_output, args.device,
                           args.table, snapshot_dir,
                           lambda t="": print(t))
        if result.problems:
            raise RuntimeError("snapshot refused (fail-closed): "
                               + "; ".join(result.problems[:10]))
        print(f"snapshot: {len(result.rows)} rows, checksum "
              f"{result.manifest['snapshot_sha256'][:16]}...")

        targets, notes = select_targets(result.rows, args.binding_bit,
                                        args.internal_bit)
        for note in notes:
            print(f"select: {note}")
        write_work_csv(work_output, targets)
        work_detail_output.write_text(json.dumps({
            "schema": "gpu-m2d.g4-t2.work.v1",
            "run_id": run_id,
            "device": args.device,
            "selection_policy": "binding: bank-linked page, then the data "
                                "input binding, then lowest PA; internal: "
                                "bank-linked page, then largest allocation, "
                                "then lowest PA; byte = midpoint of the "
                                "resident range; bits are fixed constants",
            "binding_bit": args.binding_bit,
            "internal_bit": args.internal_bit,
            "notes": notes,
            "snapshot_manifest": result.manifest,
            "targets": targets,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(work_output, 0o644)
        os.chmod(work_detail_output, 0o644)

        release.write_text(f"snapshot_complete target_tgid={process.pid}\n",
                           encoding="utf-8")
        os.chmod(release, 0o644)

        finished = drain_until_marker(process, observer,
                                      PASS_MARKER.encode(), captured,
                                      args.timeout_seconds)
        if not finished:
            if process.poll() is None:
                process.terminate()
            raise RuntimeError("runner did not report " + PASS_MARKER)
        exit_deadline = time.monotonic() + args.timeout_seconds
        while process.poll() is None and time.monotonic() < exit_deadline:
            observer.poll(50)
        if process.poll() is None:
            process.terminate()
            raise RuntimeError("runner exceeded timeout")
        for _ in range(20):
            observer.poll(50)
        drain_pipe(process, captured)
        output = captured.decode("utf-8", errors="replace")

        # ---------------- post-run verification ----------------
        if process.returncode != 0:
            failures.append(f"runner exit code {process.returncode}")
        allocated, lifecycle, times = parse_harness_output(output)
        skeleton = [name for name in lifecycle if name not in T2_VARIABLE_EVENTS]
        expected_skeleton = [name for name in T2_SKELETON
                             if not (name in ("HOLD_BEGIN", "HOLD_END")
                                     and args.hold_seconds <= 0)]
        if skeleton != expected_skeleton:
            failures.append(f"lifecycle skeleton mismatch: {skeleton}")

        final_segments, final_failures = build_final_ledger(
            observer.rows, allocated, times.get("PROCESS_END", 0))
        failures.extend(final_failures)
        final_map_rows = map_rows_from_segments(final_segments, run_id,
                                                g1_5_run_id, args.device,
                                                registry)
        write_map_csv(map_output, final_map_rows)
        failures.extend(diff_map_rows(gate_map_rows, final_map_rows,
                                      "va-pa map"))

        # mappings alive across the whole injection window
        work_begin_ns = times.get("INJECTION_WORK_BEGIN", 0)
        gate_wait_ns = times.get("INJECTION_GATE_WAIT", 0)
        work_end_ns = times.get("INJECTION_WORK_END", 0)
        for segment in final_segments:
            name = str(segment["allocation_id"])
            if not (gate_wait_ns and work_end_ns
                    and int(segment["mapped_at_ns"]) < gate_wait_ns
                    and int(segment["unmapped_at_ns"] or 0) > work_end_ns):
                failures.append(f"{name}: mapping not alive across the "
                                "injection window")

        # registry stability between gate and end (no realloc mid-run)
        if registry_output.read_text() != registry_gate_copy.read_text():
            failures.append("allocation registry changed between the "
                            "injection gate and the end of the run")

        result_rows = read_csv_rows(result_output, RESULT_FIELDS)
        failures.extend(verify_result_rows(targets, result_rows))
        failures.extend(cross_check_target_events(output, result_rows))

        kernel_uuids = sorted({str(row["gpu_uuid"]) for row in observer.rows
                               if row["event_type"] == "PTE_HEADER"
                               and row["gpu_uuid"]})
        if len(kernel_uuids) > 1:
            failures.append(f"multiple kernel GPU UUIDs: {kernel_uuids}")
        elif kernel_uuids and normalize_gpu_uuid(kernel_uuids[0]) != \
                normalize_gpu_uuid(device_uuid):
            failures.append(f"kernel GPU UUID {kernel_uuids[0]} != device "
                            f"UUID {device_uuid}")
        if len({str(row["address_space_id"]) for row in observer.rows
                if row["event_type"] in ("MAP_RETURN", "PTE_HEADER")}) > 1:
            failures.append("multiple UVM address-space IDs observed")
        if observer.lost_event_count:
            failures.append(f"lost BPF events: {observer.lost_event_count}")

        chain_evidence = []
        for target, row in zip(targets, result_rows):
            chain_evidence.append({
                "target_id": target["target_id"],
                "kind": "binding" if target["target_id"].startswith("t1")
                        else "trt_internal",
                "allocation_id": target["allocation_id"],
                "semantic_label": target["semantic_label"],
                "byte_offset": target["byte_offset"],
                "bit": target["bit"],
                "xor_mask": int(row["xor_mask"]),
                "before": int(row["before"]),
                "after": int(row["after"]),
                "xor_verified": int(row["after"])
                    == int(row["before"]) ^ int(row["xor_mask"]),
                "guard_bytes_unchanged": row["guard_bytes_unchanged"] == "1",
                "reverse_map_ok": row["reverse_map_ok"] == "1",
                "restored_byte_ok": row["restored_byte_ok"] == "1",
                "restore_guard_ok": row["restore_guard_ok"] == "1",
                "sanity_matches_clean": row["sanity_matches_clean"] == "1",
                "injected_outcome": row["injected_outcome"],
                "chain": target["chain"],
            })
        extra.update({
            "g1_5_run_id": g1_5_run_id,
            "registry_allocation_count": len(registry),
            "map_row_count": len(final_map_rows),
            "selection_notes": notes,
            "work_output": str(work_output),
            "work_sha256": sha256(work_output),
            "result_output": str(result_output),
            "result_sha256": sha256(result_output),
            "map_output": str(map_output),
            "map_output_sha256": sha256(map_output) if final_map_rows else "",
            "map_gate_output": str(map_gate_output),
            "map_gate_output_sha256": sha256(map_gate_output),
            "snapshot_dir": str(snapshot_dir),
            "snapshot_sha256": result.manifest.get("snapshot_sha256", ""),
            "snapshot_manifest": result.manifest,
            "chain_evidence": chain_evidence,
            "injection_window_ns": {
                "injection_gate_wait": gate_wait_ns,
                "work_begin": work_begin_ns,
                "work_end": work_end_ns,
            },
            "contract_sha256": contract_hash,
            "nvidia_module_sha256": contract["nvidia_module_sha256"],
            "nvidia_uvm_module_sha256": contract["nvidia_uvm_module_sha256"],
        })
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

    return finalize(failures, run_dir, summary_output, test_log,
                    event_output, observer, args, run_id, device_uuid,
                    started_ns, extra, uid, gid)


# --------------------------------------------------------------------------
# self-test (offline: no root, no GPU, no observer)
# --------------------------------------------------------------------------

def self_test() -> int:
    import tempfile

    page = 2 << 20
    quiet = lambda t="": None

    def snap_row(alloc, label, va_page, start, end, pa, linked=1,
                 size=None, base=None):
        base = base if base is not None else va_page
        return {
            "allocation_id": alloc, "semantic_label": label,
            "allocation_va_base": f"{base:#x}",
            "allocation_size_bytes": size if size is not None else (end - start),
            "va_page_base": f"{va_page:#x}",
            "byte_start_in_page": f"{start:#x}",
            "byte_end_in_page": f"{end:#x}",
            "fb_pa_page_base": f"{pa:#x}", "page_size": page,
            "in_universe": 1, "bank_linked": linked,
            "channel_root": f"{pa:#x}" if linked else "",
            "channel_size": 2 if linked else 0, "row_class_count": 0,
            "same_row_site_nodes": 0, "valid_anchor_masks": "0x1f9dc0",
        }

    # --- selection: linked data page wins; byte math crosses pages
    v_base = 0x700000000000
    rows = [
        snap_row("bind-data", "TENSOR:data", v_base + 5 * page, 0,
                 602112, 0x20000000, linked=1, size=602112 + 3 * page,
                 base=v_base),
        snap_row("bind-prob", "TENSOR:prob", v_base + 9 * page, 0, 4,
                 0x21000000, linked=1, size=4, base=v_base + 9 * page),
        snap_row("int-0", "TENSORRT_INTERNAL_UNKNOWN", v_base + 20 * page,
                 0, page, 0x23000000, linked=1, size=12 * page,
                 base=v_base + 20 * page),
        snap_row("int-1", "TENSORRT_INTERNAL_UNKNOWN", v_base + 40 * page,
                 0, 2048, 0x24000000, linked=1, size=2048,
                 base=v_base + 40 * page),
    ]
    targets, notes = select_targets(rows, binding_bit=5, internal_bit=2)
    t1, t2 = targets
    # t1: data binding, midpoint of [0, 602112) on its page 5
    assert t1["allocation_id"] == "bind-data", t1
    assert t1["byte_offset"] == 5 * page + 301056, hex(t1["byte_offset"])
    assert t1["expected_gpu_va"] == v_base + 5 * page + 301056
    assert t1["chain"]["fb_pa_page_base"] == "0x20000000"
    # t2: largest internal allocation (12 pages) wins over the 2 KiB one
    assert t2["allocation_id"] == "int-0" and t2["bit"] == 2
    assert t2["byte_offset"] == page // 2
    # linked beats the data-label preference
    rows_unlinked = [dict(r, bank_linked=0, channel_root="")
                     if r["allocation_id"] == "bind-data" else r
                     for r in rows]
    t1b = select_targets(rows_unlinked, 5, 2)[0][0]
    assert t1b["allocation_id"] == "bind-prob", t1b
    # unlinked internal falls back with a note
    rows_no_linked_int = [dict(r, bank_linked=0, channel_root="")
                          if r["semantic_label"] == "TENSORRT_INTERNAL_UNKNOWN"
                          else r for r in rows]
    t2b, notes2 = select_targets(rows_no_linked_int, 5, 2)
    t2b = t2b[1]
    assert t2b["chain"]["bank_linked"] == 0
    assert any("UNLINKED" in n for n in notes2), notes2
    # empty category refuses
    try:
        select_targets([r for r in rows
                        if r["semantic_label"] != "TENSORRT_INTERNAL_UNKNOWN"],
                       5, 2)
        raise AssertionError("internal-less snapshot must refuse")
    except SelectionError:
        pass

    # --- work file round-trip
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work.csv"
        write_work_csv(work, targets)
        lines = [line for line in work.read_text().splitlines()
                 if line and not line.startswith("#")]
        assert lines[0] == ",".join(WORK_FIELDS), lines
        fields = lines[1].split(",")
        assert fields[0] == "t1-binding" and fields[1] == "bind-data"
        assert fields[2] == str(5 * page + 301056) and fields[3] == "5"
        assert fields[4] == f"{v_base + 5 * page + 301056:#x}"

        # --- result verification
        def result_row(target, before=0x55, after=None, **over):
            mask = 1 << target["bit"]
            row = {
                "target_id": target["target_id"],
                "allocation_id": target["allocation_id"],
                "semantic_label": target["semantic_label"],
                "byte_offset": str(target["byte_offset"]),
                "bit_in_byte": str(target["bit"]),
                "xor_mask": str(mask),
                "gpu_va": f"{target['expected_gpu_va']:#x}",
                "expected_gpu_va": f"{target['expected_gpu_va']:#x}",
                "before": str(before),
                "after": str(before ^ mask if after is None else after),
                "guard_bytes_unchanged": "1", "reverse_map_ok": "1",
                "restored_byte_ok": "1", "restore_guard_ok": "1",
                "sanity_matches_clean": "1", "injected_outcome": "BENIGN",
            }
            row.update({k: str(v) for k, v in over.items()})
            return row

        assert verify_result_rows(targets, [result_row(t) for t in targets]) \
            == []
        bad = result_row(t1, after=0x00)
        assert verify_result_rows(targets, [bad, result_row(t2)])
        bad = result_row(t1, guard_bytes_unchanged=0)
        assert any("guard" in f for f in
                   verify_result_rows(targets, [bad, result_row(t2)]))
        bad = result_row(t2, sanity_matches_clean=0)
        assert any("sanity" in f for f in
                   verify_result_rows(targets, [result_row(t1), bad]))

        # --- map diff: identical ok; PA change and key-set change caught
        gate_rows = [{"allocation_id": "a", "va_page_base": "0x1000",
                      "fb_pa_page_base": "0x2000000", "pte_valid": "true",
                      "aperture": "VIDEO",
                      "physical_coverage_status": "LOCAL_VIDEO_COMPLETE",
                      "page_size": page}]
        assert diff_map_rows(gate_rows, [dict(gate_rows[0])], "m") == []
        moved = [dict(gate_rows[0], fb_pa_page_base="0x3000000")]
        assert any("mid-run remap" in f
                   for f in diff_map_rows(gate_rows, moved, "m"))
        assert diff_map_rows(gate_rows, [], "m")

        # --- event cross-check
        output = ("GPU_M2D_EVENT,event=TARGET_FLIPPED,target_id=t1-binding,"
                  f"allocation_id=bind-data,gpu_va={v_base + 5 * page + 301056:#x},"
                  "before=85,after=117,xor_mask=32,pid=1,tgid=1\n"
                  "GPU_M2D_EVENT,event=TARGET_FLIPPED,target_id=t2-internal,"
                  f"allocation_id=int-0,gpu_va={v_base + 20 * page + page // 2:#x},"
                  "before=85,after=81,xor_mask=4,pid=1,tgid=1\n"
                  "GPU_M2D_EVENT,event=INJECTED_OUTCOME,outcome=BENIGN,pid=1\n")
        good = [result_row(t1, before=85, after=117), result_row(t2)]
        assert cross_check_target_events(output, good) == []
        flipped_wrong = ("GPU_M2D_EVENT,event=TARGET_FLIPPED,"
                         "target_id=t1-binding,allocation_id=bind-data,"
                         f"gpu_va={v_base + 5 * page + 301056:#x},"
                         "before=85,after=1,xor_mask=32,pid=1,tgid=1\n"
                         "GPU_M2D_EVENT,event=INJECTED_OUTCOME,outcome=BENIGN\n")
        assert cross_check_target_events(flipped_wrong, good)

        # --- run_build plumbing on a tiny fixture (happy + fail-closed)
        tdir = Path(tmp) / "table"
        tdir.mkdir()
        (tdir / "gddr_seed_table.csv").write_text(
            "page_index,page_base,lambda,channel_root,channel_size,super_tail,"
            "n_shoulder,n_deep,n_low,n_nodes,n_bank_classes,classified\n"
            "0,0x2000000,1000,0x2000000,1,0,0,0,10,1,1,1\n")
        (tdir / "bank_classes.csv").write_text(
            "page_index,page_base,offset,pa,bank_class,bank_size,row_class,"
            "row_size,n_deep,n_low,n_shoulder\n")
        (tdir / "page_anchors.csv").write_text(
            "page_base,candidate,seen,votes,valid\n"
            "0x2000000,0x1f9dc0,3,3,true\n")
        areg = Path(tmp) / "allocations.csv"
        areg.write_text(",".join(ALLOC_FIELDS) + "\n"
                        f"r-x,a,0,{v_base:#x},4096,512,o,ph,lt,1,TENSOR:data\n")
        amap = Path(tmp) / "map.csv"
        amap.write_text(
            "run_id,g1_5_run_id,device,gpu_uuid,allocation_id,allocation_api,"
            "va_page_base,va_page_end_exclusive,fb_pa_page_base,page_size,"
            "aperture,pte_valid,covered_allocation_bytes,"
            "physical_coverage_status\n"
            f"m,r-x,0,u,a,api,{v_base:#x},{v_base + page:#x},0x2000000,"
            f"{page},VIDEO,true,4096,LOCAL_VIDEO_COMPLETE\n")
        out = Path(tmp) / "snap"
        built = run_build(areg, amap, 0, tdir, out, quiet)
        assert not built.problems, built.problems
        assert (out / "snapshot_pages.csv").is_file() and len(built.rows) == 1
        assert built.manifest["snapshot_sha256"] == sha256(
            out / "snapshot_pages.csv")
        bad_map = Path(tmp) / "map_bad.csv"
        bad_map.write_text(amap.read_text().replace("0x2000000", "0x9000000"))
        out2 = Path(tmp) / "snap2"
        refused = run_build(areg, bad_map, 0, tdir, out2, quiet)
        assert refused.problems and any("OUTSIDE" in p for p in refused.problems)
        assert not out2.exists()

    print("g4 t2 injection self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if os.geteuid() != 0:
        raise PermissionError(
            "run through sudo; the CUDA child is dropped to the invoking user")
    if not 0 <= args.binding_bit <= 7 or not 0 <= args.internal_bit <= 7:
        raise RuntimeError("--binding-bit/--internal-bit must be in [0, 7]")
    return run_once(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(f"GPU_M2D_G4_T2_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
