#!/usr/bin/python3
"""GPU_M2D G2 read-only full-PTE observer.

Attaches read-only kprobe/kretprobe pairs to the already-loaded, unmodified
NVIDIA kernel modules and records, for one target TGID only:

  * ``uvm_api_map_external_allocation`` (nvidia_uvm)  -> VA range + GPU UUID
  * ``nvUvmInterfaceGetExternalAllocPtes`` (nvidia)   -> the complete PTE
    payload RM hands to UVM for that mapping
  * ``uvm_api_free`` (nvidia_uvm)                     -> mapping teardown

The captured PTEs are exactly the entries UVM writes into the GMMU page
tables, so a decoded (valid, aperture=VIDEO) PTE yields the local
framebuffer physical page for the corresponding GPU VA page. No driver
state, PTE, or UVM object is modified, and no other GPU process is
touched.

Ported from the REMU gpu_va_pa_mapping observer (g2p_full_pte_tracer.py
and the contract loader of candidate_c_payload_tracer.py, frozen
2026-09-08). The BPF program, event ABI, and contract checks are kept
functionally identical.

Requires root for kprobe attachment; use the orchestrator which drops
the CUDA child back to the invoking user.
"""

from __future__ import annotations

import csv
import ctypes as ct
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from bcc import BPF

from pte_decoder import decode_pte, format_uuid


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_CONTRACT = HERE / "pte_contract.json"
DEFAULT_OPEN_SOURCE_REPO = Path(
    "/data1/luojx/REMU/GPU_GDDR_addressing_reference/nvidia-open-gpu-kernel-modules-580.95.05"
)

MAX_PTES_PER_QUERY = 64
EVENT_MAP_RETURN = 1
EVENT_PTE_HEADER = 2
EVENT_PTE_ENTRY = 3
EVENT_FREE_RETURN = 4
EVENT_NAMES = {
    EVENT_MAP_RETURN: "MAP_RETURN",
    EVENT_PTE_HEADER: "PTE_HEADER",
    EVENT_PTE_ENTRY: "PTE_ENTRY",
    EVENT_FREE_RETURN: "FREE_RETURN",
}


class G2Event(ct.Structure):
    _fields_ = [
        ("timestamp_ns", ct.c_uint64),
        ("file_cookie", ct.c_uint64),
        ("rm_va_space", ct.c_uint64),
        ("map_base", ct.c_uint64),
        ("map_length", ct.c_uint64),
        ("map_offset", ct.c_uint64),
        ("query_offset", ct.c_uint64),
        ("query_size", ct.c_uint64),
        ("mapping_page_size", ct.c_uint64),
        ("num_written", ct.c_uint64),
        ("num_remaining", ct.c_uint64),
        ("h_memory", ct.c_uint64),
        ("pte_index", ct.c_uint64),
        ("raw_pte_lo", ct.c_uint64),
        ("raw_pte_hi", ct.c_uint64),
        ("pid", ct.c_uint32),
        ("tgid", ct.c_uint32),
        ("event_type", ct.c_uint32),
        ("status", ct.c_int32),
        ("pte_size", ct.c_uint32),
        ("mapping_type", ct.c_uint32),
        ("caching_type", ct.c_uint32),
        ("format_type", ct.c_uint32),
        ("element_bits", ct.c_uint32),
        ("compression_type", ct.c_uint32),
        ("gpu_uuid", ct.c_uint8 * 16),
        ("need_l2_invalidate", ct.c_uint8),
        ("payload_complete", ct.c_uint8),
        ("reserved", ct.c_uint16),
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")
    return completed.stdout.strip()


def validate_event_abi() -> None:
    expected = {
        "timestamp_ns": 0, "map_base": 24, "pte_index": 96,
        "raw_pte_lo": 104, "raw_pte_hi": 112, "pid": 120,
        "event_type": 128, "pte_size": 136, "gpu_uuid": 160,
        "payload_complete": 177,
    }
    failures = [
        f"{name}: expected {offset}, got {getattr(G2Event, name).offset}"
        for name, offset in expected.items()
        if getattr(G2Event, name).offset != offset
    ]
    if ct.sizeof(G2Event) != 184:
        failures.append(f"event size: expected 184, got {ct.sizeof(G2Event)}")
    if failures:
        raise RuntimeError("G2 observer event ABI mismatch: " + "; ".join(failures))


def load_and_validate_contract(
    path: Path, open_source_repo: Path = DEFAULT_OPEN_SOURCE_REPO
) -> tuple[dict[str, Any], str]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if contract.get("status") != "OFFLINE_SOURCE_SYMBOL_LAYOUT_AND_PTE_FORMAT_CONTRACT":
        failures.append("unexpected contract status")
    if contract.get("architecture") != platform.machine():
        failures.append("architecture mismatch")
    if contract.get("kernel_release") != platform.release():
        failures.append("kernel release mismatch")
    if command_output(["modinfo", "-F", "version", "nvidia"]) != contract.get("driver_version"):
        failures.append("driver version mismatch")

    for prefix in ("nvidia", "nvidia_uvm"):
        module = Path(contract[f"{prefix}_module_path"])
        if not module.is_file() or sha256(module) != contract.get(f"{prefix}_module_sha256"):
            failures.append(f"{prefix} module hash mismatch")
    for source_path, expected_hash in contract.get("source_hashes", {}).items():
        source = Path(source_path)
        if not source.is_file() or sha256(source) != expected_hash:
            failures.append(f"source hash mismatch: {source_path}")

    symbol_modules: dict[str, set[str]] = {}
    for line in Path("/proc/kallsyms").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            module = fields[3].strip("[]") if len(fields) >= 4 else "vmlinux"
            symbol_modules.setdefault(fields[2], set()).add(module)
    for probe in contract.get("probe_symbols", []):
        if probe["module"] not in symbol_modules.get(probe["name"], set()):
            failures.append(f"probe symbol/module mismatch: {probe['name']} expected {probe['module']}")

    if not open_source_repo.is_dir():
        failures.append(f"open-kernel source repo missing: {open_source_repo}")
    elif command_output(["git", "-C", str(open_source_repo), "rev-parse", "HEAD"]) != contract.get("open_kernel_source_commit"):
        failures.append("open-kernel source commit mismatch")
    if failures:
        raise RuntimeError("G2 observer contract failed:\n- " + "\n- ".join(failures))
    return contract, sha256(path)


def build_bpf_source() -> str:
    return r"""
#include <uapi/linux/ptrace.h>

#define EVENT_MAP_RETURN 1
#define EVENT_PTE_HEADER 2
#define EVENT_PTE_ENTRY 3
#define EVENT_FREE_RETURN 4
#define MAX_PTES_PER_QUERY 64

struct map_call_t {
    u64 file_cookie;
    u64 base;
    u64 length;
    u64 offset;
    u64 uuid_lo;
    u64 uuid_hi;
    u32 mapping_type;
    u32 caching_type;
    u32 format_type;
    u32 element_bits;
    u32 compression_type;
};

struct pte_call_t {
    u64 info_ptr;
    u64 rm_va_space;
    u64 h_memory;
    u64 query_offset;
    u64 query_size;
    struct map_call_t map;
};

struct free_call_t { u64 file_cookie; u64 base; u64 length; };

struct external_mapping_info_t {
    u32 caching_type;
    u32 mapping_type;
    u32 format_type;
    u32 element_bits;
    u32 compression_type;
    u32 padding;
    u64 pte_buffer_size;
    u64 mapping_page_size;
    u64 pte_buffer;
    u64 num_written;
    u64 num_remaining;
    u32 pte_size;
    u8 need_l2_invalidate;
    u8 tail_padding[3];
};

struct g2_event_t {
    u64 timestamp_ns;
    u64 file_cookie;
    u64 rm_va_space;
    u64 map_base;
    u64 map_length;
    u64 map_offset;
    u64 query_offset;
    u64 query_size;
    u64 mapping_page_size;
    u64 num_written;
    u64 num_remaining;
    u64 h_memory;
    u64 pte_index;
    u64 raw_pte_lo;
    u64 raw_pte_hi;
    u32 pid;
    u32 tgid;
    u32 event_type;
    s32 status;
    u32 pte_size;
    u32 mapping_type;
    u32 caching_type;
    u32 format_type;
    u32 element_bits;
    u32 compression_type;
    u8 gpu_uuid[16];
    u8 need_l2_invalidate;
    u8 payload_complete;
    u16 reserved;
};

BPF_HASH(target_tgid_map, u32, u32, 1);
BPF_HASH(map_calls, u64, struct map_call_t, 4096);
BPF_HASH(pte_calls, u64, struct pte_call_t, 4096);
BPF_HASH(free_calls, u64, struct free_call_t, 4096);
BPF_PERF_OUTPUT(g2_events);

static __always_inline int target_matches(u64 pid_tgid)
{
    u32 key = 0;
    u32 tgid = pid_tgid >> 32;
    const u32 *target = target_tgid_map.lookup(&key);
    return target && *target == tgid;
}

static __always_inline void fill_common(struct g2_event_t *event, u64 key, u32 type)
{
    event->timestamp_ns = bpf_ktime_get_ns();
    event->pid = (u32)key;
    event->tgid = key >> 32;
    event->event_type = type;
}

static __always_inline void copy_map(struct g2_event_t *event, const struct map_call_t *map)
{
    event->file_cookie = map->file_cookie;
    event->map_base = map->base;
    event->map_length = map->length;
    event->map_offset = map->offset;
    __builtin_memcpy(event->gpu_uuid, &map->uuid_lo, 8);
    __builtin_memcpy(event->gpu_uuid + 8, &map->uuid_hi, 8);
    event->mapping_type = map->mapping_type;
    event->caching_type = map->caching_type;
    event->format_type = map->format_type;
    event->element_bits = map->element_bits;
    event->compression_type = map->compression_type;
}

static __always_inline void copy_pte(struct g2_event_t *event,
                                     const struct pte_call_t *call,
                                     const struct external_mapping_info_t *info)
{
    copy_map(event, &call->map);
    event->rm_va_space = call->rm_va_space;
    event->h_memory = call->h_memory;
    event->query_offset = call->query_offset;
    event->query_size = call->query_size;
    event->mapping_page_size = info->mapping_page_size;
    event->num_written = info->num_written;
    event->num_remaining = info->num_remaining;
    event->pte_size = info->pte_size;
    event->need_l2_invalidate = info->need_l2_invalidate;
}

int trace_map_entry(struct pt_regs *ctx)
{
    u64 key = bpf_get_current_pid_tgid();
    u64 params = PT_REGS_PARM1(ctx);
    struct map_call_t call = {};
    if (!target_matches(key)) return 0;
    call.file_cookie = PT_REGS_PARM2(ctx);
    bpf_probe_read_kernel(&call.base, 8, (void *)(params + 0));
    bpf_probe_read_kernel(&call.length, 8, (void *)(params + 8));
    bpf_probe_read_kernel(&call.offset, 8, (void *)(params + 16));
    bpf_probe_read_kernel(&call.uuid_lo, 8, (void *)(params + 24));
    bpf_probe_read_kernel(&call.uuid_hi, 8, (void *)(params + 32));
    bpf_probe_read_kernel(&call.mapping_type, 4, (void *)(params + 40));
    bpf_probe_read_kernel(&call.caching_type, 4, (void *)(params + 44));
    bpf_probe_read_kernel(&call.format_type, 4, (void *)(params + 48));
    bpf_probe_read_kernel(&call.element_bits, 4, (void *)(params + 52));
    bpf_probe_read_kernel(&call.compression_type, 4, (void *)(params + 56));
    map_calls.update(&key, &call);
    return 0;
}

int trace_map_return(struct pt_regs *ctx)
{
    u64 key = bpf_get_current_pid_tgid();
    const struct map_call_t *call;
    struct g2_event_t event = {};
    if (!target_matches(key)) return 0;
    call = map_calls.lookup(&key);
    if (!call) return 0;
    fill_common(&event, key, EVENT_MAP_RETURN);
    copy_map(&event, call);
    event.status = (s32)PT_REGS_RC(ctx);
    g2_events.perf_submit(ctx, &event, sizeof(event));
    map_calls.delete(&key);
    return 0;
}

int trace_pte_entry(struct pt_regs *ctx)
{
    u64 key = bpf_get_current_pid_tgid();
    struct pte_call_t call = {};
    const struct map_call_t *map;
    if (!target_matches(key)) return 0;
    call.rm_va_space = PT_REGS_PARM1(ctx);
    call.h_memory = PT_REGS_PARM2(ctx);
    call.query_offset = PT_REGS_PARM3(ctx);
    call.query_size = PT_REGS_PARM4(ctx);
    call.info_ptr = PT_REGS_PARM5(ctx);
    map = map_calls.lookup(&key);
    if (map) __builtin_memcpy(&call.map, map, sizeof(call.map));
    pte_calls.update(&key, &call);
    return 0;
}

int trace_pte_return(struct pt_regs *ctx)
{
    u64 key = bpf_get_current_pid_tgid();
    const struct pte_call_t *call;
    struct external_mapping_info_t info = {};
    struct g2_event_t event = {};
    u64 address;
    if (!target_matches(key)) return 0;
    call = pte_calls.lookup(&key);
    if (!call) return 0;
    bpf_probe_read_kernel(&info, sizeof(info), (void *)call->info_ptr);
    fill_common(&event, key, EVENT_PTE_HEADER);
    copy_pte(&event, call, &info);
    event.status = (s32)PT_REGS_RC(ctx);
    event.payload_complete = event.status == 0 && info.num_written <= MAX_PTES_PER_QUERY &&
                             (info.pte_size == 8 || info.pte_size == 16);
    g2_events.perf_submit(ctx, &event, sizeof(event));
    if (event.payload_complete && info.pte_buffer) {
#pragma unroll
        for (int index = 0; index < MAX_PTES_PER_QUERY; ++index) {
            if ((u64)index >= info.num_written)
                break;
            event.event_type = EVENT_PTE_ENTRY;
            event.pte_index = index;
            event.raw_pte_lo = 0;
            event.raw_pte_hi = 0;
            address = info.pte_buffer + (u64)index * info.pte_size;
            if (bpf_probe_read_kernel(&event.raw_pte_lo, 8, (void *)address) != 0)
                break;
            if (info.pte_size == 16 &&
                bpf_probe_read_kernel(&event.raw_pte_hi, 8, (void *)(address + 8)) != 0)
                break;
            g2_events.perf_submit(ctx, &event, sizeof(event));
        }
    }
    pte_calls.delete(&key);
    return 0;
}

int trace_free_entry(struct pt_regs *ctx)
{
    u64 key = bpf_get_current_pid_tgid();
    u64 params = PT_REGS_PARM1(ctx);
    struct free_call_t call = {};
    if (!target_matches(key)) return 0;
    call.file_cookie = PT_REGS_PARM2(ctx);
    bpf_probe_read_kernel(&call.base, 8, (void *)(params + 0));
    bpf_probe_read_kernel(&call.length, 8, (void *)(params + 8));
    free_calls.update(&key, &call);
    return 0;
}

int trace_free_return(struct pt_regs *ctx)
{
    u64 key = bpf_get_current_pid_tgid();
    const struct free_call_t *call;
    struct g2_event_t event = {};
    if (!target_matches(key)) return 0;
    call = free_calls.lookup(&key);
    if (!call) return 0;
    fill_common(&event, key, EVENT_FREE_RETURN);
    event.file_cookie = call->file_cookie;
    event.map_base = call->base;
    event.map_length = call->length;
    event.status = (s32)PT_REGS_RC(ctx);
    g2_events.perf_submit(ctx, &event, sizeof(event));
    free_calls.delete(&key);
    return 0;
}
"""


CSV_FIELDS = [
    "timestamp_ns", "pid", "tgid", "event_type", "status",
    "address_space_id", "rm_va_space_id", "map_base", "map_length",
    "map_offset", "gpu_uuid", "query_offset", "query_size",
    "mapping_page_size", "num_written", "num_remaining", "pte_size",
    "pte_index", "payload_complete", "raw_pte_lo", "raw_pte_hi",
    "raw_entry", "high_word_zero", "valid", "aperture",
    "physical_page_base", "gpu_local_pa", "need_l2_invalidate",
    "contract_sha256",
]


class G2Observer:
    def __init__(self, contract: dict[str, Any], contract_sha256: str,
                 target_tgid: int, output_path: Path):
        if os.geteuid() != 0:
            raise PermissionError("runtime G2 observer kprobe attachment requires root")
        self.contract = contract
        self.contract_sha256 = contract_sha256
        self.target_tgid = target_tgid
        self.output_path = output_path
        self.rows: list[dict[str, object]] = []
        self.lost_event_count = 0
        self.cookie_salt = hashlib.sha256(f"g2:{contract_sha256}:{target_tgid}".encode()).digest()
        self.bpf = BPF(text=build_bpf_source())
        self.attached: list[tuple[str, str]] = []
        try:
            for probe in contract["probe_symbols"]:
                self.bpf.attach_kprobe(event=probe["name"], fn_name=probe["entry_fn"])
                self.attached.append(("kprobe", probe["name"]))
                self.bpf.attach_kretprobe(event=probe["name"], fn_name=probe["return_fn"])
                self.attached.append(("kretprobe", probe["name"]))
        except Exception:
            self.close()
            raise
        self.set_target_tgid(target_tgid)
        self.bpf["g2_events"].open_perf_buffer(self._on_event, lost_cb=self._on_lost, page_cnt=256)

    def _cookie(self, kind: str, value: int) -> str:
        if not value:
            return ""
        digest = hashlib.sha256(self.cookie_salt + kind.encode() + value.to_bytes(8, "little")).hexdigest()
        return f"{kind}-{digest[:16]}"

    def set_target_tgid(self, target_tgid: int) -> None:
        self.bpf["target_tgid_map"][ct.c_uint(0)] = ct.c_uint(target_tgid)
        self.target_tgid = target_tgid
        self.cookie_salt = hashlib.sha256(f"g2:{self.contract_sha256}:{target_tgid}".encode()).digest()

    def _on_event(self, cpu: int, data: int, size: int) -> None:
        del cpu, size
        event = ct.cast(data, ct.POINTER(G2Event)).contents
        row: dict[str, object] = {
            "timestamp_ns": event.timestamp_ns,
            "pid": event.pid,
            "tgid": event.tgid,
            "event_type": EVENT_NAMES.get(event.event_type, f"UNKNOWN_{event.event_type}"),
            "status": event.status,
            "address_space_id": self._cookie("uvmfile", event.file_cookie),
            "rm_va_space_id": self._cookie("rmvas", event.rm_va_space),
            "map_base": f"0x{event.map_base:x}" if event.map_base else "",
            "map_length": event.map_length,
            "map_offset": event.map_offset,
            "gpu_uuid": format_uuid(bytes(event.gpu_uuid)) if any(event.gpu_uuid) else "",
            "query_offset": event.query_offset,
            "query_size": event.query_size,
            "mapping_page_size": event.mapping_page_size,
            "num_written": event.num_written,
            "num_remaining": event.num_remaining,
            "pte_size": event.pte_size,
            "pte_index": event.pte_index if event.event_type == EVENT_PTE_ENTRY else "",
            "payload_complete": str(bool(event.payload_complete)).lower(),
            "need_l2_invalidate": event.need_l2_invalidate,
            "contract_sha256": self.contract_sha256,
        }
        if event.event_type == EVENT_PTE_ENTRY:
            decoded = decode_pte(event.raw_pte_lo, event.pte_size, self.contract, event.raw_pte_hi)
            row.update({
                "raw_pte_lo": decoded["raw_pte_lo"],
                "raw_pte_hi": decoded["raw_pte_hi"],
                "raw_entry": decoded["raw_entry"],
                "high_word_zero": str(decoded["high_word_zero"]).lower(),
                "valid": str(decoded["valid"]).lower(),
                "aperture": decoded["aperture"],
                "physical_page_base": decoded["physical_base"] or "",
                "gpu_local_pa": str(decoded["gpu_local_pa"]).lower(),
            })
        else:
            row.update({name: "" for name in (
                "raw_pte_lo", "raw_pte_hi", "raw_entry", "high_word_zero",
                "valid", "aperture", "physical_page_base", "gpu_local_pa",
            )})
        self.rows.append(row)

    def _on_lost(self, cpu: int, count: int) -> None:
        del cpu
        self.lost_event_count += count

    def poll(self, timeout_ms: int = 50) -> None:
        self.bpf.perf_buffer_poll(timeout=timeout_ms)

    def close(self) -> None:
        if hasattr(self, "bpf"):
            for probe_type, symbol in reversed(getattr(self, "attached", [])):
                try:
                    if probe_type == "kretprobe":
                        self.bpf.detach_kretprobe(event=symbol)
                    else:
                        self.bpf.detach_kprobe(event=symbol)
                except Exception:
                    pass
            self.bpf.cleanup()
        if hasattr(self, "output_path"):
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(self.rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-bpf", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    validate_event_abi()
    contract, contract_hash = load_and_validate_contract(DEFAULT_CONTRACT)
    if args.emit_bpf:
        args.emit_bpf.write_text(build_bpf_source(), encoding="utf-8")
    if args.self_check:
        print(f"G2_OBSERVER_SELF_CHECK_PASS contract_sha256={contract_hash}")
