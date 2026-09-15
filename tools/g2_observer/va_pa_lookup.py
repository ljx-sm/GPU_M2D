#!/usr/bin/python3
"""GPU VA <-> local framebuffer PA lookup over the G2 map.

Consumes the canonical page-level map produced by the TensorRT probe
(``artifacts/g2/gpu_va_pa_map.csv``) and answers the two questions the
injection stages need, always fail-closed:

- forward:  (GPU, gpu_va)              -> framebuffer PA page + in-page offset
- reverse:  (GPU, fb_pa_page_base)     -> every VA page and allocation on it

A PA is only ever returned for rows whose captured PTE is valid and whose
decoded aperture is VIDEO, on the exact GPU (by index or UUID) named in the
query. Unmapped VAs, foreign-GPU addresses, stale addresses from earlier
runs, inactive allocations, and non-local pages are rejected, never
guaranteed.

``--self-test`` exercises the accept and rejection cases against the real
map (plus synthetic corruption cases) and is wired into ``make check``.

Exit codes: 0 lookup answered, 2 lookup rejected, 1 usage or map error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_MAP = PROJECT / "artifacts/g2/gpu_va_pa_map.csv"

REQUIRED_COLUMNS = (
    "device", "gpu_uuid", "allocation_id", "active_at_snapshot",
    "va_page_base", "va_page_end_exclusive", "fb_pa_page_base", "page_size",
    "aperture", "pte_valid", "allocation_api", "cuda_buffer_id",
    "allocation_size_bytes", "semantic_label", "run_id",
)


def parse_int(text: str, what: str) -> int:
    try:
        return int(text, 0)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is not an integer: {text!r}")


class VaPaMap:
    """Loaded map with load-time integrity checks; invalid rows are
    excluded from every lookup and reported, never silently used."""

    def __init__(self, rows: list[dict[str, str]], source: str):
        self.source = source
        self.invalid_rows: list[str] = []
        self.rows: list[dict[str, str]] = []
        for row in rows:
            problem = self._row_problem(row)
            if problem:
                self.invalid_rows.append(problem)
            else:
                self.rows.append(row)
        conflicts: dict[tuple[str, str], set[str]] = {}
        for row in self.rows:
            key = (row["device"], row["va_page_base"])
            conflicts.setdefault(key, set()).add(row["fb_pa_page_base"])
        self.conflicts = {key: pas for key, pas in conflicts.items() if len(pas) > 1}

    @staticmethod
    def _row_problem(row: dict[str, str]) -> str | None:
        missing = [name for name in REQUIRED_COLUMNS if name not in row]
        if missing:
            return f"row missing columns {missing}"
        try:
            va_base = parse_int(row["va_page_base"], "va_page_base")
            va_end = parse_int(row["va_page_end_exclusive"], "va_page_end_exclusive")
            page = parse_int(row["page_size"], "page_size")
            parse_int(row["fb_pa_page_base"], "fb_pa_page_base")
        except ValueError as error:
            return f"unparsable row: {error}"
        if page <= 0 or page & (page - 1):
            return f"non-power-of-two page size {row['page_size']}"
        if va_end != va_base + page:
            return (f"va page extent {row['va_page_base']}..{row['va_page_end_exclusive']} "
                    f"disagrees with page size {row['page_size']}")
        return None

    @classmethod
    def load(cls, path: Path) -> "VaPaMap":
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
        if not rows:
            raise ValueError(f"map is empty: {path}")
        loaded = cls(rows, str(path))
        if loaded.invalid_rows and not loaded.rows:
            raise ValueError(f"every row is invalid in {path}")
        return loaded

    def device_matches(self, row: dict[str, str], device: str) -> bool:
        if device.startswith("GPU-"):
            return row["gpu_uuid"] == device or f"GPU-{row['gpu_uuid']}" == device
        return row["device"] == device

    def forward(self, device: str, gpu_va: int,
                include_inactive: bool = False) -> tuple[list[dict[str, object]], str]:
        hits: list[dict[str, object]] = []
        for row in self.rows:
            if not self.device_matches(row, device):
                continue
            if row["aperture"] != "VIDEO" or row["pte_valid"] != "true":
                continue  # never resolve a PA through a non-local page
            va_base = int(row["va_page_base"], 16)
            va_end = int(row["va_page_end_exclusive"], 16)
            if not (va_base <= gpu_va < va_end):
                continue
            if row["active_at_snapshot"] != "true" and not include_inactive:
                continue
            hits.append({
                "run_id": row["run_id"],
                "allocation_id": row["allocation_id"],
                "allocation_api": row["allocation_api"],
                "cuda_buffer_id": row["cuda_buffer_id"],
                "semantic_label": row["semantic_label"],
                "active_at_snapshot": row["active_at_snapshot"] == "true",
                "va_page_base": row["va_page_base"],
                "fb_pa_page_base": row["fb_pa_page_base"],
                "page_size": int(row["page_size"]),
                "fb_pa_byte": f"0x{int(row['fb_pa_page_base'], 16) + (gpu_va - va_base):x}",
                "in_page_offset": gpu_va - va_base,
            })
        return hits, (f"{len(hits)} mapping(s)" if hits
                      else f"no active local-VIDEO mapping covers VA {gpu_va:#x} "
                           f"on GPU {device} in {self.source}")

    def reverse(self, device: str, fb_pa_page_base: int) -> tuple[list[dict[str, object]], str]:
        matches: list[dict[str, object]] = []
        for row in self.rows:
            if not self.device_matches(row, device):
                continue
            if int(row["fb_pa_page_base"], 16) != fb_pa_page_base:
                continue
            matches.append({
                "run_id": row["run_id"],
                "allocation_id": row["allocation_id"],
                "semantic_label": row["semantic_label"],
                "active_at_snapshot": row["active_at_snapshot"] == "true",
                "va_page_base": row["va_page_base"],
                "va_page_end_exclusive": row["va_page_end_exclusive"],
                "page_size": int(row["page_size"]),
                "aperture": row["aperture"],
                "pte_valid": row["pte_valid"],
            })
        return matches, (f"{len(matches)} VA page(s)" if matches
                         else f"PA page {fb_pa_page_base:#x} is not mapped on GPU {device} "
                              f"in {self.source} (a hit on another GPU only means the "
                              f"PA namespaces are independent)")


def print_result(name: str, hits: list[dict[str, object]], message: str,
                 as_json: bool) -> None:
    if as_json:
        print(json.dumps({"query": name, "result": hits, "message": message},
                         indent=2, sort_keys=True))
    else:
        print(f"query: {name}")
        for hit in hits:
            print(f"  {hit}")
        print(f"result: {message}")


def self_test(map_path: Path) -> int:
    failures: list[str] = []

    def expect(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}: {name}" + (f" — {detail}" if detail and not condition else ""))
        if not condition:
            failures.append(name)

    def synthetic(rows_extra: list[dict[str, str]]) -> VaPaMap:
        base_row = {
            "run_id": "r", "device": "9", "gpu_uuid": "uu", "allocation_id": "a",
            "allocation_api": "cudaMalloc-binding", "cuda_buffer_id": "",
            "allocation_size_bytes": "8", "semantic_label": "TENSOR:data",
            "active_at_snapshot": "true",
            "va_page_base": "0x1000", "va_page_end_exclusive": "0x2000",
            "fb_pa_page_base": "0x5000", "page_size": "4096",
            "aperture": "VIDEO", "pte_valid": "true",
        }
        return VaPaMap([dict(base_row, **extra) for extra in rows_extra], "synthetic")

    print("synthetic integrity cases:")
    m = synthetic([{"va_page_end_exclusive": "0x3000"},
                   {"va_page_base": "0x9000", "va_page_end_exclusive": "0xa000",
                    "aperture": "SYSMEM"}])
    expect("corrupt extent row excluded from lookups", m.forward("9", 0x1800)[0] == [])
    expect("corrupt row reported", len(m.invalid_rows) == 1, str(m.invalid_rows))
    m = synthetic([{"va_page_base": "0x1000", "va_page_end_exclusive": "0x2000",
                    "fb_pa_page_base": "0x5000"},
                   {"va_page_base": "0x1000", "va_page_end_exclusive": "0x2000",
                    "fb_pa_page_base": "0x7000", "allocation_id": "b"}])
    expect("same VA page -> two PAs detected as conflict", len(m.conflicts) == 1)
    m = synthetic([{"va_page_base": "0x1000", "va_page_end_exclusive": "0x2000"}])
    hits, _ = m.forward("9", 0x1800)
    expect("synthetic forward hit", len(hits) == 1 and hits[0]["fb_pa_byte"] == "0x5800")
    m = synthetic([{"va_page_base": "0x1000", "va_page_end_exclusive": "0x2000",
                    "active_at_snapshot": "false"}])
    expect("inactive allocation rejected by default", m.forward("9", 0x1800)[0] == [])
    expect("inactive allocation visible only when explicitly requested",
           len(m.forward("9", 0x1800, include_inactive=True)[0]) == 1)
    m = synthetic([{"va_page_base": "0x1000", "va_page_end_exclusive": "0x2000",
                    "aperture": "SYSMEM"}])
    expect("non-VIDEO aperture never yields a PA", m.forward("9", 0x1800)[0] == [])

    if not map_path.is_file():
        print(f"real map absent ({map_path}); synthetic cases only")
        print("G2_VA_PA_LOOKUP_SELF_TEST_" + ("PASS" if not failures else "FAIL"))
        return 0 if not failures else 1

    real = VaPaMap.load(map_path)
    print(f"real map cases ({map_path}):")
    expect("no invalid rows in the accepted map", not real.invalid_rows,
           str(real.invalid_rows))
    expect("no VA-page PA conflicts in the accepted map", not real.conflicts,
           str(real.conflicts))
    devices = sorted({row["device"] for row in real.rows})
    expect("map covers at least one device", len(devices) >= 1)

    sample = next(row for row in real.rows
                  if row["device"] == devices[0] and row["active_at_snapshot"] == "true")
    va_base = int(sample["va_page_base"], 16)
    page = int(sample["page_size"])

    hits, message = real.forward(sample["device"], va_base)
    expect("forward: page base resolves", len(hits) == 1
           and hits[0]["fb_pa_page_base"] == sample["fb_pa_page_base"], message)
    hits, message = real.forward(sample["device"], va_base + page // 2)
    expect("forward: mid-page VA resolves into the same PA page",
           len(hits) == 1 and hits[0]["in_page_offset"] == page // 2, message)

    unmapped_va = max(int(row["va_page_end_exclusive"], 16) for row in real.rows
                      if row["device"] == devices[0])
    hits, message = real.forward(devices[0], unmapped_va)
    expect("forward: VA past the mapped window rejected", hits == [], message)

    other = next(d for d in devices if d != sample["device"]) if len(devices) > 1 else None
    if other is not None:
        hits, message = real.forward(other, va_base)
        expect("forward: VA on the wrong GPU rejected", hits == [], message)

    pa_page = int(sample["fb_pa_page_base"], 16)
    matches, message = real.reverse(sample["device"], pa_page)
    expect("reverse: PA page resolves to at least one VA page", len(matches) >= 1, message)
    shared = [pa for pa in {row["fb_pa_page_base"] for row in real.rows
                            if row["device"] == sample["device"]}
              if sum(1 for row in real.rows
                     if row["device"] == sample["device"]
                     and row["fb_pa_page_base"] == pa) > 1]
    if shared:
        matches, _ = real.reverse(sample["device"], int(shared[0], 16))
        expect("reverse: shared PA page is one-to-many", len(matches) > 1)
    if other is not None:
        matches, message = real.reverse(other, pa_page)
        expect("reverse: PA on the wrong GPU rejected", matches == [], message)

    stale = None
    trt_root = PROJECT / "artifacts/g2/observer/trt"
    current_run_ids = {row["run_id"] for row in real.rows}
    for run_dir in sorted(trt_root.glob("run_trt_gpu*"), reverse=True):
        alloc = run_dir / "g1_5_allocations.csv"
        if not alloc.is_file() or run_dir.name in current_run_ids:
            continue
        with alloc.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                stale = (row["device"], int(row["gpu_va"], 16))
                break
        if stale:
            break
    if stale is not None:
        hits, message = real.forward(stale[0], stale[1])
        expect("forward: stale VA from an earlier run rejected", hits == [], message)
    else:
        print("  SKIP: stale VA from an earlier run (no older run artifacts kept)")

    print("G2_VA_PA_LOOKUP_SELF_TEST_" + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--forward", metavar="GPU_VA", type=lambda t: parse_int(t, "GPU_VA"),
                        help="resolve a GPU virtual address to its framebuffer PA page")
    parser.add_argument("--reverse", metavar="FB_PA_PAGE",
                        type=lambda t: parse_int(t, "FB_PA_PAGE"),
                        help="resolve a framebuffer PA page to its VA page(s)")
    parser.add_argument("--gpu", required=False,
                        help="CUDA device index or GPU-UUID that owns the address")
    parser.add_argument("--include-inactive", action="store_true",
                        help="also report mappings whose allocation was inactive at snapshot")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--self-test", action="store_true",
                        help="run accept/rejection cases and exit")
    args = parser.parse_args()

    if args.self_test:
        return self_test(args.map)
    if not args.forward and not args.reverse:
        parser.error("one of --forward/--reverse/--self-test is required")
    if not args.gpu:
        parser.error("--gpu is required for lookups (device index or GPU-UUID)")

    try:
        loaded = VaPaMap.load(args.map)
    except (OSError, ValueError) as error:
        print(f"GPU_M2D_G2_LOOKUP_ERROR: {error}", file=sys.stderr)
        return 1
    if loaded.conflicts:
        print(f"GPU_M2D_G2_LOOKUP_ERROR: conflicting PA entries in {loaded.source}: "
              f"{loaded.conflicts}", file=sys.stderr)
        return 1

    if args.forward:
        hits, message = loaded.forward(args.gpu, args.forward, args.include_inactive)
        print_result(f"forward gpu={args.gpu} va={args.forward:#x}", hits, message, args.json)
        if not hits:
            return 2
    if args.reverse is not None:
        page_hint = next((int(row["page_size"]) for row in loaded.rows), 4096)
        if args.reverse % page_hint:
            print(f"GPU_M2D_G2_LOOKUP_REJECTED: PA {args.reverse:#x} is not page-aligned "
                  f"to {page_hint}; a PA claim exists only at page granularity", file=sys.stderr)
            return 2
        matches, message = loaded.reverse(args.gpu, args.reverse)
        print_result(f"reverse gpu={args.gpu} pa={args.reverse:#x}", matches, message, args.json)
        if not matches:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
