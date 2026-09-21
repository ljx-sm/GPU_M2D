#!/usr/bin/env python3
"""GPU_M2D G4: dual-addressing snapshot -- join the three validated legs.

    G1.5 (allocation registry)  (allocation_id, byte, bit) <-> GPU VA
    G2   (eBPF observer)        (allocation_id, GPU VA page) <-> fb PA page
    G3   (EMT table v4 + API)   GPU PA page -> GDDR relation classes

This tool folds ONE captured G1.5+G2 run for ONE device into the per-run
snapshot G5 consumes, keeping the fail-closed discipline of every leg:

  * every ACTIVE allocation's full VA range must be covered by observed
    pages (pte_valid, aperture=VIDEO, LOCAL_VIDEO_COMPLETE) -- a gap
    breaks the chain and the snapshot is refused (exit 2, nothing written);
  * every resident PA page must be inside the G3 table universe -- the
    driver handed out memory the table never measured; refused (the
    relation must be built, never extrapolated);
  * two different VA pages may not alias one PA page inside one snapshot
    (refused; revisit deliberately if G5 ever wants aliasing);
  * byte residency comes from the registry VA bounds, so several small
    allocations may share one 2 MiB page as DISJOINT byte ranges -- the
    overlap check enforces it;
  * unlinked pages (bank drew none of the 1024 uniform reps) and pages
    without anchors are carried as honest UNKNOWN columns + warnings,
    never as guessed relations.

Outputs (--out DIR):
  snapshot_pages.csv  one row per (allocation, VA page): both addresses,
                      byte residency, GDDR classes + provenance fields
  manifest.json       input/table hashes, counts, the mapping checksum
                      (sha256 of snapshot_pages.csv), warnings

    python3 build_snapshot.py \
        --allocations artifacts/g1_5/gpu0_allocations.csv \
        --va-pa artifacts/g2/gpu_va_pa_map.csv --device 0 \
        --table artifacts/g3/table_v4 --out artifacts/g4/snapshot_trt_gpu0
    python3 build_snapshot.py --snapshot artifacts/g4/snapshot_trt_gpu0 \
        --lookup-va 0x76632bc00000
    python3 build_snapshot.py --snapshot ... --lookup-pa 0x1f000000

Exit codes: 0 ok, 2 fail-closed, 1 error (repo convention). --self-test
pins the semantics on synthetic fixtures (registry + map + table).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "g3_probe"))
from query_table import Table, page_of  # noqa: E402

PAGE_SIZE = 2 << 20
PAGE_MASK = PAGE_SIZE - 1

ALLOC_FIELDS = ["run_id", "allocation_id", "device", "gpu_va", "size_bytes",
                "alignment_bytes", "owner", "allocation_phase", "lifetime",
                "active_at_injection", "semantic_label"]
MAP_FIELDS = ["run_id", "g1_5_run_id", "device", "gpu_uuid", "allocation_id",
              "allocation_api", "va_page_base", "va_page_end_exclusive",
              "fb_pa_page_base", "page_size", "aperture", "pte_valid",
              "covered_allocation_bytes", "physical_coverage_status"]
SNAPSHOT_FIELDS = ["device", "gpu_uuid", "run_id", "g1_5_run_id",
                   "allocation_id", "semantic_label", "allocation_phase",
                   "allocation_va_base", "allocation_size_bytes",
                   "va_page_base", "byte_start_in_page", "byte_end_in_page",
                   "fb_pa_page_base", "page_size", "in_universe",
                   "bank_linked", "channel_root", "channel_size",
                   "row_class_count", "same_row_site_nodes",
                   "valid_anchor_masks"]


def _hx(text: str) -> int:
    return int(text, 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Allocation:
    run_id: str
    allocation_id: str
    va: int
    size: int
    label: str
    phase: str
    active: bool


def load_allocations(path: Path, device: int, say) -> list[Allocation]:
    """ACTIVE registry allocations for the device (inactive counted, skipped)."""
    active: list[Allocation] = []
    skipped = 0
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["device"]) != device:
                continue
            if row["active_at_injection"] != "1":
                skipped += 1
                continue
            active.append(Allocation(
                run_id=row["run_id"], allocation_id=row["allocation_id"],
                va=_hx(row["gpu_va"]), size=int(row["size_bytes"]),
                label=row["semantic_label"], phase=row["allocation_phase"],
                active=True))
    say(f"registry: {len(active)} active allocation(s) on device {device} "
        f"({skipped} inactive skipped)")
    return active


def load_map(path: Path, device: int, run_ids: set[str],
             alloc_ids: set[str], say):
    """(allocation_id -> va_page -> row) for the device + linked G1.5 runs."""
    pages: dict[str, dict[int, dict]] = defaultdict(dict)
    uuid = None
    map_run = None
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["device"]) != device:
                continue
            if row["g1_5_run_id"] not in run_ids:
                continue
            if row["allocation_id"] not in alloc_ids:
                say(f"INTEGRITY: map row for {row['allocation_id']} has no "
                    "ACTIVE registry allocation in this run")
                return None, None, None
            uuid = uuid or row["gpu_uuid"]
            map_run = map_run or row["run_id"]
            page = _hx(row["va_page_base"])
            if page in pages[row["allocation_id"]]:
                say(f"INTEGRITY: duplicate map row {row['allocation_id']} "
                    f"va_page {page:#x}")
                return None, None, None
            pages[row["allocation_id"]][page] = row
    say(f"va-pa map: {sum(len(v) for v in pages.values())} page row(s) "
        f"across {len(pages)} allocation(s), run {map_run}, uuid {uuid}")
    return pages, uuid, map_run


def build_rows(allocs: list[Allocation], pages: dict, tab: Table, say):
    """The join itself. Returns (rows, problems, warnings)."""
    problems: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    residency: dict[int, list[tuple]] = defaultdict(list)   # va_page -> ranges
    pa_owners: dict[int, int] = {}                          # pa_page -> va_page

    for alloc in sorted(allocs, key=lambda a: (a.va, a.allocation_id)):
        if alloc.size <= 0:
            problems.append(f"{alloc.allocation_id}: non-positive size")
            continue
        first = alloc.va >> 21
        last = (alloc.va + alloc.size - 1) >> 21
        for page_no in range(first, last + 1):
            va_page = page_no << 21
            mrow = pages.get(alloc.allocation_id, {}).get(va_page)
            if mrow is None:
                problems.append(f"{alloc.allocation_id}: VA page "
                                f"{va_page:#x} has no observed map row")
                continue
            if mrow["pte_valid"] != "true":
                problems.append(f"{alloc.allocation_id}: VA page "
                                f"{va_page:#x} PTE not valid")
                continue
            if mrow["aperture"] != "VIDEO":
                problems.append(f"{alloc.allocation_id}: VA page "
                                f"{va_page:#x} aperture {mrow['aperture']}")
                continue
            if mrow["physical_coverage_status"] != "LOCAL_VIDEO_COMPLETE":
                problems.append(f"{alloc.allocation_id}: VA page "
                                f"{va_page:#x} coverage "
                                f"{mrow['physical_coverage_status']}")
                continue
            if int(mrow["page_size"]) != PAGE_SIZE:
                problems.append(f"{alloc.allocation_id}: VA page "
                                f"{va_page:#x} page_size "
                                f"{mrow['page_size']} != 2 MiB")
                continue
            pa_page = _hx(mrow["fb_pa_page_base"])
            if pa_page & PAGE_MASK:
                problems.append(f"{alloc.allocation_id}: PA "
                                f"{pa_page:#x} not 2 MiB aligned")
                continue
            if not tab.in_universe(pa_page):
                problems.append(f"{alloc.allocation_id}: PA page "
                                f"{pa_page:#x} OUTSIDE the G3 table universe "
                                "(fail-closed -- build the table for it, "
                                "never extrapolate)")
                continue
            if pa_owners.get(pa_page, va_page) != va_page:
                problems.append(f"PA aliasing: {pa_page:#x} backs VA pages "
                                f"{pa_owners[pa_page]:#x} and {va_page:#x}")
                continue
            pa_owners[pa_page] = va_page
            start = max(alloc.va, va_page) - va_page
            end = min(alloc.va + alloc.size, va_page + PAGE_SIZE) - va_page
            overlap = False
            for other_start, other_end, other_id in residency[va_page]:
                if start < other_end and other_start < end:
                    problems.append(f"residency overlap on VA page "
                                    f"{va_page:#x}: {alloc.allocation_id} "
                                    f"[{start:#x},{end:#x}) vs {other_id} "
                                    f"[{other_start:#x},{other_end:#x})")
                    overlap = True
            if overlap:
                continue
            residency[va_page].append((start, end, alloc.allocation_id))

            trow = tab.pages.get(pa_page, {})
            row_classes = sum(1 for members in tab.row_members.values()
                              if members and page_of(members[0]) == pa_page)
            row_nodes = sum(len(members)
                            for members in tab.row_members.values()
                            if members and page_of(members[0]) == pa_page)
            masks = " ".join(f"{m:#x}"
                             for m, _ in tab.anchors.get(pa_page, [])[:3])
            rows.append({
                "device": "", "gpu_uuid": "", "run_id": alloc.run_id,
                "g1_5_run_id": alloc.run_id,
                "allocation_id": alloc.allocation_id,
                "semantic_label": alloc.label,
                "allocation_phase": alloc.phase,
                "allocation_va_base": f"{alloc.va:#x}",
                "allocation_size_bytes": alloc.size,
                "va_page_base": f"{va_page:#x}",
                "byte_start_in_page": f"{start:#x}",
                "byte_end_in_page": f"{end:#x}",
                "fb_pa_page_base": f"{pa_page:#x}",
                "page_size": PAGE_SIZE,
                "in_universe": 1,
                "bank_linked": int(bool(trow.get("channel_root"))),
                "channel_root": trow.get("channel_root", ""),
                "channel_size": trow.get("channel_size", ""),
                "row_class_count": row_classes,
                "same_row_site_nodes": row_nodes,
                "valid_anchor_masks": masks,
            })
            if not trow.get("channel_root"):
                warnings.append(f"{pa_page:#x}: unlinked page (bank UNKNOWN)")
            if not masks:
                warnings.append(f"{pa_page:#x}: no consensus-valid anchor")
    rows.sort(key=lambda r: (int(r["va_page_base"], 16),
                             int(r["byte_start_in_page"], 16)))
    return rows, problems, warnings


def write_outputs(out: Path, rows, manifest) -> None:
    out.mkdir(parents=True, exist_ok=True)
    snap = out / "snapshot_pages.csv"
    with snap.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=SNAPSHOT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest["snapshot_sha256"] = _sha256(snap)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def print_stats(rows, allocs, problems, warnings, say) -> None:
    if problems:
        say(f"PROBLEMS ({len(problems)}, fail-closed -- nothing written):")
        for text in problems[:10]:
            say(f"  {text}")
        if len(problems) > 10:
            say(f"  ... {len(problems) - 10} more")
        return
    by_label_bytes: Counter = Counter()
    by_label_allocs: Counter = Counter()
    for a in allocs:
        by_label_bytes[a.label] += a.size
        by_label_allocs[a.label] += 1
    pa_pages = {int(r["fb_pa_page_base"], 16) for r in rows}
    linked = {int(r["fb_pa_page_base"], 16) for r in rows
              if str(r["bank_linked"]) == "1"}
    row_pages = {int(r["fb_pa_page_base"], 16) for r in rows
                 if int(r["same_row_site_nodes"]) > 0}
    row_bytes = sum(int(r["byte_end_in_page"], 16)
                    - int(r["byte_start_in_page"], 16) for r in rows
                    if int(r["same_row_site_nodes"]) > 0)
    anchors = {int(r["fb_pa_page_base"], 16) for r in rows
               if r["valid_anchor_masks"]}
    say(f"snapshot: {len(rows)} page row(s), {len(pa_pages)} distinct PA "
        f"page(s), {sum(a.size for a in allocs)} resident byte(s) over "
        f"{len(allocs)} allocation(s)")
    for label in sorted(by_label_bytes):
        say(f"  {label}: {by_label_allocs[label]} alloc(s), "
            f"{by_label_bytes[label]} byte(s)")
    say(f"  bank: {len(linked)}/{len(pa_pages)} PA pages linked "
        "(rest unlinked = bank UNKNOWN, warned)")
    say(f"  anchors: {len(anchors)}/{len(pa_pages)} pages with "
        "consensus-valid anchor masks")
    say(f"  GDDR same-row sites: {len(row_pages)} resident page(s) host "
        f"measured row classes ({row_bytes} resident byte(s) on them)")
    if warnings:
        say(f"  warnings ({len(warnings)}):")
        for text in sorted(set(warnings))[:5]:
            say(f"    {text}")
        if len(set(warnings)) > 5:
            say(f"    ... {len(set(warnings)) - 5} more")


def lookup_va(rows, addr: int, say) -> bool:
    page = page_of(addr)
    off = addr - page
    hits = [r for r in rows
            if int(r["va_page_base"], 16) == page
            and int(r["byte_start_in_page"], 16) <= off
            < int(r["byte_end_in_page"], 16)]
    say(f"lookup VA {addr:#x}:")
    if not hits:
        say("  no resident allocation byte at this address "
            "(outside every registry range)")
        return False
    for r in hits:
        say(f"  {r['allocation_id']} [{r['semantic_label']}] byte "
            f"{off - int(r['byte_start_in_page'], 16):#x} of "
            f"{int(r['allocation_size_bytes']):#x} -> PA page "
            f"{r['fb_pa_page_base']} +{off:#x} | bank "
            f"{'known' if str(r['bank_linked']) == '1' else 'UNKNOWN'} "
            f"(component {r['channel_root'] or '-'}) | row sites "
            f"{r['same_row_site_nodes']} | anchors "
            f"{r['valid_anchor_masks'] or '-'}")
    return True


def lookup_pa(rows, addr: int, say) -> bool:
    page = page_of(addr)
    hits = [r for r in rows if int(r["fb_pa_page_base"], 16) == page]
    say(f"lookup PA {addr:#x} (page {page:#x}):")
    if not hits:
        say("  not resident in this snapshot")
        return False
    for r in hits:
        say(f"  -> {r['allocation_id']} [{r['semantic_label']}] VA page "
            f"{r['va_page_base']} bytes "
            f"[{r['byte_start_in_page']},{r['byte_end_in_page']}) | bank "
            f"{'known' if str(r['bank_linked']) == '1' else 'UNKNOWN'} "
            f"(component {r['channel_root'] or '-'}) | row sites "
            f"{r['same_row_site_nodes']} | anchors "
            f"{r['valid_anchor_masks'] or '-'}")
    say("  (element/bit arithmetic from the byte offset is G1.5's validated "
        "registry job; this row is the VA<->PA<->GDDR join)")
    return True


def load_snapshot(out: Path, say):
    snap = out / "snapshot_pages.csv"
    if not snap.is_file():
        say(f"INTEGRITY: missing {snap}")
        return None
    with snap.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# driver + self-test
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocations", type=Path,
                        help="G1.5 registry CSV (gpu{N}_allocations.csv)")
    parser.add_argument("--va-pa", type=Path,
                        help="aggregated G2 gpu_va_pa_map.csv")
    parser.add_argument("--device", type=int)
    parser.add_argument("--table", type=Path,
                        default=Path("artifacts/g3/table_v4"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--snapshot", type=Path,
                        help="existing snapshot dir (for lookups)")
    parser.add_argument("--lookup-va", type=_hx)
    parser.add_argument("--lookup-pa", type=_hx)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    say = lambda t="": print(t)

    if args.snapshot:
        if not (args.lookup_va or args.lookup_pa):
            parser.error("--snapshot needs --lookup-va/--lookup-pa")
        rows = load_snapshot(args.snapshot, say)
        if rows is None:
            return 1
        ok = True
        if args.lookup_va:
            ok = lookup_va(rows, args.lookup_va, say) and ok
        if args.lookup_pa:
            ok = lookup_pa(rows, args.lookup_pa, say) and ok
        return 0 if ok else 2
    if args.lookup_va or args.lookup_pa:
        parser.error("lookups without a build need --snapshot DIR")

    if args.allocations is None or args.va_pa is None \
            or args.device is None or args.out is None:
        parser.error("build needs --allocations, --va-pa, --device, --out")
    problems: list[str] = []
    tab = Table(args.table, problems)
    if problems:
        say(f"INTEGRITY: {'; '.join(problems)}")
        return 1
    allocs = load_allocations(args.allocations, args.device, say)
    pages, uuid, map_run = load_map(args.va_pa, args.device,
                                    {a.run_id for a in allocs},
                                    {a.allocation_id for a in allocs}, say)
    if pages is None:
        return 2
    rows, problems, warnings = build_rows(allocs, pages, tab, say)
    print_stats(rows, allocs, problems, warnings, say)
    if problems:
        return 2
    manifest = {
        "schema": "gpu-m2d.g4-dualaddr.snapshot.v1",
        "device": args.device,
        "gpu_uuid": uuid,
        "run_id": map_run,
        "g1_5_run_id": sorted({a.run_id for a in allocs}),
        "table_dir": str(args.table),
        "inputs": {
            "allocations_sha256": _sha256(args.allocations),
            "va_pa_sha256": _sha256(args.va_pa),
            "gddr_seed_table_sha256":
                _sha256(args.table / "gddr_seed_table.csv"),
            "bank_classes_sha256":
                _sha256(args.table / "bank_classes.csv"),
        },
        "counts": {
            "allocations": len(allocs),
            "rows": len(rows),
            "resident_bytes": sum(a.size for a in allocs),
            "pa_pages": len({r["fb_pa_page_base"] for r in rows}),
            "unlinked_pages":
                len({r["fb_pa_page_base"] for r in rows
                     if str(r["bank_linked"]) == "0"}),
            "row_site_pages":
                len({r["fb_pa_page_base"] for r in rows
                     if int(r["same_row_site_nodes"]) > 0}),
        },
        "warnings": sorted(set(warnings)),
    }
    write_outputs(args.out, rows, manifest)
    say(f"output: {args.out}/snapshot_pages.csv + manifest.json "
        f"(mapping checksum {manifest['snapshot_sha256'][:16]}...)")
    if args.lookup_va:
        lookup_va(rows, args.lookup_va, say)
    if args.lookup_pa:
        lookup_pa(rows, args.lookup_pa, say)
    return 0


def self_test() -> int:
    import io
    import contextlib
    import tempfile

    P = lambda n: n << 21
    VA = 0x700000000000
    V0, V3 = VA, VA + 3 * PAGE_SIZE
    quiet = lambda t: None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # ---- G3 table fixture: comp(P0,P1) / P2 alone / P3 unlinked /
        # 0x2ae00000 with a 2-node row class
        tdir = root / "table"
        tdir.mkdir()
        (tdir / "gddr_seed_table.csv").write_text(
            "page_index,page_base,lambda,channel_root,channel_size,super_tail,"
            "n_shoulder,n_deep,n_low,n_nodes,n_bank_classes,classified\n"
            f"0,{P(0):#x},1000,{P(0):#x},2,0,0,2,10,3,1,1\n"
            f"1,{P(1):#x},1010,{P(0):#x},2,0,0,2,10,1,1,1\n"
            f"2,{P(2):#x},1020,{P(2):#x},1,0,0,0,10,1,1,1\n"
            f"3,{P(3):#x},1030,,0,0,0,0,10,1,1,1\n"
            f"343,0x2ae00000,1040,0x2ae00000,1,0,0,0,10,2,1,1\n")
        (tdir / "bank_classes.csv").write_text(
            "page_index,page_base,offset,pa,bank_class,bank_size,row_class,"
            "row_size,n_deep,n_low,n_shoulder\n"
            f"0,{P(0):#x},0x0,{P(0):#x},{P(0):#x},3,{P(0):#x},2,1,1,0\n"
            f"0,{P(0):#x},0x200,{P(0) + 0x200:#x},{P(0):#x},3,{P(0):#x},"
            f"2,0,1,0\n"
            "343,0x2ae00000,0xd3880,0x2aed3880,0x2aed3880,2,0x2aed3880,"
            "2,0,1,0\n"
            "343,0x2ae00000,0xd7b00,0x2aed7b00,0x2aed3880,2,0x2aed3880,"
            "2,0,1,0\n")
        (tdir / "page_anchors.csv").write_text(
            "page_base,candidate,seen,votes,valid\n"
            f"{P(0):#x},0x1f9dc0,3,3,true\n{P(1):#x},0x1f9dc0,3,3,true\n"
            "0x2ae00000,0x1f9dc0,3,3,true\n")
        problems: list[str] = []
        tab = Table(tdir, problems)
        assert not problems, problems

        # ---- registry fixture: a-big spans 4 pages (last partial), b-small
        # shares a-big's last page disjointly, d-bind sits on the row-class
        # page, c-dead is inactive
        def alloc_text(extra=""):
            return ",".join(ALLOC_FIELDS) + "\n" + "\n".join([
                f"r-a,a-big,0,{V0:#x},{3 * PAGE_SIZE + 0x1000},512,o,ph,lt,1,T:a",
                f"r-a,b-small,0,{V3 + 0x1000:#x},2048,512,o,ph,lt,1,T:b",
                f"r-a,d-bind,0,{VA + P(9):#x},16,512,o,ph,lt,1,T:d",
                f"r-a,c-dead,0,{VA + P(11):#x},4096,512,o,ph,lt,0,T:c",
            ] + ([extra] if extra else [])) + "\n"

        def map_row(alloc, va_page, pa_page):
            return (f"m,r-a,0,uuid,{alloc},api,{va_page:#x},"
                    f"{va_page + PAGE_SIZE:#x},{pa_page:#x},{PAGE_SIZE},"
                    f"VIDEO,true,{PAGE_SIZE},LOCAL_VIDEO_COMPLETE")

        a_pas = [P(0), P(1), P(2), P(3)]
        map_lines = [",".join(MAP_FIELDS)]
        for i in range(4):
            map_lines.append(map_row("a-big", V0 + i * PAGE_SIZE, a_pas[i]))
        map_lines.append(map_row("b-small", V3, P(3)))
        map_lines.append(map_row("d-bind", VA + P(9), 0x2ae00000))

        def run_build(alloc_csv: Path, map_csv: Path):
            allocs = load_allocations(alloc_csv, 0, quiet)
            pg, _, _ = load_map(map_csv, 0,
                                {"r-a"},
                                {a.allocation_id for a in allocs}, quiet)
            assert pg is not None
            return allocs, build_rows(allocs, pg, tab, quiet)

        areg = root / "allocations.csv"
        areg.write_text(alloc_text())
        good_map = root / "map_good.csv"
        good_map.write_text("\n".join(map_lines) + "\n")

        # happy path: 6 rows; shared page carries disjoint byte ranges;
        # inactive allocation never appears
        allocs, (rows, probs, warns) = run_build(areg, good_map)
        assert not probs, probs
        assert len(rows) == 6, len(rows)
        assert all(r["allocation_id"] != "c-dead" for r in rows)
        last = [r for r in rows if r["va_page_base"] == f"{V3:#x}"]
        assert {(r["allocation_id"], r["byte_start_in_page"],
                 r["byte_end_in_page"]) for r in last} == {
            ("a-big", "0x0", "0x1000"), ("b-small", "0x1000", "0x1800")}, last
        # unlinked PA page -> bank UNKNOWN + warning, still a valid row
        unlinked = [r for r in rows if r["fb_pa_page_base"] == f"{P(3):#x}"]
        assert len(unlinked) == 2 and all(str(r["bank_linked"]) == "0"
                                          for r in unlinked)
        assert any("unlinked" in w for w in warns), warns
        # row-class page annotation
        d_row = [r for r in rows if r["allocation_id"] == "d-bind"]
        assert d_row and int(d_row[0]["same_row_site_nodes"]) == 2 \
            and d_row[0]["valid_anchor_masks"] == "0x1f9dc0", d_row
        out = root / "snap"
        write_outputs(out, rows, {"schema": "x"})
        assert (out / "snapshot_pages.csv").is_file()
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["snapshot_sha256"] == _sha256(
            out / "snapshot_pages.csv")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_stats(rows, allocs, [], warns, print)
        assert "bank: 4/5 PA pages linked" in buffer.getvalue(), \
            buffer.getvalue()

        # lookups: forward / reverse round-trips
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = lookup_va(rows, V3 + 0x1200, print)
        assert ok and "b-small" in buffer.getvalue(), buffer.getvalue()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = lookup_va(rows, V3 + 0x1800, print)
        assert not ok, buffer.getvalue()      # past b's end: not resident
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = lookup_pa(rows, P(3) + 0x1000, print)
        assert ok and "b-small" in buffer.getvalue(), buffer.getvalue()

        # ---- fail-closed paths: each refuses with the right diagnosis
        def expect_fail(alloc_extra, map_text, needle, name):
            areg2 = root / f"a_{name}.csv"
            areg2.write_text(alloc_text(alloc_extra))
            mmap = root / f"m_{name}.csv"
            mmap.write_text(map_text)
            _, (_, probs, _) = run_build(areg2, mmap)
            joined = "; ".join(probs)
            assert probs and needle in joined, (name, joined)

        # a missing observed page
        expect_fail("", "\n".join(map_lines[:4] + map_lines[5:]) + "\n",
                    "no observed map row", "gap")
        # PA page outside the table universe
        outside = map_lines[:1] + [map_row("a-big", V0, 0x90000000)] \
            + map_lines[2:]
        expect_fail("", "\n".join(outside) + "\n",
                    "OUTSIDE the G3 table universe", "outside")
        # invalid PTE
        invalid = map_lines[:1] + [map_lines[1].replace("VIDEO,true",
                                                        "VIDEO,false")] \
            + map_lines[2:]
        expect_fail("", "\n".join(invalid) + "\n", "PTE not valid", "pte")
        # PA aliasing: e-bind's page also backs P(0)
        expect_fail(f"r-a,e-bind,0,{VA + P(10):#x},256,512,o,ph,lt,1,T:e",
                    "\n".join(map_lines
                              + [map_row("e-bind", VA + P(10), P(0))]) + "\n",
                    "PA aliasing", "alias")
        # residency overlap: b moved onto a-big's tail bytes
        overlap = root / "a_overlap.csv"
        overlap.write_text(alloc_text().replace(
            f"r-a,b-small,0,{V3 + 0x1000:#x},2048",
            f"r-a,b-small,0,{V3:#x},2048"))
        mmap = root / "m_overlap.csv"
        mmap.write_text("\n".join(map_lines) + "\n")
        _, (_, probs, _) = run_build(overlap, mmap)
        assert any("residency overlap" in p for p in probs), probs
        # a map allocation missing from the registry refuses at load
        dangling = root / "m_dangling.csv"
        dangling.write_text("\n".join(
            map_lines + [map_row("ghost", VA + P(12), P(2))]) + "\n")
        allocs2 = load_allocations(areg, 0, quiet)
        pg, _, _ = load_map(dangling, 0, {"r-a"},
                            {a.allocation_id for a in allocs2}, quiet)
        assert pg is None
    print("build_snapshot self-test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
