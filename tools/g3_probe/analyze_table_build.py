#!/usr/bin/env python3
"""GPU_M2D G3 S5-T1: table-build (anchor-expansion) analyzer.

The T0 seed table is honest but sparse (220/2048 pages with a channel
component, 38/2048 pages with a classified node). The table-build
workload densifies it in ONE pre-planned run -- no closed form, the
GeForge-style empirical class table only:

  self          fresh per-page lambda (the classification reference)
  anchor_sweep  (p, p^M_c) per page x candidate: DEEP = same-bank
                different-row edge + a valid anchor for double probes;
                the page x mask matrix separates universally
                bank-preserving masks from seed-dependent ones
  classify      (p, rep) page starts: cross-page DEEP = same bank
                (hence same channel), different row; cross-page LOW =
                anything else. The run measures that NO cross-page
                shoulder band exists (see the model note below), so the
                channel story reduces to same-bank page sets.
  bank_map      (p, p^y) and (p^M, p^y) per lattice offset y at
                anchor-valid pages: with bank(x0)=bank(x0^M) and the two
                rows distinct, y is same-bank iff EITHER probe is deep,
                and a low among same-bank pairs pins y's row to
                row(x0) vs row(x0^M)
  repeat        the drift anchor for the late-section additive step

MODEL REVISION (measured on the first T1 run, supersedes S4b-1's
three-band story for cross-page pairs): every pair value references
max(lambda_a, lambda_b), not the global calibration baseline. The
per-page lambda spreads ~120 cycles -- wider than the conflict
amplitude -- so any global gate lands inside the lambda spread and
manufactures a "shoulder" band out of slow pages. Lambda-referenced,
the classify section is bimodal with an EMPTY +30..+80 valley: low at
d ~ 0 (96%) and deep at d >= +80 (0.25% ~= 1/384, matching the AD102
prior of 24 channels x 16 banks). The historical cross-page "shallow
band" (S3b's 185, the census re-probe's 228) shows the SAME
lambda-referenced distribution for its shoulder and low verdicts --
selection bias over the lambda spread, not a physical regime. Deep is
the only cross-page class evidence; low is compatible with
same-channel-different-bank, so the old C3 rule (low inside a channel
component) is invalid and the analyzer unions pages on deep edges
only. In-page, the valley sits at +45..+70; gates: low < 0.35*amp,
deep >= 0.60*amp above the lambda reference.

Sections are reconstructed from summary.json's table_build_selection and
verified against every row's own PA (fail-closed, exit 2). Outputs
table_build_edges.csv (the build_bank_table.py --t1 source, highest
priority), anchor_validity.csv, bank_map_pages.csv and
channel_partition.csv next to the inputs.

    python3 analyze_table_build.py RUN_DIR

Exit codes: 0 ok, 2 integrity failure, 1 error (repo convention).
`--self-test` pins the splitter, the three-band relabel, the channel
partition with a planted C3, the anchor-validity matrix and the
double-probe bank map (including the anchor-invalid void) on a synthetic
run planned by the real planner.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_census import (  # noqa: E402
    LOW_FRACTION,
    SYMMETRY_FRACTION,
    UnionFind,
)
from g3_pool import (  # noqa: E402
    PoolMap,
    parse_pool_map,
)

# Lambda-referenced band gates (fractions of the calibration amplitude):
# low below LOW_FRACTION, deep at/above DEEP_FRACTION, mid between. The
# deep edge sits in the measured empty valley (cross-page +30..+80,
# in-page +45..+70), NOT at the census's global 0.90 -- deeps ride the
# slower endpoint's lambda, and a global gate manufactures shoulders out
# of lambda-slow pages (see the module docstring).
DEEP_FRACTION = 0.60
# AD102 prior (G3_SURVEY): 12 x32-bit MCs, one package each with 2 x16
# channels -> ~24 channels, 24 x 16 = 384 banks. The cross-page deep
# rate over random pairs should sit near 1/384. Sanity anchor, no gate.
BANK_PRIOR = 384


def _hex_list(values: list[str]) -> list[int]:
    return [int(value, 16) for value in values]


def load_rows(run_dir: Path, name: str) -> list[dict[str, str]]:
    with (run_dir / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(line for line in source
                                   if not line.startswith("#")))


def split_sections(rows: list[dict[str, str]], pool: PoolMap,
                   sel: dict) -> tuple[dict[str, list[dict[str, str]]],
                                       list[str]]:
    """Slice the result rows into the table-build sections by the counts
    implied by the recorded selection, verifying each row's PA shape."""
    n_pages = len(pool.pa_pages)
    candidates = _hex_list(sel.get("anchor_candidates", []))
    reps = _hex_list(sel.get("reps", []))
    lattice = _hex_list(sel.get("bank_map_offsets", []))
    anchor = int(sel.get("bank_map_anchor", "0x0"), 16)
    bank_pages = _hex_list(sel.get("bank_map_pages", []))
    repeat_count = int(sel.get("repeat_pages", 0))

    classify_count = sum(
        1 for page in pool.pa_pages
        for rep in reps if rep != page.fb_pa_page_base)
    bank_count = len(bank_pages) * (
        len(lattice) + len(lattice) - (1 if anchor in lattice else 0))
    expected = 3 + n_pages + n_pages * len(candidates) + classify_count \
        + bank_count + repeat_count
    problems: list[str] = []
    if len(rows) != expected:
        problems.append(f"row count {len(rows)} != table-build plan "
                        f"{expected} (pages {n_pages}, cands {len(candidates)}, "
                        f"reps {len(reps)}, bank {bank_count}, repeat "
                        f"{repeat_count})")
        return {}, problems

    cursor = 0

    def take(count: int, section: str) -> list[dict[str, str]]:
        nonlocal cursor
        part = rows[cursor:cursor + count]
        for row in part:
            row["section"] = section
        cursor += count
        return part

    sections = {
        "calibration": take(3, "calibration"),
        "self": take(n_pages, "self"),
        "anchor_sweep": take(n_pages * len(candidates), "anchor_sweep"),
        "classify": take(classify_count, "classify"),
        "bank_map": take(bank_count, "bank_map"),
        "repeat": take(repeat_count, "repeat"),
    }

    def query_pas(row: dict[str, str]) -> tuple[int, int]:
        return (pool.pa_of(int(row["chunk_a"]), int(row["ofs_a"])),
                pool.pa_of(int(row["chunk_b"]), int(row["ofs_b"])))

    for index, row in enumerate(sections["self"]):
        pa_a, pa_b = query_pas(row)
        if pa_a != pa_b or pa_a != pool.pa_pages[index].fb_pa_page_base:
            problems.append(f"self row {row['query_id']}: pa {pa_a:#x} != "
                            f"page base {pool.pa_pages[index].fb_pa_page_base:#x}")
    bank_cursor = 0
    for index, row in enumerate(sections["anchor_sweep"]):
        pa_a, pa_b = query_pas(row)
        page = pool.pa_pages[index // len(candidates)]
        cand = candidates[index % len(candidates)]
        if (pa_a != page.fb_pa_page_base
                or pa_b - page.fb_pa_page_base != cand
                or pa_a >> 21 != pa_b >> 21):
            problems.append(f"anchor_sweep row {row['query_id']}: "
                            f"{pa_a:#x}/{pa_b:#x} != page+cand "
                            f"{page.fb_pa_page_base:#x}+{cand:#x}")
    classify_pairs = [(page.fb_pa_page_base, rep)
                      for page in pool.pa_pages
                      for rep in reps
                      if rep != page.fb_pa_page_base]
    for row, want in zip(sections["classify"], classify_pairs):
        pa_a, pa_b = query_pas(row)
        if (pa_a, pa_b) != want:
            problems.append(f"classify row {row['query_id']}: "
                            f"{pa_a:#x}/{pa_b:#x} != planned "
                            f"{want[0]:#x}/{want[1]:#x}")
            break
    bank_pairs: list[tuple[int, int]] = []
    for base in bank_pages:
        for low in lattice:
            bank_pairs.append((base, base + low))
            if low in (0, anchor):
                continue  # (M, 0) is (0, M) again; (M, M) is a self pair
            bank_pairs.append((base + anchor, base + low))
    for row, want in zip(sections["bank_map"], bank_pairs):
        pa_a, pa_b = query_pas(row)
        if (pa_a, pa_b) != want:
            problems.append(f"bank_map row {row['query_id']}: "
                            f"{pa_a:#x}/{pa_b:#x} != planned "
                            f"{want[0]:#x}/{want[1]:#x}")
            break
    for index, row in enumerate(sections["repeat"]):
        pa_a, pa_b = query_pas(row)
        if pa_a != pa_b or pa_a != pool.pa_pages[index].fb_pa_page_base:
            problems.append(f"repeat row {row['query_id']} != self row {index}")
    return sections, problems


def analyze(run_dir: Path) -> int:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("query_selection", {}).get("mode") != "table-build":
        print("INTEGRITY: not a table-build run (summary query_selection.mode)")
        return 2
    sel = summary.get("table_build_selection")
    if not sel:
        print("INTEGRITY: summary lacks table_build_selection")
        return 2
    pool = PoolMap(parse_pool_map((run_dir / "pool_map.csv").read_text(
        encoding="utf-8")))
    rows = load_rows(run_dir, "result.csv")
    sections, problems = split_sections(rows, pool, sel)
    if problems:
        print(f"INTEGRITY: {'; '.join(problems[:5])}")
        return 2

    by_id = {int(row["query_id"]): row for row in rows}
    floor = int(by_id[0]["cycles_a"])
    baseline = int(by_id[1]["cycles_a"])
    conflict = int(by_id[2]["cycles_a"])
    amplitude = conflict - baseline
    if amplitude <= 0:
        print(f"INTEGRITY: degenerate calibration {floor}/{baseline}/{conflict}")
        return 2
    page_size = pool.pages[0].page_size
    lambdas = {pool.pa_pages[index].fb_pa_page_base: int(row["cycles_a"])
               for index, row in enumerate(sections["self"])}

    repeat_deltas = sorted(
        int(row["cycles_a"]) - lambdas[pool.pa_pages[index].fb_pa_page_base]
        for index, row in enumerate(sections["repeat"]))
    late_offset = -repeat_deltas[len(repeat_deltas) // 2] if repeat_deltas else 0

    def pair_pa(row: dict[str, str]) -> tuple[int, int]:
        return (pool.pa_of(int(row["chunk_a"]), int(row["ofs_a"])),
                pool.pa_of(int(row["chunk_b"]), int(row["ofs_b"])))

    def label_of(row: dict[str, str]) -> str:
        """Lambda-referenced three-way label: low / mid / deep_conflict
        (no shoulder -- measured absent, see the module docstring)."""
        pa_a, pa_b = pair_pa(row)
        if abs(int(row["cycles_a"]) - int(row["cycles_b"])) \
                > SYMMETRY_FRACTION * amplitude:
            return "asymmetric"
        reference = max(lambdas.get(pa_a & ~(page_size - 1), 0),
                        lambdas.get(pa_b & ~(page_size - 1), 0))
        d = int(row["cycles_a"]) + late_offset - reference
        if d >= DEEP_FRACTION * amplitude:
            return "deep_conflict"
        if d < LOW_FRACTION * amplitude:
            return "low"
        return "mid"

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    wall = (summary.get("ended_wall_time_ns", 0)
            - summary.get("started_wall_time_ns", 0)) / 1e9
    rate = len(rows) / wall if wall > 0 else 0.0
    say("=== S5-T1 table-build run ===")
    say(f"run {run_dir.name}: {len(rows)} queries in {wall:.0f}s "
        f"({rate:.0f}/s) at "
        f"{summary.get('rate_mhz_during_work', 'n/a')} MHz; calibration "
        f"{floor}/{baseline}/{conflict} (amp {amplitude}); late-section "
        f"offset {late_offset:+d} cyc (repeat n={len(repeat_deltas)}, "
        f"spread {repeat_deltas[0]}..{repeat_deltas[-1]})"
        if repeat_deltas else
        f"run {run_dir.name}: {len(rows)} queries; calibration "
        f"{floor}/{baseline}/{conflict}; no repeat block")

    # -- edges CSV (the build_bank_table --t1 source) ------------------
    edge_rows: list[dict[str, str]] = []
    class_counts: Counter = Counter()
    for row in sections["anchor_sweep"] + sections["classify"] \
            + sections["bank_map"]:
        pa_a, pa_b = pair_pa(row)
        label = label_of(row)
        class_counts[label] += 1
        edge_rows.append({
            "query_id": row["query_id"], "pa_a": f"0x{pa_a:x}",
            "pa_b": f"0x{pa_b:x}", "section": row["section"],
            "cycles_a": row["cycles_a"], "cycles_b": row["cycles_b"],
            "late_offset": late_offset,
            "corrected_cycles": int(row["cycles_a"]) + late_offset,
            "class": label,
        })
    with (run_dir / "table_build_edges.csv").open("w", encoding="utf-8",
                                                  newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)
    say(f"classified pairs: {len(edge_rows)} {dict(class_counts)} "
        "(deep = fresh same-bank edges, the densification feed)")

    # -- same-bank page graph (classify deep edges only) ----------------
    # No cross-page shoulder band exists (module docstring), so pages
    # union only through shared banks; a component IS a same-bank page
    # set. A cross-page low inside one is a real contradiction (same
    # bank + two distinct 2 MiB pages forces different rows -> deep).
    pages_in_run = [page.fb_pa_page_base for page in pool.pa_pages]
    bank_page_uf = UnionFind(pages_in_run)
    n_cross_deep = 0
    for row in sections["classify"]:
        pa_a, pa_b = pair_pa(row)
        if label_of(row) == "deep_conflict":
            bank_page_uf.union(pa_a & ~(page_size - 1), pa_b & ~(page_size - 1))
            n_cross_deep += 1
    components = bank_page_uf.groups()
    comp_of = {page: root for root, members in components.items()
               for page in members}
    contradictions = []
    low_cross = 0
    for row in sections["classify"]:
        pa_a, pa_b = pair_pa(row)
        if label_of(row) == "low":
            low_cross += 1
            if comp_of.get(pa_a & ~(page_size - 1)) == \
                    comp_of.get(pa_b & ~(page_size - 1)):
                contradictions.append((pa_a, pa_b))
    sizes = sorted((len(members) for members in components.values()
                    if len(members) > 1), reverse=True)
    multi = {root: members for root, members in components.items()
             if len(members) > 1}
    assigned = sum(len(members) for members in multi.values())
    n_classify = len(sections["classify"])
    say("")
    say(f"same-bank page graph (cross-page deep edges only): "
        f"{n_cross_deep} deep edges ({n_cross_deep / n_classify:.2%} of "
        f"classify pairs; random-pair prior 1/{BANK_PRIOR} = "
        f"{1 / BANK_PRIOR:.2%} for 24ch x 16bank) -> {len(multi)} "
        f"components over {assigned}/{len(pages_in_run)} pages, sizes "
        f"{sizes[:12]}{' ...' if len(sizes) > 12 else ''}")
    say("  model note: cross-page low is COMPATIBLE with same channel "
        "different bank (no shoulder band) -- the S4b-1/T0 channel "
        "components built on shoulder edges were lambda-spread artifacts")
    if low_cross:
        say(f"  cross-page lows {low_cross}, inside a same-bank component: "
            f"{len(contradictions)} ({len(contradictions) / low_cross:.1%})"
            " -- these mark bad deep edges" if contradictions else
            f"  cross-page lows {low_cross}, inside a same-bank component: "
            f"{len(contradictions)}")
    with (run_dir / "channel_partition.csv").open("w", encoding="utf-8",
                                                  newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["page_base", "component_root", "component_size"])
        for page in pages_in_run:
            root = comp_of.get(page)
            writer.writerow([f"0x{page:x}",
                             f"0x{root:x}" if root is not None else "",
                             len(components[root]) if root is not None else ""])

    # -- anchor validity matrix ----------------------------------------
    candidates = _hex_list(sel.get("anchor_candidates", []))
    validity: dict[int, set[int]] = defaultdict(set)   # page -> cand set
    cand_totals: Counter = Counter()
    valid_pages: set[int] = set()
    with (run_dir / "anchor_validity.csv").open("w", encoding="utf-8",
                                                newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["page_base", "candidate", "corrected_cycles",
                         "class"])
        for index, row in enumerate(sections["anchor_sweep"]):
            page = pool.pa_pages[index // len(candidates)].fb_pa_page_base
            cand = candidates[index % len(candidates)]
            label = label_of(row)
            writer.writerow([f"0x{page:x}", f"0x{cand:x}",
                             int(row["cycles_a"]) + late_offset, label])
            if label == "deep_conflict":
                validity[page].add(cand)
                cand_totals[cand] += 1
                valid_pages.add(page)
    depth_hist = Counter(len(v) for v in validity.values())
    say("")
    say(f"anchor validity: {len(valid_pages)}/{len(pages_in_run)} pages "
        f"have >=1 valid anchor (deep); valid-count histogram "
        f"{dict(sorted(depth_hist.items()))}; anchor-less pages are the "
        f"T1 residue (page- vs mask-specific structure below)")
    top = cand_totals.most_common(5)
    say(f"  per-candidate valid pages: "
        + ", ".join(f"0x{cand:x}={count}" for cand, count in top)
        + (" ..." if len(cand_totals) > 5 else ""))
    if len(cand_totals) >= 2:
        (c1, n1), (c2, n2) = cand_totals.most_common(2)
        both = sum(1 for page in validity if c1 in validity[page]
                   and c2 in validity[page])
        say(f"  co-occurrence of the top two: 0x{c1:x}&0x{c2:x} both "
            f"valid on {both} pages (of {n1}/{n2}) -- near-zero means "
            f"mask-specific rotation, high means page-specific")

    # -- bank map (double probe) ----------------------------------------
    lattice = _hex_list(sel.get("bank_map_offsets", []))
    anchor = int(sel.get("bank_map_anchor", "0x0"), 16)
    bank_bases = _hex_list(sel.get("bank_map_pages", []))
    by_page_pairs: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in sections["bank_map"]:
        pa_a, pa_b = pair_pa(row)
        page = pa_a & ~(page_size - 1)
        in_a, in_b = pa_a & (page_size - 1), pa_b & (page_size - 1)
        low = in_b if in_a in (0, anchor) else in_a
        role = "anchor" if in_a == anchor else "base"
        by_page_pairs[page].setdefault(low, {})[role] = row
    same_bank_sets: dict[int, set[int]] = {}
    map_rows: list[list] = []
    for page in bank_bases:
        pairs = by_page_pairs.get(page, {})
        anchor_row = pairs.get(anchor, {}).get("base")
        anchor_label = label_of(anchor_row) if anchor_row else "missing"
        writer_rows = []
        if anchor_label != "deep_conflict":
            say(f"  bank map page 0x{page:x}: anchor (0,M) is "
                f"{anchor_label}, NOT deep -> map void (the mined "
                f"validity did not reproduce this run)")
        else:
            bank: set[int] = set()
            for low in lattice:
                base = label_of(pairs[low]["base"]) if low in pairs else "missing"
                anch = (label_of(pairs[low]["anchor"])
                        if low in pairs and "anchor" in pairs[low]
                        else ("n/a" if low in (0, anchor) else "missing"))
                is_bank = base == "deep_conflict" or anch == "deep_conflict"
                if is_bank:
                    bank.add(low)
                if base == "deep_conflict" or anch == "deep_conflict":
                    row_state = ("rowM" if base != "low" and anch == "low"
                                 else "row0" if base == "low" else "other_row")
                else:
                    row_state = "-"
                writer_rows.append([f"0x{page:x}", f"0x{low:x}", base,
                                    anch, int(is_bank), row_state])
            same_bank_sets[page] = bank
            bank_list = " ".join(f"0x{low:x}" for low in sorted(bank))
            say(f"  bank map page 0x{page:x}: {len(bank)}/{len(lattice)} "
                f"lattice offsets same-bank as 0: {bank_list}")
        for row_values in writer_rows:
            map_rows.append(row_values)
    with (run_dir / "bank_map_pages.csv").open("w", encoding="utf-8",
                                               newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["page_base", "offset", "base_class",
                         "anchor_class", "same_bank", "row_state"])
        writer.writerows(map_rows)
    pages_mapped = sorted(same_bank_sets)
    if len(pages_mapped) >= 2:
        jacc = []
        for i, left in enumerate(pages_mapped):
            for right in pages_mapped[i + 1:]:
                a, b = same_bank_sets[left], same_bank_sets[right]
                jacc.append(len(a & b) / len(a | b) if a | b else 1.0)
        ordered = sorted(jacc)
        say(f"  cross-page same-bank Jaccard (mapped pages): p50 "
            f"{ordered[len(ordered) // 2]:.2f}, min {ordered[0]:.2f}, "
            f"max {ordered[-1]:.2f} -- high = one shared in-page hash, "
            f"low = the per-page seed the S4 verdict predicted")

    say("")
    say(f"outputs: table_build_edges.csv, anchor_validity.csv, "
        f"bank_map_pages.csv, channel_partition.csv in {run_dir}")
    (run_dir / "table_build_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return 0


def self_test() -> int:
    import contextlib
    import io
    import tempfile

    from g3_pool import plan_table_build_queries

    page_size = 2 * 1024 * 1024
    fields = ["run_id", "device", "gpu_uuid", "chunk_index", "allocation_id",
              "va_page_base", "va_page_end_exclusive", "fb_pa_page_base",
              "page_size", "aperture", "pte_valid", "raw_pte_lo",
              "raw_pte_hi", "mapped_at_ns", "unmapped_at_ns", "source",
              "confidence"]
    rows_map = [
        (0, 0x7F0000000000, 0x120000000),
        (0, 0x7F0000000000 + page_size, 0x120400000),
        (1, 0x7F0008000000, 0x120200000),
        (1, 0x7F0008000000 + page_size, 0x120600000),
    ]
    text = ",".join(fields) + "\n" + "".join(
        f"r,0,uuid,{chunk},alloc,0x{va:x},0x{va + page_size:x},0x{fb:x},"
        f"{page_size},VIDEO,true,0x1,0x0,1,2,ebpf,c\n"
        for chunk, va, fb in rows_map)
    pool = PoolMap(parse_pool_map(text))
    # pages in PA order: p0=0x120000000 p1=0x120200000 p2=0x120400000
    # p3=0x120600000; reps are p0 and p1. Channels: {p0, p2} and {p1, p3}.
    cands = (0xd0100, 0x2000)
    lattice = (0xd0100, 0x2000, 0x200)
    plan, meta = plan_table_build_queries(
        pool, rep_pas=(0x120000000, 0x120200000, 0x900000000),
        anchor_candidates=cands, bank_map_probes=cands,
        bank_map_pages=(0x120000000, 0x120200000), repeat_pages=4)
    # The splitter must accept the meta dict the orchestrator really
    # embeds (the off-pool rep is dropped and reported, never planned).
    assert meta["dropped"]["reps"] == ["0x900000000"]
    sel = dict(meta)

    # calibration 1015/1018/1142 -> amp 124; lambda-referenced gates:
    # low d < +43.4, deep d >= +74.4, mid between.
    page_values = {0x120000000: 1015, 0x120200000: 1025,
                   0x120400000: 1016, 0x120600000: 1041}
    # anchor sweep truth: cand0 deep at p0 (valid anchor) and low
    # elsewhere; cand1 deep at p2 only.
    sweep_deep = {(0, 0xd0100), (2, 0x2000)}
    # classify pairs in planner order: (p0,p1), (p1,p0), (p2,p0), (p2,p1),
    # (p3,p0), (p3,p1). p2 lands DEEP against BOTH reps (same bank as
    # both -- toy geometry) -> one same-bank component {p0,p1,p2}; the
    # (p0,p1) lows then fall INSIDE it: the planted contradiction (two
    # rows, both directions of the same page pair). p3 is a lone low
    # page (no bank shared with any rep).
    same_bank = {(0x120000000, 0x120400000), (0x120200000, 0x120400000)}
    c3_plant = (0x120000000, 0x120200000)

    def classify_value(page_a: int, page_b: int) -> int:
        base = max(page_values[page_a & ~(page_size - 1)],
                   page_values[page_b & ~(page_size - 1)])
        pair = (min(page_a & ~(page_size - 1), page_b & ~(page_size - 1)),
                max(page_a & ~(page_size - 1), page_b & ~(page_size - 1)))
        if pair == c3_plant:
            return base + 10            # low inside the component: plant
        if pair in same_bank:
            return base + 100           # deep: same bank, different row
        return base + 10                 # low

    result_lines = ["query_id,chunk_a,ofs_a,chunk_b,ofs_b,cycles_a,cycles_b"]
    for typed in plan:
        pa_a = pool.pa_of(typed.query.chunk_a, typed.query.ofs_a)
        pa_b = pool.pa_of(typed.query.chunk_b, typed.query.ofs_b)
        if typed.section == "calibration":
            value = (1015, 1018, 1142)[
                ("floor", "baseline", "conflict").index(typed.role)]
        elif typed.section in ("self", "repeat"):
            value = page_values[pa_a]
        elif typed.section == "anchor_sweep":
            page_idx = next(i for i, p in enumerate(pool.pa_pages)
                            if p.fb_pa_page_base == (pa_a & ~(page_size - 1)))
            value = 1140 if (page_idx, pa_b - pa_a) in sweep_deep else \
                page_values[pa_a] + 8
        elif typed.section == "classify":
            value = classify_value(pa_a, pa_b)
        else:  # bank_map
            page = pa_a & ~(page_size - 1)
            assert page == 0x120000000 or page == 0x120200000
            in_a, in_b = (pa_a & (page_size - 1), pa_b & (page_size - 1))
            low = in_b if in_a in (0, 0xd0100) else in_a
            if page == 0x120200000:
                value = page_values[page] + 50    # mid: anchor NOT deep
            elif low == 0xd0100:
                value = 1140                       # the anchor itself: deep
            elif low == 0x2000:
                value = 1015 if in_a == 0 else 1140  # base low, anchor deep
            else:                                  # 0x200 column probe
                value = 1015 if in_a == 0 else 1140
        result_lines.append(
            f"{len(result_lines) - 1},{typed.query.chunk_a},"
            f"{typed.query.ofs_a},{typed.query.chunk_b},"
            f"{typed.query.ofs_b},{value},{value}")
    rows = [dict(zip(("query_id", "chunk_a", "ofs_a", "chunk_b", "ofs_b",
                      "cycles_a", "cycles_b"), line.split(",")))
            for line in result_lines[1:]]
    sections, problems = split_sections(rows, pool, sel)
    assert problems == [], problems
    counts = {name: len(part) for name, part in sections.items()}
    assert counts == {"calibration": 3, "self": 4, "anchor_sweep": 8,
                      "classify": 6, "bank_map": 10, "repeat": 4}, counts

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "pool_map.csv").write_text(text)
        (run / "result.csv").write_text("\n".join(result_lines) + "\n")
        (run / "summary.json").write_text(json.dumps({
            "query_selection": {"mode": "table-build"},
            "table_build_selection": sel,
            "rate_mhz_during_work": 2500.0,
            "started_wall_time_ns": 0, "ended_wall_time_ns": 5_000_000_000}))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = analyze(run)
        assert code == 0
        report = buffer.getvalue()
        # same-bank page graph: deeps (p2,p0) and (p2,p1) -> {p0,p1,p2};
        # both (p0,p1) low rows land inside it -> planted contradiction
        assert "1 components over 3/4 pages" in report, report
        assert "inside a same-bank component: 2" in report, report
        # anchor validity: 2 pages valid (p0 cand0, p2 cand1)
        assert "2/4 pages" in report, report
        assert "0xd0100&0x2000 both valid on 0 pages" in report, report
        # bank map: p0 anchor deep -> 3 same-bank offsets; p1 void
        assert "bank map page 0x120000000: 3/3 lattice offsets" in report, \
            report
        assert "0x200 0x2000 0xd0100" in report, report   # sorted offsets
        assert "NOT deep -> map void" in report, report
        # deep edges: 2 sweep + 2 classify (p2 vs both reps) + 3 bank-map
        edges = list(csv.DictReader(
            (run / "table_build_edges.csv").open(encoding="utf-8")))
        assert len(edges) == 24, len(edges)
        assert Counter(e["class"] for e in edges)["deep_conflict"] == 7, \
            Counter(e["class"] for e in edges)
        bank_map = list(csv.DictReader(
            (run / "bank_map_pages.csv").open(encoding="utf-8")))
        p0_rows = [r for r in bank_map if r["page_base"] == "0x120000000"]
        assert len(p0_rows) == 3
        by_off = {r["offset"]: r for r in p0_rows}
        assert by_off["0x2000"]["same_bank"] == "1"
        assert by_off["0x2000"]["row_state"] == "row0"
        assert by_off["0xd0100"]["row_state"] == "other_row"
        # integrity: a swapped row must fail closed
        bad = list(rows)
        bad[3], bad[4] = bad[4], bad[3]
        _, bad_problems = split_sections(bad, pool, sel)
        assert bad_problems and bad_problems[0].startswith("self row"), \
            bad_problems
    print("analyze_table_build self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="g3pool table-build run directory")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir is None:
        parser.error("run_dir is required outside --self-test")
    return analyze(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
