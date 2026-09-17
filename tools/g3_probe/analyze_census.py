#!/usr/bin/env python3
"""GPU_M2D G3 S4b-1: per-page latency census + re-probe + row-pilot analyzer.

S4b-0 left three questions that pair data cannot answer (a pair's single
scalar mixes both endpoints) and one label debt (273 suspect conflicts
kept on the global gate). The census workload answers them with queries
that have ONE endpoint semantics:

  self          (p, p) per pool page -- the clean per-page lambda; the
                channel question becomes: does the per-page lambda
                distribution form ~12 discrete bands (a per-channel access
                path) or stay continuous (hash-like mixing)?
  self_second   (p+K, p+K) -- is lambda a page property or an offset
                property?
  repeat        the first self pairs again, late in the run -- drift check
  reprobe       the S4b-0 suspect conflicts, judged with clean lambdas:
                reproduced on the global gate, local shoulder (same-channel
                candidate), or not reproduced (low)
  row_pilot     at anchor-valid pages, all pairs of {0, M, probe, M|probe}:
                conflict edges are same-bank-different-row (transitive on
                the bank), low edges inside a bank group are same-row, so
                union-find recovers (bank, row) classes directly -- the
                pool-local class table that unblocks G4/G5 as a fallback
                deliverable even without a closed-form mapping.

Sections are reconstructed from summary.json's census_selection counts and
verified against the work rows' own PAs (fail-closed: any mismatch is an
integrity failure, exit 2). Outputs page_lambdas.csv, reprobe_verdicts.csv
and row_pilot_classes.csv next to the inputs.

    python3 analyze_census.py RUN_DIR [--compare-old OLD_RUN_DIR]

Exit codes: 0 ok, 2 integrity failure, 1 error (repo convention).
`--self-test` pins the section splitter, the verdict rule, the lambda
band clustering, and the pilot (bank,row) union-find on a synthetic census.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_pool import (  # noqa: E402
    CENSUS_PILOT_PROBES,
    CENSUS_SELF_SECOND_OFFSET,
    PAIR_SCAN_ANCHOR,
    PoolMap,
    parse_pool_map,
)

LOW_FRACTION = 0.35   # below local base + 0.35*amplitude -> low
HIGH_FRACTION = 0.70  # above baseline + 0.70*amplitude  -> elevated (global)
DEEP_FRACTION = 0.90  # above baseline + 0.90*amplitude  -> full conflict
SYMMETRY_FRACTION = 0.25
CLUSTER_GAP = 4       # lambda-distribution gap (cycles) that splits bands
MIN_BAND_SHARE = 1 / 30  # a "channel band" holds >= ~3% of pages


def classify_pair(value: int, lambda_max: int, hi_global: int,
                  amplitude: int) -> str:
    """The S4b-0 asymmetric rule: local low/mid boundary, global conflict
    gate. `lambda_max` is the slower endpoint's census lambda."""
    if value >= hi_global:
        return "conflict"
    if value < lambda_max + LOW_FRACTION * amplitude:
        return "low"
    return "mid"


def classify3(value: int, lambda_max: int, hi_global: int,
              deep_gate: int, amplitude: int) -> str:
    """Three-regime classifier. The census re-probes showed the old
    'conflict' class was bimodal: a reproducible shallow band at
    ~0.70-0.85*amplitude (partial penalty -- same-channel candidate) and
    the full row-conflict at ~0.95+*amplitude (the calibration anchor's
    own regime). Splitting them is the S4b-1 deliverable for the solver:
      low / mid / shoulder / deep_conflict."""
    if value >= deep_gate:
        return "deep_conflict"
    if value >= hi_global:
        return "shoulder"
    if value < lambda_max + LOW_FRACTION * amplitude:
        return "low"
    return "mid"


def gap_clusters(values: list[int], gap: int = CLUSTER_GAP) \
        -> list[tuple[int, int, int]]:
    """Split a sorted value list at gaps >= gap cycles."""
    if not values:
        return []
    clusters: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value - clusters[-1][-1] >= gap:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [(c[0], c[-1], len(c)) for c in clusters]


class UnionFind:
    def __init__(self, items: list[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return {root: sorted(members) for root, members in out.items()}


def load_summary(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("query_selection", {}).get("mode") != "census":
        raise ValueError("not a census run (summary query_selection.mode)")
    if "census_selection" not in summary:
        raise ValueError("summary lacks census_selection")
    return summary


def load_rows(run_dir: Path, name: str) -> list[dict[str, str]]:
    with (run_dir / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(line for line in source
                                   if not line.startswith("#")))


def pilot_offset_list() -> list[int]:
    """The planner's contractual pilot offsets, in order."""
    return [0, PAIR_SCAN_ANCHOR] + list(CENSUS_PILOT_PROBES) \
        + [PAIR_SCAN_ANCHOR ^ probe for probe in CENSUS_PILOT_PROBES]


def split_sections(rows: list[dict[str, str]], pool: PoolMap,
                   census_sel: dict) -> tuple[dict[str, list[dict[str, str]]],
                                              list[str]]:
    """Slice the result rows into the census sections by the counts the
    orchestrator recorded, verifying each row's shape against its own PA."""
    n_pages = len(pool.pa_pages)
    stride = int(census_sel.get("second_stride", 4))
    second_count = len(range(0, n_pages, stride)) if stride > 0 else 0
    repeat_count = min(int(census_sel.get("repeat_pages", 0)), n_pages)
    reprobe_count = int(census_sel.get("reprobe_pairs_planned", 0))
    pilot_planned = int(census_sel.get("pilot_pages_planned", 0))
    n_offsets = len(pilot_offset_list())
    pilot_count = pilot_planned * n_offsets * (n_offsets - 1) // 2
    expected = (3 + n_pages + second_count + repeat_count + reprobe_count
                + pilot_count)
    problems: list[str] = []
    if len(rows) != expected:
        problems.append(f"row count {len(rows)} != census plan {expected}")
        return {}, problems

    cursor = 0

    def take(count: int, section: str) -> list[dict[str, str]]:
        nonlocal cursor
        part = rows[cursor:cursor + count]
        for row in part:
            row["section"] = section
        cursor += count
        return part

    calibration = take(3, "calibration")
    self_rows = take(n_pages, "self")
    second_rows = take(second_count, "self_second")
    repeat_rows = take(repeat_count, "repeat")
    reprobe_rows = take(reprobe_count, "reprobe")
    pilot_rows = take(pilot_count, "row_pilot")

    def query_pas(row: dict[str, str]) -> tuple[int, int]:
        return (pool.pa_of(int(row["chunk_a"]), int(row["ofs_a"])),
                pool.pa_of(int(row["chunk_b"]), int(row["ofs_b"])))

    for index, row in enumerate(self_rows):
        pa_a, pa_b = query_pas(row)
        if pa_a != pa_b or pa_a != pool.pa_pages[index].fb_pa_page_base:
            problems.append(f"self row {row['query_id']}: pa {pa_a:#x} "
                            f"!= page base {pool.pa_pages[index].fb_pa_page_base:#x}")
    for index, row in enumerate(second_rows):
        pa_a, pa_b = query_pas(row)
        page = pool.pa_pages[index * stride]
        if pa_a != pa_b or pa_a - page.fb_pa_page_base != CENSUS_SELF_SECOND_OFFSET:
            problems.append(f"self_second row {row['query_id']}: {pa_a:#x}")
    for index, row in enumerate(repeat_rows):
        if (row["chunk_a"], row["ofs_a"], row["chunk_b"], row["ofs_b"]) != \
           (self_rows[index]["chunk_a"], self_rows[index]["ofs_a"],
                self_rows[index]["chunk_b"], self_rows[index]["ofs_b"]):
            problems.append(f"repeat row {row['query_id']} != self row {index}")
    offsets = pilot_offset_list()
    offset_index = {offset: index for index, offset in enumerate(offsets)}
    page_size = pool.pages[0].page_size
    pilot_pages = {int(base, 16) & ~(page_size - 1)
                   for base in census_sel.get("pilot_pages", [])}
    for row in reprobe_rows + pilot_rows:
        pa_a, pa_b = query_pas(row)
        if row["section"] == "pilot":
            if pa_a >> 21 != pa_b >> 21 or (pa_a & ~(page_size - 1)) not in pilot_pages:
                problems.append(f"pilot row {row['query_id']}: {pa_a:#x}")
            else:
                for pa in (pa_a, pa_b):
                    if (pa & (page_size - 1)) not in offset_index:
                        problems.append(f"pilot row {row['query_id']}: "
                                        f"offset {pa & (page_size - 1):#x}")
    sections = {"calibration": calibration, "self": self_rows,
                "self_second": second_rows, "repeat": repeat_rows,
                "reprobe": reprobe_rows, "row_pilot": pilot_rows}
    return sections, problems


def analyze_pilot(pilot_rows: list[dict[str, str]], pool: PoolMap,
                  amplitude: int, hi_global: int, lambdas: dict[int, int],
                  correction: int = 0) \
        -> tuple[dict[int, dict[int, tuple[int, int]]], list[str], list[dict[str, str]]]:
    """Per pilot page: (bank, row) classes via union-find. Only DEEP
    conflicts (>= 0.90*amplitude: full row-conflict) prove same-bank-
    different-row and may union banks -- the shallow shoulder band
    (~0.70-0.85*amplitude) is a partial-penalty regime that must not be
    treated as same-bank evidence. Low edges inside a bank group union
    rows (same row OR same bank+row ambiguity is resolved by the bank
    group). `correction` is the late-section offset for values measured
    after the self block."""
    page_size = pool.pages[0].page_size
    offsets = pilot_offset_list()
    deep_gate = hi_global + (DEEP_FRACTION - HIGH_FRACTION) * amplitude
    by_page: dict[int, list[tuple[int, int, str, dict[str, str]]]] = {}
    out_rows: list[dict[str, str]] = []
    for row in pilot_rows:
        pa_a = pool.pa_of(int(row["chunk_a"]), int(row["ofs_a"]))
        pa_b = pool.pa_of(int(row["chunk_b"]), int(row["ofs_b"]))
        page = pa_a & ~(page_size - 1)
        left = offset_index_of(pa_a & (page_size - 1))
        right = offset_index_of(pa_b & (page_size - 1))
        value = int(row["cycles_a"]) + correction
        label = classify3(value, lambdas.get(page, 0), hi_global, deep_gate,
                          amplitude)
        if abs(int(row["cycles_a"]) - int(row["cycles_b"])) \
                > SYMMETRY_FRACTION * amplitude:
            label = "asymmetric"
        by_page.setdefault(page, []).append((left, right, label, row))
        out_rows.append({
            "page": f"0x{page:x}", "offset_i": f"0x{offsets[left]:x}",
            "offset_j": f"0x{offsets[right]:x}",
            "xor": f"0x{offsets[left] ^ offsets[right]:x}",
            "cycles_a": row["cycles_a"], "cycles_b": row["cycles_b"],
            "corrected_cycles": value, "class": label})
    classes: dict[int, dict[int, tuple[int, int]]] = {}
    contradictions: list[str] = []
    for page, edges in by_page.items():
        bank_uf = UnionFind(list(range(len(offsets))))
        for left, right, label, _ in edges:
            if label == "deep_conflict":
                bank_uf.union(left, right)
        row_uf = UnionFind(list(range(len(offsets))))
        for left, right, label, _ in edges:
            if label == "low" and bank_uf.find(left) == bank_uf.find(right):
                row_uf.union(left, right)
        for left, right, label, _ in edges:
            if label == "deep_conflict" and row_uf.find(left) == row_uf.find(right):
                contradictions.append(
                    f"page 0x{page:x}: deep conflict inside a row group "
                    f"({offsets[left]:#x}, {offsets[right]:#x})")
        page_classes: dict[int, tuple[int, int]] = {}
        bank_roots = {}
        for item in range(len(offsets)):
            bank_roots.setdefault(bank_uf.find(item), len(bank_roots))
        row_roots = {}
        for item in range(len(offsets)):
            row_roots.setdefault(row_uf.find(item), len(row_roots))
        for item in range(len(offsets)):
            page_classes[item] = (bank_roots[bank_uf.find(item)],
                                  row_roots[row_uf.find(item)])
        classes[page] = page_classes
    return classes, contradictions, out_rows


def offset_index_of(in_page: int) -> int:
    return pilot_offset_list().index(in_page)


def cluster_structure(lambdas: dict[int, int], clusters: list[tuple[int, int, int]],
                      chunk_of_page: dict[int, int],
                      min_share: float = MIN_BAND_SHARE) -> list[dict[str, object]]:
    """Per lambda cluster (>= min_share of pages): is the membership a
    coarse PA structure (channel-like) or fine-grained per-page placement?
    A channel observable makes chunks homogeneous and shows bit effects;
    per-page placement mixes pages inside one 8 MiB chunk."""
    pages = sorted(lambdas)
    n = len(pages)
    out = []
    for lo, hi, count in clusters:
        if count < n * min_share:
            continue
        members = {p for p in pages if lo <= lambdas[p] <= hi}
        chunk_state: dict[int, list[bool]] = {}
        for page, chunk in chunk_of_page.items():
            chunk_state.setdefault(chunk, []).append(page in members)
        all_in = sum(1 for flags in chunk_state.values() if all(flags))
        all_out = sum(1 for flags in chunk_state.values() if not any(flags))
        mixed = len(chunk_state) - all_in - all_out
        run = best = 1
        for left, right in zip(pages, pages[1:]):
            run = run + 1 if (left in members and right in members) else 1
            best = max(best, run)
        top_bit = pages[-1].bit_length()
        bit_effects = []
        for bit in range(21, top_bit):
            ones = [p for p in pages if (p >> bit) & 1]
            zeros = [p for p in pages if not (p >> bit) & 1]
            if len(ones) < 50 or len(zeros) < 50:
                continue
            share1 = sum(p in members for p in ones) / len(ones)
            share0 = sum(p in members for p in zeros) / len(zeros)
            bit_effects.append((abs(share1 - share0), bit, share1, share0))
        bit_effects.sort(reverse=True)
        out.append({"range": (lo, hi), "pages": count,
                    "share": count / n, "chunks_all_in": all_in,
                    "chunks_all_out": all_out, "chunks_mixed": mixed,
                    "max_adjacent_run": best,
                    "top_bit_effects": bit_effects[:3]})
    return out


def compare_lambdas(new_lambdas: dict[int, int], old_lambdas: dict[int, int],
                    new_rate: float, old_rate: float) -> dict[str, float]:
    """Cross-run lambda comparison, ns-normalized (unlocked clocks make
    raw cycles incomparable; ns is still only indicative)."""
    common = sorted(set(new_lambdas) & set(old_lambdas))
    if not common:
        return {"common_pages": 0}
    new_ns = [new_lambdas[p] / (new_rate / 1000.0) for p in common]
    old_ns = [old_lambdas[p] / (old_rate / 1000.0) for p in common]
    deltas = [n - o for n, o in zip(new_ns, old_ns)]
    mean_new = sum(new_ns) / len(new_ns)
    mean_old = sum(old_ns) / len(old_ns)
    cov = sum((n - mean_new) * (o - mean_old) for n, o in zip(new_ns, old_ns))
    var_new = sum((n - mean_new) ** 2 for n in new_ns)
    var_old = sum((o - mean_old) ** 2 for o in old_ns)
    pearson = cov / (var_new * var_old) ** 0.5 if var_new and var_old else 0.0
    ordered = sorted(deltas)
    return {"common_pages": len(common),
            "median_delta_ns": ordered[len(ordered) // 2],
            "mean_abs_delta_ns": sum(abs(d) for d in deltas) / len(deltas),
            "new_le_old_share": sum(d <= 0 for d in deltas) / len(deltas),
            "pearson_r": pearson}


def analyze(run_dir: Path, compare_old: Path | None) -> int:
    summary = load_summary(run_dir)
    pool = PoolMap(parse_pool_map((run_dir / "pool_map.csv").read_text(
        encoding="utf-8")))
    rows = load_rows(run_dir, "result.csv")
    census_sel = summary["census_selection"]
    sections, problems = split_sections(rows, pool, census_sel)
    if problems:
        print(f"INTEGRITY: {'; '.join(problems[:5])}")
        return 2

    by_id = {int(row["query_id"]): row for row in rows}
    floor = int(by_id[0]["cycles_a"])
    baseline = int(by_id[1]["cycles_a"])
    conflict = int(by_id[2]["cycles_a"])
    amplitude = conflict - baseline
    hi_global = baseline + HIGH_FRACTION * amplitude
    if amplitude <= 0:
        print(f"INTEGRITY: degenerate calibration {floor}/{baseline}/{conflict}")
        return 2

    page_size = pool.pages[0].page_size
    lambdas = {pool.pa_pages[index].fb_pa_page_base: int(row["cycles_a"])
               for index, row in enumerate(sections["self"])}

    # page_lambdas.csv
    second_by_page = {}
    for index, row in enumerate(sections["self_second"]):
        page = pool.pa_pages[index * int(census_sel.get("second_stride", 4))]
        second_by_page[page.fb_pa_page_base] = int(row["cycles_a"])
    repeat_by_page = {}
    for index, row in enumerate(sections["repeat"]):
        page = pool.pa_pages[index]
        repeat_by_page[page.fb_pa_page_base] = int(row["cycles_a"])
    with (run_dir / "page_lambdas.csv").open("w", encoding="utf-8",
                                             newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["fb_pa_page_base", "self_cycles", "second_cycles",
                         "repeat_cycles"])
        for base, value in sorted(lambdas.items()):
            writer.writerow([f"0x{base:x}", value,
                             second_by_page.get(base, ""),
                             repeat_by_page.get(base, "")])

    values = sorted(lambdas.values())
    clusters = gap_clusters(values)
    band_clusters = [c for c in clusters if c[2] >= len(values) * MIN_BAND_SHARE]
    print(f"census run: {run_dir.name}")
    print(f"calibration {floor}/{baseline}/{conflict} (amplitude {amplitude}), "
          f"{len(values)} pages, rate "
          f"{summary.get('rate_mhz_during_work', 'n/a')} MHz")
    print(f"\nper-page lambda (self-pair): range {values[0]}-{values[-1]}, "
          f"p50 {values[len(values) // 2]}, "
          f"p95 {values[int(len(values) * 0.95)]}, "
          f"IQR {values[len(values) // 4]}-{values[(len(values)) * 3 // 4]}")
    print(f"gap-{CLUSTER_GAP} clusters: {len(clusters)} "
          f"({', '.join(f'{lo}-{hi}x{n}' for lo, hi, n in clusters[:12])}"
          f"{' ...' if len(clusters) > 12 else ''})")
    if len(band_clusters) >= 8:
        print(f"BAND TEST: {len(band_clusters)} clusters hold >= "
              f"{MIN_BAND_SHARE:.0%} of pages each -- consistent with a "
              f"discrete channel-band structure")
    else:
        print(f"BAND TEST: only {len(band_clusters)} cluster(s) clear the "
              f"{MIN_BAND_SHARE:.0%} share bar at gap-{CLUSTER_GAP} -- no "
              f"discrete ~12-band structure resolvable; the lambda "
              f"distribution is continuous/hash-like, matching S4b-0")

    # Cluster structure: channel-like (coarse, chunk-homogeneous, bit
    # effects) or fine-grained per-page placement?
    chunk_of_page = {page.fb_pa_page_base: page.chunk_index
                     for page in pool.pages}
    for entry in cluster_structure(lambdas, clusters, chunk_of_page):
        lo, hi = entry["range"]
        bits = ", ".join(f"bit {bit} |d|={delta:.2f}"
                         for delta, bit, _, _ in entry["top_bit_effects"])
        print(f"  cluster {lo}-{hi}: {entry['pages']} pages "
              f"({entry['share']:.0%}); chunks {entry['chunks_all_in']} "
              f"all-in / {entry['chunks_mixed']} mixed / "
              f"{entry['chunks_all_out']} all-out; max adjacent run "
              f"{entry['max_adjacent_run']}; bit effects: {bits or 'none'}")
        coarse = (entry["chunks_mixed"] <= 0.2 * len(chunk_of_page)
                  and entry["top_bit_effects"]
                  and entry["top_bit_effects"][0][0] >= 0.25)
        print(f"    -> {'coarse PA structure (channel-like)' if coarse else 'fine-grained per-page placement: mixed chunks / no dominant bit -- NOT a channel observable'}")

    # Late-section offset: everything measured after the self block
    # (self_second, repeat, reprobe, row_pilot) reads a few cycles below
    # the same pages inside it. The self block's own position-bucket
    # medians separate a step from a ramp; the repeat block measures the
    # step and anchors an additive correction back into the calibration
    # frame.
    self_by_pos = [int(row["cycles_a"]) for row in sections["self"]]

    def _median(items: list[int]) -> int:
        items = sorted(items)
        return items[len(items) // 2] if items else 0

    if len(self_by_pos) >= 8:
        bucket_medians = [
            _median(self_by_pos[b * len(self_by_pos) // 8:
                                (b + 1) * len(self_by_pos) // 8])
            for b in range(8)]
    else:
        bucket_medians = [_median(self_by_pos)]
    flat = max(bucket_medians) - min(bucket_medians) <= 6
    if repeat_by_page:
        late_deltas = sorted(repeat_by_page[p] - lambdas[p]
                             for p in repeat_by_page)
        late_offset = -late_deltas[len(late_deltas) // 2]
        print(f"\nlate-section offset (repeat block n={len(late_deltas)}): "
              f"{late_offset:+d} cyc (spread {late_deltas[0]}.."
              f"{late_deltas[-1]}); self-block bucket medians "
              f"{bucket_medians[0]}..{bucket_medians[-1]} "
              f"({'flat -> step, not ramp' if flat else 'NOT flat -- treat the offset as approximate'})")
        print(f"  applied additively to self_second / reprobe / row_pilot "
              f"values (they are measured after the step)")
    else:
        late_offset = 0
        print("\nlate-section offset: no repeat block; no correction applied")
    if second_by_page:
        deltas = sorted(second_by_page[p] + late_offset - lambdas[p]
                        for p in second_by_page)
        verdict = ("page-stable" if abs(deltas[len(deltas) // 2]) <= 4
                   and abs(deltas[-1]) <= 16 else "offset-dependent")
        print(f"second-offset self-pairs (n={len(deltas)}, 1 MiB into the "
              f"page), corrected: delta vs page start median "
              f"{deltas[len(deltas) // 2]}, range {deltas[0]}..{deltas[-1]} "
              f"-- lambda is {verdict}")

    if compare_old is not None:
        import analyze_local_recal
        old_stats, old_problems = analyze_local_recal.analyze(compare_old, None,
                                                              None)
        if old_problems:
            print(f"compare-old: integrity failure in old run "
                  f"({old_problems[0]}); skipping comparison")
        else:
            # The old run is an S3/S3b run, not a census: read its summary
            # directly for the clock rate only.
            old_summary = json.loads((compare_old / "summary.json")
                                     .read_text(encoding="utf-8"))
            old_rate = float(old_summary.get("rate_mhz_during_work") or 1.0)
            new_rate = float(summary.get("rate_mhz_during_work") or 1.0)
            stats = compare_lambdas(
                {base >> 21: value for base, value in lambdas.items()},
                old_stats["lambdas"], new_rate, old_rate)
            if stats["common_pages"]:
                print(f"\nvs S4b-0 pair-lambda ({compare_old.name}): "
                      f"{stats['common_pages']} common pages, median delta "
                      f"{stats['median_delta_ns']:.1f} ns, mean |delta| "
                      f"{stats['mean_abs_delta_ns']:.1f} ns, census<=old "
                      f"{stats['new_le_old_share']:.0%}, r="
                      f"{stats['pearson_r']:.2f}")

    # reprobe verdicts (late-section corrected, three-regime)
    deep_gate = baseline + DEEP_FRACTION * amplitude
    verdict_rows: list[dict[str, str]] = []
    verdict_counts: Counter = Counter()
    corrected_values: list[int] = []
    for row in sections["reprobe"]:
        pa_a = pool.pa_of(int(row["chunk_a"]), int(row["ofs_a"]))
        pa_b = pool.pa_of(int(row["chunk_b"]), int(row["ofs_b"]))
        lambda_a = lambdas.get(pa_a & ~(page_size - 1))
        lambda_b = lambdas.get(pa_b & ~(page_size - 1))
        corrected = int(row["cycles_a"]) + late_offset
        if lambda_a is None or lambda_b is None:
            verdict = "page_uncovered"
            lambda_max = ""
        elif abs(int(row["cycles_a"]) - int(row["cycles_b"])) \
                > SYMMETRY_FRACTION * amplitude:
            verdict = "asymmetric"
        else:
            lambda_max = max(lambda_a, lambda_b)
            verdict = classify3(corrected, lambda_max, hi_global, deep_gate,
                                amplitude)
            corrected_values.append(corrected)
        verdict_counts[verdict] += 1
        verdict_rows.append({
            "query_id": row["query_id"], "pa_a": f"0x{pa_a:x}",
            "pa_b": f"0x{pa_b:x}", "cycles_a": row["cycles_a"],
            "cycles_b": row["cycles_b"], "late_offset": late_offset,
            "corrected_cycles": corrected, "verdict": verdict,
            "lambda_a": lambda_a if lambda_a is not None else "",
            "lambda_b": lambda_b if lambda_b is not None else "",
            "local_low_edge": (f"{lambda_max + LOW_FRACTION * amplitude:.0f}"
                               if lambda_max != "" else ""),
            "local_conflict_edge": (f"{lambda_max + HIGH_FRACTION * amplitude:.0f}"
                                    if lambda_max != "" else "")})
    if verdict_rows:
        with (run_dir / "reprobe_verdicts.csv").open("w", encoding="utf-8",
                                                     newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(verdict_rows[0]))
            writer.writeheader()
            writer.writerows(verdict_rows)
    print(f"\nsuspect-conflict re-probes (corrected by {late_offset:+d}): "
          f"{dict(verdict_counts)}")
    if corrected_values:
        ordered = sorted(corrected_values)
        print(f"  corrected value band: p10 {ordered[int(len(ordered) * 0.1)]}, "
              f"p50 {ordered[len(ordered) // 2]}, "
              f"p90 {ordered[int(len(ordered) * 0.9)]} "
              f"(shoulder gate {hi_global:.0f}, deep gate {deep_gate:.0f}, "
              f"calibration conflict {conflict})")
    if verdict_counts.get("shoulder"):
        print(f"  {verdict_counts['shoulder']} land in the reproducible "
              f"SHALLOW band (~0.70-0.85 amplitude: partial penalty, "
              f"same-channel candidates -- the old conflict label conflated "
              f"them with full row conflicts)")
    if verdict_counts.get("low"):
        print(f"  {verdict_counts['low']} at low even after correction -- "
              f"those global-gate labels were not reproducible")

    # row pilot
    if sections["row_pilot"]:
        classes, contradictions, pilot_out = analyze_pilot(
            sections["row_pilot"], pool, amplitude, hi_global, lambdas,
            correction=late_offset)
        with (run_dir / "row_pilot_classes.csv").open("w", encoding="utf-8",
                                                      newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(pilot_out[0]))
            writer.writeheader()
            writer.writerows(pilot_out)
        print(f"\nrow pilot: {len(classes)} anchor-valid page(s), "
              f"{len(pilot_out)} pairs, {len(contradictions)} "
              f"contradiction(s)")
        offsets = pilot_offset_list()
        for page, page_classes in sorted(classes.items()):
            anchor_ok = any(
                row["class"] == "deep_conflict" for row in pilot_out
                if row["page"] == f"0x{page:x}"
                and row["offset_i"] == "0x0"
                and row["offset_j"] == f"0x{PAIR_SCAN_ANCHOR:x}")
            groups: dict[tuple[int, int], list[str]] = {}
            for item, cr in page_classes.items():
                groups.setdefault(cr, []).append(f"0x{offsets[item]:x}")
            print(f"  page 0x{page:x}: anchor(0,M) deep={anchor_ok}; "
                  f"{len(set(cr[0] for cr in page_classes.values()))} bank "
                  f"group(s), {len(set(cr[1] for cr in page_classes.values()))} "
                  f"row group(s)")
            for cr in sorted(groups, key=lambda cr: (cr[0], cr[1])):
                members = groups[cr]
                if len(members) > 1 or cr[0] == 0:
                    print(f"    bank {cr[0]} row {cr[1]}: {' '.join(members)}")
        for line in contradictions[:5]:
            print(f"  CONTRADICTION: {line}")
    print(f"\noutputs: page_lambdas.csv, reprobe_verdicts.csv, "
          f"row_pilot_classes.csv in {run_dir}")
    return 0


def self_test() -> int:
    # Verdict rule bands (amp 124, global gate 1104.8).
    assert classify_pair(1120, 1015, 1105, 124) == "conflict"
    assert classify_pair(1090, 1041, 1105, 124) == "mid"      # local edge 1084
    assert classify_pair(1030, 1040, 1105, 124) == "low"      # local edge 1083
    # Three-regime bands (amp 124, shoulder gate 1104.8, deep gate 1129.6).
    assert classify3(1135, 1015, 1105, 1130, 124) == "deep_conflict"
    assert classify3(1110, 1015, 1105, 1130, 124) == "shoulder"
    assert classify3(1090, 1041, 1105, 1130, 124) == "mid"
    assert classify3(1020, 1015, 1105, 1130, 124) == "low"
    # Band clustering.
    assert gap_clusters([1, 2, 3, 9, 10, 20]) == [(1, 3, 3), (9, 10, 2),
                                                  (20, 20, 1)]

    # Lambda comparison math on a fabricated overlap.
    stats = compare_lambdas({0: 1015, 1: 1040, 2: 1016, 3: 1041},
                            {0: 1018, 1: 1044, 2: 1020, 3: 1046},
                            2500.0, 2500.0)
    assert stats["common_pages"] == 4
    assert stats["new_le_old_share"] == 1.0     # census lambda <= pair bound
    assert stats["pearson_r"] > 0.95            # strongly rank-correlated

    # Union-find groups.
    uf = UnionFind([0, 1, 2, 3])
    uf.union(0, 1)
    uf.union(1, 2)
    assert sorted(uf.groups().values())[0] == [0, 1, 2]

    # End-to-end on a synthetic census run: the plan is generated by the
    # real planner, values hand-assigned per section.
    import tempfile
    from g3_pool import plan_census_queries
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
    reprobe = [(0x120001234, 0x120401234),  # fast pages: 1135 -> deep conflict
               (0x120201234, 0x120601234),  # slow pages: 1090 -> mid
               (0x120001238, 0x120201238)]  # fast/slow:  1030 -> low
    plan, dropped = plan_census_queries(pool, second_stride=2, repeat_pages=2,
                                        reprobe_pairs=reprobe,
                                        pilot_pages=(0x120000000,))
    assert dropped == {"reprobe": [], "row_pilot": []}
    # Per-section values. calibration 1015/1018/1142 (amp 124, shoulder gate
    # 1104.8, deep gate 1129.6, local low edge = max(lambda) + 43.4).
    page_values = {0x120000000: 1015, 0x120200000: 1040,
                   0x120400000: 1016, 0x120600000: 1041}
    second_values = {0x120000000: 1016, 0x120400000: 1017}
    repeat_values = {0x120000000: 1016, 0x120200000: 1041}
    reprobe_values = [1135, 1090, 1030]
    # Pilot truth: bank = bit 12, row differs iff xor touches the anchor
    # mask M. Conflicts are exactly {0,0x200,0x800} x {M,M|0x200,M|0x800};
    # one non-anchor pair is given a shoulder value to test exclusion.
    def toy_label(xor: int) -> str:
        if xor & (1 << 12):
            return "low"                    # bank left -> no conflict
        if xor & PAIR_SCAN_ANCHOR:
            return "deep_conflict"
        return "low"
    result_lines = ["query_id,chunk_a,ofs_a,chunk_b,ofs_b,cycles_a,cycles_b"]
    values_by_section: dict[str, list[int]] = {}
    for section in ("calibration", "self", "self_second", "repeat",
                    "reprobe", "row_pilot"):
        values_by_section[section] = []
    for typed in plan:
        section = typed.section
        if section == "calibration":
            value = (1015, 1018, 1142)[("floor", "baseline", "conflict")
                                        .index(typed.role)]
        elif section == "self":
            page = pool.pa_pages[typed.sample_index]
            value = page_values[page.fb_pa_page_base]
        elif section == "self_second":
            page = pool.pa_pages[typed.sample_index]
            value = second_values[page.fb_pa_page_base]
        elif section == "repeat":
            page = pool.pa_pages[typed.sample_index]
            value = repeat_values[page.fb_pa_page_base]
        elif section == "reprobe":
            value = reprobe_values[typed.base_index]
        else:
            ofs_a = typed.query.ofs_a & (page_size - 1)
            ofs_b = typed.query.ofs_b & (page_size - 1)
            xor = ofs_a ^ ofs_b
            value = {"deep_conflict": 1140, "low": 1018, "mid": 1070}[toy_label(xor)]
            if xor == 0x20000 ^ 0x100000:
                value = 1070
        values_by_section[section].append(value)
        result_lines.append(
            f"{len(result_lines) - 1},{typed.query.chunk_a},{typed.query.ofs_a},"
            f"{typed.query.chunk_b},{typed.query.ofs_b},{value},{value}")
    offsets = pilot_offset_list()
    # The planted shoulder must be inside the pilot block (else the toy
    # assertion below is vacuous).
    assert any(t.query.ofs_a & (page_size - 1) == 0x20000
               and t.query.ofs_b & (page_size - 1) == 0x100000
               for t in plan if t.section == "row_pilot")
    census_sel = {
        "second_stride": 2, "repeat_pages": 2,
        "reprobe_pairs_planned": 3, "pilot_pages_planned": 1,
        "pilot_pages": ["0x120000000"], "dropped": dropped,
    }
    summary = {"query_selection": {"mode": "census", "distinct_fb_pages": 4},
               "census_selection": census_sel, "rate_mhz_during_work": 2500.0}
    rows = [dict(zip(("query_id", "chunk_a", "ofs_a", "chunk_b", "ofs_b",
                      "cycles_a", "cycles_b"), line.split(",")))
            for line in result_lines[1:]]
    sections, problems = split_sections(rows, pool, census_sel)
    assert problems == [], problems
    assert [len(sections[s]) for s in
            ("calibration", "self", "self_second", "repeat", "reprobe",
             "row_pilot")] == [3, 4, 2, 2, 3, 66]
    assert values_by_section["calibration"] == [1015, 1018, 1142]

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "pool_map.csv").write_text(text)
        (run / "result.csv").write_text("\n".join(result_lines) + "\n")
        (run / "summary.json").write_text(json.dumps(summary))
        # Re-probe verdicts through the full analyze() path.
        import io as _io
        import contextlib
        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = analyze(run, None)
        assert code == 0
        report = buffer.getvalue()
        assert "BAND TEST" in report
        assert "gap-4 clusters: 2" in report
        verdicts = list(csv.DictReader(
            (run / "reprobe_verdicts.csv").open(encoding="utf-8")))
        assert [v["verdict"] for v in verdicts] == \
            ["deep_conflict", "mid", "low"], verdicts
        pilot = list(csv.DictReader(
            (run / "row_pilot_classes.csv").open(encoding="utf-8")))
        anchor = [r for r in pilot if r["offset_i"] == "0x0"
                  and r["offset_j"] == f"0x{PAIR_SCAN_ANCHOR:x}"]
        assert anchor and anchor[0]["class"] == "deep_conflict"
        shoulder = [r for r in pilot if r["offset_i"] == "0x20000"
                    and r["offset_j"] == "0x100000"]
        assert shoulder and shoulder[0]["class"] == "mid"
        # Toy structure recovered: bank = bit12 gives one 10-offset bank
        # (all bit12=0 offsets, joined by the M-side conflicts) split into
        # exactly two row groups (xor touches M), plus the bit12 bank with
        # 0x1000/M^0x1000. The shoulder pair stays excluded from grouping.
        assert "bank 0 row 0: 0x0 0x200 0x800 0x20000 0x100000" in report
        assert "bank 0 row 1: 0xd0100 0xd0300 0xd0900 0xf0100 0x1d0100" \
            in report
        assert "2 bank group(s), 4 row group(s)" in report
        assert "0 contradiction(s)" in report
        # Splitter integrity: a swapped row must be caught.
        bad = list(rows)
        bad[3], bad[4] = bad[4], bad[3]
        _, bad_problems = split_sections(bad, pool, census_sel)
        assert bad_problems and "self row" in bad_problems[0]
    print("analyze_census self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="g3pool census run directory")
    parser.add_argument("--compare-old", type=Path, default=None,
                        help="an S3/S3b run dir: compare census lambdas "
                             "against its S4b-0 pair-lambda estimates")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir is None:
        parser.error("run_dir is required outside --self-test")
    return analyze(args.run_dir, args.compare_old)


if __name__ == "__main__":
    sys.exit(main())
