#!/usr/bin/env python3
"""GPU_M2D G3 pool query selector.

Turns a PA-annotated pool map (pool_map.csv written by the S2 orchestrator:
one row per GMMU page of every pool chunk, with the observed framebuffer
PA base of that page) into work CSVs for the pool harness.

This is the S2/S3 bridge: because G2 proved in-page offsets translate 1:1
into framebuffer PA, a query address is fully described by
(page, in-page offset), and its PA is

    fb_pa = fb_pa_page_base(page) + (va - va_page_base(page)).

PA bits [0:21) are steerable by the in-page offset, bits 21+ by choosing
pages -- so "measure the pair whose PAs differ in exactly one bit" becomes
a pure selection problem over the observed page table, before any timing
runs.

Modes:
  sanity        S2 smoke workload: the S1-verified in-page floor/baseline/
                conflict triple plus N cross-page pairs spread over the
                observed PA range (informational outcomes).
  single-bit    S3 workload: for a target PA bit, pairs of addresses whose
                PAs differ in exactly that bit.
  bit-scan      S3 collection workload: the calibration triple, then one
                single-bit pair per PA bit -- in-page bits from several
                base pages (nonlinear-hash detection), page-level bits
                spread over the pool. Classifying each pair's timing
                (analyze_bit_scan.py) decomposes PA bits into bank-hash /
                row / column roles.
  pair-scan     S3b workload: anchored and two-bit probes around the S1
                conflict anchor (a same-bank different-row in-page mask),
                which separates column bits from bank-hash bits that the
                single-bit scan cannot tell apart (analyze_pair_scan.py).
  census        S4b-1 workload: per-page self-pairs (a clean per-page
                lambda), suspect-conflict re-probes, and a row-class
                pilot at anchor-valid pages (analyze_census.py).
  table-build   S5-T1 workload: anchor-expansion collection that
                densifies the seed table -- per-page self lambdas, a
                per-page anchor-candidate sweep, every page classified
                against one representative page per seed channel
                component, and a double-probe bank/row map at
                anchor-valid pages (analyze_table_build.py).

Run with --self-test to pin the arithmetic on a synthetic pool map.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

POOL_MAP_FIELDS = [
    "run_id", "device", "gpu_uuid", "chunk_index", "allocation_id",
    "va_page_base", "va_page_end_exclusive", "fb_pa_page_base", "page_size",
    "aperture", "pte_valid", "raw_pte_lo", "raw_pte_hi",
    "mapped_at_ns", "unmapped_at_ns", "source", "confidence",
]
WORK_HEADER = "query_id,chunk_a,ofs_a,chunk_b,ofs_b"

# S1-verified in-page pairs on GPU 0 (cycles at 2520 MHz): same-address
# floor ~1021, different-bank ~1014, row-conflict ~1145.
S1_IN_PAGE_CANDIDATES = (0, 8192, 852224)


@dataclass(frozen=True)
class Page:
    chunk_index: int
    va_page_base: int
    va_page_end: int
    fb_pa_page_base: int
    page_size: int


@dataclass(frozen=True)
class Query:
    chunk_a: int
    ofs_a: int
    chunk_b: int
    ofs_b: int


@dataclass(frozen=True)
class TypedQuery:
    """One pair-scan query plus its role in the S3b matrix."""
    query: Query
    section: str  # calibration | anchor_base | anchored_bit | pair |
    #             # anchor_sweep | page_triple
    bit: int | None = None
    bit2: int | None = None
    base_index: int | None = None
    sample_index: int | None = None
    role: str | None = None  # page_triple: anchor | single | anchored


def parse_int(value: str) -> int:
    return int(value, 0) if value.startswith(("0x", "0X")) else int(value)


def parse_pool_map(text: str) -> list[Page]:
    """Parses pool_map.csv content into Page records (ints resolved)."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != POOL_MAP_FIELDS:
        raise ValueError(f"unexpected pool map header: {reader.fieldnames}")
    pages: list[Page] = []
    for row in reader:
        page = Page(
            chunk_index=int(row["chunk_index"]),
            va_page_base=parse_int(row["va_page_base"]),
            va_page_end=parse_int(row["va_page_end_exclusive"]),
            fb_pa_page_base=parse_int(row["fb_pa_page_base"]),
            page_size=parse_int(row["page_size"]),
        )
        if page.va_page_end != page.va_page_base + page.page_size or page.page_size <= 0:
            raise ValueError(f"corrupt page row: {page}")
        if row["pte_valid"] != "true" or row["aperture"] != "VIDEO":
            raise ValueError(f"non-local page in pool map: {row}")
        pages.append(page)
    if not pages:
        raise ValueError("pool map is empty")
    return pages


class PoolMap:
    """Chunk-grouped view of the pool pages with PA arithmetic."""

    def __init__(self, pages: list[Page]):
        self.pages = sorted(pages, key=lambda page: (page.chunk_index, page.va_page_base))
        self.chunk_pages: dict[int, list[Page]] = {}
        for page in self.pages:
            self.chunk_pages.setdefault(page.chunk_index, []).append(page)
        for chunk_index, chunk in self.chunk_pages.items():
            cursor = chunk[0].va_page_base
            for page in chunk:
                if page.va_page_base != cursor:
                    raise ValueError(f"chunk {chunk_index} pages do not tile VA")
                cursor = page.va_page_end
        # Distinct observed PA pages (edge pages shared by two chunks
        # appear once per chunk; selection dedupes on fb_pa).
        self.pa_pages: list[Page] = sorted(
            {page.fb_pa_page_base: page for page in self.pages}.values(),
            key=lambda page: page.fb_pa_page_base)

    def page_for_va(self, chunk_index: int, va: int) -> Page:
        for page in self.chunk_pages[chunk_index]:
            if page.va_page_base <= va < page.va_page_end:
                return page
        raise ValueError(f"va 0x{va:x} not covered by chunk {chunk_index}")

    def chunk_base_va(self, chunk_index: int) -> int:
        return self.chunk_pages[chunk_index][0].va_page_base

    def pa_of(self, chunk_index: int, offset: int) -> int:
        """Framebuffer PA of chunk_base_va(chunk) + offset (G2: in-page
        offsets translate 1:1)."""
        va = self.chunk_base_va(chunk_index) + offset
        page = self.page_for_va(chunk_index, va)
        return page.fb_pa_page_base + (va - page.va_page_base)

    def query_pa(self, query: Query) -> tuple[int, int]:
        return self.pa_of(query.chunk_a, query.ofs_a), self.pa_of(query.chunk_b, query.ofs_b)


def page_start_query(page: Page, pool: PoolMap) -> Query:
    """Query addressing the first byte of one PA page."""
    offset = page.va_page_base - pool.chunk_base_va(page.chunk_index)
    return Query(page.chunk_index, offset, page.chunk_index, offset)


def select_sanity_queries(pool: PoolMap, cross_page: int = 8) -> list[Query]:
    """S2 smoke workload.

    The three in-page candidates reproduce the S1-calibrated regimes
    inside one observed page; the cross-page pairs probe one page each,
    spread evenly across the observed PA range.
    """
    first = pool.pa_pages[0]
    chunk = first.chunk_index
    offset = first.va_page_base - pool.chunk_base_va(chunk)
    queries: list[Query] = []
    for candidate in S1_IN_PAGE_CANDIDATES:
        in_page = candidate % first.page_size
        queries.append(Query(chunk, offset, chunk, offset + in_page))
    if cross_page > 0:
        sampled = [pool.pa_pages[0]]
        if cross_page > 1 and len(pool.pa_pages) > 1:
            step = (len(pool.pa_pages) - 1) / (cross_page - 1)
            indexes = {round(i * step) for i in range(cross_page)}
            sampled = [pool.pa_pages[index] for index in sorted(indexes)]
        anchor = page_start_query(sampled[0], pool)
        for page in sampled[1:]:
            queries.append(Query(anchor.chunk_a, anchor.ofs_a,
                                 page.chunk_index,
                                 page.va_page_base - pool.chunk_base_va(page.chunk_index)))
    return queries


def select_single_bit_pairs(pool: PoolMap, bit: int, limit: int = 64) -> list[Query]:
    """S3 workload: pairs of addresses whose PAs differ in exactly `bit`.

    Bits at or above the page-size LSB are steered by pairing whole pages
    that differ in exactly that PA bit; lower bits are steerable inside a
    single page by offset arithmetic.
    """
    mask = 1 << bit
    queries: list[Query] = []
    page_size = pool.pages[0].page_size
    if bit >= page_size.bit_length() - 1:
        by_pa = {page.fb_pa_page_base: page for page in pool.pa_pages}
        for base_pa, page in sorted(by_pa.items()):
            partner_pa = base_pa ^ mask
            if partner_pa < base_pa:
                continue  # emit each unordered pair once
            partner = by_pa.get(partner_pa)
            if partner is None:
                continue
            left = page_start_query(page, pool)
            right = page_start_query(partner, pool)
            queries.append(Query(left.chunk_a, left.ofs_a, right.chunk_a, right.ofs_a))
            if len(queries) >= limit:
                break
    else:
        # In-page: PA low bits equal the offset low bits (untranslated).
        first = pool.pa_pages[0]
        chunk = first.chunk_index
        offset = first.va_page_base - pool.chunk_base_va(chunk)
        for low in range(0, first.page_size - mask, 256):
            if (low ^ mask) >= first.page_size or low & mask:
                continue
            queries.append(Query(chunk, offset + low, chunk, offset + (low ^ mask)))
            if len(queries) >= limit:
                break
    return queries


def _base_pages(pool: PoolMap, count: int) -> list[Page]:
    """`count` base pages spread evenly over the pool's PA pages."""
    if count <= 1:
        return [pool.pa_pages[0]]
    return [pool.pa_pages[min(j * (len(pool.pa_pages) - 1) // (count - 1),
                              len(pool.pa_pages) - 1)]
            for j in range(count)]


def _page_partners(by_pa: dict[int, Page], mask: int, limit: int) -> list[Page]:
    """Up to `limit` left pages whose XOR-`mask` partner is also pooled,
    evenly spread over the available unordered pairs."""
    partners = [pa for pa in sorted(by_pa)
                if (pa ^ mask) in by_pa and pa < (pa ^ mask)]
    if not partners:
        return []
    stride = max(1, len(partners) // limit)
    return [by_pa[pa] for pa in partners[::stride][:limit]]


def select_bit_scan_queries(pool: PoolMap, in_page_bases: int = 4,
                            pairs_per_bit: int = 64) -> list[Query]:
    """S3 collection workload.

    Query order is contractual for analyze_bit_scan.py: ids 0..2 are the
    calibration triple (floor / different-bank baseline / in-page
    row-conflict) on the first PA page, then one single-bit pair per PA
    bit. In-page bits [0, log2(page_size)) are probed from several base
    pages (with a linear bank hash the class is base-independent; votes
    across bases expose nonlinear hashing). Page-level bits from the page
    shift up to the pool's top PA get up to `pairs_per_bit` pairs each,
    evenly spread over the available XOR partners.
    """
    page_size = pool.pages[0].page_size

    def page_offset(page: Page, low: int) -> tuple[int, int]:
        return page.chunk_index, \
            page.va_page_base - pool.chunk_base_va(page.chunk_index) + low

    queries: list[Query] = []
    first = pool.pa_pages[0]
    for candidate in S1_IN_PAGE_CANDIDATES:
        chunk, offset = page_offset(first, 0)
        queries.append(Query(chunk, offset, chunk, offset + candidate))

    for bit in range(page_size.bit_length() - 1):
        mask = 1 << bit
        if mask >= page_size:
            continue
        for page in _base_pages(pool, in_page_bases):
            chunk, offset = page_offset(page, 0)
            queries.append(Query(chunk, offset, chunk, offset + mask))

    by_pa = {page.fb_pa_page_base: page for page in pool.pa_pages}
    top_bit = max(by_pa).bit_length() - 1
    for bit in range(page_size.bit_length() - 1, top_bit + 1):
        for left in _page_partners(by_pa, 1 << bit, pairs_per_bit):
            right = by_pa[left.fb_pa_page_base ^ (1 << bit)]
            queries.append(Query(page_offset(left, 0)[0],
                                 page_offset(left, 0)[1],
                                 page_offset(right, 0)[0],
                                 page_offset(right, 0)[1]))
    return queries


# S3b anchor: the S1-verified in-page row-conflict offset (0xd0100).
# Pairs differing by exactly this mask are same-bank different-row, so
# XOR-ing it into any pair makes the row differ unconditionally.
PAIR_SCAN_ANCHOR = S1_IN_PAGE_CANDIDATES[2]


def plan_pair_scan_queries(pool: PoolMap, in_page_bases: int = 4,
                           page_samples: int = 16,
                           anchor_samples: int = 128) -> list[TypedQuery]:
    """S3b collection plan (pair-scan workload).

    The single-bit scan cannot separate a column bit (flip keeps bank and
    row -> low) from a bank-hash bit (flip leaves the bank -> low). The
    anchor M solves this: wherever (x, x^M) is verified same-bank, the
    anchored probe (x, x^M^(1<<b)) conflicts exactly when flipping b kept
    the bank, so column bits stay conflict while bank bits drop to low.
    Section order is contractual for analyze_pair_scan.py:

      calibration   ids 0..2: floor / different-bank / anchor conflict
      anchor_base   (base, base^M) per in-page base page -- anchor validity
      anchored_bit  (base, base^M^(1<<b)) per in-page bit x base
      pair          (base, base^(1<<b1)^(1<<b2)) per in-page bit pair x
                    base -- two bank bits cancel under a linear hash and
                    the pair returns to conflict
      anchor_sweep  (p, p^M) over evenly sampled pool pages -- where the
                    anchor keeps the bank across the PA range
      page_triple   per page-level bit and sampled partner x:
                    (x, x^M) anchor validity, (x, x^(1<<b)) a fresh
                    single-bit vote, (x, x^(1<<b)^M) the anchored vote
    """
    anchor = PAIR_SCAN_ANCHOR
    low_bits = pool.pages[0].page_size.bit_length() - 1

    def page_query(page: Page, low_a: int, low_b: int) -> Query:
        offset = page.va_page_base - pool.chunk_base_va(page.chunk_index)
        return Query(page.chunk_index, offset + low_a,
                     page.chunk_index, offset + low_b)

    plan: list[TypedQuery] = []
    first = pool.pa_pages[0]
    for index, candidate in enumerate(S1_IN_PAGE_CANDIDATES):
        plan.append(TypedQuery(page_query(first, 0, candidate), "calibration",
                               role=("floor", "baseline", "conflict")[index]))

    bases = _base_pages(pool, in_page_bases)
    for index, page in enumerate(bases):
        plan.append(TypedQuery(page_query(page, 0, anchor), "anchor_base",
                               base_index=index))
    for bit in range(low_bits):
        for index, page in enumerate(bases):
            plan.append(TypedQuery(page_query(page, 0, anchor ^ (1 << bit)),
                                   "anchored_bit", bit=bit, base_index=index))
    for b1 in range(low_bits):
        for b2 in range(b1 + 1, low_bits):
            for index, page in enumerate(bases):
                plan.append(TypedQuery(page_query(page, 0, (1 << b1) | (1 << b2)),
                                       "pair", bit=b1, bit2=b2,
                                       base_index=index))

    if anchor_samples > 0:
        step = max(1, len(pool.pa_pages) // anchor_samples)
        for index, page in enumerate(pool.pa_pages[::step][:anchor_samples]):
            plan.append(TypedQuery(page_query(page, 0, anchor),
                                   "anchor_sweep", sample_index=index))

    by_pa = {page.fb_pa_page_base: page for page in pool.pa_pages}
    top_bit = max(by_pa).bit_length() - 1
    for bit in range(low_bits, top_bit + 1):
        for index, left in enumerate(_page_partners(by_pa, 1 << bit,
                                                    page_samples)):
            right = by_pa[left.fb_pa_page_base ^ (1 << bit)]
            offset_l = left.va_page_base - pool.chunk_base_va(left.chunk_index)
            offset_r = right.va_page_base - pool.chunk_base_va(right.chunk_index)
            plan.append(TypedQuery(Query(left.chunk_index, offset_l,
                                         left.chunk_index, offset_l + anchor),
                                   "page_triple", bit=bit, sample_index=index,
                                   role="anchor"))
            plan.append(TypedQuery(Query(left.chunk_index, offset_l,
                                         right.chunk_index, offset_r),
                                   "page_triple", bit=bit, sample_index=index,
                                   role="single"))
            plan.append(TypedQuery(Query(left.chunk_index, offset_l,
                                         right.chunk_index, offset_r + anchor),
                                   "page_triple", bit=bit, sample_index=index,
                                   role="anchored"))
    return plan


def select_pair_scan_queries(pool: PoolMap, in_page_bases: int = 4,
                             page_samples: int = 16,
                             anchor_samples: int = 128) -> list[Query]:
    return [typed.query for typed in
            plan_pair_scan_queries(pool, in_page_bases=in_page_bases,
                                   page_samples=page_samples,
                                   anchor_samples=anchor_samples)]


# S4b-1 census: in-page probe offsets for the row-class pilot. Each probe
# is measured alone and XOR-ed with the S1 anchor M, so the pilot sees both
# row states of every probe bit at an anchor-valid page: bits 9/11 are
# S3b-classified column bits (keep bank and row), 12/17/20 are hash bits
# that do NOT belong to the anchor mask (M covers 8/16/18/19), so M|probe
# stays inside the page and never collapses onto M.
CENSUS_PILOT_PROBES = (0x200, 0x800, 0x1000, 0x20000, 0x100000)
CENSUS_SELF_SECOND_OFFSET = 1 << 20  # 1 MiB into the page: lambda stability
# at a PA-distant cache line of the same page


def _pa_query(by_pa: dict[int, Page], pool: PoolMap,
              pa: int) -> Query | None:
    """Reverse lookup: a byte PA observed in a previous run -> the query
    addressing it in THIS pool (page base + in-page offset). None when the
    page is not backed by this pool."""
    page_size = pool.pages[0].page_size
    page = by_pa.get(pa & ~(page_size - 1))
    if page is None:
        return None
    in_page = pa & (page_size - 1)
    base = page.va_page_base - pool.chunk_base_va(page.chunk_index)
    return Query(page.chunk_index, base + in_page,
                 page.chunk_index, base + in_page)


def plan_census_queries(pool: PoolMap, second_stride: int = 4,
                        second_offset: int = CENSUS_SELF_SECOND_OFFSET,
                        repeat_pages: int = 64,
                        reprobe_pairs: Sequence[tuple[int, int]] = (),
                        pilot_pages: Sequence[int] = (),
                        pilot_probes: Sequence[int] = CENSUS_PILOT_PROBES
                        ) -> tuple[list[TypedQuery], dict[str, list[str]]]:
    """S4b-1 census plan. Section order is contractual for
    analyze_census.py:

      calibration  ids 0..2: floor / different-bank / anchor conflict
      self         (p, p) per pool page -- a clean per-page lambda: ONE
                   endpoint per scalar, unlike pair data where both pages
                   mix into one value (S4b-0 could only bound lambda)
      self_second  (p+K, p+K) every stride-th page -- is lambda a page
                   property or an offset property?
      repeat       the first repeat_pages self queries again, late in the
                   list -- within-run drift / DVFS check
      reprobe      the S4b-0 suspect-conflict pairs (kept conflict but
                   below their own local conflict threshold), re-measured
                   with clean census lambdas available for both pages
      row_pilot    at anchor-valid pages (S3b anchor_sweep conflicts):
                   every unordered pair of {0, M, probe, M|probe} -- the
                   transitive (bank,row) class clustering pilot

    reprobe_pairs and pilot_pages are byte PAs / PA page bases from the
    previous runs; they are resolved against THIS run's pool map at plan
    time, and unresolvable entries are reported (never silently dropped).
    """
    anchor = PAIR_SCAN_ANCHOR
    by_pa = {page.fb_pa_page_base: page for page in pool.pa_pages}
    page_size = pool.pages[0].page_size
    if second_offset <= 0 or second_offset >= page_size:
        raise ValueError("second_offset must lie inside a page")

    def page_query(page: Page, low_a: int, low_b: int) -> Query:
        offset = page.va_page_base - pool.chunk_base_va(page.chunk_index)
        return Query(page.chunk_index, offset + low_a,
                     page.chunk_index, offset + low_b)

    plan: list[TypedQuery] = []
    first = pool.pa_pages[0]
    for index, candidate in enumerate(S1_IN_PAGE_CANDIDATES):
        plan.append(TypedQuery(page_query(first, 0, candidate), "calibration",
                               role=("floor", "baseline", "conflict")[index]))

    for index, page in enumerate(pool.pa_pages):
        plan.append(TypedQuery(page_query(page, 0, 0), "self",
                               sample_index=index))
    if second_stride > 0:
        for index, page in enumerate(pool.pa_pages):
            if index % second_stride == 0:
                plan.append(TypedQuery(page_query(page, second_offset,
                                                  second_offset),
                                       "self_second", sample_index=index))
    if repeat_pages > 0:
        for index, page in enumerate(pool.pa_pages[:repeat_pages]):
            plan.append(TypedQuery(page_query(page, 0, 0), "repeat",
                                   sample_index=index, role="drift"))

    dropped: dict[str, list[str]] = {"reprobe": [], "row_pilot": []}
    for index, (pa_a, pa_b) in enumerate(reprobe_pairs):
        query = _pa_query(by_pa, pool, pa_a)
        other = _pa_query(by_pa, pool, pa_b)
        if query is None or other is None:
            dropped["reprobe"].append(f"0x{pa_a:x}/0x{pa_b:x}")
            continue
        plan.append(TypedQuery(Query(query.chunk_a, query.ofs_a,
                                     other.chunk_a, other.ofs_a),
                               "reprobe", base_index=index))

    for base_index, pa_base in enumerate(pilot_pages):
        page = by_pa.get(pa_base)
        if page is None:
            dropped["row_pilot"].append(f"0x{pa_base:x}")
            continue
        offsets = [0, anchor] + [probe for probe in pilot_probes] \
            + [anchor ^ probe for probe in pilot_probes]
        for left in range(len(offsets)):
            for right in range(left + 1, len(offsets)):
                plan.append(TypedQuery(
                    page_query(page, offsets[left], offsets[right]),
                    "row_pilot", bit=left, bit2=right,
                    base_index=base_index))
    return plan, dropped


def select_census_queries(pool: PoolMap, **kwargs) -> list[Query]:
    plan, _ = plan_census_queries(pool, **kwargs)
    return [typed.query for typed in plan]


# S5-T1 anchor candidates: same-bank different-row in-page masks. Curated
# from the S1 tight conflict cluster (>=1120 cycles: 215 points, median
# 1124, spread 29) by greedy descending S1 value with a 0x1000 minimum
# separation, seeded with the two cross-run-validated masks -- 0xd0100
# (S1 targeted check 1143-1145; the S3b anchor and the S4b-1 deep pilot)
# and 0xd0300 (the S4b-1 pilot's M|0x200 partner). S3b measured that a
# single mask keeps the bank on only ~27% of pages, so the sweep carries
# a spread of candidates: per page, any DEEP hit is a bank-class edge and
# a valid anchor for double-probe classification.
TABLE_BUILD_ANCHOR_CANDIDATES = (0xd0100, 0xd0300, 0xd7b00, 0x11e300,
                                 0x119980, 0xd3880, 0x1fdc80, 0x1b5f00,
                                 0x1f9dc0, 0x37680, 0xb0b00, 0x1c00c0,
                                 0x1cd000, 0x30ec0, 0x4c0c0, 0x5a7c0,
                                 0x8d780, 0xeacc0, 0x12efc0, 0x135840,
                                 0x1388c0, 0x1b8d00, 0x22a00, 0x42bc0)
# Column probe folded into the bank-map lattice: bit 9 is the hardware
# column bit (S4b-1 pilot), so 0x200 pairs inside one bank class are
# same-row evidence.
TABLE_BUILD_COLUMN_PROBE = 0x200


def sample_uniform_reps(pool: PoolMap, count: int, seed: int = 7) -> list[int]:
    """Uniformly drawn rep page bases for big-pool table builds. The
    seed-table reps are one per v0 channel component and land
    bank-clustered (the S5-T2 R-b diagnostic: rep-rep deep edges 64 vs
    null 6), so a fresh large pool draws its reps uniformly instead --
    with 384 banks, count R links a fraction 1-(1-1/384)^R of pages to
    a same-bank rep (R=1024 -> ~93%, R=1536 -> ~98%). The fixed default
    seed makes the selection reproducible run to run."""
    rng = random.Random(seed)
    bases = sorted(page.fb_pa_page_base for page in pool.pa_pages)
    return sorted(rng.sample(bases, min(count, len(bases))))


def plan_table_build_queries(pool: PoolMap,
                             rep_pas: Sequence[int] = (),
                             anchor_candidates: Sequence[int] =
                             TABLE_BUILD_ANCHOR_CANDIDATES,
                             bank_map_pages: Sequence[int] = (),
                             bank_map_anchor: int = PAIR_SCAN_ANCHOR,
                             bank_map_probes: Sequence[int] =
                             TABLE_BUILD_ANCHOR_CANDIDATES,
                             repeat_pages: int = 64
                             ) -> tuple[list[TypedQuery], dict[str, object]]:
    """S5-T1 table-build plan. Section order is contractual for
    analyze_table_build.py:

      calibration   ids 0..2: floor / different-bank / anchor conflict
      self          (p, p) per pool page -- the fresh per-page lambda
                    that anchors the three-band classifier
      anchor_sweep  (p, p^M_c) per pool page x candidate mask: any DEEP
                    hit is a same-bank different-row edge and marks M_c
                    a valid anchor at p; the matrix answers whether
                    validity is page- or mask-specific (S3b only ever
                    swept one mask)
      classify      (p, rep) page starts, every pool page x one
                    representative page per seed channel component:
                    cross-page shoulder/deep proves same channel
                    (shoulder = same channel different bank; deep =
                    same bank, rarer), cross-page low proves different
                    channel -- the page->channel assignment and the
                    component merges come out of this one section
      bank_map      at anchor-valid pages (mined from the S3b sweep):
                    (p, p^y) and (p^M, p^y) per lattice offset y. With
                    the anchor M valid, bank(x0)=bank(x0^M) and the two
                    rows differ, so y is same-bank iff EITHER probe is
                    deep (a same-row hit with one endpoint is a deep
                    with the other); a low among same-bank pairs splits
                    rows
      repeat        the first repeat_pages self pairs again, late in
                    the list -- the late-section step measured by the
                    census shows up here too; the analyzer anchors the
                    additive correction on this block

    rep_pas and bank_map_pages are byte PA page bases from the seed
    table / the S3b anchor sweep; each is resolved against THIS run's
    pool map and unresolvable entries are reported in meta['dropped'],
    never silently dropped.
    """
    anchor_candidates = tuple(anchor_candidates)
    by_pa = {page.fb_pa_page_base: page for page in pool.pa_pages}

    def page_query(page: Page, low_a: int, low_b: int) -> Query:
        offset = page.va_page_base - pool.chunk_base_va(page.chunk_index)
        return Query(page.chunk_index, offset + low_a,
                     page.chunk_index, offset + low_b)

    def cross_query(left: Page, right: Page) -> Query:
        return Query(left.chunk_index,
                     left.va_page_base - pool.chunk_base_va(left.chunk_index),
                     right.chunk_index,
                     right.va_page_base - pool.chunk_base_va(right.chunk_index))

    plan: list[TypedQuery] = []
    first = pool.pa_pages[0]
    for index, candidate in enumerate(S1_IN_PAGE_CANDIDATES):
        plan.append(TypedQuery(page_query(first, 0, candidate), "calibration",
                               role=("floor", "baseline", "conflict")[index]))

    for index, page in enumerate(pool.pa_pages):
        plan.append(TypedQuery(page_query(page, 0, 0), "self",
                               sample_index=index))

    for page_index, page in enumerate(pool.pa_pages):
        for cand_index, mask in enumerate(anchor_candidates):
            plan.append(TypedQuery(page_query(page, 0, mask), "anchor_sweep",
                                   bit=cand_index, sample_index=page_index))

    dropped: dict[str, list[str]] = {"reps": [], "bank_map": []}
    reps: list[Page] = []
    for pa_base in rep_pas:
        page = by_pa.get(pa_base)
        if page is None:
            dropped["reps"].append(f"0x{pa_base:x}")
        elif page not in reps:
            reps.append(page)
    for page_index, page in enumerate(pool.pa_pages):
        for rep_index, rep in enumerate(reps):
            if page.fb_pa_page_base == rep.fb_pa_page_base:
                continue  # a page is not classified against itself
            plan.append(TypedQuery(cross_query(page, rep), "classify",
                                   base_index=rep_index,
                                   sample_index=page_index))

    lattice = tuple(dict.fromkeys(
        (*bank_map_probes, TABLE_BUILD_COLUMN_PROBE)))
    bank_pages: list[Page] = []
    for pa_base in bank_map_pages:
        page = by_pa.get(pa_base)
        if page is None:
            dropped["bank_map"].append(f"0x{pa_base:x}")
        else:
            bank_pages.append(page)
    for bank_index, page in enumerate(bank_pages):
        for offset_index, low in enumerate(lattice):
            plan.append(TypedQuery(page_query(page, 0, low), "bank_map",
                                   bit=offset_index, base_index=bank_index,
                                   role="base"))
            if low in (0, bank_map_anchor):
                continue  # (M, 0) is (0, M) again; (M, M) is a self pair
            plan.append(TypedQuery(page_query(page, bank_map_anchor, low),
                                   "bank_map", bit=offset_index,
                                   base_index=bank_index, role="anchor"))

    if repeat_pages > 0:
        for index, page in enumerate(pool.pa_pages[:repeat_pages]):
            plan.append(TypedQuery(page_query(page, 0, 0), "repeat",
                                   sample_index=index, role="drift"))

    meta: dict[str, object] = {
        "pages": len(pool.pa_pages),
        "anchor_candidates": [f"0x{mask:x}" for mask in anchor_candidates],
        "reps": [f"0x{page.fb_pa_page_base:x}" for page in reps],
        "reps_requested": len(rep_pas),
        "bank_map_pages": [f"0x{page.fb_pa_page_base:x}"
                           for page in bank_pages],
        "bank_map_anchor": f"0x{bank_map_anchor:x}",
        "bank_map_offsets": [f"0x{low:x}" for low in lattice],
        "repeat_pages": min(repeat_pages, len(pool.pa_pages)),
        "dropped": dropped,
    }
    return plan, meta


def select_table_build_queries(pool: PoolMap, **kwargs) -> list[Query]:
    plan, _ = plan_table_build_queries(pool, **kwargs)
    return [typed.query for typed in plan]


def plan_predict_check_queries(pool: PoolMap,
                               pair_pas: Sequence[tuple[int, int, str]],
                               repeat_pages: int = 32
                               ) -> tuple[list[TypedQuery], dict[str, object]]:
    """S5-T2 R-e prediction-check plan. Section order is contractual
    for validate_table.py --r-e-check:

      calibration  ids 0..2: floor / different-bank / anchor conflict
      self         (p, p) per DISTINCT page touched by a kept pair, in
                   PA order -- the lambda references
      predict      one row per kept (pa_a, pa_b, predicted) triple in
                   input order; validate_table scores the measured
                   label against the embedded prediction
      repeat       the first repeat_pages touched pages again, late in
                   the list -- the drift anchor for the late-section
                   step

    pair_pas entries are (pa_a, pa_b, predicted) byte PAs from the
    table's transitive closure; each PA must fall inside THIS run's
    pool or the pair is reported in meta['dropped'], never silently
    dropped.
    """
    import bisect
    bases = [page.fb_pa_page_base for page in pool.pa_pages]

    def page_of(pa: int) -> Page | None:
        index = bisect.bisect_right(bases, pa) - 1
        if index < 0:
            return None
        page = pool.pa_pages[index]
        return page if pa < page.fb_pa_page_base + page.page_size else None

    def half(pa: int, page: Page) -> tuple[int, int]:
        offset = page.va_page_base - pool.chunk_base_va(page.chunk_index)
        return page.chunk_index, offset + (pa - page.fb_pa_page_base)

    dropped: list[str] = []
    kept: list[tuple[int, int, int, int, str]] = []
    kept_pas: list[tuple[int, int, str]] = []
    for pa_a, pa_b, predicted in pair_pas:
        page_a, page_b = page_of(pa_a), page_of(pa_b)
        if page_a is None or page_b is None:
            dropped.append(f"0x{pa_a:x}/0x{pa_b:x}")
            continue
        chunk_a, ofs_a = half(pa_a, page_a)
        chunk_b, ofs_b = half(pa_b, page_b)
        kept.append((chunk_a, ofs_a, chunk_b, ofs_b, predicted))
        kept_pas.append((pa_a, pa_b, predicted))

    touched_bases = {page.fb_pa_page_base
                     for pa_a, pa_b, _ in kept_pas
                     for page in (page_of(pa_a), page_of(pa_b))}
    self_pages = sorted(touched_bases)
    self_page_of = {page.fb_pa_page_base: page
                    for page in pool.pa_pages}

    plan: list[TypedQuery] = []
    first = pool.pa_pages[0]
    for index, candidate in enumerate(S1_IN_PAGE_CANDIDATES):
        offset = first.va_page_base - pool.chunk_base_va(first.chunk_index)
        plan.append(TypedQuery(Query(first.chunk_index, offset,
                                     first.chunk_index, offset + candidate),
                               "calibration",
                               role=("floor", "baseline", "conflict")[index]))
    for index, base in enumerate(self_pages):
        page = self_page_of[base]
        offset = page.va_page_base - pool.chunk_base_va(page.chunk_index)
        plan.append(TypedQuery(Query(page.chunk_index, offset,
                                     page.chunk_index, offset),
                               "self", sample_index=index))
    for index, (chunk_a, ofs_a, chunk_b, ofs_b, _) in enumerate(kept):
        plan.append(TypedQuery(Query(chunk_a, ofs_a, chunk_b, ofs_b),
                               "predict", sample_index=index))
    repeat_of = self_pages[:repeat_pages]
    for index, base in enumerate(repeat_of):
        page = self_page_of[base]
        offset = page.va_page_base - pool.chunk_base_va(page.chunk_index)
        plan.append(TypedQuery(Query(page.chunk_index, offset,
                                     page.chunk_index, offset),
                               "repeat", sample_index=index, role="drift"))

    meta: dict[str, object] = {
        "pairs": len(kept),
        "pair_pa_a": [f"0x{pa_a:x}" for pa_a, _, _ in kept_pas],
        "pair_pa_b": [f"0x{pa_b:x}" for _, pa_b, _ in kept_pas],
        "pair_predicted": [predicted for _, _, predicted in kept_pas],
        "self_pages": [f"0x{base:x}" for base in self_pages],
        "repeat_pages": len(repeat_of),
        "repeat_of": [f"0x{base:x}" for base in repeat_of],
        "dropped": {"pairs": dropped},
    }
    return plan, meta


def write_work_csv(path: Path | None, queries: list[Query],
                   mode: str = "sanity") -> str:
    lines = [f"# mode={mode}", WORK_HEADER]
    for index, query in enumerate(queries):
        lines.append(f"{index},{query.chunk_a},{query.ofs_a},"
                     f"{query.chunk_b},{query.ofs_b}")
    text = "\n".join(lines) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def self_test() -> int:
    page_size = 2 * 1024 * 1024
    chunk0_va = 0x7F0000000000
    chunk1_va = 0x7F0008000000
    # PA pattern: chunk0 pages map to 0x120000000 / 0x120400000 (differ in
    # bit 22 = 0x400000), chunk1 pages to 0x120200000 / 0x120600000, so
    # page 0x120000000 ^ page 0x124000000 would be bit 26 -- absent; and
    # 0x120000000 ^ 0x120400000 is bit 22 -- present.
    rows = [
        (0, chunk0_va, 0x120000000),
        (0, chunk0_va + page_size, 0x120400000),
        (1, chunk1_va, 0x120200000),
        (1, chunk1_va + page_size, 0x120600000),
    ]
    text = ",".join(POOL_MAP_FIELDS) + "\n" + "".join(
        f"r,0,uuid,{chunk},alloc,0x{va:x},0x{va_end:x},0x{fb:x},{page_size},VIDEO,true,"
        f"0x1,0x0,1,2,ebpf,definition_level_gmmu_pte_payload\n"
        for chunk, va, fb in rows
        for va_end in [va + page_size]
    )
    pool = PoolMap(parse_pool_map(text))

    # PA arithmetic: in-page offset is untranslated, chunk-relative offsets
    # cross pages inside the chunk.
    assert pool.pa_of(0, 0) == 0x120000000
    assert pool.pa_of(0, 0x1234) == 0x120001234
    assert pool.pa_of(0, page_size) == 0x120400000
    assert pool.pa_of(1, page_size + 0x10) == 0x120600010

    sanity = select_sanity_queries(pool, cross_page=2)
    kinds = [(q.chunk_a, q.chunk_b) for q in sanity]
    assert kinds.count((0, 0)) == 3, "in-page triple missing"
    assert any(a != b for a, b in kinds), "cross-page pairs missing"
    for query in sanity:
        pa_a, pa_b = pool.query_pa(query)
        assert pa_a >= 0x120000000 and pa_b >= 0x120000000
    # The three in-page candidates must differ only inside one page.
    for query in sanity[:3]:
        pa_a, pa_b = pool.query_pa(query)
        assert pa_a // page_size == pa_b // page_size

    # Page-level single bit: bit 22 pairs 0x120000000<->0x120400000 and
    # 0x120200000<->0x120600000; bit 26 has no partner pages.
    bit22 = select_single_bit_pairs(pool, 22)
    assert len(bit22) == 2, bit22
    for query in bit22:
        pa_a, pa_b = pool.query_pa(query)
        assert bin(pa_a ^ pa_b).count("1") == 1 and (pa_a ^ pa_b) == (1 << 22)
    assert select_single_bit_pairs(pool, 26) == []

    # In-page single bit (bit 16 = 0x10000) stays inside one page.
    bit16 = select_single_bit_pairs(pool, 16, limit=4)
    assert 0 < len(bit16) <= 4
    for query in bit16:
        pa_a, pa_b = pool.query_pa(query)
        assert pa_a // page_size == pa_b // page_size
        assert (pa_a ^ pa_b) == (1 << 16)
        assert query.chunk_a == query.chunk_b

    # CSV round-trip.
    csv_text = write_work_csv(None, sanity)
    lines = [line for line in csv_text.splitlines() if line and not line.startswith("#")]
    assert lines[0] == WORK_HEADER
    assert len(lines) - 1 == len(sanity)
    for index, query in enumerate(sanity):
        parts = lines[index + 1].split(",")
        assert (int(parts[0]), int(parts[1]), int(parts[2]),
                int(parts[3]), int(parts[4])) == (
            index, query.chunk_a, query.ofs_a, query.chunk_b, query.ofs_b)

    # Bit-scan: controls first, in-page bits per base page, page-level bits
    # only where the XOR partner is also in the pool.
    scan = select_bit_scan_queries(pool, in_page_bases=2, pairs_per_bit=4)
    assert scan[:3] == sanity[:3], "bit-scan must open with the calibration triple"
    assert len(scan) == 3 + 21 * 2 + 4, len(scan)  # 21 in-page bits, 2 bases;
    # the fixture's four fbs differ only in bits 21/22 -> 2 pairs per bit
    for query in scan[3:3 + 21 * 2]:
        pa_a, pa_b = pool.query_pa(query)
        assert query.chunk_a == query.chunk_b
        assert pa_b - pa_a == (pa_a ^ pa_b) and bin(pa_a ^ pa_b).count("1") == 1
    for query in scan[3 + 21 * 2:]:
        pa_a, pa_b = pool.query_pa(query)
        xor = pa_a ^ pa_b
        assert xor in (1 << 21, 1 << 22), hex(xor)
        assert query.chunk_a != query.chunk_b or query.ofs_a != query.ofs_b

    # Pair-scan: calibration triple first, then the anchored sections; the
    # anchored-bit xors carry the anchor mask, page triples come in
    # anchor/single/anchored order with the anchor folded into the offsets.
    plan = plan_pair_scan_queries(pool, in_page_bases=2, page_samples=2,
                                  anchor_samples=4)
    pscan = [typed.query for typed in plan]
    assert pscan[:3] == sanity[:3], "pair-scan must open with the calibration triple"
    sections = [typed.section for typed in plan]
    assert sections.count("anchor_base") == 2
    assert sections.count("anchored_bit") == 21 * 2
    assert sections.count("pair") == 210 * 2
    assert sections.count("anchor_sweep") == 4  # fixture has 4 PA pages
    assert sections.count("page_triple") == 2 * 2 * 3  # bits 21/22, 2 partners
    for typed in plan:
        pa_a, pa_b = pool.query_pa(typed.query)
        xor = pa_a ^ pa_b
        if typed.section in ("anchor_base", "anchor_sweep"):
            assert xor == PAIR_SCAN_ANCHOR, hex(xor)
        elif typed.section == "anchored_bit":
            assert xor == (PAIR_SCAN_ANCHOR ^ (1 << typed.bit)), hex(xor)
        elif typed.section == "pair":
            assert xor == ((1 << typed.bit) | (1 << typed.bit2)), hex(xor)
        elif typed.section == "page_triple":
            if typed.role == "anchor":
                assert xor == PAIR_SCAN_ANCHOR
            elif typed.role == "single":
                assert xor == (1 << typed.bit)
            else:
                assert xor == ((1 << typed.bit) | PAIR_SCAN_ANCHOR)
    triples = [typed for typed in plan if typed.section == "page_triple"]
    for bit in (21, 22):
        roles = [typed.role for typed in triples if typed.bit == bit]
        assert roles == ["anchor", "single", "anchored"] * 2, roles

    # Census: calibration triple, one self-pair per PA page, strided second
    # offsets, a drift repeat block, PA-level reprobe reverse lookup with
    # honest drop reporting, and the 12-offset row pilot per pilot page.
    reprobe = [(0x120001234, 0x120401234),   # both pages backed
               (0x120001234, 0x900000000)]   # second page absent -> dropped
    census, dropped = plan_census_queries(
        pool, second_stride=2, repeat_pages=2, reprobe_pairs=reprobe,
        pilot_pages=(0x120000000, 0xDEAD0000))
    cqueries = [typed.query for typed in census]
    assert cqueries[:3] == sanity[:3], "census must open with the calibration triple"
    sections = [typed.section for typed in census]
    assert sections.count("self") == 4            # every PA page
    assert sections.count("self_second") == 2     # stride 2 over 4 pages
    assert sections.count("repeat") == 2
    assert sections.count("reprobe") == 1         # one dropped, reported
    assert dropped["reprobe"] == ["0x120001234/0x900000000"]
    assert dropped["row_pilot"] == ["0xdead0000"]
    n_probes = len(CENSUS_PILOT_PROBES)
    assert sections.count("row_pilot") == (2 + 2 * n_probes) * (2 + 2 * n_probes - 1) // 2
    # Self-pairs address the first byte of each PA page; second-offset
    # pairs move 1 MiB into the same page.
    for typed in census:
        if typed.section == "self":
            pa_a, pa_b = pool.query_pa(typed.query)
            assert pa_a == pa_b and pa_a & (page_size - 1) == 0
        elif typed.section == "self_second":
            pa_a, pa_b = pool.query_pa(typed.query)
            assert pa_a == pa_b and pa_a & (page_size - 1) == CENSUS_SELF_SECOND_OFFSET
    # The resolved reprobe pair addresses exactly the requested bytes.
    reprobe_query = next(typed.query for typed in census
                         if typed.section == "reprobe")
    assert pool.query_pa(reprobe_query) == (0x120001234, 0x120401234)
    # Pilot xors: (0, M) conflicts by construction, probe pairs carry one
    # probe bit or one probe bit folded into the anchor.
    pilot = [typed for typed in census if typed.section == "row_pilot"]
    for typed in pilot:
        pa_a, pa_b = pool.query_pa(typed.query)
        xor = (pa_a ^ pa_b) & (page_size - 1)
        probes = {0, PAIR_SCAN_ANCHOR}
        probes |= set(CENSUS_PILOT_PROBES)
        probes |= {PAIR_SCAN_ANCHOR ^ probe for probe in CENSUS_PILOT_PROBES}
        assert xor in {a ^ b for a in probes for b in probes if a != b}, hex(xor)
    assert any((pool.query_pa(t.query)[0] ^ pool.query_pa(t.query)[1])
               == PAIR_SCAN_ANCHOR for t in pilot)

    # Table-build: calibration triple, one self per PA page, the anchor
    # sweep at every page x candidate, cross-page classify pairs against
    # resolvable reps only (same-page pairs skipped, off-pool reps
    # reported), the double-probe bank map with the (M,0)/(M,M)
    # degenerates dropped, and a late repeat block.
    cands = (0xd0100, 0x2000)
    plan, meta = plan_table_build_queries(
        pool, rep_pas=(0x120000000, 0x120200000, 0x900000000),
        anchor_candidates=cands, bank_map_probes=cands,
        bank_map_pages=(0x120400000, 0xDEAD0000), repeat_pages=2)
    tqueries = [typed.query for typed in plan]
    assert tqueries[:3] == sanity[:3], "table-build must open with the triple"
    sections = [typed.section for typed in plan]
    assert sections == (["calibration"] * 3 + ["self"] * 4
                        + ["anchor_sweep"] * 8 + ["classify"] * 6
                        + ["bank_map"] * 5 + ["repeat"] * 2), sections
    assert meta["dropped"] == {"reps": ["0x900000000"],
                               "bank_map": ["0xdead0000"]}
    assert meta["reps"] == ["0x120000000", "0x120200000"]
    assert meta["bank_map_pages"] == ["0x120400000"]
    assert meta["anchor_candidates"] == ["0xd0100", "0x2000"]
    # pa_pages PA order: 0x120000000, 0x120200000, 0x120400000, 0x120600000
    # -> reps are pages 0 and 1; classify skips the two same-page pairs.
    for typed in plan:
        pa_a, pa_b = pool.query_pa(typed.query)
        if typed.section == "anchor_sweep":
            assert pa_a >> 21 == pa_b >> 21
            assert pa_a ^ pa_b == cands[typed.bit]
            assert pa_a & (page_size - 1) == 0
        elif typed.section == "classify":
            assert pa_a >> 21 != pa_b >> 21
            assert pa_a & (page_size - 1) == 0 and pa_b & (page_size - 1) == 0
            assert pool.pa_pages[typed.sample_index].fb_pa_page_base == pa_a
    # bank map: lattice = candidates + column probe, base probes for every
    # offset, anchor probes except the two degenerates; y=0xd0100 keeps the
    # (0, M) anchor-validation probe itself.
    bank = [typed for typed in plan if typed.section == "bank_map"]
    lattice = (0xd0100, 0x2000, TABLE_BUILD_COLUMN_PROBE)
    assert [t.bit for t in bank] == [0, 1, 1, 2, 2], [t.bit for t in bank]
    seen = {}
    for typed in bank:
        pa_a, pa_b = pool.query_pa(typed.query)
        assert pa_a >> 21 == pa_b >> 21 == 0x120400000 >> 21
        pair = {pa_a & (page_size - 1), pa_b & (page_size - 1)}
        fixed = {PAIR_SCAN_ANCHOR} if typed.role == "anchor" else {0}
        assert pair == fixed | {lattice[typed.bit]}, (pair, typed.role)
        seen.setdefault(typed.bit, []).append(typed.role)
    assert seen[0] == ["base"], seen[0]          # y == M: anchor probe is (M, M)
    assert seen[1] == ["base", "anchor"]
    assert seen[2] == ["base", "anchor"]
    repeat = [typed for typed in plan if typed.section == "repeat"]
    for typed in repeat:
        pa_a, pa_b = pool.query_pa(typed.query)
        assert pa_a == pa_b and pa_a & (page_size - 1) == 0

    # Predict-check: calibration triple, one self per DISTINCT touched
    # page in PA order, one row per kept pair (PA round-trip exact,
    # including in-page offsets), repeat block over the first touched
    # pages, off-pool pairs reported.
    pplan, pmeta = plan_predict_check_queries(
        pool,
        pair_pas=[(0x120000000 + 0x200, 0x120000000 + 0xd0300, "low"),
                  (0x120000000, 0x120400000 + 0xd0100, "deep_conflict"),
                  (0x120600000 + 0x1000, 0x900000000, "deep_conflict")],
        repeat_pages=2)
    psections = [typed.section for typed in pplan]
    assert psections == (["calibration"] * 3 + ["self"] * 2
                         + ["predict"] * 2 + ["repeat"] * 2), psections
    assert pmeta["dropped"] == {"pairs": ["0x120601000/0x900000000"]}
    assert pmeta["pair_pa_a"] == ["0x120000200", "0x120000000"]
    assert pmeta["pair_pa_b"] == ["0x1200d0300", "0x1204d0100"]
    assert pmeta["pair_predicted"] == ["low", "deep_conflict"]
    assert pmeta["self_pages"] == ["0x120000000", "0x120400000"], \
        pmeta["self_pages"]
    assert pmeta["repeat_of"] == ["0x120000000", "0x120400000"]
    self_rows = [typed for typed in pplan if typed.section == "self"]
    for typed, base in zip(self_rows, pmeta["self_pages"]):
        pa_a, pa_b = pool.query_pa(typed.query)
        assert pa_a == pa_b == int(base, 16)
    predict_rows = [typed for typed in pplan if typed.section == "predict"]
    for typed, pa_hex_a, pa_hex_b in zip(predict_rows, pmeta["pair_pa_a"],
                                         pmeta["pair_pa_b"]):
        assert pool.query_pa(typed.query) == (int(pa_hex_a, 16),
                                              int(pa_hex_b, 16))

    # uniform rep sampling: deterministic, sorted, distinct, clamped
    universe = {0x120000000, 0x120400000, 0x120200000, 0x120600000}
    reps = sample_uniform_reps(pool, 3, seed=7)
    assert len(reps) == 3 and reps == sorted(reps) \
        and len(set(reps)) == 3 and set(reps) <= universe, reps
    assert sample_uniform_reps(pool, 3, seed=7) == reps
    assert sample_uniform_reps(pool, 99) == sorted(universe)

    print("g3_pool self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-map", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode",
                        choices=("sanity", "single-bit", "bit-scan", "pair-scan",
                                 "census", "table-build"),
                        default="sanity")
    parser.add_argument("--bit", type=int)
    parser.add_argument("--cross-page", type=int, default=8)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--in-page-bases", type=int, default=4)
    parser.add_argument("--pairs-per-bit", type=int, default=64)
    parser.add_argument("--page-samples", type=int, default=16,
                        help="pair-scan mode: page-level partners per bit")
    parser.add_argument("--anchor-samples", type=int, default=128,
                        help="pair-scan mode: anchor-validity sweep pages")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.pool_map or not args.output:
        parser.error("--pool-map and --output are required outside --self-test")
    pool = PoolMap(parse_pool_map(args.pool_map.read_text(encoding="utf-8")))
    if args.mode == "sanity":
        queries = select_sanity_queries(pool, cross_page=args.cross_page)
    elif args.mode == "bit-scan":
        queries = select_bit_scan_queries(pool, in_page_bases=args.in_page_bases,
                                          pairs_per_bit=args.pairs_per_bit)
    elif args.mode == "pair-scan":
        queries = select_pair_scan_queries(pool, in_page_bases=args.in_page_bases,
                                           page_samples=args.page_samples,
                                           anchor_samples=args.anchor_samples)
    elif args.mode == "census":
        queries = select_census_queries(pool)
    elif args.mode == "table-build":
        # Shape-debug aid only: the real run's reps / bank-map pages come
        # from the seed table and the S3b sweep via the orchestrator.
        queries = select_table_build_queries(pool, repeat_pages=0)
    else:
        if args.bit is None or not 0 <= args.bit < 40:
            parser.error("--bit must be in [0, 40)")
        queries = select_single_bit_pairs(pool, args.bit, limit=args.limit)
    write_work_csv(args.output, queries, mode=args.mode)
    print(f"wrote {len(queries)} queries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
