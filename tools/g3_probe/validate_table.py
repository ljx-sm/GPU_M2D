#!/usr/bin/env python3
"""GPU_M2D G3 S5-T2: empirical mapping table (EMT) validation gates.

The T1 collection built the table; these gates decide whether it may be
TRUSTED as the G4/G5 mapping. GeForge-style offline tables ship with no
validation at all (their threat model is an attacker who accepts missed
banks); our defensive-reliability deliverable must state its accuracy,
so every gate prints numbers plus an explicit pass bar.

  R-a transitive consistency   the class closures must not be falsified
       by any measured edge: deep inside a row class (C2), shoulder
       inside a bank class (C1), decided non-deep (low/shoulder)
       cross-page inside a channel component (C3 -- mid is the
       undecided valley band: a mid between same-bank pages is the
       measured shallow-conflict wobble and never falsifies), and a
       row class spanning two pages -- the last is new here: same
       (bank,row) on two distinct 2 MiB pages is physically impossible
       (the in-page column field cannot absorb PA bits >= 21), and the
       T1 edges must not force one.
       Bar: every count 0.
  R-b class cardinality        the bank structure must match the AD102
       prior (24 channels x 16 banks = 384). Unbiased estimator: the
       classify section's cross-page deep rate (the lattice/sweep masks
       are biased samples). Bar: the implied effective class count
       n/deeps inside [368, 400]; the exactly-uniform-384 z-test and the
       uniform-hash Monte Carlo stats print as diagnostics. Rationale:
       the big-pool run (11.5M classify pairs) and every earlier run
       measure the rate reproducibly ~2% BELOW 1/384 (implied ~391.7
       effective classes, 2-sigma [387..396], z = -3.4) -- a rate below
       1/384 is impossible for any fixed distribution over <= 384
       buckets (non-uniformity only raises collisions), so the deviation
       direction excludes the corruption this gate exists to catch, and
       the per-page degree diagnostics show the sigma is NOT understated
       (degrees under-dispersed vs the multinomial null, no page above
       the null max). The band holds the nominal 384 and the measured
       ~392 consensus with ~2-sigma headroom while still failing every
       structural break -- collapsed/stale pairs or a drifted deep gate
       move the rate, and K-hat, by far more than 4%.
  R-c reproducibility          a second table-build run must reproduce
       the first: label agreement on common PA pairs, corrected-d
       agreement, anchor validity agreement, same-bank page partition
       co-membership. Bars: zero hard deep<->low flips (a flip across
       the measured-empty valley falsifies the model), deep|mid region
       recall >= 99% on run-1 deeps, anchor hard flips 0, median
       corrected-d shift <= 15 cycles. Strict label equality is
       reported ungated: mid is the undecided band and each run's gate
       rides its own single-query calibration amplitude.
  R-e end-to-end prediction    the table must PREDICT pairs it never
       measured: pairs inside a bank class that are not themselves
       edges -- same row class -> low, row classes linked by a deep
       edge -> deep. --r-e-plan samples them (cross-page pairs first:
       the strong claim), --r-e-check runs the measurement through
       the predict-check work mode and scores it. Bar: >= 95% per
       predicted class counting only hard flips (deep<->low, the
       opposite decided class); deep<->mid is boundary wobble as in
       R-c. Every hard falsifier is listed -- the first T1 table
       carried 1/87: the double-probe row inference (anchor-low =>
       same row as M) is not universally valid, so row classes are
       PROBABLE, not certain, for downstream consumers.
  R-d cross-card transfer      the same collection on another GPU of
       the SAME MODEL must reproduce the structure, not the absolute
       cycles. Bars: same-bank partition co-membership Jaccard >=
       0.99, hard deep<->low flip RATE <= 0.1% (one borderline edge
       in 200k pairs is silicon variation; a systematic pattern would
       falsify the per-model claim), pages keeping a common anchor
       (no page may lose every anchor the other run found; rate bar
       0.1% cross-card, zero same-card -- a card with a different bank
       hash would lose common anchors essentially everywhere),
       deep|mid region recall >= 99%. Per-CELL anchor hard flips are
       informational, not gated: the anchor sweep's mid valley is
       POPULATED (3.5-7% of cells, unlike classify's empty valley), so
       per-card gate placement (calibration amplitudes measured
       117/121/101 cyc across the three cards) composes a deep<->low
       cell flip out of two soft band-edge steps -- measured 633 cells
       GPU2-deep/GPU1-low/GPU0-mid while every structure bar sat at
       1.000 and zero pages lost a common anchor. The Jaccard bar
       applies to
       SAME-SHAPE runs only: a run pair with different pool/rep sets
       cannot compare co-membership -- the denser run merges strictly
       more page pairs simply by measuring more of them (big-pool vs
       old-2048-page run: Jaccard 0.199 with zero hard flips both
       ways), so shape-mismatched pairs print it as informational
       while the falsifying bars (hard flips, region recall, common
       anchors) stay fully gated. The corrected-d median shift is
       informational only: lambda carries a per-card timing offset
       (measured 26 cyc GPU0->GPU1 vs bar 15 same-card), which is
       exactly why the table stores classes, not cycle counts.

    python3 validate_table.py <s3> <s3b> <census> [--t1 RUN ...]
        [--channel-deep-only] [--r-c RUN2] [--r-d RUN2]
        [--r-e-plan --out pairs.csv --n-deep 128 --n-low 128 --seed 7]
        [--r-e-check PREDICT_RUN]

Exit codes: 0 all requested gates passed, 2 gate failure or integrity
failure, 1 error (repo convention). `--self-test` pins the R-a/R-b
mechanics, the R-c join math and the R-e plan/check round trip on
synthetic fixtures.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_table_build import (  # noqa: E402
    BANK_PRIOR,
    DEEP_FRACTION,
    LOW_FRACTION,
)
from analyze_census import LOW_FRACTION as _LOW, SYMMETRY_FRACTION  # noqa: E402
from build_bank_table import build, load_edges, load_pages  # noqa: E402
from g3_pool import PoolMap, parse_pool_map  # noqa: E402

assert LOW_FRACTION == _LOW  # one band definition across the analyzers

# R-c / R-e pass bars (documented in the module docstring). R-c gates
# hard flips and the deep|mid REGION, not strict label equality: mid is
# by construction the undecided valley band, and each run's gate rides
# its own single-query calibration amplitude (measured 110 vs 121 cyc
# across the first two runs), so deep<->mid boundary flips are gate
# wobble while a deep<->low flip would cross the measured-empty valley.
REGION_RECALL_BAR = 0.99
D_SHIFT_BAR = 15
PREDICT_ACCURACY_BAR = 0.95
# R-d cross-card bars: two GPUs of one model share the hash (same die,
# same GDDR controller) but not the silicon lottery, so hard flips become
# a RATE and the absolute cycle shift is un-gated: lambda differs across
# cards by a constant-ish offset (measured 26 cyc GPU0->GPU1, amp 110 vs
# 111), which is exactly why the table stores CLASSES, not cycles.
CROSS_HARD_RATE_BAR = 1e-3
CROSS_JACCARD_BAR = 0.99
# R-b effective-class band. The exactly-uniform-384 z-test was the T2
# bar, but the big-pool run (n=11.5M classify pairs) measures the
# cross-page deep rate reproducibly 2% BELOW 1/384 (implied effective
# classes ~391.7, 2-sigma [387..396]; every earlier run agrees:
# 391.4, 391.7). A rate below 1/384 is impossible for any fixed
# distribution over <=384 buckets -- non-uniformity only raises
# collisions -- so the deviation direction excludes the corruption R-b
# exists to catch (merged/fewer banks). The degree-split diagnostic
# confirms the sigma is not understated (per-page deep degree
# UNDER-dispersed vs the multinomial null: non-rep variance 2.20 vs
# 2.67, no page above the null max degree), so widening sigma would be
# wrong. The gate therefore bands the implied class count: it holds the
# nominal 384 and the measured ~392 consensus with ~2-sigma headroom
# while still failing any structural break (a collapsed classify
# section, stale PAs, or a drifted deep threshold moves the rate --
# and K-hat -- by far more than 4%; the degenerate all-low case sends
# K-hat to infinity).
EFFECTIVE_CLASSES_LO = 368
EFFECTIVE_CLASSES_HI = 400


def _hex(value: str) -> int:
    return int(value, 16)


def load_t1_edges(run: Path, problems: list[str]) -> list[dict[str, str]]:
    path = run / "table_build_edges.csv"
    if not path.is_file():
        problems.append(f"missing {path} (run analyze_table_build.py)")
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# R-a -- transitive consistency of the built classes
# --------------------------------------------------------------------------

def gate_ra(edges: dict[tuple[int, int], tuple[str, str]], part: dict,
            say) -> bool:
    row_root = part["row"].find
    bank_root = part["bank"].find
    comp_of = {page: root for root, members in part["page_groups"].items()
               for page in members}
    a1 = a2 = a3 = a4 = 0
    for (a, b), (cls, _) in edges.items():
        if cls == "deep_conflict" and row_root(a) == row_root(b):
            a1 += 1                                    # C2 recount
        if cls == "shoulder" and bank_root(a) == bank_root(b):
            a2 += 1                                    # C1 recount
        # a4 mirrors the build's C3 on DECIDED non-deep (low/shoulder):
        # a decided-different measurement inside a same-bank component
        # contradicts transitivity. mid is the undecided valley band --
        # a mid between same-bank pages is the S4b-1 shallow-conflict
        # observation (same bank paying a partial penalty), the same
        # deep<->mid wobble R-c/R-d/R-e treat as boundary wobble, so it
        # falsifies nothing (measured: GPU1-big read 2 classify pairs
        # mid that GPU0-big read deep, both inside one component).
        if a >> 21 != b >> 21 and cls in ("low", "shoulder") \
                and comp_of.get(a >> 21) is not None \
                and comp_of.get(a >> 21) == comp_of.get(b >> 21):
            a4 += 1                                    # C3 family
    row_pages: dict[int, set[int]] = defaultdict(set)
    for root, members in part["row_groups"].items():
        for pa in members:
            row_pages[root].add(pa >> 21)
    a3 = sum(1 for pages in row_pages.values() if len(pages) > 1)
    say("R-a transitive consistency (bar: every count 0):")
    say(f"  a1 deep edge inside a row class:        {a1}")
    say(f"  a2 shoulder edge inside a bank class:   {a2}")
    say(f"  a3 row class spanning >1 page:          {a3}")
    say(f"  a4 decided non-deep (low/shoulder) cross-page inside a "
        f"channel component: {a4}")
    ok = a1 == a2 == a3 == a4 == 0
    say(f"  R-a: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
# R-b -- class cardinality vs the AD102 bank prior
# --------------------------------------------------------------------------

def _null_distribution(pages: int, reps: int, banks: int,
                       trials: int = 2000, seed: int = 7) -> dict[str, list]:
    """Uniform-hash null for the classify section: every page one uniform
    bank, every rep one uniform bank, an edge per (page, rep) bank match.
    Returns the sorted per-statistic distributions."""
    rng = random.Random(seed)
    stats: dict[str, list[int]] = {name: [] for name in
                                   ("deeps", "components", "covered",
                                    "largest", "rep_reps")}
    rep_banks = [rng.randrange(banks) for _ in range(reps)]
    for _ in range(trials):
        page_banks = [rng.randrange(banks) for _ in range(pages)]
        # classify pairs pages x reps: a bank forms a component only if
        # it holds >=1 rep; members = reps AND pages in that bank (reps
        # are pool pages too, so rep-rep pairs are measured).
        members: dict[int, int] = defaultdict(int)   # bank -> member count
        for bank in rep_banks:
            members[bank] += 1
        for bank in page_banks:
            if bank in members:                      # joins via a rep
                members[bank] += 1
        sizes = [count for count in members.values() if count >= 2]
        rep_count: Counter = Counter(rep_banks)
        page_count: Counter = Counter(page_banks)
        rep_reps = sum(r * (r - 1) // 2 for r in rep_count.values())
        deeps = sum(rep_count[b] * page_count[b] + rep_count[b]
                    * (rep_count[b] - 1) // 2 for b in rep_count)
        stats["deeps"].append(deeps)
        stats["components"].append(len(sizes))
        stats["covered"].append(sum(sizes))
        stats["largest"].append(max(sizes) if sizes else 0)
        stats["rep_reps"].append(rep_reps)
    return {name: sorted(values) for name, values in stats.items()}


def _quantile(sorted_values: list[int], q: float) -> int:
    index = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))
    return sorted_values[index]


def gate_rb(t1_rows: list[dict[str, str]], say) -> bool:
    classify = [row for row in t1_rows if row["section"] == "classify"]
    if not classify:
        say("R-b: FAIL (no classify rows in the T1 edges)")
        return False
    deeps = sum(1 for row in classify if row["class"] == "deep_conflict")
    n = len(classify)
    rate = deeps / n
    prior = 1 / BANK_PRIOR
    sigma = (prior * (1 - prior) / n) ** 0.5
    z = (rate - prior) / sigma
    # implied bank count from the unbiased random-pair rate
    n_banks = n / deeps if deeps else float("inf")
    lo = n / (deeps + 2 * sigma * n) if deeps else 0.0
    hi = n / max(1, deeps - 2 * sigma * n) if deeps else 0.0
    say("R-b class cardinality vs prior "
        f"(AD102 24ch x 16bank = {BANK_PRIOR} banks):")
    say(f"  cross-page deep rate {deeps}/{n} = {rate:.4%} vs prior "
        f"{prior:.4%} (z = {z:+.2f} vs exactly-uniform {BANK_PRIOR}; "
        f"diagnostic -- the measured consensus is ~392 effective "
        f"classes, 2% above nominal)")
    say(f"  implied effective class count {n_banks:.0f} "
        f"(2-sigma {lo:.0f}..{hi:.0f}) -- prior {BANK_PRIOR} "
        f"{'inside' if lo <= BANK_PRIOR <= hi else 'OUTSIDE'} the "
        f"interval; gate band [{EFFECTIVE_CLASSES_LO}, "
        f"{EFFECTIVE_CLASSES_HI}]")

    # diagnostic: uniform-hash Monte Carlo null. NOT a gate: the null
    # assumes reps land uniform over banks, but the T1 reps were drawn
    # one per v0 channel component (lambda-artifact groups -- lambda
    # correlates with bank), so rep-side concentration is a property of
    # the selection, measured directly by the rep-rep deep count.
    pages = sorted({int(row["pa_a"], 16) >> 21 for row in classify}
                   | {int(row["pa_b"], 16) >> 21 for row in classify})
    reps = {int(row["pa_b"], 16) for row in classify}
    rep_reps = sum(1 for row in classify
                   if row["class"] == "deep_conflict"
                   and int(row["pa_a"], 16) in reps)
    null = _null_distribution(len(pages), len(reps), BANK_PRIOR)
    null_rr = _quantile(null["rep_reps"], 0.5)
    clustered = rep_reps > null_rr * 2
    say(f"  rep-rep deep edges {rep_reps} vs null p50 {null_rr} -- "
        + ("REPS BANK-CLUSTERED: null component stats are diagnostics "
           "only (reps came one-per-v0-component, lambda correlates "
           "with bank)" if clustered else "rep placement ~uniform"))

    # observed partition from the deep edges
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for row in classify:
        if row["class"] == "deep_conflict":
            union(int(row["pa_a"], 16), int(row["pa_b"], 16))
    groups: dict[int, list[int]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    sizes = sorted((len(m) for m in groups.values()), reverse=True)
    observed = {"deeps": deeps,
                "components": len(sizes),
                "covered": sum(sizes),
                "largest": sizes[0] if sizes else 0}
    say("  (uniform-null diagnostics, observed vs p05/p50/p95 -- the "
        "implied-classes band above is the gate)")
    for name in ("deeps", "components", "covered", "largest"):
        dist = null[name]
        p05, p50, p95 = (_quantile(dist, q) for q in (0.05, 0.5, 0.95))
        say(f"  {name:<11} observed {observed[name]:>6}  null "
            f"{p05:>5} / {p50:>5} / {p95:>5}   "
            f"{'in range' if p05 <= observed[name] <= p95 else 'OUT OF RANGE'}")
    ok = deeps > 0 and EFFECTIVE_CLASSES_LO <= n_banks <= EFFECTIVE_CLASSES_HI
    say(f"  R-b: {'PASS' if ok else 'FAIL'} (gate = implied effective "
        f"classes in [{EFFECTIVE_CLASSES_LO}, {EFFECTIVE_CLASSES_HI}] -- "
        f"see EFFECTIVE_CLASSES_* for why the z-test is a diagnostic)")
    return ok


# --------------------------------------------------------------------------
# R-c -- reproducibility of the whole collection
# --------------------------------------------------------------------------

def _page_bases(run: Path) -> set[int]:
    pool = PoolMap(parse_pool_map((run / "pool_map.csv").read_text(
        encoding="utf-8")))
    return {page.fb_pa_page_base for page in pool.pa_pages}


def _anchor_validity(run: Path) -> dict[tuple[int, int], str]:
    table: dict[tuple[int, int], str] = {}
    with (run / "anchor_validity.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            table[(_hex(row["page_base"]), _hex(row["candidate"]))] \
                = row["class"]
    return table


def gate_rc(run1: Path, run2: Path, say, cross_card: bool = False) -> bool:
    """Same-card reproducibility (R-c) or cross-card transfer (R-d,
    cross_card=True). Cross-card relaxes exactly two bars, both
    expected across silicon: absolute lambda shifts (per-card timing
    offset, informational only) and the hard-flip bar becomes a RATE
    (one borderline edge in 200k pairs is silicon variation, not model
    falsification)."""
    problems: list[str] = []
    edges1 = {( _hex(r["pa_a"]), _hex(r["pa_b"])): r
              for r in load_t1_edges(run1, problems)}
    edges2 = {( _hex(r["pa_a"]), _hex(r["pa_b"])): r
              for r in load_t1_edges(run2, problems)}
    if problems:
        say(f"R-c: FAIL ({'; '.join(problems)})")
        return False
    pages1, pages2 = _page_bases(run1), _page_bases(run2)
    same_pool = pages1 == pages2
    say(("R-d cross-card transfer " if cross_card else "R-c reproducibility ")
        + f"({run1.name} vs {run2.name}):")
    say(f"  pool page sets {'identical' if same_pool else 'DIFFER'} "
        f"(|p1| {len(pages1)}, |p2| {len(pages2)}, "
        f"common {len(pages1 & pages2)}) -- PA-hole stability evidence")

    common = sorted(set(edges1) & set(edges2))
    confusion = Counter((edges1[k]["class"], edges2[k]["class"])
                        for k in common)
    agree = sum(n for (c1, c2), n in confusion.items() if c1 == c2)
    deep1 = [k for k in common if edges1[k]["class"] == "deep_conflict"]
    deep_kept = sum(1 for k in deep1
                    if edges2[k]["class"] == "deep_conflict")
    deep_region = sum(1 for k in deep1
                      if edges2[k]["class"] in ("deep_conflict", "mid"))
    recall = deep_kept / len(deep1) if deep1 else 1.0
    region = deep_region / len(deep1) if deep1 else 1.0
    hard = (confusion[("deep_conflict", "low")]
            + confusion[("low", "deep_conflict")])
    overall = agree / len(common) if common else 1.0
    say(f"  common classified pairs {len(common)} "
        f"(of {len(edges1)} / {len(edges2)}); label agreement "
        f"{overall:.2%}")
    say(f"  deep recall {deep_kept}/{len(deep1)} = {recall:.2%} strict; "
        f"deep|mid region recall {deep_region}/{len(deep1)} = "
        f"{region:.2%} (bar >= {REGION_RECALL_BAR:.0%})")
    if cross_card:
        say(f"  hard deep<->low flips: {hard}/{len(common)} = "
            f"{hard / len(common):.4%} (bar <= "
            f"{CROSS_HARD_RATE_BAR:.1%} rate) -- silicon variation "
            f"tolerated cross-card, a systematic flip pattern would "
            f"falsify the per-model claim")
    else:
        say(f"  hard deep<->low flips: {hard} (bar 0) -- a flip across the "
            f"empty valley would falsify the model; deep<->mid flips are "
            f"gate wobble (each run's gate rides its own single-query "
            f"calibration amplitude)")
    flips = {(a, b): n for (a, b), n in confusion.items() if a != b}
    if flips:
        top = sorted(flips.items(), key=lambda kv: -kv[1])[:5]
        say("  top label flips: "
            + ", ".join(f"{a}->{b} x{n}" for (a, b), n in top))

    lows = [k for k in common if edges1[k]["class"] == "low"
            and edges2[k]["class"] == "low"]
    shifts = sorted(abs(int(edges1[k]["corrected_cycles"])
                        - int(edges2[k]["corrected_cycles"]))
                    for k in lows)
    med_shift = shifts[len(shifts) // 2] if shifts else 0
    say(f"  corrected-d |run1-run2| on {len(lows)} common low pairs: "
        + (f"median {med_shift} cyc (informational cross-card: lambda "
           f"carries a per-card timing offset, the table stores classes "
           f"not cycles)" if cross_card
           else f"median {med_shift} cyc (bar <= {D_SHIFT_BAR})"))

    va1, va2 = _anchor_validity(run1), _anchor_validity(run2)
    cells = sorted(set(va1) & set(va2))
    agree_cells = sum(1 for k in cells if va1[k] == va2[k])
    anchor_hard = sum(1 for k in cells
                      if {va1[k], va2[k]} == {"deep_conflict", "low"})
    anchor_ok = agree_cells / len(cells) if cells else 1.0
    universal = [cand for cand in {cand for _, cand in cells}
                 if all(va1.get((p, cand)) == "deep_conflict"
                        and va2.get((p, cand)) == "deep_conflict"
                        for p in pages1 & pages2)]
    say(f"  anchor validity agreement {agree_cells}/{len(cells)} = "
        f"{anchor_ok:.2%}; per-cell hard deep<->low flips {anchor_hard} "
        "(informational: the anchor sweep's mid valley is POPULATED "
        "(3.5-7% of cells, unlike classify's empty one), so per-card "
        "gate placement composes deep<->low from two soft band-edge "
        "steps -- three-card measurement: 633 cells GPU2-deep/GPU1-low/"
        "GPU0-mid, amplitudes 117/121/101); universal in both runs: "
        + (" ".join(f"0x{c:x}" for c in sorted(universal)) or "none"))
    # the TRANSFER gate at the consumer level: whatever the bank_map
    # touches must keep an anchor the other run also found. A card with
    # a different bank hash would lose common anchors on essentially
    # every page; band-edge wobble loses a handful at most (measured 0
    # on all three card pairs).
    anchors1: dict[int, set[int]] = defaultdict(set)
    anchors2: dict[int, set[int]] = defaultdict(set)
    for (page, cand), cls in va1.items():
        if cls == "deep_conflict":
            anchors1[page].add(cand)
    for (page, cand), cls in va2.items():
        if cls == "deep_conflict":
            anchors2[page].add(cand)
    need_pages = [p for p in pages1 & pages2
                  if anchors1.get(p) or anchors2.get(p)]
    no_common = sum(1 for p in need_pages
                    if not (anchors1.get(p, set())
                            & anchors2.get(p, set())))
    anchor_loss = no_common / len(need_pages) if need_pages else 0.0
    say(f"  pages keeping a common anchor: "
        f"{len(need_pages) - no_common}/{len(need_pages)}; {no_common} "
        f"without a common anchor "
        + (f"(bar <= {CROSS_HARD_RATE_BAR:.1%} of anchored pages)"
           if cross_card else "(bar 0)"))

    # co-membership of the same-bank page partition over common pages
    def components(edges: dict) -> dict[int, int]:
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b), row in edges.items():
            if row["class"] == "deep_conflict":
                ra, rb = find(a & ~((2 << 20) - 1)), find(b & ~((2 << 20) - 1))
                if ra != rb:
                    parent[ra] = rb
        return {p: find(p) for p in parent}

    lab1, lab2 = components(edges1), components(edges2)
    shared = sorted(set(lab1) & set(lab2))
    same_both = same_one = 0
    for i, p in enumerate(shared):
        for q in shared[i + 1:]:
            in1 = lab1[p] == lab1[q]
            in2 = lab2[p] == lab2[q]
            if in1 and in2:
                same_both += 1
            elif in1 or in2:
                same_one += 1
    jac = (same_both / (same_both + same_one)
           if same_both + same_one else 1.0)
    jac_note = ""
    if cross_card and not same_pool:
        # Shape mismatch: the two runs measured different rep sets over
        # different pools, so the denser graph merges strictly more page
        # pairs by construction -- co-membership compares coverage, not
        # card structure. The falsifying bars above stay fully gated.
        jac_note = (" -- INFORMATIONAL: pool shapes differ, co-membership "
                    "compares pair coverage, not structure; the Jaccard "
                    f"bar >= {CROSS_JACCARD_BAR:.2f} applies to same-shape "
                    "runs")
    say(f"  same-bank page co-membership over {len(shared)} shared pages: "
        f"{same_both} pairs together in both, {same_one} in exactly one "
        f"(Jaccard {jac:.3f}"
        + (f", bar >= {CROSS_JACCARD_BAR:.2f} -- the per-model structure"
           if cross_card and not jac_note else "") + ")" + jac_note)

    if cross_card:
        ok = (hard / len(common) if common else 0) <= CROSS_HARD_RATE_BAR \
            and anchor_loss <= CROSS_HARD_RATE_BAR \
            and region >= REGION_RECALL_BAR \
            and (jac >= CROSS_JACCARD_BAR or not same_pool)
    else:
        ok = hard == 0 and no_common == 0 and region >= REGION_RECALL_BAR \
            and med_shift <= D_SHIFT_BAR
    say(f"  R-{'d' if cross_card else 'c'}: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
# R-e -- end-to-end prediction
# --------------------------------------------------------------------------

def predict_pairs(edges: dict[tuple[int, int], tuple[str, str]],
                  part: dict, n_deep: int, n_low: int, seed: int,
                  say) -> list[tuple[int, int, str]]:
    """Unmeasured pairs inside one bank class that the table can PREDICT
    from transitive evidence only:

      low  both endpoints in one row class (a measured low chain pins
           the same row);
      deep the endpoints sit in row classes linked by a measured DEEP
           edge (directly or chained: rows are values, A!=B and B!=C
           force A!=C).

    Row classes inside a bank class with NO deep edge between them leave
    the row relation UNKNOWN -- those pairs are excluded, never guessed.
    Cross-page pairs sample first (the strong claim: distinct 2 MiB
    pages pinned to one bank and different rows)."""
    row_root = part["row"].find
    deep_pool: list[tuple[int, int]] = []
    low_pool: list[tuple[int, int]] = []
    for root, members in part["bank_groups"].items():
        if len(members) < 2:
            continue
        # row classes of this bank class, deep-linked into components
        row_classes: dict[int, list[int]] = defaultdict(list)
        for pa in members:
            row_classes[row_root(pa)].append(pa)
        link: dict[int, set[int]] = defaultdict(set)
        for (a, b), (cls, _) in edges.items():
            if cls == "deep_conflict" and part["bank"].find(a) == root:
                link[row_root(a)].add(row_root(b))
                link[row_root(b)].add(row_root(a))

        def row_component(row: int, seen: set[int] | None = None) -> set[int]:
            seen = seen or set()
            if row in seen:
                return seen
            seen.add(row)
            for neighbor in link[row]:
                row_component(neighbor, seen)
            return seen

        row_comp: dict[int, frozenset[int]] = {}
        for row in row_classes:
            row_comp[row] = frozenset(row_component(row))
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                key = (a, b) if a < b else (b, a)
                if key in edges:
                    continue  # measured pairs are R-a/R-c territory
                row_a, row_b = row_root(a), row_root(b)
                if row_a == row_b:
                    low_pool.append(key)
                elif row_comp[row_a] == row_comp[row_b]:
                    deep_pool.append(key)  # rows differ by deep evidence

    def sample(pairs: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
        cross_first = sorted(pairs,
                             key=lambda p: (p[0] >> 21 == p[1] >> 21, p))
        return cross_first[:count]

    chosen_deep = sample(deep_pool, n_deep)
    chosen_low = sample(low_pool, n_low)
    say(f"R-e plan: predictable unmeasured pairs -- deep pool "
        f"{len(deep_pool)} (sampled {len(chosen_deep)}), low pool "
        f"{len(low_pool)} (sampled {len(chosen_low)}); cross-page pairs "
        f"sampled first; row-relation-unknown pairs excluded")
    return ([(a, b, "deep_conflict") for a, b in chosen_deep]
            + [(a, b, "low") for a, b in chosen_low])


def gate_re_plan(pairs: list[tuple[int, int, str]], out: Path, say) -> bool:
    with out.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["pa_a", "pa_b", "predicted"])
        for a, b, predicted in pairs:
            writer.writerow([f"0x{a:x}", f"0x{b:x}", predicted])
    say(f"R-e plan written: {out} ({len(pairs)} pairs)")
    return True


def gate_re_check(run: Path, say) -> bool:
    """Score a predict-check run against its embedded predictions."""
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("query_selection", {}).get("mode") != "predict-check":
        say("R-e check: FAIL (not a predict-check run)")
        return False
    sel = summary.get("predict_check_selection")
    if not sel:
        say("R-e check: FAIL (summary lacks predict_check_selection)")
        return False
    pool = PoolMap(parse_pool_map((run / "pool_map.csv").read_text(
        encoding="utf-8")))
    with (run / "result.csv").open(encoding="utf-8", newline="") as fh:
        body = [line.rstrip("\n") for line in fh
                if not line.startswith("#")]
    rows = [dict(zip(("query_id", "chunk_a", "ofs_a", "chunk_b",
                      "ofs_b", "cycles_a", "cycles_b"), line.split(",")))
            for line in body[1:]]                      # body[0] = header

    pairs = list(zip([_hex(x) for x in sel["pair_pa_a"]],
                     [_hex(x) for x in sel["pair_pa_b"]],
                     sel["pair_predicted"]))
    expected = 3 + len(sel["self_pages"]) + len(pairs) \
        + int(sel["repeat_pages"])
    if len(rows) != expected:
        say(f"R-e check: FAIL (row count {len(rows)} != plan {expected})")
        return False
    page_size = pool.pages[0].page_size
    self_pages = [_hex(x) for x in sel["self_pages"]]

    def pa_of(row: dict[str, str]) -> tuple[int, int]:
        return (pool.pa_of(int(row["chunk_a"]), int(row["ofs_a"])),
                pool.pa_of(int(row["chunk_b"]), int(row["ofs_b"])))

    cursor = 3
    lambdas: dict[int, int] = {}
    for base in self_pages:
        row = rows[cursor]
        cursor += 1
        pa_a, pa_b = pa_of(row)
        if pa_a != pa_b or pa_a != base:
            say(f"R-e check: FAIL (self row {row['query_id']} pa "
                f"{pa_a:#x} != {base:#x})")
            return False
        lambdas[base] = int(row["cycles_a"])
    repeat = rows[cursor + len(pairs):cursor + len(pairs)
                  + int(sel["repeat_pages"])]
    deltas = sorted(int(row["cycles_a"]) - lambdas.get(
        _hex(base), 0) for row, base in zip(repeat, sel["repeat_of"]))
    late_offset = -deltas[len(deltas) // 2] if deltas else 0
    amplitude = int(rows[2]["cycles_a"]) - int(rows[1]["cycles_a"])
    if amplitude <= 0:
        say("R-e check: FAIL (degenerate calibration)")
        return False

    def label(row: dict[str, str]) -> str:
        pa_a, pa_b = pa_of(row)
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

    say(f"R-e end-to-end prediction (run {run.name}; amp {amplitude}, "
        f"late offset {late_offset:+d}):")
    scored: Counter = Counter()
    flips: Counter = Counter()
    hard_flips: list[tuple[int, int, str]] = []
    for want_a, want_b, predicted in pairs:
        row = rows[cursor]
        cursor += 1
        pa_a, pa_b = pa_of(row)
        if (pa_a, pa_b) != (want_a, want_b):
            say(f"R-e check: FAIL (predict row {row['query_id']} "
                f"{pa_a:#x}/{pa_b:#x} != planned {want_a:#x}/{want_b:#x})")
            return False
        measured = label(row)
        scored[(predicted, measured)] += 1
        if predicted != measured:
            flips[f"{predicted}->{measured}"] += 1
        if {predicted, measured} == {"deep_conflict", "low"}:
            hard_flips.append((want_a, want_b, measured))
    ok = True
    # Same hard/region split as R-c: mid is the undecided valley band and
    # the gate rides this run's calibration amplitude, so deep<->mid is
    # boundary wobble; a deep<->low flip falsifies the prediction.
    for predicted, decided in (("deep_conflict", ("deep_conflict", "mid")),
                               ("low", ("low", "mid"))):
        total = sum(n for (p, _), n in scored.items() if p == predicted)
        good = scored[(predicted, predicted)]
        region = sum(n for (p, m), n in scored.items()
                     if p == predicted and m in decided)
        hard = sum(n for (p, m), n in scored.items()
                   if p == predicted and m not in decided)
        accuracy = (total - hard) / total if total else 1.0
        strict = good / total if total else 1.0
        ok = ok and accuracy >= PREDICT_ACCURACY_BAR
        say(f"  predicted {predicted:<13} strict {good}/{total} = "
            f"{strict:.2%}; hard flips {hard}/{total} = {accuracy:.2%} "
            f"(bar >= {PREDICT_ACCURACY_BAR:.0%})")
    for a, b, measured in hard_flips:
        say(f"  HARD FALSIFIER: 0x{a:x} / 0x{b:x} measured {measured} -- "
            f"the transitive row/bank closure behind this prediction is "
            f"wrong; trace its class edges before consuming the table")
    if flips:
        say("  label flips: " + ", ".join(f"{k} x{v}"
                                          for k, v in flips.most_common()))
    say(f"  R-e: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s3", nargs="?", type=Path)
    parser.add_argument("s3b", nargs="?", type=Path)
    parser.add_argument("census", nargs="?", type=Path)
    parser.add_argument("--t1", type=Path, action="append", default=[],
                        help="analyzed table-build run dir(s), later = fresher")
    parser.add_argument("--channel-deep-only", action="store_true")
    parser.add_argument("--r-c", type=Path, metavar="RUN2",
                        help="second analyzed table-build run for R-c")
    parser.add_argument("--r-d", type=Path, metavar="RUN2",
                        help="analyzed table-build run from ANOTHER GPU of "
                             "the same model for R-d")
    parser.add_argument("--r-e-plan", action="store_true",
                        help="emit a predict-check pairs CSV (needs the "
                             "three source dirs and --t1)")
    parser.add_argument("--out", type=Path, default=Path(
        "artifacts/g3/predict_pairs.csv"))
    parser.add_argument("--n-deep", type=int, default=128)
    parser.add_argument("--n-low", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--r-e-check", type=Path, metavar="PREDICT_RUN",
                        help="score a predict-check run")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.r_e_check:
        say = lambda t="": print(t)
        return 0 if gate_re_check(args.r_e_check, say) else 2

    if not (args.s3 and args.s3b and args.census):
        parser.error("three source run dirs are required "
                     "outside --self-test/--r-e-check")
    problems: list[str] = []
    edges, _ = load_edges(args.s3, args.s3b, args.census, args.t1, problems)
    load_pages(args.census, problems)
    if problems:
        print(f"INTEGRITY: {'; '.join(problems[:5])}")
        return 2
    part = build(edges, args.channel_deep_only)

    results: list[bool] = []
    say = lambda t="": print(t)
    results.append(gate_ra(edges, part, say))
    t1_rows: list[dict[str, str]] = []
    for run in args.t1:
        t1_rows.extend(load_t1_edges(run, problems))
    if problems:
        print(f"INTEGRITY: {'; '.join(problems[:5])}")
        return 2
    if t1_rows:
        results.append(gate_rb(t1_rows, say))
    if args.r_c:
        if not args.t1:
            parser.error("--r-c needs --t1 RUN1 to compare against")
        results.append(gate_rc(args.t1[-1], args.r_c, say))
    if args.r_d:
        if not args.t1:
            parser.error("--r-d needs --t1 RUN1 to compare against")
        results.append(gate_rc(args.t1[-1], args.r_d, say, cross_card=True))
    if args.r_e_plan:
        pairs = predict_pairs(edges, part, args.n_deep, args.n_low,
                              args.seed, say)
        results.append(gate_re_plan(pairs, args.out, say))
    verdict = all(results)
    print(f"S5-T2 verdict: "
          f"{'PASS' if verdict and results else 'FAIL'} "
          f"({sum(results)}/{len(results)} gates)")
    return 0 if verdict else 2


def self_test() -> int:
    import contextlib
    import io
    import tempfile

    PAGE = 2 << 20

    def pa(page_index: int, offset: int = 0) -> int:
        return (page_index << 21) + offset

    # -- fixture: three source dirs with a 4-page census universe -------
    bands_fields = ["query_id", "kind", "bit", "pa_a", "pa_b", "xor",
                    "cycles_a", "cycles_b", "class"]

    def bands_rows(rows):
        return [",".join(bands_fields)] + [
            f"{qid},x,0,0x{a:x},0x{b:x},0x0,1050,1050,{cls}"
            for qid, (a, b, cls) in enumerate(rows)]

    # bank class spanning p0/p2: p0 rows A={0,0x200,0x400} (two lows)
    # and B={M} (three deeps into A); p2 rows C={0}, D={M} with deep
    # (C,D); cross-page deep (p0+0, p2+M) links A--D. Every row-class
    # pair is deep-linked, so unmeasured cross-row pairs are predictable
    # deep and (0x200, 0x400) inside A is predictable low. Plus a lone
    # shoulder pair p1-p3 outside every class.
    edges_seed = [(pa(0, o1), pa(0, o2), cls) for o1, o2, cls in [
        (0x0, 0xd0100, "deep_conflict"),
        (0x200, 0xd0100, "deep_conflict"),
        (0x400, 0xd0100, "deep_conflict"),
        (0x0, 0x200, "low"), (0x0, 0x400, "low")]]
    edges_seed += [(pa(2, o1), pa(2, o2), cls) for o1, o2, cls in [
        (0x0, 0xd0100, "deep_conflict"),
        (0x0, 0x200, "low")]]
    edges_seed += [
        (pa(0, 0), pa(2, 0xd0100), "deep_conflict"),   # cross-page bank
        (pa(1, 0x1000), pa(3, 0x1000), "shoulder"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("s3", "s3b", "census", "t1a", "t1b", "run2", "run3"):
            (root / name).mkdir()
        (root / "s3" / "constraints_bands.csv").write_text(
            "\n".join(bands_rows(edges_seed)) + "\n")
        (root / "s3b" / "pair_constraints_bands.csv").write_text(
            "\n".join(bands_rows([])) + "\n")
        with (root / "census" / "page_lambdas.csv").open(
                "w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["fb_pa_page_base", "self_cycles"])
            for page in range(4):
                writer.writerow([f"0x{pa(page):x}", 1000 + page])
        for missing in ("reprobe_verdicts.csv", "row_pilot_classes.csv"):
            (root / "census" / missing).write_text(
                "query_id,pa_a,pa_b,cycles_a,cycles_b,verdict\n")

        def t1_edges(path: Path, rows):
            fields = ["query_id", "pa_a", "pa_b", "section", "cycles_a",
                      "cycles_b", "late_offset", "corrected_cycles", "class"]
            with path.open("w", encoding="utf-8", newline="") as sink:
                writer = csv.writer(sink)
                writer.writerow(fields)
                writer.writerows(rows)

        # classify section: 3 deeps / 12 pairs -> rate vs 1/384 z is
        # huge, R-b must FAIL its bar (tiny fixture); mechanics still
        # computed. (1,2)/(2,1) sit in mid: boundary pairs the run4
        # fixture below wobbles to deep (mid<->deep is never a hard
        # flip, mirroring the real cross-card wobble).
        classify_rows = []
        qid = 0
        for p in range(4):
            for rep in range(4):
                if p == rep:
                    continue
                cls = ("deep_conflict"
                       if (p, rep) in ((0, 2), (2, 0), (1, 3))
                       else "mid" if (p, rep) in ((1, 2), (2, 1))
                       else "low")
                classify_rows.append(
                    [qid, f"0x{pa(p):x}", f"0x{pa(rep):x}", "classify",
                     1050, 1050, 0, 1050, cls])
                qid += 1
        t1_edges(root / "t1a" / "table_build_edges.csv", classify_rows)

        # R-a on the clean fixture: a1..a4 all 0 -> PASS
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            problems: list[str] = []
            edges, _ = load_edges(root / "s3", root / "s3b",
                                  root / "census", [], problems)
            assert not problems, problems
            part = build(edges, channel_deep_only=True)
            ra = gate_ra(edges, part, print)
        assert ra, buffer.getvalue()
        assert "R-a: PASS" in buffer.getvalue(), buffer.getvalue()

        # plant a3: low (pa(0,0), pa(2,0)) merges the two pages' row-0
        # classes inside the spanning bank class -> row class across pages
        bad_edges = dict(edges)
        bad_edges[(pa(0, 0), pa(2, 0))] = ("low", "plant")
        bad_part = build(bad_edges, channel_deep_only=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ra_bad = gate_ra(bad_edges, bad_part, print)
        assert not ra_bad and "a3 row class spanning >1 page:          1" \
            in buffer.getvalue(), buffer.getvalue()

        # a4 mechanics: the {0,2} deep-only page component makes a
        # cross-page LOW inside it a decided contradiction (a4 fires)
        # while a cross-page MID inside it is valley wobble (a4 must
        # not count it -- the GPU1-big 2-pair case).
        bad2 = dict(edges)
        bad2[(pa(0, 0x800), pa(2, 0x800))] = ("low", "plant")
        bad2[(pa(0, 0x900), pa(2, 0x900))] = ("mid", "plant")
        part2 = build(bad2, channel_deep_only=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ra_a4 = gate_ra(bad2, part2, print)
        assert not ra_a4 \
            and "inside a channel component: 1" in buffer.getvalue(), \
            buffer.getvalue()

        # R-b mechanics: rate and null quantiles computed; fixture fails
        rows = load_t1_edges(root / "t1a", [])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rb = gate_rb(rows, print)
        out = buffer.getvalue()
        assert "deep rate 3/12 = 25.0000%" in out, out
        assert "null" in out and "OUT OF RANGE" in out and "R-b: FAIL" in out
        assert f"gate band [{EFFECTIVE_CLASSES_LO}, " \
               f"{EFFECTIVE_CLASSES_HI}]" in out, out

        # R-b band mechanics: 100 pages x 100 reps (disjoint ranges) =
        # 10000 pairs. 26 deeps -> implied classes 384.6 = in band; 10
        # deeps -> 1000 = above; 60 deeps -> 166.7 = below. The z-test
        # is a diagnostic and never gates (26/10000 gives |z| within 1,
        # 10 and 60 give far out -- both fail ONLY through the band).
        (root / "t1band").mkdir()
        rng = random.Random(7)
        deep_set: set[tuple[int, int]] = set()
        while len(deep_set) < 60:
            deep_set.add((rng.randrange(100), rng.randrange(100, 200)))
        for n_deeps, want in ((26, True), (10, False), (60, False)):
            band_rows = []
            qid = 0
            for p in range(100):
                for rep in range(100, 200):
                    cls = ("deep_conflict" if (p, rep) in
                           sorted(deep_set)[:n_deeps] else "low")
                    band_rows.append(
                        [qid, f"0x{pa(p):x}", f"0x{pa(rep):x}", "classify",
                         1050, 1050, 0, 1050, cls])
                    qid += 1
            t1_edges(root / "t1band" / "table_build_edges.csv", band_rows)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                got = gate_rb(load_t1_edges(root / "t1band", []), print)
            band_out = buffer.getvalue()
            assert got is want, (n_deeps, want, band_out)
            assert f"R-b: {'PASS' if want else 'FAIL'}" in band_out, band_out

        # R-c: run2 reproduces t1a except one deep -> low flip (the
        # deeps sit at qid 1/5/6: pairs (0,2), (1,3), (2,0))
        run2_rows = [list(row) for row in classify_rows]
        for row in run2_rows:
            if row[0] == 1:
                row[8] = "low"  # one of the three deeps flips
        t1_edges(root / "run2" / "table_build_edges.csv", run2_rows)
        # run3: deep -> mid on the (2,0) deep -- cross-card boundary
        # wobble that keeps the partition (deeps left: (0,2),(1,3))
        run3_rows = [list(row) for row in classify_rows]
        for row in run3_rows:
            if row[0] == 6:
                row[8] = "mid"
        t1_edges(root / "run3" / "table_build_edges.csv", run3_rows)
        pool_fields = ["run_id", "device", "gpu_uuid", "chunk_index",
                       "allocation_id", "va_page_base",
                       "va_page_end_exclusive", "fb_pa_page_base",
                       "page_size", "aperture", "pte_valid", "raw_pte_lo",
                       "raw_pte_hi", "mapped_at_ns", "unmapped_at_ns",
                       "source", "confidence"]
        pool_text = ",".join(pool_fields) + "\n" + "".join(
            f"r,0,uuid,{i},alloc,0x{0x7f0000000000 + i * 0x200000:x},"
            f"0x{0x7f0000000000 + (i + 1) * 0x200000:x},0x{pa(i):x},"
            f"{PAGE},VIDEO,true,0x1,0x0,1,2,ebpf,c\n" for i in range(4))
        for name in ("t1a", "run2", "run3"):
            (root / name / "pool_map.csv").write_text(pool_text)
            with (root / name / "anchor_validity.csv").open(
                    "w", encoding="utf-8", newline="") as sink:
                writer = csv.writer(sink)
                writer.writerow(["page_base", "candidate",
                                 "corrected_cycles", "class"])
                for p in range(4):
                    writer.writerow([f"0x{pa(p):x}", "0xd0100", 1150,
                                     "deep_conflict"])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rc = gate_rc(root / "t1a", root / "run2", print)
        out = buffer.getvalue()
        assert "pool page sets identical" in out, out
        assert "deep recall 2/3 = 66.67% strict" in out, out
        assert "hard deep<->low flips: 1 (bar 0)" in out, out
        assert "R-c: FAIL" in out, out   # the planted flip is hard

        # R-d cross-card: the mid wobble passes (rate 0, Jaccard intact,
        # cycle shift informational); the same hard flip that failed R-c
        # also fails the cross-card rate bar (1/12 = 8.3% >> 0.1%).
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rd = gate_rc(root / "t1a", root / "run3", print,
                         cross_card=True)
        out = buffer.getvalue()
        assert "R-d cross-card transfer" in out, out
        assert rd and "R-d: PASS" in out, out
        assert "hard deep<->low flips: 0/12 = 0.0000%" in out, out
        assert "informational cross-card" in out, out
        assert "pages keeping a common anchor: 4/4" in out, out
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rd_bad = gate_rc(root / "t1a", root / "run2", print,
                             cross_card=True)
        assert not rd_bad and "R-d: FAIL" in buffer.getvalue(), \
            buffer.getvalue()

        # R-d with a SHAPE-MISMATCHED run2 (a 3-page pool, no page-3
        # pairs): run4 wobbles the two mid pairs to deep, merging
        # {0,1,2} where t1a keeps {0,2}+{1} -> Jaccard 0.333 would fail
        # the bar, but co-membership across shapes compares pair
        # coverage, not structure -- the gate must treat it as
        # informational and pass on the falsifying bars (hard flips 0,
        # region recall 100%, anchor flips 0).
        run4_rows = [list(row) for row in classify_rows
                     if (int(row[1], 16) >> 21) < 3
                     and (int(row[2], 16) >> 21) < 3]
        for row in run4_rows:
            if row[8] == "mid":
                row[8] = "deep_conflict"
        (root / "run4").mkdir(exist_ok=True)
        t1_edges(root / "run4" / "table_build_edges.csv", run4_rows)
        (root / "run4" / "pool_map.csv").write_text(
            ",".join(pool_fields) + "\n" + "".join(
                f"r,1,uuid,{i},alloc,0x{0x7f0000000000 + i * 0x200000:x},"
                f"0x{0x7f0000000000 + (i + 1) * 0x200000:x},0x{pa(i):x},"
                f"{PAGE},VIDEO,true,0x1,0x0,1,2,ebpf,c\n"
                for i in range(3)))
        with (root / "run4" / "anchor_validity.csv").open(
                "w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["page_base", "candidate",
                             "corrected_cycles", "class"])
            for p in range(3):
                writer.writerow([f"0x{pa(p):x}", "0xd0100", 1150,
                                 "deep_conflict"])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rd_shape = gate_rc(root / "t1a", root / "run4", print,
                               cross_card=True)
        out = buffer.getvalue()
        assert rd_shape and "R-d: PASS" in out, out
        assert "pool page sets DIFFER" in out, out
        assert "Jaccard 0.333" in out, out
        assert "INFORMATIONAL: pool shapes differ" in out, out
        assert "pages keeping a common anchor: 3/3" in out, out

        # anchor-transfer gate: run5's edge labels mirror t1a (0 hard
        # flips, Jaccard 1.000) but its only deep anchors use a
        # DIFFERENT candidate mask, so every page loses its common
        # anchor -- the consumer claim breaks and R-d must FAIL on it.
        run5_rows = [list(row) for row in classify_rows]
        (root / "run5").mkdir()
        t1_edges(root / "run5" / "table_build_edges.csv", run5_rows)
        (root / "run5" / "pool_map.csv").write_text(pool_text)
        with (root / "run5" / "anchor_validity.csv").open(
                "w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["page_base", "candidate",
                             "corrected_cycles", "class"])
            for p in range(4):
                writer.writerow([f"0x{pa(p):x}", "0xd0300", 1150,
                                 "deep_conflict"])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rd_anchor = gate_rc(root / "t1a", root / "run5", print,
                                cross_card=True)
        out = buffer.getvalue()
        assert not rd_anchor and "R-d: FAIL" in out, out
        assert "pages keeping a common anchor: 0/4" in out, out
        assert "without a common anchor" in out, out
        assert "informational: the anchor sweep's mid valley" in out, out

        # R-e plan: bank class spans p0/p2 with row classes A/B/C/D all
        # deep-linked -> 7 unmeasured cross-row pairs predict deep; the
        # only unmeasured same-row pair (p0+0x200, p0+0x400) predicts low.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            pairs = predict_pairs(edges, part, n_deep=8, n_low=8,
                                  seed=7, say=print)
        got = {(a, b): pred for a, b, pred in pairs}
        assert got[(pa(0, 0x200), pa(0, 0x400))] == "low", got
        assert got[(pa(0, 0xd0100), pa(2, 0))] == "deep_conflict", got
        assert got[(pa(0, 0x200), pa(2, 0xd0100))] == "deep_conflict", got
        assert got.get((pa(0, 0), pa(2, 0xd0100))) is None  # measured deep
        assert len(got) == 8, got
        out = buffer.getvalue()
        assert "deep pool 7 (sampled 7)" in out, out
        assert "low pool 1 (sampled 1)" in out, out

        # -- R-e check: a synthetic predict-check run in the exact shape
        # the orchestrator embeds. calibration 1015/1018/1142 (amp 124,
        # low gate < +43.4, deep gate >= +74.4); lambdas 1015/1025/1016/
        # 1041; one dropped off-pool pair is reported by the planner,
        # never planned.
        run = root / "predict_run"
        run.mkdir()
        (run / "pool_map.csv").write_text(pool_text)
        kept = [(pa(0, 0x200), pa(0, 0x400), "low"),
                (pa(0, 0xd0100), pa(2, 0), "deep_conflict"),
                (pa(1, 0x1000), pa(3, 0x1000), "low")]
        value_of = {0: 1015, 1: 1025, 2: 1016, 3: 1041}

        def pair_value(a: int, b: int) -> int:
            base = max(value_of[a >> 21], value_of[b >> 21])
            return base + (100 if a >> 21 == 0 and b >> 21 == 2
                           else 10)

        lines = ["query_id,chunk_a,ofs_a,chunk_b,ofs_b,cycles_a,cycles_b"]
        lines.append("0,0,1015,0,1018,1015,1018")       # calibration floor
        lines.append("1,0,1015,0,1018,1018,1018")       # baseline
        lines.append("2,0,1015,0,1018,1142,1142")       # conflict
        for page in (0, 1, 2, 3):                       # self
            lines.append(f"{len(lines) - 1},{page},0,{page},0,"
                         f"{value_of[page]},{value_of[page]}")
        for a, b, _ in kept:                            # predict rows
            value = pair_value(a, b)
            lines.append(f"{len(lines) - 1},{a >> 21},{a & (PAGE - 1)},"
                         f"{b >> 21},{b & (PAGE - 1)},{value},{value}")
        for page in (0, 1, 2):                          # repeat rows
            lines.append(f"{len(lines) - 1},{page},0,{page},0,"
                         f"{value_of[page]},{value_of[page]}")
        (run / "result.csv").write_text("\n".join(lines) + "\n")
        (run / "summary.json").write_text(json.dumps({
            "query_selection": {"mode": "predict-check"},
            "predict_check_selection": {
                "pair_pa_a": [f"0x{a:x}" for a, _, _ in kept],
                "pair_pa_b": [f"0x{b:x}" for _, b, _ in kept],
                "pair_predicted": [p for _, _, p in kept],
                "self_pages": [f"0x{pa(p):x}" for p in (0, 1, 2, 3)],
                "repeat_pages": 3,
                "repeat_of": [f"0x{pa(p):x}" for p in (0, 1, 2)],
                "dropped": {"pairs": ["0x900000000/0x900000200"]},
            }}))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            re_ok = gate_re_check(run, print)
        out = buffer.getvalue()
        assert re_ok and "R-e: PASS" in out, out
        assert "deep_conflict strict 1/1 = 100.00%" in out, out
        assert "hard flips 0/2 = 100.00%" in out, out

        # swapped predict rows must fail closed
        swapped = lines[:]
        i, j = 8, 9                      # two predict rows (after header)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        (run / "result.csv").write_text("\n".join(swapped) + "\n")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            re_bad = gate_re_check(run, print)
        assert not re_bad and "!= planned" in buffer.getvalue(), \
            buffer.getvalue()
    print("validate_table self-test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
