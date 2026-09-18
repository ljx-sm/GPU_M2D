#!/usr/bin/env python3
"""GPU_M2D G3 S5-T3: EMT query API -- consume the canonical mapping table.

The S5-T1/T2/T3.0 deliverable is a measured relation table over the card's
PA universe; this tool is the consumer surface G4/G5 call into. It answers
relation queries, runs the fail-closed coverage check, and selects fault
sites under the agreed (simplified) fault model:

  SEU                     any single bit, any in-universe PA
  MCU same-row            same bank, same row, different column   [measured]
  MCU same-bank-diff-row  same bank, different row                [measured]
  MCU row-adjacent        same bank, assumed neighboring row      [assumed]
  MCU column-adjacent     FOLDED into same-row (column distance is
                          unobservable in the timing channel: column
                          changes never alter latency)
  MCU DQ-adjacent         UNSUPPORTED (R3; bits below the ~32 B burst)

Provenance labels -- every claim carries one:

  measured    a direct measurement backs the pair: an anchor sweep cell
              (page_base, page_base^M read deep on a majority of cards),
              a row-class edge (in-bank low), or a page pair inside one
              same-bank component (see below).
  transitive  closure over measured edges (the same-bank relation is an
              equivalence, so closure is sound; R-e validated the closure
              on never-measured pairs: deep 0/128 hard flips).
  assumed     a structural prior, never a measurement: row-adjacency via
              the GeForge App. B monotonic-row-stripe model (within one
              bank, ordering pages by PA orders their rows).
  unknown     no claim. Never guessed.

Soundness arguments the labels rest on (measured facts, see README):

  * distinct page components => distinct banks. The classify sections are
    COMPLETE over pages x reps: a page joins a component exactly when some
    rep shares its bank, so two pages of one bank always share a component
    (both deep-link to that bank's rep in the same run). Merging runs only
    unions components, preserving the property.
  * cross-page same component => different row, physically: two 2 MiB
    pages cannot share a (bank,row) -- the in-page column field cannot
    absorb PA bits >= 21 (validate_table gate a3).
  * node-level (in-page) bank classes do NOT get the completeness
    argument: in-page pairs were sparsely sampled, so two nodes in
    different in-page bank classes are UNKNOWN, not different-bank.
  * different row is only claimed from a direct deep measurement (anchor
    cell) or the cross-page physical argument -- never from "different row
    classes" (row inequality is not transitive through unmeasured pairs).

Coverage discipline (fail-closed): a PA whose 2 MiB page is outside the
table universe FAILS (exit 2) -- the driver handed out memory the table
never measured and the relation must be built, not guessed. Unlinked
pages (bank drew none of the 1024 uniform reps) are covered-but-unknown:
a warning by default, a failure under --require-linked.

    python3 query_table.py --table artifacts/g3/table_v4 \
        --query 0x1ee00000 0x2ae00000
    python3 query_table.py --table ... --check 0x1eed0100 0x2aed3880
    python3 query_table.py --table ... --check-pool <g3_run_dir>
    python3 query_table.py --table ... --select row-adjacent \
        --near 0x1ee00000 --count 4
    python3 query_table.py --table ... --annotate-pool <g3_run_dir> \
        --out g4_gddr_snapshot.csv
    python3 query_table.py --table ... --build-anchors <run> <run> <run> \
        --out <table_dir>/page_anchors.csv

Exit codes: 0 ok, 2 fail-closed (coverage / unsupported model / no
candidates), 1 error (repo convention). `--self-test` pins the semantics
on synthetic fixtures.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_pool import PoolMap, parse_pool_map  # noqa: E402

PAGE_MASK = (2 << 20) - 1

SELECT_MODELS = ("seu", "same-bank-diff-row", "same-row-diff-col",
                 "row-adjacent")
REFUSED_MODELS = {
    "dq-adjacent": "unsupported per research-plan R3: the DQ/burst domain "
                   "(in-page bits below the ~32 B burst) has no observable "
                   "structure in the timing channel",
    "column-adjacent": "folded into 'same-row-diff-col': column DISTANCE is "
                       "unobservable in the timing channel (column changes "
                       "never alter latency); the simplified G5 fault model "
                       "uses same-row random-column sites instead",
}

ROW_PROBABLE_NOTE = ("row classes carry the R-e ~1% falsifier "
                     "(0x2aed3880/0x2aed7b00, reproducible)")

# The one reproducible row-class falsifier R-e measured (predict-check:
# both members read anchor-low vs M -- same-row inference -- yet the pair
# itself read deep, reproducibly). deep is hard evidence, so this exact
# pair is NOT a same-row site no matter what the row-class closure says;
# the closure keeps it (the lows were measured too) but every consumer
# surface must flag it.
KNOWN_ROW_FALSIFIERS = frozenset([
    (0x2aed3880, 0x2aed7b00), (0x2aed7b00, 0x2aed3880)])


def _hx(text: str) -> int:
    return int(text, 16)


def page_of(pa: int) -> int:
    return pa & ~PAGE_MASK


# --------------------------------------------------------------------------
# table loading
# --------------------------------------------------------------------------

class Table:
    """The canonical EMT plus its optional per-page anchor consensus."""

    def __init__(self, table_dir: Path, problems: list[str]):
        self.dir = table_dir
        seed = table_dir / "gddr_seed_table.csv"
        classes = table_dir / "bank_classes.csv"
        if not seed.is_file() or not classes.is_file():
            problems.append(f"missing table files in {table_dir} "
                            "(expected gddr_seed_table.csv + "
                            "bank_classes.csv; build with "
                            "build_bank_table.py)")
            return
        # page_base -> row dict of the per-page table
        self.pages: dict[int, dict[str, str]] = {}
        with seed.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                self.pages[_hx(row["page_base"])] = row
        # component -> sorted member pages (unlinked pages excluded)
        self.components: dict[str, list[int]] = defaultdict(list)
        for base, row in self.pages.items():
            if row["channel_root"]:
                self.components[row["channel_root"]].append(base)
        for members in self.components.values():
            members.sort()
        # pa -> node row; row_class -> member pas (multi-member classes only)
        self.nodes: dict[int, dict[str, str]] = {}
        self.row_members: dict[str, list[int]] = defaultdict(list)
        with classes.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                pa = _hx(row["pa"])
                self.nodes[pa] = row
                if int(row["row_size"]) >= 2:
                    self.row_members[row["row_class"]].append(pa)
        for members in self.row_members.values():
            members.sort()
        # optional per-page anchor consensus (page_anchors.csv, built by
        # --build-anchors from the T1 runs' anchor_validity.csv files)
        self.anchors: dict[int, list[tuple[int, int]]] = defaultdict(list)
        anchors_path = table_dir / "page_anchors.csv"
        if anchors_path.is_file():
            with anchors_path.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    if row["valid"] == "true":
                        self.anchors[_hx(row["page_base"])].append(
                            (_hx(row["candidate"]), int(row["votes"])))
        for masks in self.anchors.values():
            masks.sort(key=lambda mv: (-mv[1], mv[0]))

    def in_universe(self, pa: int) -> bool:
        return page_of(pa) in self.pages

    def linked(self, pa: int) -> bool:
        row = self.pages.get(page_of(pa))
        return bool(row and row["channel_root"])

    def anchor_mask(self, pa_a: int, pa_b: int) -> int | None:
        """The consensus-valid anchor mask iff (pa_a, pa_b) is a sweep cell
        (page_base, page_base^M) -- the only in-page pairs whose bank+row
        relation the sweep measured directly."""
        for base, other in ((pa_a, pa_b), (pa_b, pa_a)):
            if page_of(base) != page_of(other) or base != page_of(base):
                continue
            mask = other ^ base
            if any(mask == m for m, _ in self.anchors.get(page_of(base), [])):
                return mask
        return None


# --------------------------------------------------------------------------
# relation query
# --------------------------------------------------------------------------

def _page_note(tab: Table, pa: int) -> str:
    base = page_of(pa)
    row = tab.pages.get(base)
    if row is None:
        return f"{base:#x} (OUTSIDE the table universe)"
    if row["channel_root"]:
        return (f"{base:#x} (component {row['channel_root']}, "
                f"{row['channel_size']} pages)")
    return f"{base:#x} (covered, bank UNKNOWN -- unlinked page)"


def relation(tab: Table, pa_a: int, pa_b: int, say) -> dict[str, str]:
    """Print and return the (bank, row, column) relation with provenance."""
    out: dict[str, str] = {"same_bank": "unknown", "row": "unknown",
                           "column": "n/a", "bank_provenance": "-",
                           "row_provenance": "-"}
    say(f"relation {pa_a:#x} / {pa_b:#x}:")
    say(f"  pages: {_page_note(tab, pa_a)} | {_page_note(tab, pa_b)}")
    mask = tab.anchor_mask(pa_a, pa_b)
    node_a, node_b = tab.nodes.get(pa_a), tab.nodes.get(pa_b)
    if page_of(pa_a) == page_of(pa_b):
        base = page_of(pa_a)
        same_row_class = (node_a and node_b
                          and node_a["row_class"] == node_b["row_class"]
                          and int(node_a["row_size"]) >= 2)
        if same_row_class:
            out.update(same_bank="yes", bank_provenance="measured",
                       row="same", row_provenance="measured",
                       column="different")
            say("  same bank: YES -- same row implies same bank "
                "(row class is built inside a bank class)")
            say(f"  row:      SAME -- row class {node_a['row_class']} "
                f"(measured in-bank low; {ROW_PROBABLE_NOTE})")
            if (pa_a, pa_b) in KNOWN_ROW_FALSIFIERS:
                out["row_provenance"] = "measured-contradicted"
                say("  KNOWN FALSIFIER: the predict-check run measured "
                    "this exact pair DEEP (reproducible) -- the row-class "
                    "inference is wrong here; do NOT site same-row faults "
                    "on this pair")
            say("  column:   DIFFERENT -- same row, differing in-page "
                "offsets (column field includes bit 9, hardware-confirmed)")
        elif mask is not None:
            out.update(same_bank="yes", bank_provenance="measured",
                       row="different", row_provenance="measured")
            say(f"  same bank: YES -- anchor mask {mask:#x} is "
                "consensus-valid on this page (sweep cell read deep)")
            say("  row:      DIFFERENT -- direct anchor sweep cell "
                "(measured deep)")
            say("  column:   n/a (different rows)")
        else:
            same_bank_class = (node_a and node_b
                               and node_a["bank_class"] == node_b["bank_class"]
                               and int(node_a["bank_size"]) >= 2)
            if same_bank_class:
                out.update(same_bank="yes",
                           bank_provenance="transitive")
                say("  same bank: YES -- same in-page bank class "
                    "(transitive closure over deep edges)")
            else:
                say("  same bank: UNKNOWN -- in-page hash; only swept/"
                    "lattice offsets carry node knowledge, and distinct "
                    "in-page bank classes prove nothing (sparse sampling)")
            say("  row:      UNKNOWN -- no direct deep measurement between "
                "these nodes; differing row classes are not row inequality")
            say("  column:   n/a (row relation unknown)")
        say(f"  page context: {_page_note(tab, base)}")
    else:
        row_a, row_b = tab.pages.get(page_of(pa_a)), tab.pages.get(page_of(pa_b))
        if row_a is None or row_b is None:
            say("  same bank: UNKNOWN -- a page is outside the table "
                "universe (fail-closed: --check refuses such pools)")
        elif not (row_a["channel_root"] and row_b["channel_root"]):
            say("  same bank: UNKNOWN -- an unlinked page (its bank drew "
                "none of the 1024 uniform reps; an incremental re-run "
                "resolves it)")
        elif row_a["channel_root"] == row_b["channel_root"]:
            out.update(same_bank="yes", bank_provenance="transitive",
                       row="different", row_provenance="measured")
            say("  same bank: YES -- same page component (transitive "
                "closure over the COMPLETE classify sections; R-e "
                "validated the closure: deep 0/128 hard flips)")
            say("  row:      DIFFERENT -- physical: two 2 MiB pages "
                "cannot share a (bank,row) (gate a3)")
            say("  column:   n/a (cross-page)")
        else:
            out.update(same_bank="no", bank_provenance="measured")
            say("  same bank: NO -- distinct components are distinct "
                "banks: the classify sections measured every page against "
                "every rep, so same-bank pages always join one component")
            say("  row:      n/a (different bank; row identity is "
                "per-bank)")
            say("  column:   n/a (cross-page)")
    return out


# --------------------------------------------------------------------------
# coverage (fail-closed)
# --------------------------------------------------------------------------

def check_pas(tab: Table, pas: list[int], require_linked: bool,
              say) -> bool:
    outside = [pa for pa in pas if not tab.in_universe(pa)]
    unlinked = sorted({page_of(pa) for pa in pas
                       if tab.in_universe(pa) and not tab.linked(pa)})
    say(f"coverage: {len(pas) - len(outside)}/{len(pas)} PAs in the table "
        f"universe ({len(tab.pages)} pages)")
    if unlinked:
        say(f"  WARNING: {len(unlinked)} distinct pages covered but "
            f"UNLINKED (bank unknown; sample "
            + " ".join(f"{p:#x}" for p in unlinked[:5]) + ")")
    if outside:
        say("  FAIL: outside the universe (fail-closed -- the relation "
            "must be measured, never extrapolated; sample "
            + " ".join(f"{p:#x}" for p in outside[:5]) + ")")
        return False
    if require_linked and unlinked:
        say("  FAIL: --require-linked and unlinked pages present")
        return False
    say("  PASS")
    return True


def check_pool(tab: Table, run: Path, require_linked: bool, say) -> bool:
    pool = PoolMap(parse_pool_map((run / "pool_map.csv").read_text(
        encoding="utf-8")))
    pas = [page.fb_pa_page_base for page in pool.pages]
    say(f"pool {run.name}: {len(pas)} pages")
    return check_pas(tab, pas, require_linked, say)


# --------------------------------------------------------------------------
# fault-site selection (the simplified G5 fault model)
# --------------------------------------------------------------------------

def select_sites(tab: Table, model: str, near: int | None, count: int,
                 say) -> bool:
    if model in REFUSED_MODELS:
        say(f"select {model}: REFUSED -- {REFUSED_MODELS[model]}")
        return False
    if model == "seu":
        if near is None:
            say("select seu: pass --near PA (any single bit is a valid "
                "SEU site; no relation needed)")
            return False
        if not tab.in_universe(near):
            say(f"select seu: {near:#x} outside the table universe "
                "(fail-closed)")
            return False
        say(f"select seu: {near:#x} OK (in universe; SEU needs no "
            "relation -- provenance n/a)")
        return True

    if near is None:
        say(f"select {model}: pass --near PA")
        return False
    if not tab.in_universe(near):
        say(f"select {model}: {near:#x} outside the table universe "
            "(fail-closed)")
        return False
    base = page_of(near)
    found = 0

    if model == "same-bank-diff-row":
        masks = tab.anchors.get(base, [])[:3]
        for mask, votes in masks:
            other = base ^ mask
            say(f"  IN-PAGE  {base:#x} / {other:#x} -- anchor {mask:#x} "
                f"({votes}-card consensus; bank+row both MEASURED)")
            found += 1
            if found >= count:
                break
        row = tab.pages.get(base, {})
        root = row.get("channel_root", "")
        if root and found < count:
            others = sorted((p for p in tab.components.get(root, [])
                             if p != base), key=lambda p: abs(p - base))
            for other in others[:count - found]:
                say(f"  CROSS-PG {base:#x} / {other:#x} -- same component "
                    "(bank TRANSITIVE, R-e validated; row DIFFERENT by the "
                    "physical page argument)")
                found += 1
        if not root and not masks:
            say("  no same-bank partner: page unlinked and no valid "
                "anchor (bank unknown -- incremental re-run resolves)")

    elif model == "same-row-diff-col":
        pairs: list[tuple[int, int]] = []
        excluded = 0
        for members in tab.row_members.values():
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    if (a, b) in KNOWN_ROW_FALSIFIERS:
                        excluded += 1        # hard deep evidence wins
                        continue
                    pairs.append((a, b))
        if near is not None:
            pairs.sort(key=lambda ab: min(abs(page_of(ab[0]) - base),
                                          abs(page_of(ab[1]) - base)))
        say(f"  measured same-row sites available: {len(pairs)} pairs "
            f"(on {len(tab.row_members)} row classes -- the bank-map "
            "lattice + pilot pages; this is the honest siting limit for "
            "same-row faults)"
            + (f"; {excluded} pair excluded as the known R-e falsifier"
               if excluded else ""))
        for a, b in pairs[:count]:
            say(f"  {a:#x} / {b:#x} -- same row class (MEASURED; "
                f"{ROW_PROBABLE_NOTE})")
            found += 1

    elif model == "row-adjacent":
        root = tab.pages.get(base, {}).get("channel_root", "")
        if not root:
            say("  page unlinked (bank unknown) -- row-adjacent selection "
                "needs a same-bank component")
            return False
        members = tab.components.get(root, [])
        rank = members.index(base)
        # PA-order neighbors inside the same bank component, under the
        # monotonic-row-stripe prior; in-page row boundaries are unmapped,
        # so only the cross-page selector is offered.
        neighbors = [members[rank + k] for k in range(1, count + 1)
                     if rank + k < len(members)]
        neighbors += [members[rank - k] for k in range(1, count + 1)
                      if rank - k >= 0]
        for other in neighbors[:count]:
            say(f"  CROSS-PG {base:#x} / {other:#x} -- rank-"
                f"{abs(members.index(other) - rank)} PA neighbor in the "
                "same bank component (bank TRANSITIVE; row-adjacency "
                "ASSUMED: GeForge App. B monotonic-row-stripe prior -- "
                "timing carries no row-distance information)")
            found += 1

    if found == 0:
        say(f"select {model}: no candidates (fail-closed -- G4/G5 must "
            "not silently proceed without sites)")
        return False
    say(f"select {model}: {found} candidate pair(s) -- every line above "
        "carries its provenance; log it with the fault record")
    return True


# --------------------------------------------------------------------------
# G4 seam: annotate a run's pool with the GDDR relation snapshot
# --------------------------------------------------------------------------

def annotate_pool(tab: Table, run: Path, out: Path, say) -> bool:
    pool = PoolMap(parse_pool_map((run / "pool_map.csv").read_text(
        encoding="utf-8")))
    rows = []
    outside = 0
    for page in pool.pages:
        base = page.fb_pa_page_base
        trow = tab.pages.get(base)
        if trow is None:
            outside += 1
        n_row_nodes = sum(1 for rc, members in tab.row_members.items()
                          if members and page_of(members[0]) == base)
        masks = " ".join(f"{m:#x}" for m, _ in tab.anchors.get(base, [])[:3])
        rows.append({
            "va_page_base": f"{page.va_page_base:#x}",
            "fb_pa_page_base": f"{base:#x}",
            "page_size": page.page_size,
            "in_universe": int(trow is not None),
            "channel_root": trow["channel_root"] if trow else "",
            "channel_size": trow["channel_size"] if trow else "",
            "same_row_sites": n_row_nodes,
            "valid_anchor_masks": masks,
        })
    if outside:
        say(f"annotate-pool: FAIL -- {outside}/{len(rows)} pool pages "
            "outside the table universe (fail-closed; incremental "
            "table build required before G4 consumes this pool)")
        return False
    with out.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    linked = sum(1 for r in rows if r["channel_root"])
    withrow = sum(1 for r in rows if r["same_row_sites"])
    say(f"annotate-pool: {len(rows)} pages -> {out}")
    say(f"  in universe {len(rows)}/{len(rows)}; bank-known {linked}/"
        f"{len(rows)}; pages with measured same-row sites {withrow}")
    say("  (this CSV is the G3 leg of the G4 mapping snapshot: "
        "VA page <-> PA page <-> GDDR relation classes for THIS run)")
    return True


# --------------------------------------------------------------------------
# anchor consensus builder
# --------------------------------------------------------------------------

def build_anchors(runs: list[Path], out: Path, say) -> bool:
    cells: dict[tuple[int, int], list[str]] = defaultdict(list)
    for run in runs:
        path = run / "anchor_validity.csv"
        if not path.is_file():
            say(f"build-anchors: FAIL (missing {path})")
            return False
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                cells[(_hx(row["page_base"]),
                       _hx(row["candidate"]))].append(row["class"])
    # strict majority among the runs that measured the page; per-cell
    # hard flips are informational (the sweep valley is populated), so
    # the consensus smooths exactly the band-edge wobble R-d documented.
    rows = []
    valid_by_page: dict[int, set[int]] = defaultdict(set)
    for (page, cand), classes in sorted(cells.items()):
        votes = sum(1 for c in classes if c == "deep_conflict")
        valid = votes * 2 > len(classes)
        if valid:
            valid_by_page[page].add(cand)
        rows.append({"page_base": f"{page:#x}", "candidate": f"{cand:#x}",
                     "seen": len(classes), "votes": votes,
                     "valid": str(valid).lower()})
    with out.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=["page_base", "candidate",
                                                  "seen", "votes", "valid"])
        writer.writeheader()
        writer.writerows(rows)
    pages = {p for p, _ in cells}
    no_anchor = [p for p in pages if not valid_by_page.get(p)]
    universal = None
    for page in pages:
        masks = valid_by_page.get(page, set())
        universal = masks if universal is None else (universal & masks)
    say(f"build-anchors: {len(rows)} cells over {len(pages)} pages -> "
        f"{out}; pages without any valid anchor: {len(no_anchor)}; "
        "universal masks: "
        + (" ".join(f"{m:#x}" for m in sorted(universal or [])) or "none"))
    return not no_anchor


# --------------------------------------------------------------------------
# driver + self-test
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path,
                        default=Path("artifacts/g3/table_v4"))
    parser.add_argument("--query", nargs=2, metavar="PA")
    parser.add_argument("--check", nargs="+", type=_hx, metavar="PA")
    parser.add_argument("--check-pool", type=Path, metavar="RUN")
    parser.add_argument("--require-linked", action="store_true")
    parser.add_argument("--select", metavar="MODEL",
                        help="seu | same-bank-diff-row | same-row-diff-col "
                             "| row-adjacent (dq-adjacent / column-adjacent "
                             "are refused)")
    parser.add_argument("--near", type=_hx)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--annotate-pool", type=Path, metavar="RUN")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--build-anchors", nargs="+", type=Path,
                        metavar="RUN")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    problems: list[str] = []
    tab = Table(args.table, problems)
    if problems:
        print(f"INTEGRITY: {'; '.join(problems)}")
        return 1
    say = lambda t="": print(t)
    if args.query:
        relation(tab, _hx(args.query[0]), _hx(args.query[1]), say)
    ok = True
    if args.check:
        ok = check_pas(tab, args.check, args.require_linked, say) and ok
    if args.check_pool:
        ok = check_pool(tab, args.check_pool, args.require_linked, say) and ok
    if args.select:
        ok = select_sites(tab, args.select, args.near, args.count,
                          say) and ok
    if args.annotate_pool:
        if not args.out:
            parser.error("--annotate-pool needs --out")
        ok = annotate_pool(tab, args.annotate_pool, args.out, say) and ok
    if args.build_anchors:
        if not args.out:
            parser.error("--build-anchors needs --out")
        ok = build_anchors(args.build_anchors, args.out, say) and ok
    if not any((args.query, args.check, args.check_pool, args.select,
                args.annotate_pool, args.build_anchors)):
        parser.error("nothing to do (see --help)")
    return 0 if ok else 2


def self_test() -> int:
    import io
    import contextlib
    import tempfile

    P = lambda n: n << 21                       # page bases

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tdir = root / "table"
        tdir.mkdir()
        seed_head = ("page_index,page_base,lambda,channel_root,"
                     "channel_size,super_tail,n_shoulder,n_deep,n_low,"
                     "n_nodes,n_bank_classes,classified\n")
        seed_rows = [
            f"0,{P(0):#x},1000,{P(0):#x},2,0,0,2,10,3,1,1",
            f"1,{P(1):#x},1010,{P(0):#x},2,0,0,2,10,1,1,1",
            f"2,{P(2):#x},1020,{P(2):#x},1,0,0,0,10,1,1,1",
            f"3,{P(3):#x},1030,,0,0,0,0,10,1,1,1",
            "343,0x2ae00000,1040,0x2ae00000,1,0,0,0,10,2,1,1",
        ]
        (tdir / "gddr_seed_table.csv").write_text(
            seed_head + "\n".join(seed_rows) + "\n")
        cls_head = ("page_index,page_base,offset,pa,bank_class,bank_size,"
                    "row_class,row_size,n_deep,n_low,n_shoulder\n")
        cls_rows = [
            f"0,{P(0):#x},0x0,{P(0):#x},{P(0):#x},3,{P(0):#x},2,1,1,0",
            f"0,{P(0):#x},0x200,{P(0) + 0x200:#x},{P(0):#x},3,{P(0):#x},"
            f"2,0,1,0",
            f"0,{P(0):#x},0xd0100,{P(0) + 0xd0100:#x},{P(0):#x},3,"
            f"{P(0) + 0xd0100:#x},1,1,0,0",
            f"1,{P(1):#x},0x0,{P(1):#x},{P(0):#x},3,{P(1):#x},1,1,0,0",
            f"2,{P(2):#x},0x0,{P(2):#x},{P(2):#x},1,{P(2):#x},1,0,0,0",
            f"3,{P(3):#x},0x0,{P(3):#x},{P(3):#x},1,{P(3):#x},1,0,0,0",
            # the known R-e falsifier pair sits in one row class; every
            # consumer surface must flag it and selection must drop it
            "343,0x2ae00000,0xd3880,0x2aed3880,0x2aed3880,2,0x2aed3880,"
            "2,0,1,0",
            "343,0x2ae00000,0xd7b00,0x2aed7b00,0x2aed3880,2,0x2aed3880,"
            "2,0,1,0",
        ]
        (tdir / "bank_classes.csv").write_text(
            cls_head + "\n".join(cls_rows) + "\n")
        (tdir / "page_anchors.csv").write_text(
            "page_base,candidate,seen,votes,valid\n"
            f"{P(0):#x},0x1f9dc0,3,3,true\n"
            f"{P(0):#x},0xd0100,3,1,false\n"
            f"{P(1):#x},0x1f9dc0,3,3,true\n")
        problems: list[str] = []
        tab = Table(tdir, problems)
        assert not problems, problems

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rel = relation(tab, P(0), P(1), print)
        out = buffer.getvalue()
        assert "same bank: YES -- same page component" in out, out
        assert "row:      DIFFERENT -- physical" in out, out
        assert rel["same_bank"] == "yes" and rel["row"] == "different", rel

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rel = relation(tab, P(0), P(2), print)
        out = buffer.getvalue()
        assert "same bank: NO -- distinct components" in out, out
        assert rel["same_bank"] == "no", rel

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rel = relation(tab, P(0), P(3), print)
        out = buffer.getvalue()
        assert "same bank: UNKNOWN -- an unlinked page" in out, out
        assert rel["same_bank"] == "unknown", rel

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rel = relation(tab, P(0), P(0) + 0x200, print)
        out = buffer.getvalue()
        assert "row:      SAME -- row class" in out, out
        assert "column:   DIFFERENT" in out, out
        assert rel["row"] == "same" and rel["column"] == "different", rel

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rel = relation(tab, P(0), P(0) ^ 0x1f9dc0, print)
        out = buffer.getvalue()
        assert "anchor mask 0x1f9dc0 is consensus-valid" in out, out
        assert rel["row"] == "different", rel

        # in-page unknown: an offset the sweep never measured
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rel = relation(tab, P(0) + 0x200, P(0) + 0xd0100, print)
        out = buffer.getvalue()
        assert "same bank: YES -- same in-page bank class" in out, out
        assert "row:      UNKNOWN" in out, out
        assert rel["row"] == "unknown", rel

        # coverage: in-universe passes, outside fails closed, unlinked
        # escalates under --require-linked
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = check_pas(tab, [P(0) + 0x123, P(2)], False, print)
        assert ok and "PASS" in buffer.getvalue(), buffer.getvalue()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = check_pas(tab, [P(0), P(9)], False, print)
        assert not ok and "outside the universe" in buffer.getvalue()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = check_pas(tab, [P(3)], True, print)
        assert not ok and "--require-linked" in buffer.getvalue()

        # selectors
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = select_sites(tab, "same-bank-diff-row", P(0), 4, print)
        out = buffer.getvalue()
        assert ok and "IN-PAGE" in out and "CROSS-PG" in out, out
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = select_sites(tab, "same-row-diff-col", P(0), 4, print)
        out = buffer.getvalue()
        assert ok and f"{P(0):#x} / {P(0) + 0x200:#x}" in out, out

        # the known falsifier: query flags it, selection excludes it
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rel = relation(tab, 0x2aed3880, 0x2aed7b00, print)
        out = buffer.getvalue()
        assert "KNOWN FALSIFIER" in out and rel["row_provenance"] \
            == "measured-contradicted", out
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = select_sites(tab, "same-row-diff-col", 0x2ae00000, 4,
                              print)
        out = buffer.getvalue()
        assert ok and "excluded as the known R-e falsifier" in out, out
        assert "0x2aed3880 / 0x2aed7b00" not in out, out
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = select_sites(tab, "row-adjacent", P(1), 2, print)
        out = buffer.getvalue()
        assert ok and "ASSUMED: GeForge App. B" in out, out
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = select_sites(tab, "dq-adjacent", P(0), 1, print)
        assert not ok and "REFUSED" in buffer.getvalue()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = select_sites(tab, "column-adjacent", P(0), 1, print)
        assert not ok and "folded into 'same-row-diff-col'" \
            in buffer.getvalue()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = select_sites(tab, "seu", P(2), 1, print)
        assert ok, buffer.getvalue()

        # annotate-pool: a pool fully inside the universe passes; one
        # outside page fails closed
        pool_fields = ["run_id", "device", "gpu_uuid", "chunk_index",
                       "allocation_id", "va_page_base",
                       "va_page_end_exclusive", "fb_pa_page_base",
                       "page_size", "aperture", "pte_valid", "raw_pte_lo",
                       "raw_pte_hi", "mapped_at_ns", "unmapped_at_ns",
                       "source", "confidence"]

        def pool_text(pages):
            rows = [",".join(pool_fields)]
            for i in pages:
                rows.append(
                    f"r,0,uuid,{i},alloc,0x{0x7f0000000000 + i * 0x200000:x},"
                    f"0x{0x7f0000000000 + (i + 1) * 0x200000:x},{P(i):#x},"
                    f"{2 << 20},VIDEO,true,0x1,0x0,1,2,ebpf,c")
            return "\n".join(rows) + "\n"

        good = root / "pool_good"
        good.mkdir()
        (good / "pool_map.csv").write_text(pool_text([0, 1, 2]))
        snap = root / "snap.csv"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = annotate_pool(tab, good, snap, print)
        assert ok and snap.is_file(), buffer.getvalue()
        with snap.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3 and rows[0]["channel_root"] == f"{P(0):#x}" \
            and rows[2]["in_universe"] == "1", rows
        bad = root / "pool_bad"
        bad.mkdir()
        (bad / "pool_map.csv").write_text(pool_text([0, 9]))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = annotate_pool(tab, bad, root / "snap2.csv", print)
        assert not ok and "outside the table universe" \
            in buffer.getvalue()

        # anchor consensus: strict majority over the runs that saw the page
        for name, cells in (("ra", [(0, 0x1f9dc0, "deep_conflict"),
                                    (0, 0x555, "low"),
                                    (1, 0x1f9dc0, "deep_conflict")]),
                            ("rb", [(0, 0x1f9dc0, "deep_conflict"),
                                    (0, 0x555, "deep_conflict"),
                                    (1, 0x1f9dc0, "deep_conflict"),
                                    (1, 0x777, "mid")])):
            run = root / name
            run.mkdir()
            with (run / "anchor_validity.csv").open(
                    "w", encoding="utf-8", newline="") as sink:
                writer = csv.writer(sink)
                writer.writerow(["page_base", "candidate",
                                 "corrected_cycles", "class"])
                for page, cand, cls in cells:
                    writer.writerow([f"{P(page):#x}", f"{cand:#x}",
                                     1150, cls])
        out = root / "anchors.csv"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ok = build_anchors([root / "ra", root / "rb"], out, print)
        assert ok and "universal masks: 0x1f9dc0" \
            in buffer.getvalue(), buffer.getvalue()
        with out.open(encoding="utf-8", newline="") as fh:
            got = {(r["page_base"], r["candidate"]): r["valid"]
                   for r in csv.DictReader(fh)}
        assert got[(f"{P(0):#x}", "0x1f9dc0")] == "true"
        assert got[(f"{P(0):#x}", "0x555")] == "false"
        assert got[(f"{P(1):#x}", "0x1f9dc0")] == "true"
    print("query_table self-test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
