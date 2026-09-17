#!/usr/bin/env python3
"""GPU_M2D G3 S5-T0: seed the empirical GDDR mapping table (EMT v0).

Direction change (2026-09-17): the closed-form solver line is archived --
S4/S4b-2 showed the degree-2/weight-4 GF(2) model class cannot express
the seeded row fold, and GeForge (S&P'26, unlicensed repo; methodology
reference only) independently states the PA->bank hash is "extremely
difficult to reverse-engineer ... mixes (nearly) all address bits". The
G3 deliverable becomes an EMPIRICAL TABLE in the GeForge sense: measure
which physical addresses share (channel, bank, row), store the classes,
query the table. No closed form is ever produced. Our advantage over
GeForge: the G2 observer already pins every pool page's true PA each
run, so their page-anchoring problem does not exist here.

This is the pure-offline first step (zero new collection): fold
everything S3/S3b/census already measured into the seed table.

  page level  gddr_seed_table.csv  one row per pool 2 MiB page: census
              lambda, same-channel component (shoulder+deep cross-page
              union, recomputed), 32 MiB super-tail flag, evidence
              counts, classification state.
  node level  bank_classes.csv     node = (page, in-page offset) from
              every classified pair endpoint. bank class = union over
              DEEP conflicts (same bank, different row); row class =
              union over LOW pairs inside a bank class (same bank and
              same row). Singletons stay honest singletons; S5-T1
              (table-build collection) densifies them.

Consistency gates (reported; the run fails closed only on missing
inputs / off-pool pages):
  C1  shoulder edge inside a bank class -- same bank pays deep or a
      row hit, never the same-channel shoulder.
  C2  deep edge inside a row class -- same row and different row.
  C3  cross-page low inside a channel component (the S4b-2 rule).
  C4  cross-page low inside a bank class -- same bank forces the low
      to mean same row, and two distinct 2 MiB pages cannot share
      (bank, row) because the in-page column field cannot absorb PA
      bits >= 21. (A "bank class spans two channel components" check
      would be vacuous: every cross-page deep edge unions the channel
      components by construction.)
  C5  cross-source label disagreements on one PA pair -- reprobe >
      pilot > S3b > S3 priority; informational, counted.

    python3 build_bank_table.py <s3_run> <s3b_run> <census_run> \
        [--out artifacts/g3/table_v0] [--query 0xPA]

Writes gddr_seed_table.csv, bank_classes.csv and table_report.txt into
--out. Exit codes: 0 ok, 2 integrity failure, 1 error (repo
convention). `--self-test` pins the classes, all four contradiction
rules, dedup priority, and the fail-closed page check on a synthetic
fixture.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_census import UnionFind

PAGE = 2 * 1024 * 1024
CLASSES = ("low", "mid", "shoulder", "deep_conflict")
# Freshest re-measurement wins when the same PA pair appears twice.
SOURCE_PRIORITY = ("reprobe", "pilot", "s3b", "s3")


def _hex(value: str) -> int:
    return int(value, 16)


def load_edges(s3: Path, s3b: Path, census: Path, problems: list[str]) \
        -> tuple[dict[tuple[int, int], tuple[str, str]],
                 list[tuple[tuple[int, int], str, str, str]]]:
    """(pa_a, pa_b) -> (class, source) deduped by source priority, plus
    the disagreement list for C5. Integrity failures collect, never pass."""
    raw: list[tuple[str, str, int, int]] = []

    def add_bands(path: Path, source: str) -> None:
        if not path.is_file():
            problems.append(f"missing {path} (run analyze_three_band.py)")
            return
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                cls = row.get("class") or ""
                if cls not in CLASSES or not row.get("pa_a") \
                        or not row.get("pa_b"):
                    continue
                if abs(int(row["cycles_a"]) - int(row["cycles_b"])) > 999:
                    continue  # asymmetric rows carry no regime evidence
                raw.append((source, cls, _hex(row["pa_a"]), _hex(row["pa_b"])))

    add_bands(s3 / "constraints_bands.csv", "s3")
    add_bands(s3b / "pair_constraints_bands.csv", "s3b")

    verdicts = census / "reprobe_verdicts.csv"
    if verdicts.is_file():
        with verdicts.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                cls = row.get("verdict") or ""
                if cls not in CLASSES or not row.get("pa_a") \
                        or not row.get("pa_b"):
                    continue
                if abs(int(row["cycles_a"]) - int(row["cycles_b"])) > 999:
                    continue
                raw.append(("reprobe", cls, _hex(row["pa_a"]),
                            _hex(row["pa_b"])))
    else:
        problems.append(f"missing {verdicts} (run analyze_census.py)")

    pilots = census / "row_pilot_classes.csv"
    if pilots.is_file():
        with pilots.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                cls = row.get("class") or ""
                if cls not in CLASSES or not row.get("page"):
                    continue
                if abs(int(row["cycles_a"]) - int(row["cycles_b"])) > 999:
                    continue
                page = _hex(row["page"])
                raw.append(("pilot", cls, page + _hex(row["offset_i"]),
                            page + _hex(row["offset_j"])))
    else:
        problems.append(f"missing {pilots} (run analyze_census.py)")

    rank = {name: i for i, name in enumerate(SOURCE_PRIORITY)}
    edges: dict[tuple[int, int], tuple[str, str]] = {}
    disagreements: list[tuple[tuple[int, int], str, str, str]] = []
    for source, cls, pa_a, pa_b in sorted(raw, key=lambda r: rank[r[0]]):
        if pa_a == pa_b:
            continue
        key = (pa_a, pa_b) if pa_a < pa_b else (pa_b, pa_a)
        if key not in edges:
            edges[key] = (cls, source)
        elif edges[key][0] != cls:
            disagreements.append((key, edges[key][0], cls, source))
    return edges, disagreements


def load_pages(census: Path, problems: list[str]) -> dict[int, int]:
    """Pool universe: page base -> census self lambda."""
    source = census / "page_lambdas.csv"
    if not source.is_file():
        problems.append(f"missing {source} (run analyze_census.py)")
        return {}
    lambdas: dict[int, int] = {}
    with source.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            base = _hex(row["fb_pa_page_base"])
            if base & (PAGE - 1):
                problems.append(f"page base 0x{base:x} not 2 MiB aligned")
                continue
            lambdas[base >> 21] = int(row["self_cycles"])
    return lambdas


def build(edges: dict[tuple[int, int], tuple[str, str]]):
    """Bank/row/channel partition over the edge set."""
    node_pas = sorted({pa for pair, (cls, _) in edges.items() if cls != "mid"
                       for pa in pair})
    bank_uf = UnionFind(node_pas)
    for (a, b), (cls, _) in edges.items():
        if cls == "deep_conflict":
            bank_uf.union(a, b)
    row_uf = UnionFind(node_pas)
    for (a, b), (cls, _) in edges.items():
        if cls == "low" and bank_uf.find(a) == bank_uf.find(b):
            row_uf.union(a, b)

    channel_pages = sorted({pa >> 21 for (a, b), (cls, _) in edges.items()
                            if cls in ("shoulder", "deep_conflict")
                            and a >> 21 != b >> 21
                            for pa in (a, b)})
    page_uf = UnionFind(channel_pages)
    for (a, b), (cls, _) in edges.items():
        if cls in ("shoulder", "deep_conflict") and a >> 21 != b >> 21:
            page_uf.union(a >> 21, b >> 21)

    node_deg: dict[int, Counter] = defaultdict(Counter)
    page_deg: dict[int, Counter] = defaultdict(Counter)
    for (a, b), (cls, _) in edges.items():
        if cls == "mid":
            continue
        node_deg[a][cls] += 1
        node_deg[b][cls] += 1
        page_deg[a >> 21][cls] += 1
        page_deg[b >> 21][cls] += 1

    bank_groups = bank_uf.groups()
    row_groups = row_uf.groups()
    return {"nodes": node_pas, "bank": bank_uf, "row": row_uf,
            "bank_groups": bank_groups, "row_groups": row_groups,
            "page_uf": page_uf, "page_groups": page_uf.groups(),
            "node_deg": node_deg, "page_deg": page_deg}


def contradictions(edges, part, lambdas) -> dict[str, list[tuple[int, int]]]:
    comp_of = {page: root for root, members in part["page_groups"].items()
               for page in members}
    found: dict[str, list[tuple[int, int]]] = {k: [] for k in "C1 C2 C3 C4"
                                               .split()}
    for (a, b), (cls, _) in edges.items():
        if cls == "shoulder" and part["bank"].find(a) == part["bank"].find(b):
            found["C1"].append((a, b))
        if cls == "deep_conflict" and part["row"].find(a) == part["row"].find(b):
            found["C2"].append((a, b))
        if cls == "low" and a >> 21 != b >> 21:
            root_a = comp_of.get(a >> 21)
            if root_a is not None and root_a == comp_of.get(b >> 21):
                found["C3"].append((a, b))
            if part["bank"].find(a) == part["bank"].find(b):
                found["C4"].append((a, b))
    return found


def analyze(s3: Path, s3b: Path, census: Path, out: Path,
            query: int | None) -> int:
    problems: list[str] = []
    edges, disagreements = load_edges(s3, s3b, census, problems)
    lambdas = load_pages(census, problems)
    for (a, b) in edges:
        for pa in (a, b):
            if (pa >> 21) not in lambdas:
                problems.append(f"edge PA 0x{pa:x} not on a census page "
                                f"(pool-map mismatch)")
                break
        if problems:
            break
    if problems:
        print(f"INTEGRITY: {'; '.join(problems)}")
        return 2

    part = build(edges)
    found = contradictions(edges, part, lambdas)
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    n_cls = Counter(cls for cls, _ in edges.values())
    n_src = Counter(src for _, src in edges.values())
    say("=== EMT v0 seed table (S5-T0, zero new collection) ===")
    say(f"pool universe: {len(lambdas)} census pages; edges after dedup: "
        f"{len(edges)} (deep {n_cls['deep_conflict']}, shoulder "
        f"{n_cls['shoulder']}, low {n_cls['low']}, mid {n_cls['mid']} "
        f"excluded from classes) from {dict(n_src)}")
    if disagreements:
        flips = Counter(f"{kept}->{dropped}"
                        for _, kept, dropped, _ in disagreements)
        say(f"C5 cross-source disagreements: {len(disagreements)} "
            f"{dict(flips)} (priority {' > '.join(SOURCE_PRIORITY)} keeps "
            f"the freshest)")
    else:
        say("C5 cross-source disagreements: 0")

    bank_multi = {root: members for root, members in
                  part["bank_groups"].items() if len(members) > 1}
    bank_sizes = sorted((len(m) for m in part["bank_groups"].values()),
                        reverse=True)
    rows_in_multi = Counter()
    for root, members in bank_multi.items():
        for pa in members:
            row_root = part["row"].find(pa)
            rows_in_multi[(root, row_root)] += 1
    multi_page = sum(1 for m in bank_multi.values()
                     if len({pa >> 21 for pa in m}) > 1)
    say(f"nodes: {len(part['nodes'])}; bank classes: "
        f"{len(part['bank_groups'])} total, {len(bank_multi)} with >=2 "
        f"nodes, largest {bank_sizes[0] if bank_sizes else 0} "
        f"({multi_page} spanning >1 page); "
        f"row classes inside multi-node bank classes: "
        f"{len(rows_in_multi)}")
    comp_sizes = sorted((len(m) for m in part["page_groups"].values()),
                        reverse=True)
    say(f"channel components (recomputed, cross-page shoulder+deep): "
        f"{len(part['page_groups'])} over {len(comp_sizes) and sum(comp_sizes)}"
        f" pages, sizes {comp_sizes[:8]}"
        f"{' ...' if len(comp_sizes) > 8 else ''}")

    prior = s3 / "channel_components.csv"
    if prior.is_file():
        with prior.open(encoding="utf-8", newline="") as fh:
            old = defaultdict(set)
            for row in csv.DictReader(fh):
                old[int(row["component_root"], 16)].add(
                    int(row["page_base"], 16) >> 21)
        new = {root: set(members) for root, members in
               part["page_groups"].items()}
        verdict = ("MATCH" if {frozenset(v) for v in old.values()} ==
                   {frozenset(v) for v in new.values()} else "MISMATCH")
        old_pages = {p for members in old.values() for p in members}
        new_pages = {p for members in new.values() for p in members}
        say(f"cross-check vs S4b-2 {prior.name}: {verdict} "
            f"({len(old)} old vs {len(new)} components, "
            f"{len(old_pages)} vs {len(new_pages)} pages) -- a MISMATCH "
            f"is the expected correction when re-probe demotions (C5) "
            f"drop S4b-2 shoulder edges: that graph only ADDED reprobe "
            f"shoulders, it never demoted the bands labels")

    say("")
    n_cross_low = sum(1 for (a, b), (cls, _) in edges.items()
                      if cls == "low" and a >> 21 != b >> 21)
    for key in ("C1", "C2", "C3", "C4"):
        pairs = found[key]
        extra = (f" of {n_cross_low} cross-page lows "
                 f"({len(pairs) / n_cross_low:.1%})"
                 if key == "C3" and n_cross_low else "")
        say(f"{key}: {len(pairs)} contradiction(s){extra}")
        for a, b in pairs[:5]:
            say(f"    0x{a:x} / 0x{b:x}")
    say("  (every C4 edge is also a C3 edge -- a bank class spans pages "
        "only via cross-page deep edges, which union the channel "
        "components; C4 marks the same-bank diagnosis)")
    say("")
    pages_classified = {pa >> 21 for root, members in bank_multi.items()
                        for pa in members}
    pages_channel = {p for members in part["page_groups"].values()
                     for p in members}
    n_pool = len(lambdas)
    singleton = sum(1 for m in part["bank_groups"].values()
                    if len(m) == 1)
    say(f"coverage (T1 sizing): channel component on "
        f"{len(pages_channel)}/{n_pool} pages "
        f"({len(pages_channel) / n_pool:.0%}); classified bank-class nodes "
        f"on {len(pages_classified)}/{n_pool} pages; singleton nodes "
        f"{singleton} of {len(part['nodes'])}")

    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "gddr_seed_table.csv"
    comp_of = {page: root for root, members in part["page_groups"].items()
               for page in members}
    comp_size = {root: len(members)
                 for root, members in part["page_groups"].items()}
    page_nodes: dict[int, set[int]] = defaultdict(set)
    page_banks: dict[int, set[int]] = defaultdict(set)
    for pa in part["nodes"]:
        page = pa >> 21
        page_nodes[page].add(pa)
        page_banks[page].add(part["bank"].find(pa))
    with table_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["page_index", "page_base", "lambda",
                         "channel_root", "channel_size", "super_tail",
                         "n_shoulder", "n_deep", "n_low", "n_nodes",
                         "n_bank_classes", "classified"])
        for page in sorted(lambdas):
            root = comp_of.get(page)
            writer.writerow([page, f"0x{page << 21:x}",
                             lambdas[page],
                             f"0x{root:x}" if root is not None else "",
                             comp_size.get(root, "") if root is not None else "",
                             int(page % 16 == 7),
                             part["page_deg"][page]["shoulder"],
                             part["page_deg"][page]["deep_conflict"],
                             part["page_deg"][page]["low"],
                             len(page_nodes.get(page, ())),
                             len(page_banks.get(page, ())),
                             int(page in pages_classified)])
    say(f"output: {table_path}")

    classes_path = out / "bank_classes.csv"
    bank_size = {root: len(m) for root, m in part["bank_groups"].items()}
    row_size = {}
    for root, members in part["row_groups"].items():
        for pa in members:
            row_size[pa] = len(members)
    with classes_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["page_index", "page_base", "offset", "pa",
                         "bank_class", "bank_size", "row_class",
                         "row_size", "n_deep", "n_low", "n_shoulder"])
        for pa in part["nodes"]:
            bank_root = part["bank"].find(pa)
            row_root = part["row"].find(pa)
            writer.writerow([pa >> 21, f"0x{(pa >> 21) << 21:x}",
                             f"0x{pa & (PAGE - 1):x}", f"0x{pa:x}",
                             f"0x{bank_root:x}", bank_size[bank_root],
                             f"0x{row_root:x}", row_size.get(pa, 1),
                             part["node_deg"][pa]["deep_conflict"],
                             part["node_deg"][pa]["low"],
                             part["node_deg"][pa]["shoulder"]])
    say(f"output: {classes_path}")

    report_path = out / "table_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"output: {report_path}")

    if query is not None:
        page, offset = query >> 21, query & (PAGE - 1)
        say("")
        say(f"query 0x{query:x}: page {page} "
            f"(0x{page << 21:x}) lambda {lambdas.get(page, '?')}")
        root = comp_of.get(page)
        say(f"  channel: "
            + (f"component 0x{root:x} ({comp_size[root]} pages)"
               if root is not None else "unclassified"))
        node = next((pa for pa in part["nodes"]
                     if pa >> 21 == page and pa == query), None)
        if node is not None:
            bank_root = part["bank"].find(node)
            row_root = part["row"].find(node)
            say(f"  node: bank class 0x{bank_root:x} "
                f"({bank_size[bank_root]} nodes), row class 0x{row_root:x}")
        else:
            say("  node: unmeasured at this exact offset "
                "(S5-T1 densifies)")
        members = sorted(page_banks.get(page, ()))
        n_single = sum(1 for r in members
                       if len(part["bank_groups"][r]) == 1)
        say(f"  bank classes touching page: {len(members)} "
            f"({n_single} singletons = unmeasured, T1 densifies)")
    return 0


def self_test() -> int:
    import contextlib
    import io
    import tempfile

    def pa(page_index: int, offset: int = 0) -> int:
        return (page_index << 21) + offset

    bands_fields = ["query_id", "kind", "bit", "pa_a", "pa_b", "xor",
                    "cycles_a", "cycles_b", "class"]

    def bands_rows(rows):
        return [",".join(bands_fields)] + [
            f"{qid},x,0,0x{a:x},0x{b:x},0x0,1050,1050,{cls}"
            for qid, (a, b, cls) in enumerate(rows)]

    # page 0: the clean pilot structure -- one bank class of 4 nodes,
    # two row classes of 2 (rows {0,0x200} and {M,M|0x200}).
    # page 1: C2 plant -- deep tree into e, low path a-b-m-m2, then a
    #         deep (a,m2) inside that row class.
    # page 2: C1 plant -- deep (0,M) + deep (M,E) then shoulder (0,E).
    # pages 3-5: channel comp via shoulder path; C3 plant low (3,5).
    # pages 6-7: second channel comp; C4 plant -- in-page deeps grow a
    #         bank class across the pages, then a cross-page low inside.
    p0 = [(pa(0, o1), pa(0, o2), cls) for o1, o2, cls in [
        (0x0, 0xd0100, "deep_conflict"), (0x200, 0xd0300, "deep_conflict"),
        (0x200, 0xd0100, "deep_conflict"),
        (0x0, 0x200, "low"), (0xd0100, 0xd0300, "low")]]
    p1 = [(pa(1, o1), pa(1, o2), cls) for o1, o2, cls in [
        (0x0, 0xd4400, "deep_conflict"), (0x200, 0xd4400, "deep_conflict"),
        (0xd0100, 0xd4400, "deep_conflict"),
        (0xd0300, 0xd4400, "deep_conflict"),
        (0x0, 0x200, "low"), (0x200, 0xd0100, "low"),
        (0xd0100, 0xd0300, "low"),
        (0x0, 0xd0300, "deep_conflict")]]
    p2 = [(pa(2, o1), pa(2, o2), cls) for o1, o2, cls in [
        (0x0, 0xd0100, "deep_conflict"), (0xd0100, 0xd4400, "deep_conflict"),
        (0x0, 0xd4400, "shoulder")]]
    cross = [
        (pa(3, 0x1000), pa(4, 0x1000), "shoulder"),
        (pa(4, 0x1000), pa(5, 0x1000), "shoulder"),
        (pa(3, 0x1000), pa(5, 0x1000), "low"),               # C3 plant
        (pa(6, 0x1000), pa(7, 0x1000), "shoulder"),
        # C4 plant: bank class across p6/p7 via a cross-page deep edge
        # plus in-page deeps -- then a cross-page low inside the class.
        # (A C4 edge is always also a C3 edge: bank classes span pages
        # only via cross-page deep edges, which union channel comps.)
        (pa(6, 0x3000), pa(7, 0x3000), "deep_conflict"),
        (pa(6, 0x2000), pa(6, 0x3000), "deep_conflict"),
        (pa(7, 0x2000), pa(7, 0x3000), "deep_conflict"),
        (pa(6, 0x2000), pa(7, 0x2000), "low"),               # C4 + C3
        # C5 plant: s3 says low, pilot says shoulder -> pilot wins
        (pa(5, 0x1000), pa(5, 0x3000), "low"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "s3").mkdir()
        (root / "s3b").mkdir()
        (root / "census").mkdir()
        (root / "s3" / "constraints_bands.csv").write_text(
            "\n".join(bands_rows(p0 + p1 + p2 + cross)) + "\n")
        (root / "s3b" / "pair_constraints_bands.csv").write_text(
            "\n".join(bands_rows([])) + "\n")
        with (root / "census" / "page_lambdas.csv").open(
                "w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["fb_pa_page_base", "self_cycles"])
            for page in range(8):
                writer.writerow([f"0x{pa(page):x}", 1000 + page])
        with (root / "census" / "reprobe_verdicts.csv").open(
                "w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["query_id", "pa_a", "pa_b", "cycles_a",
                             "cycles_b", "verdict"])
            writer.writerow([90, f"0x{pa(4, 0x1000):x}",
                             f"0x{pa(5, 0x1000):x}", 1114, 1114,
                             "shoulder"])
        with (root / "census" / "row_pilot_classes.csv").open(
                "w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["page", "offset_i", "offset_j", "xor",
                             "cycles_a", "cycles_b", "corrected_cycles",
                             "class"])
            writer.writerow([f"0x{pa(5):x}", "0x1000", "0x3000",
                             "0x2000", 1114, 1114, 1114, "shoulder"])

        out = root / "table"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = analyze(root / "s3", root / "s3b", root / "census",
                           out, None)
        assert code == 0
        report = buffer.getvalue()
        assert "C1: 1 contradiction(s)" in report, report
        assert "C2: 1 contradiction(s)" in report, report
        assert "C3: 2 contradiction(s)" in report, report  # plant + C4 edge
        assert "C4: 1 contradiction(s)" in report, report
        assert "C5 cross-source disagreements: 1" in report, report
        assert "shoulder->low" in report, report          # C5 direction
        assert "2 channel" in report or \
            "channel components (recomputed, cross-page shoulder+deep): 2 " \
            in report, report

        nodes = {row["pa"]: row for row in csv.DictReader(
            (out / "bank_classes.csv").open(encoding="utf-8"))}
        page0 = [row for row in nodes.values()
                 if row["page_index"] == "0"]
        assert len(page0) == 4
        assert len({row["bank_class"] for row in page0}) == 1
        assert len({row["row_class"] for row in page0}) == 2
        assert all(int(row["bank_size"]) == 4 for row in page0)
        pages = {row["page_index"]: row for row in csv.DictReader(
            (out / "gddr_seed_table.csv").open(encoding="utf-8"))}
        assert len(pages) == 8
        assert pages["7"]["super_tail"] == "1"
        assert pages["0"]["classified"] == "1"
        assert pages["3"]["channel_size"] == "3"
        assert pages["6"]["channel_size"] == "2"
        assert pages["0"]["channel_root"] == ""

        # fail-closed: an edge off the census pages refuses the run
        bad = root / "s3" / "constraints_bands.csv"
        bad.write_text("\n".join(bands_rows(
            [(pa(99, 0), pa(99, 0x200), "low")])) + "\n")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = analyze(root / "s3", root / "s3b", root / "census",
                           out, None)
        assert code == 2 and "INTEGRITY" in buffer.getvalue()
    print("build_bank_table self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s3", nargs="?", type=Path,
                        help="S3 run dir (constraints_bands.csv)")
    parser.add_argument("s3b", nargs="?", type=Path,
                        help="S3b run dir (pair_constraints_bands.csv)")
    parser.add_argument("census", nargs="?", type=Path,
                        help="census run dir (page_lambdas / reprobe / "
                             "row_pilot)")
    parser.add_argument("--out", type=Path,
                        default=Path("artifacts/g3/table_v0"),
                        help="output dir (default artifacts/g3/table_v0)")
    parser.add_argument("--query", type=str, default=None,
                        help="print the table rows for one PA, e.g. 0x1ee0d0100")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not (args.s3 and args.s3b and args.census):
        parser.error("three run dirs are required outside --self-test")
    query = int(args.query, 16) if args.query else None
    return analyze(args.s3, args.s3b, args.census, args.out, query)


if __name__ == "__main__":
    sys.exit(main())
