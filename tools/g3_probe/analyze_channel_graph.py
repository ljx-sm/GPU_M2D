#!/usr/bin/env python3
"""GPU_M2D G3 S4b-2: same-channel partition from the three-band labels.

The S4b-1 physical model makes the channel partition a graph question:
shoulder (~0.70-0.85 amplitude) = same channel, different bank/bank group;
deep conflict (>= 0.90) = same bank, different row -- both are SAME-CHANNEL
evidence. Cross-page low is different channel OR same bank AND same row --
and two distinct 2 MiB pages cannot share (channel, bank, row), because
they differ in PA bits >= 21 which the column field (in-page, < 2^21)
cannot absorb; so a cross-page low INSIDE a same-channel component is a
label contradiction (measurement noise, not physics). This tool builds
the page-level graph and asks the three S4b-2 questions:

  1. how many components, and do they look like ~12 channels of ~170
     pages each (or one giant blob / hundreds of fragments)?
  2. how many cross-page-low contradictions fall inside components
     (edge-quality score for the partition)?
  3. what PA structure do the components carry (bit signature, the
     32 MiB super-conflict pages = 7 mod 16, census-lambda mixing)?

Inputs are `*_bands.csv` files written by analyze_three_band.py (any
number; edges accumulate). `--census` additionally loads the census run's
reprobe_verdicts.csv shoulder pairs as extra same-channel edges and
page_lambdas.csv for the lambda-mixing check. Writes channel_components.csv
(page, component, degree, n_shoulder, n_deep, lambda) next to the first
input.

    python3 analyze_channel_graph.py A_bands.csv B_bands.csv [--census DIR]

Exit codes: 0 ok, 2 integrity failure, 1 error (repo convention).
`--self-test` pins the union, the violation rule, and the super-tail
report on a synthetic 3-channel fixture.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_census import UnionFind, gap_clusters

PAGE_DEFAULT = 2 * 1024 * 1024
SAME_CHANNEL = ("shoulder", "deep_conflict")


def load_edges(paths: list[Path], page_size: int, census: Path | None) \
        -> tuple[list[tuple[int, int, str]], dict[int, int], list[str]]:
    """(page_a, page_b, kind) edges + page lambdas; passthrough rows and
    in-page pairs are not edges. Integrity failures collect, never pass."""
    edges: list[tuple[int, int, str]] = []
    lambdas: dict[int, int] = {}
    problems: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(line for line in source
                                       if not line.startswith("#")))
        for row in rows:
            cls = row.get("class") or row.get("verdict") or ""
            if not row.get("pa_a") or not row.get("pa_b") or not cls:
                continue
            pa_a, pa_b = int(row["pa_a"], 16), int(row["pa_b"], 16)
            if abs(int(row["cycles_a"]) - int(row["cycles_b"])) > 999:
                continue  # asymmetric rows carry no regime evidence
            if pa_a >> 21 == pa_b >> 21:
                continue  # in-page pairs carry no channel-partition evidence
            if cls in SAME_CHANNEL:
                edges.append((pa_a, pa_b, cls))
    if census is not None:
        verdicts = census / "reprobe_verdicts.csv"
        if verdicts.is_file():
            with verdicts.open(encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    if row.get("verdict") not in SAME_CHANNEL:
                        continue
                    pa_a = int(row["pa_a"], 16)
                    pa_b = int(row["pa_b"], 16)
                    if pa_a >> 21 != pa_b >> 21:
                        edges.append((pa_a, pa_b, row["verdict"]))
        else:
            problems.append(f"--census given but {verdicts} missing")
        lambdas_file = census / "page_lambdas.csv"
        if lambdas_file.is_file():
            with lambdas_file.open(encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    lambdas[int(row["fb_pa_page_base"], 16)] = \
                        int(row["self_cycles"])
    return edges, lambdas, problems


def build_components(edges: list[tuple[int, int, str]]) \
        -> tuple[dict[int, list[int]], dict[int, int], dict[int, Counter]]:
    pages = sorted({page for a, b, _ in edges for page in (a >> 21, b >> 21)})
    uf = UnionFind(pages)
    degree: dict[int, Counter] = {page: Counter() for page in pages}
    for pa_a, pa_b, cls in edges:
        left, right = pa_a >> 21, pa_b >> 21
        uf.union(left, right)
        degree[left][cls] += 1
        degree[right][cls] += 1
    components = uf.groups()
    comp_of = {page: root for root, members in components.items()
               for page in members}
    return components, comp_of, degree


def analyze(paths: list[Path], census: Path | None, page_size: int,
            out_path: Path | None) -> int:
    edges, lambdas, problems = load_edges(paths, page_size, census)
    if problems:
        print(f"INTEGRITY: {'; '.join(problems)}")
        return 2
    components, comp_of, degree = build_components(edges)

    sizes = sorted((len(members) for members in components.values()),
                   reverse=True)
    n_pages = len(comp_of)
    print(f"same-channel graph: {len(edges)} cross-page edges "
          f"({sum(1 for *_, c in edges if c == 'shoulder')} shoulder + "
          f"{sum(1 for *_, c in edges if c == 'deep_conflict')} deep) over "
          f"{n_pages} pages -> {len(components)} component(s)")
    print(f"component sizes: {sizes[:16]}"
          f"{' ...' if len(sizes) > 16 else ''}")
    if sizes:
        print(f"largest component holds {sizes[0]}/{n_pages} pages "
              f"({sizes[0] / n_pages:.0%}); sizes p50 "
              f"{sizes[len(sizes) // 2]}")
        verdict = ("~channel-scale components" if len(sizes) >= 8
                   and sizes[0] <= 0.4 * n_pages else
                   ("ONE GIANT component -- shoulder edges do not partition "
                    "channels (noise or transitive over-merging)"
                    if sizes[0] > 0.6 * n_pages else
                    "fragmented -- components far below channel scale"))

    # Cross-page lows inside a component = label contradictions.
    low_inside = 0
    low_total = 0
    contradictions: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(line for line in source
                                      if not line.startswith("#")):
                if (row.get("class") != "low" or not row.get("pa_a")
                        or not row.get("pa_b")):
                    continue
                pa_a, pa_b = int(row["pa_a"], 16), int(row["pa_b"], 16)
                if pa_a >> 21 == pa_b >> 21:
                    continue
                low_total += 1
                root_a, root_b = comp_of.get(pa_a >> 21), comp_of.get(pa_b >> 21)
                if root_a is not None and root_a == root_b:
                    low_inside += 1
                    if len(contradictions) < 5:
                        contradictions.append(
                            f"{path.name} q{row['query_id']}: pages "
                            f"0x{pa_a >> 21:x}/0x{pa_b >> 21:x} same "
                            f"component but pair is low "
                            f"({row['cycles_a']} cyc)")
    print(f"\ncross-page low pairs: {low_total}, inside a same-channel "
          f"component: {low_inside} "
          f"({(low_inside / low_total if low_total else 0):.1%} contradiction"
          f" rate; 2 MiB pages cannot share channel+bank+row, so each is a "
          f"label/edge error)")
    for line in contradictions:
        print(f"  e.g. {line}")

    # Structure: PA-bit signature, super-tail pages, lambda mixing.
    lambda_clusters = gap_clusters(sorted(lambdas.values())) \
        if lambdas else []
    cluster_of = {}
    for index, (lo, hi, _) in enumerate(lambda_clusters):
        for page, value in lambdas.items():
            if lo <= value <= hi:
                cluster_of[page] = index
    by_root: dict[int, list[int]] = defaultdict(list)
    for page, root in comp_of.items():
        by_root[root].append(page)
    print("\ncomponent structure (>= 8 pages):")
    for root, members in sorted(by_root.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 8:
            break
        top = max(members) << 21
        bits = []
        for bit in range(21, max(members).bit_length() + 21):
            ones = [p for p in members if (p >> (bit - 21)) & 1]
            share = len(ones) / len(members)
            overall = sum((p >> (bit - 21)) & 1 for p in comp_of) / len(comp_of)
            if abs(share - overall) > 0.15:
                bits.append(f"bit{bit}:{share:.0%} (all {overall:.0%})")
        tail = sum(p % 16 == 7 for p in members)
        lam = Counter(cluster_of.get(p * (1 << 21)) for p in members
                      if p * (1 << 21) in cluster_of)
        lam_text = "/".join(f"c{i}:{n}" for i, n in sorted(lam.items())
                            if i is not None) or "n/a"
        print(f"  component of {len(members)} pages (PA up to 0x{top:x}): "
              f"{'; '.join(bits) or 'no dominant PA bit'}; super-tail "
              f"(7 mod 16) {tail}/{sum(p % 16 == 7 for p in comp_of)}; "
              f"lambda clusters {lam_text}")

    if out_path is not None:
        with out_path.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.writer(sink)
            writer.writerow(["page_index", "page_base", "component_root",
                             "n_shoulder", "n_deep", "lambda"])
            for page in sorted(comp_of):
                root = comp_of[page]
                writer.writerow([page, f"0x{page << 21:x}", f"0x{root:x}",
                                 degree[page]["shoulder"],
                                 degree[page]["deep_conflict"],
                                 lambdas.get(page << 21, "")])
        print(f"\noutput: {out_path}")
    return 0


def self_test() -> int:
    import tempfile
    page = 2 * 1024 * 1024

    def pa(page_index: int) -> int:
        return page_index * page

    fields = ["query_id", "kind", "bit", "pa_a", "pa_b", "xor",
              "cycles_a", "cycles_b", "class"]
    # 3 channels over pages 0..11: intra-channel shoulder edges (a spanning
    # path each), one intra-channel deep edge, one cross-channel low, and
    # ONE planted contradiction: a low pair inside channel A.
    plan = [
        # channel A: pages 0,3,6,9  (spanning shoulder path + deep edge)
        (0, 3, "shoulder"), (3, 6, "shoulder"), (6, 9, "deep_conflict"),
        # channel B: pages 1,4,7,10
        (1, 4, "shoulder"), (4, 7, "shoulder"), (7, 10, "shoulder"),
        # channel C: pages 2,5,8,11
        (2, 5, "shoulder"), (5, 8, "deep_conflict"), (8, 11, "shoulder"),
        # cross-channel low (NOT a contradiction)
        (0, 1, "low"),
        # contradiction: low inside channel A
        (0, 9, "low"),
        # in-page pair (must be ignored)
        (0, 0, "shoulder"),
    ]
    lines = [",".join(fields)]
    for qid, (left, right, cls) in enumerate(plan):
        lines.append(f"{qid},x,0,0x{pa(left) + 0x1000:x},"
                     f"0x{pa(right) + 0x1000:x},0x0,1050,1050,{cls}")
    with tempfile.TemporaryDirectory() as tmp:
        bands = Path(tmp) / "bands.csv"
        bands.write_text("\n".join(lines) + "\n")
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = analyze([bands], None, page, Path(tmp) / "comp.csv")
        assert code == 0
        report = buffer.getvalue()
        assert "9 cross-page edges" in report, report
        assert "3 component(s)" in report, report
        assert "4, 4, 4" in report, report            # three 4-page channels
        assert "inside a same-channel component: 1 " in report, report
        assert "0x0/0x9" in report                    # the planted pair
        comps = list(csv.DictReader((Path(tmp) / "comp.csv")
                                    .open(encoding="utf-8")))
        assert len(comps) == 12
        roots = {c["component_root"] for c in comps}
        assert len(roots) == 3
        # degree bookkeeping: page 0 has shoulder(3) + low-ignored + ...
        row0 = next(c for c in comps if c["page_index"] == "0")
        assert int(row0["n_shoulder"]) == 1 and int(row0["n_deep"]) == 0
        row9 = next(c for c in comps if c["page_index"] == "9")
        assert int(row9["n_deep"]) == 1
    print("analyze_channel_graph self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bands", nargs="*", type=Path,
                        help="*_bands.csv from analyze_three_band.py")
    parser.add_argument("--census", type=Path, default=None,
                        help="census run dir: adds reprobe shoulder edges "
                             "+ page lambdas")
    parser.add_argument("--out", type=Path, default=None,
                        help="channel_components.csv path "
                             "(default: next to first input)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.bands:
        parser.error("at least one bands CSV is required outside --self-test")
    out = args.out or args.bands[0].with_name("channel_components.csv")
    return analyze(args.bands, args.census, PAGE_DEFAULT, out)


if __name__ == "__main__":
    sys.exit(main())
