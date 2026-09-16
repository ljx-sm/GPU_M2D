#!/usr/bin/env python3
"""GPU_M2D G3 S4b-0: local recalibration + latency-fingerprint mining.

S4 found the mapping model class insufficient, and one diagnosed cause is
label quality: every pair is classified against ONE global calibration
triple, while the S4b-0 mining below shows the per-page access latency
(the "lambda fingerprint") drifts across a ~45-cycle range address-
dependently -- the same order as the low/mid band width. A pair that is
physically "low" on a slow-lambda page lands in the global mid band and
its information is thrown away (556 mids, 23% of all pairs).

The first version of this tool shifted BOTH band boundaries by the local
lambda; the real data rejected that: conflict values sit 30-60 cycles
BELOW lambda + amplitude (conflict-minus-amplitude lands at 981-1001
while the low band itself spans 1011-1063), so the row-conflict penalty
does not ride on the same baseline as the low path. The rule here is
therefore deliberately asymmetric, and the asymmetry is a measured fact,
not a convenience:

  - low/mid boundary is LOCAL: low iff value < max(lambda_a, lambda_b) +
    LOW_FRACTION*amplitude. Low-pair latency is the slower endpoint's
    access path, so it tracks the page lambda directly.
  - mid/conflict boundary stays GLOBAL: conflict iff value >= the
    calibration triple's conflict threshold. S3b's fresh votes reproduced
    S3's conflict labels (5/5, 6/6), so those labels are validated; a
    local shift that destroys them is wrong by construction. Conflicts
    that sit below their own local conflict threshold are counted and
    reported as S4b-1 re-probe candidates instead of relabeled.

Lambda estimation: the minimum low-classified pair value touching a page
(the pair value is an upper bound of the page's own latency, noise only
adds delay); pages without a low sample fall back to their chunk's
minimum, then to the global median.

Outputs <name>_local.csv next to the input (same schema plus global_class
and local_base columns, consumable by solve_mapping.py) and a report:
lambda consistency, distribution and gap clustering, the PA-bit lambda
table (a channel/L2-slice hash would show no single-bit effect), the
anchor-sweep cross-tab, and the residual-mid analysis. Residual mids --
pairs that stay mid after local recalibration -- are same-channel
candidates (intra-channel bank-group pipelining pays a partial penalty; a
cross-channel pair has nothing to pay), saved to
same_channel_candidates.csv for the S4b channel analysis.

    python3 analyze_local_recal.py RUN_DIR [--output CSV]

Exit codes: 0 ok, 2 integrity failure, 1 error (repo convention).
`--self-test` pins the estimator, the asymmetric classifier, and the
fallback chain.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_pool import PoolMap, parse_pool_map  # noqa: E402

LOW_FRACTION = 0.35   # below base + 0.35*amplitude  -> low (analyzer values)
HIGH_FRACTION = 0.70  # above base + 0.70*amplitude  -> conflict
CLUSTER_GAP = 4       # lambda-distribution gap (cycles) that splits clusters


def classify(cycles: int, base: int, amplitude: int) -> str:
    """The global analyzer rule, mirrored for the integrity check."""
    if cycles >= base + HIGH_FRACTION * amplitude:
        return "conflict"
    if cycles < base + LOW_FRACTION * amplitude:
        return "low"
    return "mid"


def classify_local(value: int, local_base: int, hi_global: int,
                   amplitude: int) -> str:
    """Asymmetric local rule: local low boundary, global conflict gate."""
    if value >= hi_global:
        return "conflict"
    if value < local_base + LOW_FRACTION * amplitude:
        return "low"
    return "mid"


def load_result(run_dir: Path) -> dict[int, dict[str, str]]:
    with (run_dir / "result.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(line for line in source
                                   if not line.startswith("#")))
    by_id = {int(row["query_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate query ids in result.csv")
    return by_id


def calibration_triple(by_id: dict[int, dict[str, str]]) -> tuple[int, int, int]:
    if not {0, 1, 2} <= set(by_id):
        raise ValueError("calibration queries 0..2 missing")
    return (int(by_id[0]["cycles_a"]), int(by_id[1]["cycles_a"]),
            int(by_id[2]["cycles_a"]))


def read_constraints(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(line for line in source
                                   if not line.startswith("#")))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def page_lambdas(rows: list[dict[str, str]], lo_threshold: int) \
        -> tuple[dict[int, int], dict[int, list[int]]]:
    """Per-page lambda candidates: low-classified pair values by page.

    Returns (candidates, samples): candidates[page] = min value seen
    (the estimator), samples[page] = every low value touching the page
    (for the consistency statistic).
    """
    samples: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        if row.get("class") != "low" or not row.get("pa_a"):
            continue
        value = int(row["cycles_a"])
        if value >= lo_threshold:
            continue  # defensive: only true low-band values estimate lambda
        samples[int(row["pa_a"], 16) >> 21].append(value)
        samples[int(row["pa_b"], 16) >> 21].append(value)
    candidates = {page: min(values) for page, values in samples.items()}
    return candidates, samples


def resolve_lambdas(candidates: dict[int, int],
                    chunk_of_page: dict[int, int]) -> dict[int, int]:
    """Page -> chunk -> global-median fallback chain."""
    if not candidates:
        raise ValueError("no low-classified pairs: cannot estimate lambda")
    ordered = sorted(candidates.values())
    global_median = ordered[len(ordered) // 2]
    resolved = dict(candidates)
    for page, chunk in chunk_of_page.items():
        if page in resolved:
            continue
        chunk_pages = [p for p, c in chunk_of_page.items()
                       if c == chunk and p in candidates]
        resolved[page] = (candidates[min(chunk_pages)]
                          if chunk_pages else global_median)
    return resolved


def local_base(lambdas: dict[int, int], pa_a: int, pa_b: int) -> int:
    return max(lambdas.get(pa_a >> 21, 0), lambdas.get(pa_b >> 21, 0))


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


def bit_lambda_table(lambdas: dict[int, int]) -> list[tuple[float, int, int]]:
    """(|mean lambda difference|, bit, n1) for page-level PA bits.

    A channel/L2-slice hash of many bits shows no dominant single-bit
    effect; a bit-sliced layout would. Bits are page-level (21+) because
    lambda is estimated per page.
    """
    pages = sorted(lambdas)
    top_bit = pages[-1].bit_length() + 20 if pages else 21
    table = []
    for bit in range(21, top_bit + 1):
        ones = [v for p, v in lambdas.items() if (p >> (bit - 21)) & 1]
        zeros = [v for p, v in lambdas.items() if not (p >> (bit - 21)) & 1]
        if len(ones) < 3 or len(zeros) < 3:
            continue
        delta = abs(sum(ones) / len(ones) - sum(zeros) / len(zeros))
        table.append((delta, bit, len(ones)))
    table.sort(reverse=True)
    return table


def analyze(run_dir: Path, output: Path | None,
            oracle_out: Path | None) -> tuple[dict[str, object], list[str]]:
    pool = PoolMap(parse_pool_map((run_dir / "pool_map.csv").read_text(
        encoding="utf-8")))
    by_id = load_result(run_dir)
    floor, baseline, conflict = calibration_triple(by_id)
    amplitude = conflict - baseline
    if amplitude <= 0:
        raise ValueError(f"degenerate calibration {floor}/{baseline}/{conflict}")
    lo_threshold = baseline + LOW_FRACTION * amplitude
    hi_global = baseline + HIGH_FRACTION * amplitude

    constraints_path = next((run_dir / name for name in
                             ("pair_constraints.csv", "constraints.csv")
                             if (run_dir / name).is_file()), None)
    if constraints_path is None:
        raise ValueError("no constraints CSV in run dir")
    rows = read_constraints(constraints_path)

    chunk_of_page = {page.fb_pa_page_base >> 21: page.chunk_index
                     for page in pool.pages}

    # 1. Mirror check: global classes must reproduce the published labels.
    mismatches = []
    for row in rows:
        if row.get("class") in ("", None) or row["class"] == "asymmetric":
            continue
        qid = row.get("query_id", "")
        if qid == "" or int(qid) < 3:
            continue  # calibration rows are not reclassified
        mirrored = classify(int(row["cycles_a"]), baseline, amplitude)
        if mirrored != row["class"]:
            mismatches.append(f"q{qid}: {row['class']} != {mirrored}")
    if mismatches:
        return {}, mismatches  # integrity failure: never recalibrate on top

    # 2-3. Lambda estimation and local reclassification.
    candidates, samples = page_lambdas(rows, lo_threshold)
    lambdas = resolve_lambdas(candidates, chunk_of_page)

    out_rows: list[dict[str, str]] = []
    transitions: Counter = Counter()
    residual_mids: list[dict[str, str]] = []
    suspect_conflicts = 0
    for row in rows:
        new_row = dict(row)
        qid = row.get("query_id", "")
        if (qid != "" and int(qid) >= 3 and row["class"] != "asymmetric"
                and row.get("pa_a")):
            value = int(row["cycles_a"])
            base_l = local_base(lambdas, int(row["pa_a"], 16),
                                int(row["pa_b"], 16))
            new_class = classify_local(value, base_l, hi_global, amplitude)
            transitions[(row["class"], new_class)] += 1
            if (new_class == "conflict"
                    and value < base_l + HIGH_FRACTION * amplitude):
                suspect_conflicts += 1  # kept, but flagged for S4b-1
            new_row["class"] = new_class
            new_row["global_class"] = row["class"]
            new_row["local_base"] = str(base_l)
            if new_class == "mid":
                residual_mids.append(new_row)
        elif row["class"] != "asymmetric":
            new_row["global_class"] = row["class"]
            new_row["local_base"] = str(baseline)
        out_rows.append(new_row)

    if output is None:
        output = constraints_path.with_name(
            constraints_path.name.replace(".csv", "_local.csv"))
    fields = list(out_rows[0].keys())
    for row in out_rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    if oracle_out is None:
        oracle_out = run_dir / "same_channel_candidates.csv"
    if residual_mids:
        with oracle_out.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=fields)
            writer.writeheader()
            writer.writerows(residual_mids)

    stats = {
        "run_dir": run_dir, "output": output, "floor": floor,
        "baseline": baseline, "conflict": conflict, "amplitude": amplitude,
        "hi_global": hi_global, "rows": rows, "lambdas": lambdas,
        "samples": samples, "candidates": candidates,
        "transitions": transitions, "out_rows": out_rows,
        "residual_mids": residual_mids, "suspect_conflicts": suspect_conflicts,
        "constraints_path": constraints_path,
    }
    return stats, []


def print_report(stats: dict[str, object]) -> None:
    run_dir: Path = stats["run_dir"]
    lambdas: dict[int, int] = stats["lambdas"]
    samples: dict[int, list[int]] = stats["samples"]
    candidates: dict[int, int] = stats["candidates"]
    transitions: Counter = stats["transitions"]
    rows: list[dict[str, str]] = stats["rows"]

    print(f"run: {run_dir.name}  calibration "
          f"{stats['floor']}/{stats['baseline']}/{stats['conflict']} "
          f"(amplitude {stats['amplitude']})")
    before = Counter(row["class"] for row in rows)
    after = Counter(row["class"] for row in stats["out_rows"])
    print(f"classes global: {dict(before)}  ->  local: {dict(after)}")
    print("transitions (global -> local):")
    for (old, new), count in sorted(transitions.items()):
        if old != new:
            print(f"  {old} -> {new}: {count}")
    stayed = sum(count for (old, new), count in transitions.items()
                 if old == new)
    moved = sum(count for (old, new), count in transitions.items()
                if old != new)
    print(f"  (stable {stayed}, relabeled {moved}; conflicts kept on the "
          f"global gate: {stats['suspect_conflicts']} sit below their own "
          f"local conflict threshold -> S4b-1 re-probe candidates)")

    print(f"\nlambda fingerprint: {len(candidates)} pages with own estimate "
          f"({len(lambdas) - len(candidates)} fallback), "
          f"range {min(lambdas.values())}-{max(lambdas.values())}")
    multi = [values for values in samples.values() if len(values) >= 2]
    if multi:
        spreads = sorted(max(v) - min(v) for v in multi)
        print(f"  within-page spread (n={len(multi)} pages with >=2 lows): "
              f"median {spreads[len(spreads) // 2]}, "
              f"p90 {spreads[int(len(spreads) * 0.9)]}, max {spreads[-1]}")
    values = sorted(lambdas.values())
    print(f"  gap-{CLUSTER_GAP} clusters: {gap_clusters(values)}")
    table = bit_lambda_table(lambdas)
    if table:
        tops = ", ".join(f"bit {bit} |d|={delta:.1f} (n1={n1})"
                         for delta, bit, n1 in table[:5])
        print(f"  PA-bit lambda effect top-5: {tops}")

    if "section" in rows[0]:
        sweep = [row for row in rows if row.get("section") == "anchor_sweep"]
        if sweep:
            print("  anchor sweep vs page lambda:")
            for cls in ("low", "mid", "conflict"):
                pages = [int(row["pa_a"], 16) >> 21 for row in sweep
                         if row["class"] == cls]
                have = sorted(lambdas[p] for p in pages if p in candidates)
                if pages:
                    span = (f"{have[0]}-{have[-1]} (median "
                            f"{have[len(have) // 2]})") if have else "n/a"
                    print(f"    {cls}: {len(pages)} pages, lambda {span}, "
                          f"{len(have)} with own estimate")

    mids: list[dict[str, str]] = stats["residual_mids"]
    print(f"\nresidual mids after local recalibration: {len(mids)} "
          f"(same-channel candidates; intra-channel bank-group pipelining "
          f"pays a partial penalty, cross-channel pays nothing)")
    if mids:
        excess = sorted(int(row["cycles_a"]) - int(row["local_base"])
                        for row in mids)
        print(f"  excess over local base: median "
              f"{excess[len(excess) // 2]}, range {excess[0]}-{excess[-1]} "
              f"(low band edge {LOW_FRACTION * float(stats['amplitude']):.0f}, "
              f"conflict gate {stats['hi_global']})")
    print(f"output: {stats['output']}")


def self_test() -> int:
    # Global classifier bands around a realistic triple (amp 124).
    assert classify(1014, 1018, 124) == "low"
    assert classify(1080, 1018, 124) == "mid"
    assert classify(1120, 1018, 124) == "conflict"
    # Asymmetric local rule: low boundary moves, conflict gate does not.
    assert classify_local(1070, 1055, 1105, 124) == "low"   # slow page low
    assert classify_local(1068, 1020, 1105, 124) == "mid"   # fast page mid
    assert classify_local(1110, 1055, 1105, 124) == "conflict"  # kept even
    # though 1110 < 1055 + 87 (the symmetric shift would have killed it)

    # Gap clustering.
    assert gap_clusters([1, 2, 3, 9, 10, 20]) == [(1, 3, 3), (9, 10, 2),
                                                  (20, 20, 1)]
    # Lambda estimation: minimum of touching low values, both endpoints,
    # conflicts excluded.
    rows = [{"class": "low", "cycles_a": "1050",
             "pa_a": "0x0", "pa_b": "0x200000"},
            {"class": "low", "cycles_a": "1030",
             "pa_a": "0x0", "pa_b": "0x400000"},
            {"class": "conflict", "cycles_a": "1140",
             "pa_a": "0x0", "pa_b": "0x1"}]
    candidates, samples = page_lambdas(rows, lo_threshold=1061)
    assert candidates[0] == 1030 and candidates[2] == 1030  # pages 0, 2
    assert candidates[1] == 1050
    assert len(samples[0]) == 2

    # Fallback chain: unknown page takes its chunk sibling's lambda, and a
    # chunk with no candidate at all falls back to the global median.
    # Page numbers below; chunks 0={0,1}, 1={2,3,4}, 2={5}.
    candidates = {0: 1010, 1: 1050, 2: 1045, 3: 1055}
    chunk_of_page = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
    lambdas = resolve_lambdas(candidates, chunk_of_page)
    assert lambdas[4] == 1045  # chunk-1 sibling minimum
    # global median of [1010, 1045, 1050, 1055] is 1050 (index 2 of sorted)
    assert lambdas[5] == 1050  # chunk 2 empty -> global median

    # End-to-end on a synthetic run directory: a slow-lambda page whose
    # physical lows land in the global mid band must be recovered, while
    # conflicts and true shoulders keep their labels.
    import tempfile
    page = 2 * 1024 * 1024
    fields = ["run_id", "device", "gpu_uuid", "chunk_index", "allocation_id",
              "va_page_base", "va_page_end_exclusive", "fb_pa_page_base",
              "page_size", "aperture", "pte_valid", "raw_pte_lo",
              "raw_pte_hi", "mapped_at_ns", "unmapped_at_ns", "source",
              "confidence"]
    # chunk 0: pages at PA 0x0 and 0x200000 (fast, lambda ~1020);
    # chunk 1: page at PA 0x400000 (slow, lambda ~1055).
    map_rows = [
        "r,0,u,0,a0,0x7f0000000000,0x7f0000200000,0x0,%d,VIDEO,true,0x1,0x0,1,2,e,c" % page,
        "r,0,u,0,a0,0x7f0000200000,0x7f0000400000,0x200000,%d,VIDEO,true,0x1,0x0,1,2,e,c" % page,
        "r,0,u,1,a1,0x7f0008000000,0x7f0008200000,0x400000,%d,VIDEO,true,0x1,0x0,1,2,e,c" % page,
    ]
    result = ["# provenance", "query_id,chunk_a,ofs_a,chunk_b,ofs_b,"
              "cycles_a,cycles_b"]
    # calibration: floor/baseline/conflict = 1015/1018/1142 (amp 124,
    # low band edge 1061.4, conflict gate 1104.8)
    plan = [
        (0, 0, 0, 0, 0, 1015),      # floor
        (1, 0, 0, 0, 8192, 1018),   # baseline
        (2, 0, 0, 0, 852224, 1142), # conflict
        # q3: fast-region low (1015) stays low; sets page-0/1 lambda 1015.
        (3, 0, 0, 0, 65536, 1015),
        # q4: slow-region physical low at 1070: global MID (>= 1061),
        # local low (1070 < 1055+43.4).
        (4, 1, 0, 1, 65536, 1070),
        # q5: slow-region physical low at 1055 -> global low, lambda source.
        (5, 1, 0, 1, 131072, 1055),
        # q6: slow-region conflict at 1110: global conflict, and BELOW the
        # local symmetric conflict threshold (1055+86.8=1142) -- must stay
        # conflict (the asymmetric rule) and be counted suspect.
        (6, 1, 0, 1, 852224, 1110),
        # q7: fast-region true mid 1068: global mid, local mid.
        (7, 0, 0, 0, 98304, 1068),
        # q8: fast-region marginal low at 1059: global low (< 1061.4),
        # local mid (>= 1015+43.4) -- the local boundary bites lows too.
        (8, 0, 0, 0, 32768, 1059),
    ]
    for qid, ca, oa, cb, ob, cyc in plan:
        result.append(f"{qid},{ca},{oa},{cb},{ob},{cyc},{cyc}")
    constraints = ["query_id,kind,bit,pa_a,pa_b,xor,cycles_a,cycles_b,class,"
                   "band_cycles"]
    expect_global = {3: "low", 4: "mid", 5: "low", 6: "conflict",
                     7: "mid", 8: "low"}
    for qid, ca, oa, cb, ob, cyc in plan[3:]:
        pa_a = {0: 0, 1: 0x400000}[ca] + oa
        pa_b = {0: 0, 1: 0x400000}[cb] + ob
        constraints.append(
            f"{qid},in_page,16,0x{pa_a:x},0x{pa_b:x},0x{pa_a ^ pa_b:x},"
            f"{cyc},{cyc},{expect_global[qid]},x")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "pool_map.csv").write_text(",".join(fields) + "\n"
                                          + "\n".join(map_rows) + "\n")
        (run / "result.csv").write_text("\n".join(result) + "\n")
        (run / "constraints.csv").write_text("\n".join(constraints) + "\n")
        stats, problems = analyze(run, None, None)
        assert problems == [], problems
        out_rows = {int(r["query_id"]): r for r in stats["out_rows"]}
        assert out_rows[3]["class"] == "low"       # fast low stays low
        assert out_rows[4]["class"] == "low"       # recovered from global mid
        assert out_rows[4]["global_class"] == "mid"
        assert out_rows[5]["class"] == "low"
        assert out_rows[6]["class"] == "conflict"  # global gate keeps it
        assert stats["suspect_conflicts"] == 1     # but flagged suspect
        assert out_rows[7]["class"] == "mid"       # true shoulder stays mid
        assert out_rows[8]["class"] == "mid"       # marginal low corrected
        assert out_rows[8]["global_class"] == "low"
        lambdas: dict[int, int] = stats["lambdas"]
        assert lambdas[2] == 1055, lambdas         # from q5, the slow min
        assert stats["residual_mids"]
        assert (run / "same_channel_candidates.csv").is_file()
        # mirror-check failure must be reported, never recalibrated
        bad = run / "constraints.csv"
        bad.write_text(bad.read_text().replace(",mid,x\n", ",low,x\n"))
        stats2, problems2 = analyze(run, None, None)
        assert problems2 and "!=" in problems2[0]
    print("analyze_local_recal self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="g3pool run directory (S3 or S3b)")
    parser.add_argument("--output", type=Path, default=None,
                        help="reclassified CSV path (default <input>_local.csv)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir is None:
        parser.error("run_dir is required outside --self-test")
    stats, problems = analyze(args.run_dir, args.output, None)
    if problems:
        print(f"INTEGRITY: {'; '.join(problems[:5])}")
        return 2
    print_report(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
