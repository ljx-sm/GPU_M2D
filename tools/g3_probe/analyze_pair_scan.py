#!/usr/bin/env python3
"""GPU_M2D G3 S3b pair-scan analyzer.

Interprets a g3pool `--work-mode pair-scan` run. The single-bit S3 scan
cannot separate a column bit (flip keeps bank and row -> low) from a
bank-hash bit (flip leaves the bank -> low): both pairs time low. S3b
resolves this with the S1-verified in-page row-conflict anchor M
(0xd0100): (x, x^M) is same-bank different-row, so XOR-ing M into any
pair makes the row differ unconditionally. Wherever the anchor is
verified to hold at x (its own pair times conflict):

    (x, x ^ M ^ (1<<b)) conflicts  <=>  flipping b at x^M kept the bank
    (x, x ^ M ^ (1<<b)) low        <=>  flipping b at x^M changed the bank

so a column bit stays conflict on every base while a bank-hash bit drops
to low, and mixed votes across bases are nonlinear-hash evidence exactly
like the S3 single-bit votes. Two-bit pairs (x, x^(1<<b1)^(1<<b2)) test
additivity: under a linear bank hash two bank bits cancel and the pair
returns to conflict (if the combined mask still changes the row).

The query plan is regenerated deterministically from the run's own
pool_map.csv and the recorded selection parameters, and every work.csv
row is checked against it, so a query is interpreted only through its
recomputed role. Outputs pair_constraints.csv plus the per-bit anchored
vote table, the two-bit class summary, the anchor sweep coverage, and
the page-bit anchored votes. Exit codes: 0 ok, 2 integrity failure,
1 error (repo convention). `--self-test` pins the interpretation on a
synthetic pool with a fully linear bank decoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_bit_scan import check_calibration, classify  # noqa: E402
from g3_pool import (PAIR_SCAN_ANCHOR, PoolMap, parse_pool_map,  # noqa: E402
                     plan_pair_scan_queries, write_work_csv)

CONSTRAINT_FIELDS = ["query_id", "section", "bit", "bit2", "base_index",
                     "sample_index", "role", "pa_a", "pa_b", "xor",
                     "cycles_a", "cycles_b", "class", "interpretation"]
SYMMETRY_FRACTION = 0.25  # |a-b| above 0.25*amplitude -> asymmetric pair


def load_run(run_dir: Path) -> tuple[PoolMap, list[dict[str, str]], list[str], dict]:
    pool = PoolMap(parse_pool_map((run_dir / "pool_map.csv").read_text(encoding="utf-8")))
    with (run_dir / "result.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
    work = [line for line in
            (run_dir / "work.csv").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    selection = summary["query_selection"]
    if selection.get("mode") != "pair-scan":
        raise ValueError("summary.json query_selection is not pair-scan")
    params = {key: int(selection[key])
              for key in ("in_page_bases", "page_samples", "anchor_samples")}
    return pool, rows, work, params


def analyze(run_dir: Path, output: Path | None):
    pool, rows, work, params = load_run(run_dir)
    by_id = {int(row["query_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate query ids in result.csv")
    if not {0, 1, 2} <= set(by_id):
        raise ValueError("calibration queries 0..2 missing (not a pair-scan run?)")
    plan = plan_pair_scan_queries(pool, **params)
    if set(by_id) != set(range(len(plan))):
        raise ValueError(f"result has {len(by_id)} rows but the regenerated "
                         f"plan has {len(plan)}")

    problems: list[str] = []
    work_rows = [line.split(",") for line in work[1:]]  # work[0] is the header
    if len(work_rows) != len(plan):
        problems.append(f"work.csv has {len(work_rows)} rows, plan has {len(plan)}")
    else:
        for index, typed in enumerate(plan):
            cols = [int(value) for value in work_rows[index]]
            query = typed.query
            if cols[0] != index or cols[1:] != [query.chunk_a, query.ofs_a,
                                                query.chunk_b, query.ofs_b]:
                problems.append(f"work row {index} does not match the "
                                f"regenerated pair-scan plan")
                break

    floor = int(by_id[0]["cycles_a"])
    baseline = int(by_id[1]["cycles_a"])
    conflict = int(by_id[2]["cycles_a"])
    amplitude = check_calibration(floor, baseline, conflict)

    # Pass 1: classify everything and collect anchor validity where it is
    # measured (per in-page base, per page-triple sample).
    anchor_valid_base: dict[int, bool] = {}
    anchor_valid_sample: dict[tuple[int, int], bool] = {}
    classified: list[dict[str, object]] = []
    for index, typed in enumerate(plan):
        row = by_id[index]
        query = typed.query
        pa_a, pa_b = pool.query_pa(query)
        cycles_a, cycles_b = int(row["cycles_a"]), int(row["cycles_b"])
        if abs(cycles_a - cycles_b) > SYMMETRY_FRACTION * amplitude:
            label = "asymmetric"
        else:
            label = classify(cycles_a, baseline, amplitude)
        if typed.section == "anchor_base":
            anchor_valid_base[typed.base_index] = (label == "conflict")
        elif (typed.section == "page_triple" and typed.role == "anchor"):
            anchor_valid_sample[(typed.bit, typed.sample_index)] = \
                (label == "conflict")
        classified.append({"query_id": index, "section": typed.section,
                           "bit": typed.bit, "bit2": typed.bit2,
                           "base_index": typed.base_index,
                           "sample_index": typed.sample_index,
                           "role": typed.role, "pa_a": f"0x{pa_a:x}",
                           "pa_b": f"0x{pa_b:x}", "xor": f"0x{pa_a ^ pa_b:x}",
                           "cycles_a": cycles_a, "cycles_b": cycles_b,
                           "class": label, "interpretation": ""})

    # Pass 2: anchored interpretations only where the anchor holds.
    for record in classified:
        label = str(record["class"])
        if record["section"] == "calibration":
            record["interpretation"] = f"calibration_{record['role']}"
        elif record["section"] == "anchor_base":
            record["interpretation"] = ("anchor_valid" if label == "conflict"
                                        else f"anchor_{label}")
        elif record["section"] == "anchored_bit":
            if not anchor_valid_base.get(int(record["base_index"])):
                record["interpretation"] = "anchor_invalid_at_base"
            elif label == "conflict":
                record["interpretation"] = "bank_kept"
            elif label == "low":
                record["interpretation"] = "bank_changed"
            else:
                record["interpretation"] = label
        elif record["section"] == "pair":
            record["interpretation"] = label
        elif record["section"] == "anchor_sweep":
            record["interpretation"] = ("anchor_valid" if label == "conflict"
                                        else f"anchor_{label}")
        elif record["section"] == "page_triple":
            key = (int(record["bit"]), int(record["sample_index"]))
            if record["role"] == "anchor":
                record["interpretation"] = ("anchor_valid" if label == "conflict"
                                            else f"anchor_{label}")
            elif not anchor_valid_sample.get(key):
                record["interpretation"] = "anchor_invalid_at_page"
            elif record["role"] == "anchored":
                record["interpretation"] = ("bank_kept" if label == "conflict"
                                            else "bank_changed"
                                            if label == "low" else label)
            else:
                record["interpretation"] = f"single_{label}"

    if output is None:
        output = run_dir / "pair_constraints.csv"
    with output.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=CONSTRAINT_FIELDS)
        writer.writeheader()
        writer.writerows(classified)
    return classified, problems, (floor, baseline, conflict, amplitude)


def anchored_bit_table(records) -> list[str]:
    """In-page bit -> anchored votes across bases where the anchor holds."""
    lines = [f"{'bit':>4} {'n_valid':>7}  anchored votes "
             f"(bank_kept/bank_changed/mid/invalid_base)"]
    bits = sorted({int(row["bit"]) for row in records
                   if row["section"] == "anchored_bit"})
    for bit in bits:
        rows = [row for row in records
                if row["section"] == "anchored_bit" and int(row["bit"]) == bit]
        votes = Counter(str(row["interpretation"]) for row in rows)
        lines.append(f"{bit:>4} {votes.get('bank_kept', 0) + votes.get('bank_changed', 0) + votes.get('mid', 0):>7}  "
                     f"{votes.get('bank_kept', 0)}/"
                     f"{votes.get('bank_changed', 0)}/{votes.get('mid', 0)}/"
                     f"{votes.get('anchor_invalid_at_base', 0)}")
    return lines


def pair_summary(records) -> list[str]:
    votes = Counter(str(row["class"]) for row in records
                    if row["section"] == "pair")
    lines = [f"two-bit pairs (n={sum(votes.values())}): "
             f"conflict={votes.get('conflict', 0)} low={votes.get('low', 0)} "
             f"mid={votes.get('mid', 0)} asymmetric={votes.get('asymmetric', 0)}"]
    conflict_pairs = sorted((int(row["bit"]), int(row["bit2"]))
                            for row in records
                            if row["section"] == "pair"
                            and str(row["class"]) == "conflict")
    for start in range(0, len(conflict_pairs), 12):
        chunk = conflict_pairs[start:start + 12]
        lines.append("  conflict pairs: " + " ".join(f"{a}.{b}" for a, b in chunk))
    return lines


def anchor_sweep_summary(records) -> list[str]:
    rows = [row for row in records if row["section"] == "anchor_sweep"]
    valid = [row for row in rows if str(row["interpretation"]) == "anchor_valid"]
    return [f"anchor sweep: {len(valid)}/{len(rows)} sampled pages keep the "
            f"bank under mask 0x{PAIR_SCAN_ANCHOR:x}"
            + (f" (PA {valid[0]['pa_a']}..{valid[-1]['pa_a']})" if valid else "")]


def page_bit_table(records) -> list[str]:
    lines = [f"{'bit':>4} {'n_valid':>7} {'n_samp':>7}  anchored "
             f"(kept/changed/mid)   single fresh votes (low/mid/conflict)"]
    bits = sorted({int(row["bit"]) for row in records
                   if row["section"] == "page_triple"})
    for bit in bits:
        rows = [row for row in records
                if row["section"] == "page_triple" and int(row["bit"]) == bit]
        anchor_rows = [row for row in rows if row["role"] == "anchor"]
        valid = sum(str(row["interpretation"]) == "anchor_valid"
                    for row in anchor_rows)
        anchored = Counter(str(row["interpretation"]) for row in rows
                           if row["role"] == "anchored"
                           and str(row["interpretation"]).startswith(("bank_", "mid")))
        single = Counter(str(row["interpretation"]) for row in rows
                         if row["role"] == "single"
                         and str(row["interpretation"]).startswith("single_"))
        lines.append(f"{bit:>4} {valid:>7} {len(anchor_rows):>7}  "
                     f"{anchored.get('bank_kept', 0)}/"
                     f"{anchored.get('bank_changed', 0)}/{anchored.get('mid', 0)}"
                     f"{'':>8}"
                     f"{single.get('single_low', 0)}/"
                     f"{single.get('single_mid', 0)}/"
                     f"{single.get('single_conflict', 0)}")
    return lines


def self_test() -> int:
    import tempfile
    from g3_pool import POOL_MAP_FIELDS

    # Fully linear toy decoder: bank changes iff the XOR mask has odd
    # parity on BANK_BITS; the pair conflicts iff the bank is unchanged
    # AND the mask touches a row bit. The anchor (bits 8/16/18/19) has
    # even parity and touches rows 16/19, so it is a valid anchor.
    bank_bits = {8, 17, 18, 21}
    row_bits = {16, 19, 20} | set(range(22, 33))
    floor, baseline, conflict = 1023, 1014, 1140

    def pair_cycles(xor: int) -> int:
        if xor == 0:
            return floor  # same-address pair: the calibration floor
        parity = sum((xor >> bit) & 1 for bit in bank_bits) & 1
        row_differs = any((xor >> bit) & 1 for bit in row_bits)
        return conflict if parity == 0 and row_differs else baseline

    assert pair_cycles(PAIR_SCAN_ANCHOR) == conflict, "anchor must be valid"

    page_size = 2 * 1024 * 1024
    chunk0_va, chunk1_va = 0x7F0000000000, 0x7F0008000000
    rows = [(0, chunk0_va, 0x120000000), (0, chunk0_va + page_size, 0x120400000),
            (1, chunk1_va, 0x120200000), (1, chunk1_va + page_size, 0x120600000)]
    map_text = ",".join(POOL_MAP_FIELDS) + "\n" + "".join(
        f"r,0,uuid,{chunk},alloc,0x{va:x},0x{va + page_size:x},0x{fb:x},"
        f"{page_size},VIDEO,true,0x1,0x0,1,2,ebpf,c\n"
        for chunk, va, fb in rows)
    pool = PoolMap(parse_pool_map(map_text))

    params = {"in_page_bases": 4, "page_samples": 2, "anchor_samples": 4}
    plan = plan_pair_scan_queries(pool, **params)
    result_lines = ["# provenance", "query_id,chunk_a,ofs_a,chunk_b,ofs_b,cycles_a,cycles_b"]
    for index, typed in enumerate(plan):
        pa_a, pa_b = pool.query_pa(typed.query)
        cycles = pair_cycles(pa_a ^ pa_b)
        query = typed.query
        result_lines.append(f"{index},{query.chunk_a},{query.ofs_a},"
                            f"{query.chunk_b},{query.ofs_b},{cycles},{cycles}")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "pool_map.csv").write_text(map_text, encoding="utf-8")
        write_work_csv(run / "work.csv", [typed.query for typed in plan],
                       mode="pair-scan")
        (run / "result.csv").write_text("\n".join(result_lines) + "\n",
                                        encoding="utf-8")
        (run / "summary.json").write_text(json.dumps(
            {"query_selection": {"mode": "pair-scan", **params}}),
            encoding="utf-8")
        records, problems, calibration = analyze(run, run / "pair_constraints.csv")
        assert problems == [], problems
        assert calibration == (floor, baseline, conflict, conflict - baseline)

        def interpretation(section, bit, role=None, base=None):
            for row in records:
                if row["section"] == section and row["bit"] == bit and \
                        (role is None or row["role"] == role) and \
                        (base is None or row["base_index"] == base):
                    return row["interpretation"]
            raise AssertionError(f"missing {section} bit {bit}")

        # Column bit 0: bank kept under the anchor on every base; bank bit
        # 8 changed the bank on every base; row bit 16 kept the bank.
        assert interpretation("anchored_bit", 0) == "bank_kept"
        assert interpretation("anchored_bit", 8) == "bank_changed"
        assert interpretation("anchored_bit", 17) == "bank_changed"
        assert interpretation("anchored_bit", 18) == "bank_changed"
        assert interpretation("anchored_bit", 16) == "bank_kept"

        def pair_class(b1, b2):
            for row in records:
                if row["section"] == "pair" and row["bit"] == b1 and row["bit2"] == b2:
                    return row["class"]
            raise AssertionError(f"missing pair {b1}.{b2}")

        # Two bank bits cancel but stay in-row -> low; a row bit pairs with
        # anything to conflict; a lone bank bit with a row partner leaves
        # the bank -> low.
        assert pair_class(8, 17) == "low"      # parity 2, no row bit
        assert pair_class(8, 16) == "low"      # parity 1
        assert pair_class(16, 19) == "conflict"
        assert pair_class(0, 16) == "conflict"  # column + row, parity 0

        # Anchor sweep holds everywhere under the linear decoder.
        sweep = anchor_sweep_summary(records)
        assert "4/4" in sweep[0], sweep

        # Page bits: 21 is a bank bit (anchored -> changed, single -> low);
        # 22 is a row bit (anchored -> kept, single -> conflict).
        page_lines = page_bit_table(records)
        assert any(line.startswith("  21") and " 0/2/0" in line
                   and "2/0/0" in line for line in page_lines), page_lines
        assert any(line.startswith("  22") and " 2/0/0" in line
                   and "0/0/2" in line for line in page_lines), page_lines

        # The work-integrity check must catch a corrupted work row.
        bad = (run / "work.csv").read_text(encoding="utf-8").splitlines()
        parts = bad[5].split(",")
        parts[2] = str(int(parts[2]) + 256)
        bad[5] = ",".join(parts)
        (run / "work.csv").write_text("\n".join(bad) + "\n", encoding="utf-8")
        _, problems, _ = analyze(run, run / "pair_constraints.csv")
        assert problems and "does not match" in problems[0], problems
    print("analyze_pair_scan self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="g3pool pair-scan run directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="constraints CSV path (default "
                             "<run_dir>/pair_constraints.csv)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir is None:
        parser.error("run_dir is required outside --self-test")
    records, problems, (floor, baseline, conflict, amplitude) = \
        analyze(args.run_dir, args.output)
    print(f"calibration: floor {floor} / baseline {baseline} / "
          f"conflict {conflict} (amplitude {amplitude})")
    print("\n".join(anchored_bit_table(records)))
    print("\n".join(pair_summary(records)))
    print("\n".join(anchor_sweep_summary(records)))
    print("\n".join(page_bit_table(records)))
    if problems:
        print(f"INTEGRITY: {'; '.join(problems)}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
