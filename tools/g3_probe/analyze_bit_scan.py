#!/usr/bin/env python3
"""GPU_M2D G3 S3 bit-scan analyzer.

Classifies the timing of every single-bit PA pair measured by a g3pool
`--work-mode bit-scan` run and emits the per-pair constraint records that
S4's solver consumes.

Inputs: one run directory of run_g3_pool_probe.py containing pool_map.csv
(every page's observed framebuffer PA), work.csv (queries; ids 0..2 are
the calibration triple), and result.csv (per-query min-of-iters cycles).

Classification uses the run's own calibration triple, so thresholds track
the run's boost clock. The timing primitive separates exactly one thing:
whether the pair pays the row-conflict penalty. Same-row and
different-bank pairs sit at the same low level (S2: floor 1023 vs
baseline 1014 cycles -- indistinguishable), so the honest classes are:
  - low        no conflict: flipping the bit either left the bank and row
              (column/DQ bit) or changed the bank (bank/channel-hash bit);
              disambiguating these two needs partner probes / the solver
  - mid        shoulder between the bands (same channel, different bank
              group; kept as its own class, not forced either way)
  - conflict   same bank, different row: the flipped bit is a row-address
              bit that does not participate in the bank hash
  - asymmetric both threads disagree beyond a quarter of the conflict
              amplitude -- pair excluded from constraints (noise)

Per PA bit the tool aggregates the class votes. A bit whose votes split
across base pages is evidence of a nonlinear (seeded) hash and is flagged
SPLIT. Outputs constraints.csv next to the inputs and prints the per-bit
table. Exit codes: 0 ok, 2 integrity failure, 1 error (repo convention).
`--self-test` pins the classifier on synthetic data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_pool import PoolMap, parse_pool_map  # noqa: E402

CONSTRAINT_FIELDS = ["query_id", "kind", "bit", "pa_a", "pa_b", "xor",
                     "cycles_a", "cycles_b", "class", "band_cycles"]

LOW_FRACTION = 0.35   # below baseline + 0.35*amplitude  -> same row
HIGH_FRACTION = 0.70  # above baseline + 0.70*amplitude  -> full conflict
SYMMETRY_FRACTION = 0.25  # |a-b| above 0.25*amplitude -> asymmetric


def classify(cycles: int, baseline: int, amplitude: int) -> str:
    if cycles >= baseline + HIGH_FRACTION * amplitude:
        return "conflict"
    if cycles < baseline + LOW_FRACTION * amplitude:
        return "low"
    return "mid"


def check_calibration(floor: int, baseline: int, conflict: int) -> int:
    """The calibration triple must show a clean conflict band, and the
    same-address floor must sit inside the low band (floor may legitimately
    sit a few cycles either side of the different-bank baseline)."""
    amplitude = conflict - baseline
    if amplitude <= 0 or not (baseline - max(8, amplitude // 2)
                              <= floor
                              < baseline + LOW_FRACTION * amplitude):
        raise ValueError(f"degenerate calibration: floor={floor} "
                         f"baseline={baseline} conflict={conflict}")
    return amplitude


def load_run(run_dir: Path) -> tuple[PoolMap, list[dict[str, str]]]:
    pool = PoolMap(parse_pool_map((run_dir / "pool_map.csv").read_text(encoding="utf-8")))
    with (run_dir / "result.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
    return pool, rows


def analyze(run_dir: Path, output: Path | None) -> tuple[list[dict[str, object]], str]:
    pool, rows = load_run(run_dir)
    by_id = {int(row["query_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate query ids in result.csv")
    if not {0, 1, 2} <= set(by_id):
        raise ValueError("calibration queries 0..2 missing (not a bit-scan run?)")

    floor = int(by_id[0]["cycles_a"])
    baseline = int(by_id[1]["cycles_a"])
    conflict = int(by_id[2]["cycles_a"])
    amplitude = check_calibration(floor, baseline, conflict)

    constraints: list[dict[str, object]] = []
    problems: list[str] = []
    for query_id in sorted(by_id):
        if query_id < 3:
            continue
        row = by_id[query_id]
        query = (int(row["chunk_a"]), int(row["ofs_a"]),
                 int(row["chunk_b"]), int(row["ofs_b"]))
        pa_a = pool.pa_of(*query[:2])
        pa_b = pool.pa_of(*query[2:])
        xor = pa_a ^ pa_b
        if bin(xor).count("1") != 1:
            problems.append(f"query {query_id}: PA xor {xor:#x} is not one bit")
            continue
        cycles_a, cycles_b = int(row["cycles_a"]), int(row["cycles_b"])
        if abs(cycles_a - cycles_b) > SYMMETRY_FRACTION * amplitude:
            constraints.append({"query_id": query_id, "kind": "asymmetric",
                                "bit": xor.bit_length() - 1, "pa_a": f"0x{pa_a:x}",
                                "pa_b": f"0x{pa_b:x}", "xor": f"0x{xor:x}",
                                "cycles_a": cycles_a, "cycles_b": cycles_b,
                                "class": "asymmetric", "band_cycles": ""})
            continue
        label = classify(cycles_a, baseline, amplitude)
        constraints.append({"query_id": query_id,
                            "kind": "in_page" if pa_a >> 21 == pa_b >> 21 else "page",
                            "bit": xor.bit_length() - 1, "pa_a": f"0x{pa_a:x}",
                            "pa_b": f"0x{pa_b:x}", "xor": f"0x{xor:x}",
                            "cycles_a": cycles_a, "cycles_b": cycles_b,
                            "class": label,
                            "band_cycles": f"{baseline + LOW_FRACTION * amplitude:.0f}"
                                           f"-{baseline + HIGH_FRACTION * amplitude:.0f}"})

    if output is None:
        output = run_dir / "constraints.csv"
    with output.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=CONSTRAINT_FIELDS)
        writer.writeheader()
        writer.writerows(constraints)
    return constraints, ("" if not problems else "; ".join(problems))


def per_bit_table(constraints: list[dict[str, object]]) -> list[str]:
    lines = [f"{'bit':>4} {'kind':>8} {'n':>4}  class votes (low/mid/conflict/asym)"]
    for kind in ("in_page", "page"):
        bits = sorted({int(row["bit"]) for row in constraints
                       if row["kind"] == kind})
        for bit in bits:
            votes = Counter(str(row["class"]) for row in constraints
                            if row["kind"] == kind and int(row["bit"]) == bit)
            total = sum(votes.values())
            splits = "  SPLIT" if len([c for c in votes.values() if c >= total * 0.25]) > 1 else ""
            lines.append(f"{bit:>4} {kind:>8} {total:>4}  "
                         f"{votes.get('low', 0)}/{votes.get('mid', 0)}/"
                         f"{votes.get('conflict', 0)}/{votes.get('asymmetric', 0)}{splits}")
    return lines


def self_test() -> int:
    def classify_all(triple, samples):
        floor, baseline, conflict = triple
        amplitude = check_calibration(floor, baseline, conflict)
        return [classify(cycles, baseline, amplitude) for cycles in samples]

    # Bands around a realistic triple (S2 numbers at 2675 MHz): low band
    # covers both the floor and different-bank levels.
    assert classify_all((1023, 1014, 1140), [1014, 1023, 1060]) == ["low", "low", "mid"]
    assert classify_all((1023, 1014, 1140), [1140, 1120, 1090]) == ["conflict", "conflict", "mid"]
    # Degenerate calibration must be rejected (floor below baseline is
    # physically fine and accepted).
    for bad in ((1014, 1014, 1014), (1023, 1014, 1000), (1100, 1014, 1090)):
        try:
            classify_all(bad, [1014])
            raise AssertionError(f"degenerate calibration accepted: {bad}")
        except ValueError:
            pass

    # End-to-end on a synthetic run directory.
    import tempfile
    page_size = 2 * 1024 * 1024
    fields = ["run_id", "device", "gpu_uuid", "chunk_index", "allocation_id",
              "va_page_base", "va_page_end_exclusive", "fb_pa_page_base", "page_size",
              "aperture", "pte_valid", "raw_pte_lo", "raw_pte_hi",
              "mapped_at_ns", "unmapped_at_ns", "source", "confidence"]
    rows = []
    queries = ["query_id,chunk_a,ofs_a,chunk_b,ofs_b,cycles_a,cycles_b"]
    # chunk 0 = one page; bits 0..20 in-page, bit 21 to chunk 1.
    plan = [
        (0, 0, 0, 0, 0, 1023, 1023),      # floor control
        (0, 0, 0, 0, 8192, 1014, 1014),   # baseline control
        (0, 0, 0, 0, 852224, 1140, 1140), # conflict control
        (0, 0, 0, 0, 256, 1014, 1014),    # bit 8: different bank
        (0, 0, 0, 0, 1, 1023, 1023),      # bit 0: same row
        (0, 0, 0, 0, 1048576, 1140, 1140),# bit 20: same bank, other row
        (0, 0, 0, 1, 0, 1140, 1140),      # bit 21 (page): conflict
        (0, 0, 0, 0, 128, 1080, 1030),    # asymmetric pair -> excluded
    ]
    for i, (qid, ca, oa, cb, ob, cyc_a, cyc_b) in enumerate(plan):
        queries.append(f"{i},{ca},{oa},{cb},{ob},{cyc_a},{cyc_b}")
    map_rows = [f"r,0,uuid,0,a0,0x7f0000000000,0x7f0000200000,0x120000000,"
                f"{page_size},VIDEO,true,0x1,0x0,1,2,ebpf,c",
                f"r,0,uuid,1,a1,0x7f0008000000,0x7f0008200000,0x120200000,"
                f"{page_size},VIDEO,true,0x1,0x0,1,2,ebpf,c"]
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "pool_map.csv").write_text(",".join(fields) + "\n" + "\n".join(map_rows) + "\n")
        (run / "result.csv").write_text("# provenance\n" + "\n".join(queries) + "\n")
        constraints, problems = analyze(run, run / "constraints.csv")
        by_bit = {(row["kind"], int(row["bit"])): row["class"] for row in constraints}
        assert by_bit[("in_page", 8)] == "low"    # bank interleave: no conflict on this pair
        assert by_bit[("in_page", 0)] == "low"    # column bit
        assert by_bit[("in_page", 20)] == "conflict"
        assert by_bit[("page", 21)] == "conflict"
        asym = [row for row in constraints if row["class"] == "asymmetric"]
        assert len(asym) == 1 and problems == ""
        table = per_bit_table(constraints)
        assert any("SPLIT" not in line and "in_page" in line for line in table)
    print("analyze_bit_scan self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="g3pool bit-scan run directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="constraints CSV path (default <run_dir>/constraints.csv)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir is None:
        parser.error("run_dir is required outside --self-test")
    constraints, problems = analyze(args.run_dir, args.output)
    for line in per_bit_table(constraints):
        print(line)
    usable = [row for row in constraints if row["class"] != "asymmetric"]
    print(f"\nconstraints: {len(usable)} usable, "
          f"{len(constraints) - len(usable)} asymmetric, "
          f"written to {args.output or args.run_dir / 'constraints.csv'}")
    if problems:
        print(f"INTEGRITY: {problems}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
