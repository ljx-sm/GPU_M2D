#!/usr/bin/env python3
"""GPU_M2D G3 S4b-2: three-band relabeling of the S3/S3b constraints.

S4b-1 proved the old two-band classifier conflated two regimes: a
reproducible shallow band (~0.70-0.85*amplitude: same channel, different
bank / bank group) and full row conflicts (>=0.90*amplitude: same bank,
different row). S3's conflicts were ALL shallow; S3b's split 185/71. The
S4 solver's same-bank GF(2) constraints were that mixture -- one concrete
cause of its gate FAIL.

This tool relabels an old run's constraints with the three-band rule,
using the S4b-1 CENSUS per-page lambdas (clean self-pair measurements)
for the local low/mid boundary instead of the S4b-0 pair-derived
estimates (which barely correlate with the census, r=0.11). The census
run reproduced the same 4 GiB PA hole, so its page table maps 1:1 onto
the old runs' pages; any constraint page without a census lambda is an
integrity failure (exit 2), never silently defaulted.

Band order (measured in S4b-1: neither penalty rides the lambda):
  deep_conflict  value >= baseline + 0.90*amplitude  (global)
  shoulder       value >= baseline + 0.70*amplitude  (global)
  low            value <  max(lambda_a, lambda_b) + 0.35*amplitude
  mid            otherwise

Outputs <name>_bands.csv (four classes, for the channel graph and the
record) and <name>_solver3.csv (solver view: deep_conflict -> conflict,
shoulder -> mid) next to the input.

    python3 analyze_three_band.py RUN_DIR --census CENSUS_RUN_DIR

Exit codes: 0 ok, 2 integrity failure, 1 error (repo convention).
`--self-test` pins the band rule and the census-lambda join.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_census import DEEP_FRACTION, HIGH_FRACTION, LOW_FRACTION, classify3
from g3_pool import PoolMap, parse_pool_map

BAND_FIELDS = ["query_id", "kind", "bit", "pa_a", "pa_b", "xor",
               "cycles_a", "cycles_b", "class", "lambda_a", "lambda_b",
               "lambda_max", "shoulder_gate", "deep_gate"]


def load_census_lambdas(census_dir: Path) -> tuple[dict[int, int], int]:
    """page base -> census self-pair lambda (and the page size)."""
    pool = PoolMap(parse_pool_map((census_dir / "pool_map.csv").read_text(
        encoding="utf-8")))
    lambdas: dict[int, int] = {}
    with (census_dir / "page_lambdas.csv").open(encoding="utf-8",
                                                newline="") as source:
        for row in csv.DictReader(source):
            lambdas[int(row["fb_pa_page_base"], 16)] = int(row["self_cycles"])
    page_size = pool.pages[0].page_size
    if len(lambdas) != len(pool.pa_pages):
        raise ValueError("census page_lambdas.csv does not cover the pool map")
    return lambdas, page_size


def analyze(run_dir: Path, census_dir: Path) \
        -> tuple[dict[str, object], list[str]]:
    pool = PoolMap(parse_pool_map((run_dir / "pool_map.csv").read_text(
        encoding="utf-8")))
    with (run_dir / "result.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(line for line in source
                                   if not line.startswith("#")))
    by_id = {int(row["query_id"]): row for row in rows}
    floor, baseline, conflict = (int(by_id[i]["cycles_a"]) for i in (0, 1, 2))
    amplitude = conflict - baseline
    if amplitude <= 0:
        raise ValueError(f"degenerate calibration {floor}/{baseline}/{conflict}")

    constraints_path = next((run_dir / name for name in
                             ("pair_constraints.csv", "constraints.csv")
                             if (run_dir / name).is_file()), None)
    if constraints_path is None:
        raise ValueError("no constraints CSV in run dir")
    with constraints_path.open(encoding="utf-8", newline="") as source:
        cons = list(csv.DictReader(line for line in source
                                   if not line.startswith("#")))

    try:
        lambdas, census_page_size = load_census_lambdas(census_dir)
    except (OSError, ValueError) as exc:
        return {}, [f"census load failed: {exc}"]
    page_size = pool.pages[0].page_size
    problems: list[str] = []
    if page_size != census_page_size:
        problems.append(f"page size mismatch: run {page_size} vs census "
                        f"{census_page_size}")
    con_pages = set()
    for row in cons:
        if (row.get("class") == "asymmetric" or not row.get("pa_a")
                or not row.get("pa_b")):
            continue
        con_pages.add(int(row["pa_a"], 16) & ~(page_size - 1))
        con_pages.add(int(row["pa_b"], 16) & ~(page_size - 1))
    missing = sorted(con_pages - set(lambdas))
    if missing:
        problems.append(f"{len(missing)} constraint pages lack a census "
                        f"lambda (e.g. 0x{missing[0]:x}) -- different PA "
                        f"hole; refusing to relabel")
    if problems:
        return {}, problems

    shoulder_gate = baseline + HIGH_FRACTION * amplitude
    deep_gate = baseline + DEEP_FRACTION * amplitude
    band_rows: list[dict[str, str]] = []
    transitions: Counter = Counter()
    for row in cons:
        out = {field: row.get(field, "") for field in BAND_FIELDS}
        qid = row.get("query_id", "")
        if (qid == "" or int(qid) < 3 or row["class"] == "asymmetric"
                or not row.get("pa_a") or not row.get("pa_b")):
            out["class"] = row.get("class", "asymmetric")
            band_rows.append(out)
            continue
        pa_a, pa_b = int(row["pa_a"], 16), int(row["pa_b"], 16)
        lambda_a = lambdas[pa_a & ~(page_size - 1)]
        lambda_b = lambdas[pa_b & ~(page_size - 1)]
        band = classify3(int(row["cycles_a"]), max(lambda_a, lambda_b),
                         shoulder_gate, deep_gate, amplitude)
        transitions[(row["class"], band)] += 1
        out.update({"class": band, "lambda_a": lambda_a, "lambda_b": lambda_b,
                    "lambda_max": max(lambda_a, lambda_b),
                    "shoulder_gate": f"{shoulder_gate:.1f}",
                    "deep_gate": f"{deep_gate:.1f}"})
        band_rows.append(out)

    bands_path = constraints_path.with_name(
        constraints_path.name.replace(".csv", "_bands.csv"))
    with bands_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=BAND_FIELDS)
        writer.writeheader()
        writer.writerows(band_rows)

    solver_map = {"deep_conflict": "conflict", "shoulder": "mid",
                  "mid": "mid", "low": "low"}
    solver_path = constraints_path.with_name(
        constraints_path.name.replace(".csv", "_solver3.csv"))
    with solver_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=[
            "query_id", "kind", "bit", "pa_a", "pa_b", "xor",
            "cycles_a", "cycles_b", "class"])
        writer.writeheader()
        for row in band_rows:
            writer.writerow({field: row[field] for field in (
                "query_id", "kind", "bit", "pa_a", "pa_b", "xor",
                "cycles_a", "cycles_b")}
                | {"class": solver_map.get(row["class"], row["class"])})

    return {"bands_path": bands_path, "solver_path": solver_path,
            "constraints_path": constraints_path, "rows": cons,
            "band_rows": band_rows, "transitions": transitions,
            "baseline": baseline, "conflict": conflict,
            "amplitude": amplitude, "run_dir": run_dir}, []


def print_report(stats: dict[str, object]) -> None:
    transitions: Counter = stats["transitions"]
    print(f"run: {stats['run_dir'].name}  calibration "
          f"{stats['baseline']}/{stats['conflict']} "
          f"(amplitude {stats['amplitude']})")
    print("old class -> three-band class:")
    for (old, new), count in sorted(transitions.items()):
        marker = "  <<" if old == "conflict" and new != "deep_conflict" else ""
        print(f"  {old} -> {new}: {count}{marker}")
    after = Counter(row["class"] for row in stats["band_rows"])
    print(f"bands: {dict(after)}")
    print(f"outputs: {stats['bands_path'].name}, "
          f"{stats['solver_path'].name} (solver view: "
          f"deep->conflict, shoulder->mid)")


def self_test() -> int:
    import tempfile
    page = 2 * 1024 * 1024
    fields = ["run_id", "device", "gpu_uuid", "chunk_index", "allocation_id",
              "va_page_base", "va_page_end_exclusive", "fb_pa_page_base",
              "page_size", "aperture", "pte_valid", "raw_pte_lo",
              "raw_pte_hi", "mapped_at_ns", "unmapped_at_ns", "source",
              "confidence"]
    map_text = ",".join(fields) + "\n" + "".join(
        f"r,0,u,0,a0,0x{0x7f0000000000 + i * page:x},"
        f"0x{0x7f0000000000 + (i + 1) * page:x},0x{0x120000000 + i * page:x},"
        f"{page},VIDEO,true,0x1,0x0,1,2,e,c\n" for i in range(3))
    # calibration 1015/1018/1142 (amp 124: shoulder gate 1104.8, deep 1129.6;
    # census lambdas 1015/1040/1016 -> local low edges 1058.4/1083.4/1059.4)
    result = ["query_id,chunk_a,ofs_a,chunk_b,ofs_b,cycles_a,cycles_b"]
    plan = [(0, 0, 0, 0, 0, 1015), (1, 0, 0, 0, 8192, 1018),
            (2, 0, 0, 0, 852224, 1142),
            (3, 0, 0, 0, 65536, 1140),    # deep conflict
            (4, 0, 0, 0, 131072, 1114),   # shoulder
            (5, 0, 0, 1, 0, 1070),        # slow page lambda 1040: local low
            (6, 0, 0, 1, 65536, 1090),    # lambda max 1040 -> mid (1083-1105)
            (7, 0, 0, 2, 0, 1019)]        # fast pages: low
    for qid, ca, oa, cb, ob, cyc in plan:
        result.append(f"{qid},{ca},{oa},{cb},{ob},{cyc},{cyc}")
    constraints = ["query_id,kind,bit,pa_a,pa_b,xor,cycles_a,cycles_b,class,"
                   "band_cycles"]
    old_class = {3: "conflict", 4: "conflict", 5: "low", 6: "mid", 7: "low"}
    for qid, ca, oa, cb, ob, cyc in plan[3:]:
        pa_a = 0x120000000 + oa
        pa_b = 0x120000000 + cb * page + ob
        constraints.append(
            f"{qid},in_page,16,0x{pa_a:x},0x{pa_b:x},0x{pa_a ^ pa_b:x},"
            f"{cyc},{cyc},{old_class[qid]},x")
    with tempfile.TemporaryDirectory() as tmp:
        run, census = Path(tmp) / "run", Path(tmp) / "census"
        run.mkdir()
        census.mkdir()
        (run / "pool_map.csv").write_text(map_text)
        (run / "result.csv").write_text("\n".join(result) + "\n")
        (run / "constraints.csv").write_text("\n".join(constraints) + "\n")
        (census / "pool_map.csv").write_text(map_text)
        (census / "page_lambdas.csv").write_text(
            "fb_pa_page_base,self_cycles\n"
            + "".join(f"0x{0x120000000 + i * page:x},{lam}\n" for i, lam in
                      enumerate((1015, 1040, 1016))))
        stats, problems = analyze(run, census)
        assert problems == [], problems
        bands = {int(r["query_id"]): r["class"] for r in stats["band_rows"]}
        assert bands[3] == "deep_conflict"
        assert bands[4] == "shoulder"          # the conflated old conflict
        assert bands[5] == "low"               # local lambda rescues it
        assert bands[6] == "mid"
        assert bands[7] == "low"
        solver = list(csv.DictReader((stats["solver_path"]).open(
            encoding="utf-8")))
        solver_classes = {int(r["query_id"]): r["class"] for r in solver}
        assert solver_classes[3] == "conflict"   # deep -> conflict
        assert solver_classes[4] == "mid"        # shoulder -> excluded
        assert solver_classes[6] == "mid"
        # A census page table that misses a constraint page must refuse.
        (census / "page_lambdas.csv").write_text(
            "fb_pa_page_base,self_cycles\n0x120000000,1015\n")
        stats2, problems2 = analyze(run, census)
        assert problems2 and "census" in problems2[0].lower()
    print("analyze_three_band self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="S3/S3b run directory to relabel")
    parser.add_argument("--census", type=Path, required=False,
                        help="S4b-1 census run directory (page_lambdas.csv)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.run_dir is None or args.census is None:
        parser.error("run_dir and --census are required outside --self-test")
    stats, problems = analyze(args.run_dir, args.census)
    if problems:
        print(f"INTEGRITY: {'; '.join(problems[:5])}")
        return 2
    print_report(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
