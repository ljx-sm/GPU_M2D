#!/usr/bin/env python3
"""G8-T2: derive each workload's cache-flip LIFETIME classes from measured
campaign evidence (docs/G8_CACHE_FAULT_PLAN.md §3.4).

  read-only       the engine never writes the allocation during inference:
                  a cache flip there lasts until its line is evicted
                  (removal = re-XOR to the pre-cache value)
  engine-written  the engine (or the runner's per-image staging) rewrites
                  it: a cache flip lasts at most the image it starts in
                  (removal = conditional restore)

Rule (fail-closed):
  - bindings (semantic label TENSOR:*) are ENGINE-WRITTEN by definition:
    the input binding is re-staged by the runner for every image, the
    output bindings are rewritten by every enqueue;
  - a TRT-internal allocation is READ-ONLY iff every restore_check it
    received across the workload's VERIFIED campaign runs is `exact`
    (the G5/G7 runner compares the whole allocation byte-exactly against
    its pre-trial snapshot after each trial); any `mismatch:N` makes it
    ENGINE-WRITTEN;
  - a TRT-internal allocation of the injection surface with no evidence
    is refused (no default class).

Evidence: artifacts/g7/campaign/run_L*/summary.json (status
G5_CAMPAIGN_VERIFIED) + g1_5_g5_site_result.csv. Output: one
`"cache_alloc_classes"` literal per workload for fault_model.WORKLOADS,
plus the per-allocation evidence counts.

    derive_alloc_classes.py [--root artifacts/g7/campaign] [--json OUT]
    derive_alloc_classes.py --self-test
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import tempfile
from pathlib import Path

VERIFIED = "G5_CAMPAIGN_VERIFIED"


class ClassError(RuntimeError):
    pass


def collect(root: Path) -> dict:
    """workload -> {allocation_id: Counter(exact|mismatch|skipped)}, plus
    the run list and each allocation's semantic label."""
    evidence = collections.defaultdict(lambda: collections.defaultdict(
        collections.Counter))
    labels = collections.defaultdict(dict)
    runs = collections.defaultdict(list)
    for run_dir in sorted(root.glob("run_L*")):
        try:
            summary = json.loads((run_dir / "summary.json").read_text())
        except (OSError, ValueError):
            continue
        if summary.get("status") != VERIFIED:
            continue
        workload = summary.get("workload", "")
        runs[workload].append(run_dir.name)
        with (run_dir / "g1_5_g5_site_result.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                kind = row["restore_check"].split(":")[0]
                evidence[workload][row["allocation_id"]][kind] += 1
                labels[workload][row["allocation_id"]] = row["semantic_label"]
    return {"evidence": evidence, "labels": labels, "runs": runs}


def classify(evidence: dict, labels: dict) -> dict:
    read_only, engine_written = [], []
    for alloc_id in sorted(evidence):
        counts = evidence[alloc_id]
        if labels.get(alloc_id, "").startswith("TENSOR:"):
            engine_written.append(alloc_id)
        elif counts.get("mismatch", 0) > 0 or counts.get("skipped", 0) > 0:
            engine_written.append(alloc_id)
        elif counts.get("exact", 0) > 0:
            read_only.append(alloc_id)
        else:
            raise ClassError(f"{alloc_id}: no usable restore evidence")
    return {"read_only": tuple(read_only),
            "engine_written": tuple(engine_written)}


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def run(name, status, rows):
            d = root / name
            d.mkdir()
            (d / "summary.json").write_text(json.dumps(
                {"status": status, "workload": "w"}))
            with (d / "g1_5_g5_site_result.csv").open("w", newline="") as h:
                w = csv.writer(h)
                w.writerow(["allocation_id", "semantic_label", "restore_check"])
                w.writerows(rows)

        run("run_L1_a", VERIFIED, [
            ["trt-internal-0", "TENSORRT_INTERNAL_UNKNOWN", "exact"],
            ["trt-internal-3", "TENSORRT_INTERNAL_UNKNOWN", "mismatch:12"],
            ["trt-binding-data-gpu-0", "TENSOR:data", "exact"]])
        run("run_L2_b", VERIFIED, [
            ["trt-internal-0", "TENSORRT_INTERNAL_UNKNOWN", "exact"],
            ["trt-internal-1", "TENSORRT_INTERNAL_UNKNOWN", "exact"],
            ["trt-internal-1", "TENSORRT_INTERNAL_UNKNOWN", "mismatch:1"]])
        # a non-verified run is ignored even if it contradicts
        run("run_L3_c", "G5_CAMPAIGN_FAILED", [
            ["trt-internal-0", "TENSORRT_INTERNAL_UNKNOWN", "mismatch:9"]])
        data = collect(root)
        assert data["runs"]["w"] == ["run_L1_a", "run_L2_b"]
        cls = classify(data["evidence"]["w"], data["labels"]["w"])
        assert cls["read_only"] == ("trt-internal-0",), cls
        # binding -> engine-written by definition despite `exact`; one
        # mismatch makes an internal allocation engine-written
        assert cls["engine_written"] == ("trt-binding-data-gpu-0",
                                         "trt-internal-1",
                                         "trt-internal-3"), cls
    print("derive_alloc_classes self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("artifacts/g7/campaign"))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    data = collect(args.root)
    out = {}
    for workload in sorted(data["evidence"]):
        if not workload.startswith("g7v2_"):
            continue
        cls = classify(data["evidence"][workload], data["labels"][workload])
        out[workload] = {
            "classes": {k: list(v) for k, v in cls.items()},
            "runs": data["runs"][workload],
            "evidence": {a: dict(c) for a, c in
                         sorted(data["evidence"][workload].items())},
        }
        print(f"{workload} ({len(data['runs'][workload])} verified runs)")
        print(f"    \"cache_alloc_classes\": {{\"read_only\": "
              f"{cls['read_only']!r},")
        print(f"                            \"engine_written\": "
              f"{cls['engine_written']!r}}},")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
