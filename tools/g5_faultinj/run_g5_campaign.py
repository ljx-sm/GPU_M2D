#!/usr/bin/env python3
"""GPU_M2D G5: dual-addressing fault-injection campaign (one BER level).

One campaign = one runner process = one bootstrap. The flow extends the
G4-T2 gated skeleton (observer attaches pre-context; gate-time registry;
ONLINE per-run snapshot; byte-exact restore; strict closing ledger) with
the G5 trial loop of docs/G5_FAULT_MODEL.md §6:

  1. the eBPF observer attaches before any CUDA context exists (the
     runner blocks at the pre-allocation gate; in campaign mode it has
     already preprocessed every evaluation image on the host -- CPU-only
     work, the gated window stays CUDA-only);
  2. the runner allocates the full G1.5 workload, runs the CLEAN pass
     over every evaluation image (strict; records the per-image clean
     output), writes its gate-time allocation registry and blocks at the
     campaign gate;
  3. the orchestrator builds the per-allocation PTE ledger, the gate
     VA-PA map and the dual-addressing snapshot (build_snapshot.run_build
     -- the same fail-closed join G4-T1/T2 validated);
  4. the frozen fault model (fault_model.py) samples every trial FROM
     THAT SNAPSHOT: resident-bytes total must equal the frozen R
     (26,428,428) or the campaign refuses fail-closed (the level table
     is derived from R); composition (s, d, t) is frozen per level, only
     positions randomize; every trial's sites are re-verified against
     the frozen table and for (PA byte, bit) distinctness;
  5. the work file is written and the release gate opens; per trial the
     runner flips ALL sites (each verified: after == before ^ mask,
     allocation-level guard compare, reverse chain), runs the full
     evaluation pass with the faults held in place, classifies every
     image against the clean records (DUE > SDC_TOP1 > SDC_NUMERIC >
     BENIGN, first invalid aborts the trial's remaining images),
     restores every site, re-proves the input binding byte-exact against
     the last staged image, and a strict sanity inference must reproduce
     the clean output;
  6. after teardown the strict ledger re-runs, the gate and final maps
     and registries must be identical (mid-run remap detector), the
     trial event stream must match the work file exactly, and every
     result CSV row is re-verified independently of the runner.

Restore-verification policy the orchestrator enforces per allocation
class: input binding -- restore_check must be "exact" (fail-closed);
output bindings -- "skipped:engine-owned-output" (the engine rewrites
the cell every enqueue; the flip itself was proven by the pre-pass
guard compare); TRT-internal -- "exact" or informational "mismatch:N"
(engine scratch churn; the sanity inference is the behavioral no-residue
proof).

Fail-closed: any ledger failure, snapshot problem, residency mismatch
with the frozen R, composition/distinctness violation, result
disagreement, remap, lost BPF event, lifecycle mismatch, or missing PASS
marker fails the campaign (exit 2) and voids only this level.
Co-tenant processes are never touched and never refuse the run.

Must be started through ``sudo`` from the research account (root only
attaches the probes; the CUDA child is dropped back to the invoking
user). Exit codes: 0 ok, 2 fail-closed, 1 error.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "tools/g2_observer"))
sys.path.insert(0, str(PROJECT / "tools/g3_probe"))
sys.path.insert(0, str(PROJECT / "tools/g4_dualaddr"))

from g2_observer import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_OPEN_SOURCE_REPO,
    G2Observer,
    load_and_validate_contract,
    sha256,
)
from run_g2_tensorrt_probe import (  # noqa: E402
    chown_outputs,
    child_preexec,
    drop_to_invoking_user,
    normalize_gpu_uuid,
    parse_harness_output,
    read_until_marker,
)
from run_g3_pool_probe import (  # noqa: E402
    drain_pipe,
    drain_until_marker,
)
from run_g4_t2_injection import (  # noqa: E402
    ALLOC_FIELDS,
    build_final_ledger,
    build_preledger,
    cotenancy_snapshot,
    diff_map_rows,
    map_rows_from_segments,
    query_device_uuid,
    read_csv_rows,
    write_map_csv,
)
from build_snapshot import run_build  # noqa: E402
import fault_model  # noqa: E402

G5_SCHEMA = "gpu-m2d.g5.campaign.v1"
PASS_MARKER = "GPU_M2D_G5_CAMPAIGN_PASS"
GATE_MARKER = b"GPU_M2D_EVENT,event=CAMPAIGN_GATE_WAIT,"

CAMPAIGN_SKELETON = [
    "PROCESS_READY",
    "WAIT_PRE_ALLOC_GATE",
    "PRE_ALLOC_GATE_OPEN",
    "CONTEXT_BEGIN",
    "CONTEXT_READY",
    "RUNTIME_BEGIN",
    "BINDINGS_READY",
    "CLEAN_PASS_BEGIN",
    "CLEAN_PASS_END",
    "ALLOCATION_REGISTRY_GATE_WRITTEN",
    "CAMPAIGN_GATE_WAIT",
    "CAMPAIGN_WORK_BEGIN",
    "CAMPAIGN_WORK_END",
    "SNAPSHOT_READY",
    "HOLD_BEGIN",
    "HOLD_END",
    "TEARDOWN_BEGIN",
    "PROCESS_END",
]
CAMPAIGN_VARIABLE_EVENTS = {"ALLOCATED", "FREE", "TRIAL_BEGIN",
                            "SITE_FLIPPED", "TRIAL_INJECTED_END",
                            "SITE_RESTORED", "TRIAL_SANITY_BEGIN",
                            "TRIAL_SANITY_END", "TRIAL_END"}
TRIAL_EVENT_NAMES = {"TRIAL_BEGIN", "SITE_FLIPPED", "TRIAL_INJECTED_END",
                     "SITE_RESTORED", "TRIAL_SANITY_BEGIN",
                     "TRIAL_SANITY_END", "TRIAL_END"}

OUTCOMES = ("BENIGN", "SDC_TOP1", "SDC_NUMERIC", "DUE_INVALID_OUTPUT")
IMAGE_OUTCOMES = ("IMAGE_BENIGN", "IMAGE_SDC_NUMERIC", "IMAGE_SDC_TOP1",
                  "IMAGE_DUE")
PROBABILITY_TOLERANCE = 1.0e-6

SITE_RESULT_FIELDS = [
    "run_id", "device", "trial_index", "event_index", "site_index",
    "target_id", "allocation_id", "semantic_label", "byte_offset",
    "bit_in_byte", "xor_mask", "gpu_va", "expected_gpu_va", "before",
    "after", "guard_bytes_unchanged", "reverse_map_ok", "restored_byte_ok",
    "restore_check",
]
TRIAL_RESULT_FIELDS = [
    "run_id", "device", "trial_index", "site_count", "event_count",
    "images_total", "images_evaluated", "images_benign",
    "images_sdc_numeric", "images_sdc_top1", "images_invalid",
    "injected_outcome", "sanity_class", "sanity_probability",
    "sanity_matches_clean", "restore_alloc_exact", "restore_alloc_mismatch",
    "restore_alloc_skipped", "restore_mismatch_bytes",
]
IMAGE_DETAIL_FIELDS = [
    "run_id", "device", "trial_index", "image_index", "evaluated",
    "clean_class", "injected_class", "clean_probability",
    "injected_probability", "outcome",
]
CLEAN_PASS_FIELDS = ["image_index", "path", "label", "clean_class",
                     "clean_probability"]


# --------------------------------------------------------------------------
# pure helpers (offline-testable)
# --------------------------------------------------------------------------

def verify_work_model(level: dict, campaign: list[list[dict]]) -> list[str]:
    """The frozen level echo: every trial realizes exactly (s, d, t) events
    of multiplicities 1/2/3, B sites total, and all sites of a trial are
    distinct at (PA page, in-page offset, bit) -- double XOR on one cell
    would cancel."""
    failures: list[str] = []
    expected = (level["s"], level["d"], level["t"])
    for sites in campaign:
        trial = sites[0]["trial_index"]
        sizes: dict[int, int] = {}
        for site in sites:
            sizes[site["event_index"]] = \
                sizes.get(site["event_index"], 0) + 1
        histogram = (sum(1 for v in sizes.values() if v == 1),
                     sum(1 for v in sizes.values() if v == 2),
                     sum(1 for v in sizes.values() if v == 3))
        if histogram != expected:
            failures.append(f"trial {trial}: multiplicity histogram "
                            f"{histogram} != frozen {expected}")
        if len(sites) != level["bits"]:
            failures.append(f"trial {trial}: {len(sites)} sites != frozen "
                            f"B={level['bits']}")
        if any(v not in (1, 2, 3) for v in sizes.values()):
            failures.append(f"trial {trial}: event multiplicity outside 1..3")
        used: set[tuple[str, str, int]] = set()
        for site in sites:
            chain = site["chain"]
            key = (chain["fb_pa_page_base"],
                   chain["pa_in_page_offset"], site["bit"])
            if key in used:
                failures.append(f"trial {trial}: duplicate (PA byte, bit) "
                                f"{key}")
            used.add(key)
    return failures


def verify_site_rows(campaign: list[list[dict]], rows: list[dict],
                     registry: list[dict]) -> list[str]:
    """Independent re-verification of the runner's per-site result rows,
    including the per-allocation-class restore_check policy."""
    failures: list[str] = []
    flat = [site for sites in campaign for site in sites]
    if len(rows) != len(flat):
        return [f"site result rows {len(rows)} != work sites {len(flat)}"]
    semantic = {row["allocation_id"]: row["semantic_label"]
                for row in registry}
    seen: set[str] = set()
    for site, row in zip(flat, rows):
        where = site["target_id"]
        if row["target_id"] != where or where in seen:
            failures.append(f"{where}: target_id mismatch or duplicate")
            continue
        seen.add(where)
        checks = (
            (row["trial_index"] == str(site["trial_index"]), "trial_index"),
            (row["event_index"] == str(site["event_index"]), "event_index"),
            (row["site_index"] == str(site["site_index"]), "site_index"),
            (row["allocation_id"] == site["allocation_id"],
             "allocation_id"),
            (row["byte_offset"] == str(site["byte_offset"]), "byte_offset"),
            (row["bit_in_byte"] == str(site["bit"]), "bit_in_byte"),
            (int(row["gpu_va"], 16) == site["expected_gpu_va"], "gpu_va"),
            (int(row["expected_gpu_va"], 16) == site["expected_gpu_va"],
             "expected_gpu_va"),
            (int(row["xor_mask"]) == (1 << site["bit"]), "xor_mask"),
            (int(row["after"]) ==
             int(row["before"]) ^ int(row["xor_mask"]), "after==before^mask"),
            (row["guard_bytes_unchanged"] == "1", "guard_bytes_unchanged"),
            (row["reverse_map_ok"] == "1", "reverse_map_ok"),
        )
        failures.extend(f"{where}: {name} failed"
                        for ok, name in checks if not ok)
        label = semantic.get(row["allocation_id"], "")
        if label == "TENSOR:data":
            if row["restore_check"] != "exact":
                failures.append(f"{where}: input binding restore_check "
                                f"{row['restore_check']!r} != exact")
        elif label in ("TENSOR:prob", "TENSOR:index"):
            if row["restore_check"] != "skipped:engine-owned-output":
                failures.append(f"{where}: output binding restore_check "
                                f"{row['restore_check']!r}")
        elif label == "TENSORRT_INTERNAL_UNKNOWN":
            if row["restore_check"] != "exact" and \
                    not row["restore_check"].startswith("mismatch:"):
                failures.append(f"{where}: TRT-internal restore_check "
                                f"{row['restore_check']!r}")
        else:
            failures.append(f"{where}: unknown allocation semantic "
                            f"{label!r}")
    return failures


def verify_trial_rows(rows: list[dict], campaign: list[list[dict]],
                      images_total: int) -> list[str]:
    failures: list[str] = []
    if len(rows) != len(campaign):
        return [f"trial rows {len(rows)} != trials {len(campaign)}"]
    for trial_index, (sites, row) in enumerate(zip(campaign, rows)):
        where = f"trial {trial_index}"
        if row["trial_index"] != str(trial_index):
            failures.append(f"{where}: trial_index mismatch")
            continue
        if int(row["site_count"]) != len(sites):
            failures.append(f"{where}: site_count {row['site_count']} != "
                            f"{len(sites)}")
        events = {site["event_index"] for site in sites}
        if int(row["event_count"]) != len(events):
            failures.append(f"{where}: event_count mismatch")
        if int(row["images_total"]) != images_total:
            failures.append(f"{where}: images_total "
                            f"{row['images_total']} != {images_total}")
        counter_sum = sum(int(row[name]) for name in
                          ("images_benign", "images_sdc_numeric",
                           "images_sdc_top1", "images_invalid"))
        if counter_sum != images_total:
            failures.append(f"{where}: image counters sum {counter_sum} != "
                            f"{images_total}")
        invalid = int(row["images_invalid"])
        top1 = int(row["images_sdc_top1"])
        numeric = int(row["images_sdc_numeric"])
        if invalid:
            expected_outcome = "DUE_INVALID_OUTPUT"
        elif top1:
            expected_outcome = "SDC_TOP1"
        elif numeric:
            expected_outcome = "SDC_NUMERIC"
        else:
            expected_outcome = "BENIGN"
        if row["injected_outcome"] not in OUTCOMES:
            failures.append(f"{where}: unknown outcome "
                            f"{row['injected_outcome']!r}")
        elif row["injected_outcome"] != expected_outcome:
            failures.append(f"{where}: outcome {row['injected_outcome']} "
                            f"violates precedence {expected_outcome}")
        if row["sanity_matches_clean"] != "1":
            failures.append(f"{where}: sanity_matches_clean not set")
    return failures


def verify_image_rows(rows: list[dict], trial_rows: list[dict],
                      clean_rows: list[dict]) -> list[str]:
    """Per-image re-verification: block shape (the DUE abort marks the
    remainder evaluated=0), classification re-derived from the logged
    numbers, clean classes joined against the clean pass CSV, counts tied
    back to the trial rows."""
    failures: list[str] = []
    trials = len(trial_rows)
    images = len(clean_rows)
    if len(rows) != trials * images:
        return [f"image rows {len(rows)} != {trials}*{images}"]
    clean_class = {row["image_index"]: row["clean_class"]
                   for row in clean_rows}
    if len(clean_class) != images:
        return ["clean pass CSV image_index set mismatch"]
    for trial in range(trials):
        block = rows[trial * images:(trial + 1) * images]
        for expected_index, row in enumerate(block):
            if row["trial_index"] != str(trial) or \
                    row["image_index"] != str(expected_index):
                failures.append(f"trial {trial}: image rows out of order")
                break
        counts = dict.fromkeys(IMAGE_OUTCOMES, 0)
        abort_seen = False
        for row in block:
            outcome = row["outcome"]
            if outcome not in counts:
                failures.append(f"trial {trial}: unknown image outcome "
                                f"{outcome!r}")
                break
            counts[outcome] += 1
            if row["clean_class"] != clean_class[row["image_index"]]:
                failures.append(f"trial {trial} image {row['image_index']}: "
                                "clean_class disagrees with the clean pass")
            if outcome == "IMAGE_DUE" and abort_seen:
                if row["evaluated"] != "0":
                    failures.append(f"trial {trial} image "
                                    f"{row['image_index']}: evaluated row "
                                    "after the DUE abort")
                continue
            if outcome == "IMAGE_DUE":
                abort_seen = True
                if row["evaluated"] != "1" or row["injected_class"] != "NA" \
                        or row["injected_probability"] != "NA":
                    failures.append(f"trial {trial} image "
                                    f"{row['image_index']}: DUE row shape")
                continue
            if row["evaluated"] != "1" or row["injected_class"] == "NA":
                failures.append(f"trial {trial} image {row['image_index']}: "
                                "evaluated row shape")
                continue
            top1_changed = row["injected_class"] != row["clean_class"]
            delta = abs(float(row["injected_probability"]) -
                        float(row["clean_probability"]))
            numeric_changed = top1_changed or delta > PROBABILITY_TOLERANCE
            expected = ("IMAGE_SDC_TOP1" if top1_changed else
                        "IMAGE_SDC_NUMERIC" if numeric_changed
                        else "IMAGE_BENIGN")
            if outcome != expected:
                failures.append(f"trial {trial} image {row['image_index']}: "
                                f"outcome {outcome} != re-derived {expected}")
        trial_row = trial_rows[trial]
        if (counts["IMAGE_BENIGN"] != int(trial_row["images_benign"]) or
                counts["IMAGE_SDC_NUMERIC"] !=
                int(trial_row["images_sdc_numeric"]) or
                counts["IMAGE_SDC_TOP1"] != int(trial_row["images_sdc_top1"])
                or counts["IMAGE_DUE"] != int(trial_row["images_invalid"])):
            failures.append(f"trial {trial}: image detail counts disagree "
                            "with the trial CSV")
    return failures


EVENT_LINE_PREFIX = "GPU_M2D_EVENT,event="


def parse_all_events(output: str) -> list[tuple[str, dict[str, str]]]:
    events: list[tuple[str, dict[str, str]]] = []
    for line in output.splitlines():
        if not line.startswith(EVENT_LINE_PREFIX):
            continue
        name, _, tail = line[len(EVENT_LINE_PREFIX):].partition(",")
        fields: dict[str, str] = {}
        for item in tail.split(","):
            if "=" in item:
                key, _, value = item.partition("=")
                fields[key] = value
        events.append((name, fields))
    return events


def expected_trial_sequence(campaign: list[list[dict]]
                            ) -> list[tuple[str, dict[str, str]]]:
    sequence: list[tuple[str, dict[str, str]]] = []
    for sites in campaign:
        trial = str(sites[0]["trial_index"])
        sequence.append(("TRIAL_BEGIN", {"trial_index": trial}))
        for site in sites:
            sequence.append(("SITE_FLIPPED", {"target_id":
                                              site["target_id"]}))
        sequence.append(("TRIAL_INJECTED_END", {"trial_index": trial}))
        # the runner restores in REVERSE flip order (G4-T2 convention)
        for site in reversed(sites):
            sequence.append(("SITE_RESTORED", {"target_id":
                                               site["target_id"]}))
        sequence.append(("TRIAL_SANITY_BEGIN", {"trial_index": trial}))
        sequence.append(("TRIAL_SANITY_END", {"trial_index": trial}))
        sequence.append(("TRIAL_END", {"trial_index": trial}))
    return sequence


def verify_event_structure(output: str, campaign: list[list[dict]],
                           site_rows: list[dict]) -> list[str]:
    """The observed trial event stream must be exactly the sequence the
    work file prescribes (flip order, restore order, per-trial bracket),
    and every SITE_FLIPPED must agree with the site result CSV."""
    failures: list[str] = []
    observed = [(name, fields) for name, fields in
                parse_all_events(output) if name in TRIAL_EVENT_NAMES]
    expected = expected_trial_sequence(campaign)
    if len(observed) != len(expected):
        failures.append(f"trial event count {len(observed)} != expected "
                        f"{len(expected)}")
        return failures
    site_by_id = {row["target_id"]: row for row in site_rows}
    for (name_o, fields_o), (name_e, fields_e) in zip(observed, expected):
        if name_o != name_e:
            failures.append(f"trial event order: got {name_o}, expected "
                            f"{name_e}")
            break
        if "target_id" in fields_e and \
                fields_o.get("target_id") != fields_e["target_id"]:
            failures.append(f"{name_o}: target_id {fields_o.get('target_id')}"
                            f" != {fields_e['target_id']}")
        if "trial_index" in fields_e and \
                fields_o.get("trial_index") != fields_e["trial_index"]:
            failures.append(f"{name_o}: trial_index "
                            f"{fields_o.get('trial_index')} != "
                            f"{fields_e['trial_index']}")
        if name_o == "SITE_FLIPPED":
            row = site_by_id.get(fields_o.get("target_id", ""))
            if row is not None and (
                    fields_o.get("before") != row["before"] or
                    fields_o.get("after") != row["after"] or
                    fields_o.get("xor_mask") != row["xor_mask"] or
                    fields_o.get("gpu_va", "").lower() !=
                    row["gpu_va"].lower()):
                failures.append(f"SITE_FLIPPED {fields_o.get('target_id')}: "
                                "disagrees with the site result CSV")
    return failures


def resident_bytes_of(snapshot_rows: list[dict]) -> int:
    total = 0
    for row in snapshot_rows:
        total += int(str(row["byte_end_in_page"]), 16) - \
            int(str(row["byte_start_in_page"]), 16)
    return total


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    remu_root = Path(os.environ.get("GPU_M2D_REMU_ROOT", "/data1/luojx/REMU"))
    dataset_root = Path(os.environ.get(
        "GPU_M2D_DATASET_ROOT", "/data1/luojx/datasets/REMU_stage8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runner", type=Path,
                        default=PROJECT / "build-g1.5/gpu_m2d_resnet50_int8_g1_5")
    parser.add_argument("--engine", type=Path,
                        default=remu_root / "artifacts/stage8/engines/paper_priority/resnet50_resisc45_int8_ptq.engine")
    parser.add_argument("--sample-csv", type=Path,
                        default=dataset_root / "RESISC45/splits/original_repo_1000_eval.csv")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="sanity image index inside the sample CSV")
    parser.add_argument("--table", type=Path,
                        default=PROJECT / "artifacts/g3/table_v4")
    parser.add_argument("--level",
                        choices=[entry["level"] for entry in fault_model.LEVELS],
                        help="frozen BER level (docs/G5_FAULT_MODEL.md §5); "
                             "required unless --self-test")
    parser.add_argument("--trials", type=int, default=100,
                        help="trials of this campaign (default 100)")
    parser.add_argument("--seed", type=int, default=7,
                        help="campaign RNG seed; trial seeds derive as "
                             "seed*100003 + trial_index")
    parser.add_argument("--hold-seconds", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0,
                        help="wall-clock budget for the runner (the full "
                             "campaign runs inside it)")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "artifacts/g5/campaign")
    return parser.parse_args()


def finalize(failures: list[str], run_dir: Path, summary_output: Path,
             test_log: Path, event_output: Path, observer: G2Observer,
             args: argparse.Namespace, run_id: str, device_uuid: str,
             started_ns: int, extra: dict[str, Any], uid: int, gid: int,
             ) -> int:
    status = "FAIL_CLOSED" if failures else "G5_CAMPAIGN_VERIFIED"
    summary = {
        "schema_version": G5_SCHEMA,
        "status": status,
        "meaning": "every trial's sites were XOR-flipped on device through "
                   "PA pages observed live this run and joined to the G3 "
                   "table, sampled by the frozen G5 fault model; faults "
                   "were held for a full evaluation pass, every image "
                   "classified against the clean pass, all sites restored, "
                   "and the post-restore sanity inference reproduced the "
                   "clean output",
        "failures": failures,
        "run_id": run_id,
        "device": args.device,
        "device_uuid": device_uuid,
        "target_tgid": observer.target_tgid,
        "level": args.level,
        "trials_requested": args.trials,
        "seed": args.seed,
        "table_dir": str(args.table),
        "engine_path": str(args.engine),
        "engine_sha256": sha256(args.engine) if args.engine.is_file() else "",
        "runner_path": str(args.runner),
        "runner_sha256": sha256(args.runner) if args.runner.is_file() else "",
        "event_count": len(observer.rows),
        "lost_event_count": observer.lost_event_count,
        "event_output": str(event_output),
        "event_output_sha256": sha256(event_output),
        "test_log": str(test_log),
        "test_log_sha256": sha256(test_log),
        "started_wall_time_ns": started_ns,
        "ended_wall_time_ns": time.time_ns(),
        "driver_modified": False,
        "uvm_state_modified": False,
        "pte_modified": False,
        "other_gpu_processes_stopped": False,
    }
    summary.update(extra)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True)
                              + "\n", encoding="utf-8")
    outputs = [event_output, test_log, summary_output]
    for name in ("gpu_va_pa_map.csv", "gpu_va_pa_map_gate.csv",
                 "work.csv", "work_detail.json",
                 "g1_5_g5_site_result.csv", "g1_5_g5_trial_result.csv",
                 "g1_5_g5_image_detail.csv", "g1_5_g5_clean_pass.csv",
                 "g1_5_allocations.csv", "g1_5_allocations_gate.csv"):
        candidate = run_dir / name
        if candidate.is_file():
            outputs.append(candidate)
    snapshot_dir = run_dir / "snapshot"
    if snapshot_dir.is_dir():
        for name in ("snapshot_pages.csv", "manifest.json"):
            candidate = snapshot_dir / name
            if candidate.is_file():
                outputs.append(candidate)
    chown_outputs(outputs + [run_dir], uid, gid)
    print(f"device={args.device} level={args.level} status={status} "
          f"run={run_id}")
    print(f"summary={summary_output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 2 if failures else 0


def run_once(args: argparse.Namespace) -> int:
    uid, gid = drop_to_invoking_user()
    contract, contract_hash = load_and_validate_contract(
        args.contract.resolve(), DEFAULT_OPEN_SOURCE_REPO)
    if not args.runner.is_file() or not os.access(args.runner, os.X_OK):
        raise RuntimeError(f"runner missing or not executable: {args.runner}")
    if args.trials < 1:
        raise RuntimeError("--trials must be >= 1")
    fault_model.assert_frozen_levels()
    level = fault_model.level_by_name(args.level)
    trt_runtime_dir = Path(os.environ.get(
        "GPU_M2D_REMU_ROOT", "/data1/luojx/REMU")) / \
        ".local/deps/tensorrt-8.6.1/tensorrt_libs"
    opencv_lib_dir = Path(os.environ.get(
        "GPU_M2D_REMU_ROOT", "/data1/luojx/REMU")) / ".local/deps/conda/lib"
    for required in (args.engine, args.sample_csv,
                     trt_runtime_dir / "libnvinfer.so.8",
                     args.table / "gddr_seed_table.csv",
                     args.table / "bank_classes.csv",
                     args.table / "page_anchors.csv"):
        if not required.is_file():
            raise RuntimeError(f"missing required asset: {required}")

    cotenancy = cotenancy_snapshot(args.device)
    device_uuid = query_device_uuid(args.device)
    run_dir = args.output_root / \
        f"run_{args.level}_gpu{args.device}_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chown(run_dir, uid, gid)
    event_output = run_dir / "events.csv"
    summary_output = run_dir / "summary.json"
    test_log = run_dir / "harness.log"
    registry_output = run_dir / "g1_5_allocations.csv"
    registry_gate_copy = run_dir / "g1_5_allocations_gate.csv"
    map_output = run_dir / "gpu_va_pa_map.csv"
    map_gate_output = run_dir / "gpu_va_pa_map_gate.csv"
    work_output = run_dir / "work.csv"
    work_detail_output = run_dir / "work_detail.json"
    site_result_output = run_dir / "g1_5_g5_site_result.csv"
    trial_result_output = run_dir / "g1_5_g5_trial_result.csv"
    image_detail_output = run_dir / "g1_5_g5_image_detail.csv"
    clean_pass_output = run_dir / "g1_5_g5_clean_pass.csv"
    gate = run_dir / f".gate_{os.getpid()}_{time.time_ns()}"
    release = run_dir / f".release_{os.getpid()}_{time.time_ns()}"
    run_id = run_dir.name

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(trt_runtime_dir), str(opencv_lib_dir),
                          os.environ.get("LD_LIBRARY_PATH", "")) if part)

    observer = G2Observer(contract, contract_hash, 0, event_output)
    process: subprocess.Popen[bytes] | None = None
    started_ns = time.time_ns()
    captured = bytearray()
    failures: list[str] = []
    extra: dict[str, Any] = {"cotenancy_at_start": cotenancy}
    try:
        process = subprocess.Popen(
            [
                str(args.runner.resolve()),
                "--engine", str(args.engine.resolve()),
                "--sample-csv", str(args.sample_csv.resolve()),
                "--sample-index", str(args.sample_index),
                "--device", str(args.device),
                "--output-prefix", str((run_dir / "g1_5").resolve()),
                "--observer-gate", str(gate),
                "--hold-seconds", str(args.hold_seconds),
                "--gate-timeout-seconds", str(max(5, int(args.timeout_seconds))),
                "--campaign-work", str(work_output.resolve()),
                "--campaign-release", str(release.resolve()),
                "--campaign-gate-timeout-seconds",
                str(max(600, int(args.timeout_seconds))),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=environment,
            preexec_fn=child_preexec(uid, gid),
        )
        observer.set_target_tgid(process.pid)
        # In campaign mode the runner preprocesses every evaluation image
        # BEFORE this marker (CPU-only), so the wait budget is minutes.
        prelude, gate_ready = read_until_marker(
            process, b"GPU_M2D_EVENT,event=WAIT_PRE_ALLOC_GATE,",
            min(args.timeout_seconds, 600.0))
        captured.extend(prelude)
        if not gate_ready:
            process.terminate()
            raise RuntimeError("runner did not reach the pre-allocation gate")
        gate.write_text(f"observer_ready target_tgid={process.pid}\n",
                        encoding="utf-8")
        os.chmod(gate, 0o644)

        reached = drain_until_marker(process, observer, GATE_MARKER,
                                     captured, args.timeout_seconds)
        if not reached:
            raise RuntimeError("runner did not reach the campaign gate")
        for _ in range(20):
            observer.poll(50)

        output_so_far = captured.decode("utf-8", errors="replace")
        allocated, _, _ = parse_harness_output(output_so_far)
        if not allocated:
            raise RuntimeError("no ALLOCATED events before the campaign gate")
        if not registry_output.is_file():
            raise RuntimeError("runner did not write the gate-time registry")
        registry = read_csv_rows(registry_output, ALLOC_FIELDS)
        registry_gate_copy.write_text(registry_output.read_text(),
                                      encoding="utf-8")
        allocated_by_id = {r["allocation_id"]: r for r in allocated}
        if len(allocated_by_id) != len(allocated):
            failures.append("duplicate ALLOCATED records at the gate")
        for row in registry:
            record = allocated_by_id.get(row["allocation_id"])
            if record is None:
                failures.append(f"{row['allocation_id']}: no ALLOCATED event")
                continue
            if record.get("base_va", "").lower() != row["gpu_va"].lower() or \
                    record.get("size_bytes") != row["size_bytes"]:
                failures.append(f"{row['allocation_id']}: ALLOCATED event "
                                "disagrees with the gate registry")
        if failures:
            raise RuntimeError("gate registry/ALLOCATED disagreement: "
                               + "; ".join(failures))

        g1_5_run_id = registry[0]["run_id"] if registry else ""
        gate_segments, gate_failures = build_preledger(observer.rows,
                                                       allocated)
        if gate_failures or not gate_segments:
            raise RuntimeError("campaign-gate ledger failed: "
                               + "; ".join(gate_failures))
        gate_map_rows = map_rows_from_segments(gate_segments, run_id,
                                               g1_5_run_id, args.device,
                                               registry)
        write_map_csv(map_gate_output, gate_map_rows)
        os.chmod(map_gate_output, 0o644)

        snapshot_dir = run_dir / "snapshot"
        result = run_build(registry_output, map_gate_output, args.device,
                           args.table, snapshot_dir,
                           lambda text="": print(text))
        if result.problems:
            raise RuntimeError("snapshot refused (fail-closed): "
                               + "; ".join(result.problems[:10]))
        print(f"snapshot: {len(result.rows)} rows, checksum "
              f"{result.manifest['snapshot_sha256'][:16]}...")

        # The frozen level table is derived from R: the live snapshot's
        # resident-byte total must equal the nominal R or the campaign
        # refuses (fail-closed) -- the table must be re-derived instead.
        resident_bytes = resident_bytes_of(result.rows)
        if resident_bytes != fault_model.RESIDENT_BYTES_NOMINAL:
            raise RuntimeError(
                f"snapshot residency {resident_bytes} bytes != frozen "
                f"R {fault_model.RESIDENT_BYTES_NOMINAL} -- the level "
                "table must be re-derived (docs/G5_FAULT_MODEL.md §5)")

        anchors = fault_model.load_anchors(args.table)
        anchor_pages = len(anchors)
        print(f"anchors: {anchor_pages} pages with valid consensus masks")
        campaign = fault_model.sample_campaign(
            args.level, args.trials, result.rows, anchors, args.seed,
            resident_bytes_expected=fault_model.RESIDENT_BYTES_NOMINAL)
        model_failures = verify_work_model(level, campaign)
        if model_failures:
            raise RuntimeError("sampled campaign violates the frozen model: "
                               + "; ".join(model_failures[:10]))
        total_sites = sum(len(sites) for sites in campaign)
        print(f"sampler: level {args.level} BER {level['ber']:g} "
              f"B={level['bits']} (s,d,t)=({level['s']},{level['d']},"
              f"{level['t']}), {args.trials} trials, {total_sites} sites")
        fault_model.write_work_csv(work_output, campaign)
        work_detail_output.write_text(json.dumps({
            "schema": "gpu-m2d.g5.campaign.work.v1",
            "run_id": run_id,
            "device": args.device,
            "level": level,
            "seed": args.seed,
            "trial_seed_rule": "random.Random(seed * 100003 + trial_index)",
            "trials": args.trials,
            "resident_bytes": resident_bytes,
            "anchor_pages": anchor_pages,
            "frozen_composition_echo": [level["s"], level["d"],
                                        level["t"]],
            "snapshot_manifest": result.manifest,
            "sites": [site for sites in campaign for site in sites],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(work_output, 0o644)
        os.chmod(work_detail_output, 0o644)

        release.write_text(f"campaign_ready target_tgid={process.pid}\n",
                           encoding="utf-8")
        os.chmod(release, 0o644)

        finished = drain_until_marker(process, observer,
                                      PASS_MARKER.encode(), captured,
                                      args.timeout_seconds)
        if not finished:
            if process.poll() is None:
                process.terminate()
            raise RuntimeError("runner did not report " + PASS_MARKER)
        exit_deadline = time.monotonic() + args.timeout_seconds
        while process.poll() is None and time.monotonic() < exit_deadline:
            observer.poll(50)
        if process.poll() is None:
            process.terminate()
            raise RuntimeError("runner exceeded timeout")
        for _ in range(20):
            observer.poll(50)
        drain_pipe(process, captured)
        output = captured.decode("utf-8", errors="replace")

        # ---------------- post-run verification ----------------
        if process.returncode != 0:
            failures.append(f"runner exit code {process.returncode}")
        allocated, lifecycle, times = parse_harness_output(output)
        skeleton = [name for name in lifecycle
                    if name not in CAMPAIGN_VARIABLE_EVENTS]
        expected_skeleton = [name for name in CAMPAIGN_SKELETON
                             if not (name in ("HOLD_BEGIN", "HOLD_END")
                                     and args.hold_seconds <= 0)]
        if skeleton != expected_skeleton:
            failures.append(f"lifecycle skeleton mismatch: {skeleton}")

        final_segments, final_failures = build_final_ledger(
            observer.rows, allocated, times.get("PROCESS_END", 0))
        failures.extend(final_failures)
        final_map_rows = map_rows_from_segments(final_segments, run_id,
                                                g1_5_run_id, args.device,
                                                registry)
        write_map_csv(map_output, final_map_rows)
        failures.extend(diff_map_rows(gate_map_rows, final_map_rows,
                                      "va-pa map"))

        # mappings alive across the whole campaign window
        gate_wait_ns = times.get("CAMPAIGN_GATE_WAIT", 0)
        work_begin_ns = times.get("CAMPAIGN_WORK_BEGIN", 0)
        work_end_ns = times.get("CAMPAIGN_WORK_END", 0)
        for segment in final_segments:
            name = str(segment["allocation_id"])
            if not (gate_wait_ns and work_end_ns
                    and int(segment["mapped_at_ns"]) < gate_wait_ns
                    and int(segment["unmapped_at_ns"] or 0) > work_end_ns):
                failures.append(f"{name}: mapping not alive across the "
                                "campaign window")

        # registry stability between gate and end (no realloc mid-run)
        if registry_output.read_text() != registry_gate_copy.read_text():
            failures.append("allocation registry changed between the "
                            "campaign gate and the end of the run")

        site_rows = read_csv_rows(site_result_output, SITE_RESULT_FIELDS)
        failures.extend(verify_site_rows(campaign, site_rows, registry))
        trial_rows = read_csv_rows(trial_result_output, TRIAL_RESULT_FIELDS)
        clean_rows = read_csv_rows(clean_pass_output, CLEAN_PASS_FIELDS)
        images_total = len(clean_rows)
        if images_total == 0:
            failures.append("clean pass CSV is empty")
        else:
            failures.extend(verify_trial_rows(trial_rows, campaign,
                                              images_total))
            image_rows = read_csv_rows(image_detail_output,
                                       IMAGE_DETAIL_FIELDS)
            failures.extend(verify_image_rows(image_rows, trial_rows,
                                              clean_rows))
        failures.extend(verify_event_structure(output, campaign, site_rows))

        outcome_histogram = dict.fromkeys(OUTCOMES, 0)
        image_totals = {"images_benign": 0, "images_sdc_numeric": 0,
                        "images_sdc_top1": 0, "images_invalid": 0}
        restore_totals = {"restore_alloc_exact": 0,
                          "restore_alloc_mismatch": 0,
                          "restore_alloc_skipped": 0,
                          "restore_mismatch_bytes": 0}
        for row in trial_rows:
            outcome_histogram[row["injected_outcome"]] = \
                outcome_histogram.get(row["injected_outcome"], 0) + 1
            for key in image_totals:
                image_totals[key] += int(row[key])
            for key in restore_totals:
                restore_totals[key] += int(row[key])
        if len(trial_rows) != args.trials:
            failures.append(f"trial count {len(trial_rows)} != requested "
                            f"{args.trials}")

        kernel_uuids = sorted({str(row["gpu_uuid"]) for row in observer.rows
                               if row["event_type"] == "PTE_HEADER"
                               and row["gpu_uuid"]})
        if len(kernel_uuids) > 1:
            failures.append(f"multiple kernel GPU UUIDs: {kernel_uuids}")
        elif kernel_uuids and normalize_gpu_uuid(kernel_uuids[0]) != \
                normalize_gpu_uuid(device_uuid):
            failures.append(f"kernel GPU UUID {kernel_uuids[0]} != device "
                            f"UUID {device_uuid}")
        if len({str(row["address_space_id"]) for row in observer.rows
                if row["event_type"] in ("MAP_RETURN", "PTE_HEADER")}) > 1:
            failures.append("multiple UVM address-space IDs observed")
        if observer.lost_event_count:
            failures.append(f"lost BPF events: {observer.lost_event_count}")

        extra.update({
            "g1_5_run_id": g1_5_run_id,
            "registry_allocation_count": len(registry),
            "map_row_count": len(final_map_rows),
            "snapshot_dir": str(snapshot_dir),
            "snapshot_sha256": result.manifest.get("snapshot_sha256", ""),
            "snapshot_manifest": result.manifest,
            "resident_bytes": resident_bytes,
            "anchor_pages": anchor_pages,
            "total_sites": total_sites,
            "trials_completed": len(trial_rows),
            "images_per_trial": images_total,
            "trial_outcome_histogram": outcome_histogram,
            "image_totals": image_totals,
            "restore_totals": restore_totals,
            "work_output": str(work_output),
            "work_sha256": sha256(work_output),
            "site_result_output": str(site_result_output),
            "site_result_sha256": sha256(site_result_output),
            "trial_result_output": str(trial_result_output),
            "trial_result_sha256": sha256(trial_result_output),
            "image_detail_output": str(image_detail_output),
            "image_detail_sha256": sha256(image_detail_output),
            "clean_pass_output": str(clean_pass_output),
            "clean_pass_sha256": sha256(clean_pass_output),
            "map_output": str(map_output),
            "map_output_sha256": sha256(map_output) if final_map_rows else "",
            "map_gate_output": str(map_gate_output),
            "map_gate_output_sha256": sha256(map_gate_output),
            "campaign_window_ns": {
                "campaign_gate_wait": gate_wait_ns,
                "work_begin": work_begin_ns,
                "work_end": work_end_ns,
            },
            "contract_sha256": contract_hash,
            "nvidia_module_sha256": contract["nvidia_module_sha256"],
            "nvidia_uvm_module_sha256": contract["nvidia_uvm_module_sha256"],
        })
    finally:
        gate.unlink(missing_ok=True)
        release.unlink(missing_ok=True)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        observer.close()
        test_log.write_text(captured.decode("utf-8", errors="replace"),
                            encoding="utf-8")

    return finalize(failures, run_dir, summary_output, test_log,
                    event_output, observer, args, run_id, device_uuid,
                    started_ns, extra, uid, gid)


# --------------------------------------------------------------------------
# self-test (offline: no root, no GPU, no observer traffic)
# --------------------------------------------------------------------------

def self_test() -> int:
    import tempfile

    rows_fixture, anchors_fixture = fault_model._fixture()
    level = fault_model.level_by_name("L2")
    campaign = fault_model.sample_campaign("L2", 3, rows_fixture,
                                           anchors_fixture, seed=11)
    assert verify_work_model(level, campaign) == []
    broken = [[dict(site, bit=site["bit"]) for site in campaign[0]]]
    broken[0][1]["bit"] = broken[0][0]["bit"]  # same bit, different byte:
    # distinctness is at (PA byte, bit), so rewrite the byte too
    broken[0][1]["chain"] = dict(broken[0][1]["chain"])
    broken[0][1]["chain"]["pa_in_page_offset"] = \
        broken[0][0]["chain"]["pa_in_page_offset"]
    assert verify_work_model(level, broken)

    # --- site rows: happy path + policy violations
    registry = [
        {"allocation_id": "trt-binding-data-gpu-0",
         "semantic_label": "TENSOR:data"},
        {"allocation_id": "trt-binding-prob-gpu-0",
         "semantic_label": "TENSOR:prob"},
        {"allocation_id": "trt-binding-index-gpu-0",
         "semantic_label": "TENSOR:index"},
        {"allocation_id": "trt-internal-0",
         "semantic_label": "TENSORRT_INTERNAL_UNKNOWN"},
        {"allocation_id": "trt-internal-1",
         "semantic_label": "TENSORRT_INTERNAL_UNKNOWN"},
    ]

    def site_row(site, before=0x55, **over):
        mask = 1 << site["bit"]
        row = {
            "run_id": "r", "device": "0",
            "trial_index": str(site["trial_index"]),
            "event_index": str(site["event_index"]),
            "site_index": str(site["site_index"]),
            "target_id": site["target_id"],
            "allocation_id": site["allocation_id"],
            "semantic_label": "x",
            "byte_offset": str(site["byte_offset"]),
            "bit_in_byte": str(site["bit"]),
            "xor_mask": str(mask),
            "gpu_va": f"{site['expected_gpu_va']:#x}",
            "expected_gpu_va": f"{site['expected_gpu_va']:#x}",
            "before": str(before),
            "after": str(before ^ mask),
            "guard_bytes_unchanged": "1", "reverse_map_ok": "1",
            "restored_byte_ok": "1", "restore_check": "exact",
        }
        row.update({key: str(value) for key, value in over.items()})
        return row

    flat = [site for sites in campaign for site in sites]
    good_sites = [site_row(site) for site in flat]
    assert verify_site_rows(campaign, good_sites, registry) == []
    bad = list(good_sites)
    bad[0] = site_row(flat[0], after=0)
    assert any("after" in f for f in
               verify_site_rows(campaign, bad, registry))
    # an input-binding site (TENSOR:data) with a non-exact restore_check
    data_sites = [index for index, site in enumerate(flat)
                  if site["allocation_id"] == "trt-binding-data-gpu-0"]
    if data_sites:
        bad = list(good_sites)
        bad[data_sites[0]] = site_row(flat[data_sites[0]],
                                      restore_check="mismatch:3")
        assert any("input binding restore_check" in f for f in
                   verify_site_rows(campaign, bad, registry))
    # a TRT-internal site honestly records an informational mismatch
    internal_sites = [index for index, site in enumerate(flat)
                      if site["allocation_id"].startswith("trt-internal")]
    if internal_sites:
        patched = list(good_sites)
        patched[internal_sites[0]] = site_row(
            flat[internal_sites[0]], restore_check="mismatch:17")
        assert verify_site_rows(campaign, patched, registry) == []

    # --- trial + image rows on a small synthetic pass (4 images)
    images_total = 4
    trial_rows = []
    image_rows = []
    clean_rows = [{"image_index": str(i), "path": f"img{i}.jpg",
                   "label": "7", "clean_class": "7",
                   "clean_probability": "0.5"}
                  for i in range(images_total)]
    for trial in range(len(campaign)):
        trial_rows.append({
            "run_id": "r", "device": "0", "trial_index": str(trial),
            "site_count": str(len(campaign[trial])),
            "event_count": str(level["s"] + level["d"] + level["t"]),
            "images_total": str(images_total),
            "images_evaluated": str(images_total),
            "images_benign": str(images_total), "images_sdc_numeric": "0",
            "images_sdc_top1": "0", "images_invalid": "0",
            "injected_outcome": "BENIGN", "sanity_class": "7",
            "sanity_probability": "0.5", "sanity_matches_clean": "1",
            "restore_alloc_exact": "1", "restore_alloc_mismatch": "0",
            "restore_alloc_skipped": "0", "restore_mismatch_bytes": "0",
        })
        for image in range(images_total):
            image_rows.append({
                "run_id": "r", "device": "0", "trial_index": str(trial),
                "image_index": str(image), "evaluated": "1",
                "clean_class": "7", "injected_class": "7",
                "clean_probability": "0.5", "injected_probability": "0.5",
                "outcome": "IMAGE_BENIGN",
            })
    assert verify_trial_rows(trial_rows, campaign, images_total) == []
    assert verify_image_rows(image_rows, trial_rows, clean_rows) == []

    # precedence violation: counters say numeric, outcome says BENIGN
    bad_trials = [dict(row) for row in trial_rows]
    bad_trials[0]["images_sdc_numeric"] = "1"
    bad_trials[0]["images_benign"] = str(images_total - 1)
    assert any("precedence" in f for f in
               verify_trial_rows(bad_trials, campaign, images_total))

    # image classification re-derivation: a silent top-1 change is caught
    bad_images = [dict(row) for row in image_rows]
    bad_images[1]["injected_class"] = "9"
    assert any("re-derived" in f for f in
               verify_image_rows(bad_images, trial_rows, clean_rows))

    # DUE abort shape: invalid at image 1, remainder evaluated=0
    abort_trials = [dict(row) for row in trial_rows]
    abort_trials[2] = dict(abort_trials[2],
                           images_evaluated="2", images_benign="1",
                           images_invalid="3",
                           injected_outcome="DUE_INVALID_OUTPUT")
    abort_images = []
    for trial in range(len(campaign)):
        for image in range(images_total):
            row = {"run_id": "r", "device": "0", "trial_index": str(trial),
                   "image_index": str(image), "evaluated": "1",
                   "clean_class": "7", "injected_class": "7",
                   "clean_probability": "0.5", "injected_probability": "0.5",
                   "outcome": "IMAGE_BENIGN"}
            if trial == 2:
                if image == 1:
                    row.update(outcome="IMAGE_DUE", injected_class="NA",
                               injected_probability="NA")
                elif image > 1:
                    row.update(outcome="IMAGE_DUE", evaluated="0",
                               injected_class="NA",
                               injected_probability="NA")
            abort_images.append(row)
    assert verify_trial_rows(abort_trials, campaign, images_total) == []
    assert verify_image_rows(abort_images, abort_trials, clean_rows) == []
    # an evaluated row AFTER the abort refuses
    bad_abort = [dict(row) for row in abort_images]
    bad_abort[-1]["evaluated"] = "1"
    assert any("after the DUE abort" in f for f in
               verify_image_rows(bad_abort, abort_trials, clean_rows))

    # --- event structure: exact sequence + CSV agreement. The synthetic
    # stream mirrors the runner's ACTUAL loop (flip forward, restore in
    # reverse) rather than expected_trial_sequence, so a bug in either
    # side's ordering shows up as a mismatch.
    by_id = {site["target_id"]: row for site, row in zip(flat, good_sites)}
    lines = []
    for sites in campaign:
        trial = str(sites[0]["trial_index"])
        lines.append(f"GPU_M2D_EVENT,event=TRIAL_BEGIN,trial_index={trial},"
                     "pid=1,tgid=1")
        for site in sites:
            row = by_id[site["target_id"]]
            lines.append(
                f"GPU_M2D_EVENT,event=SITE_FLIPPED,trial_index={trial},"
                f"target_id={site['target_id']},"
                f"gpu_va={row['gpu_va']},before={row['before']},"
                f"after={row['after']},xor_mask={row['xor_mask']},"
                "pid=1,tgid=1")
        lines.append(f"GPU_M2D_EVENT,event=TRIAL_INJECTED_END,trial_index="
                     f"{trial},pid=1,tgid=1")
        for site in reversed(sites):
            lines.append(f"GPU_M2D_EVENT,event=SITE_RESTORED,trial_index="
                         f"{trial},target_id={site['target_id']},pid=1,"
                         "tgid=1")
        lines.append(f"GPU_M2D_EVENT,event=TRIAL_SANITY_BEGIN,trial_index="
                     f"{trial},pid=1,tgid=1")
        lines.append(f"GPU_M2D_EVENT,event=TRIAL_SANITY_END,trial_index="
                     f"{trial},pid=1,tgid=1")
        lines.append(f"GPU_M2D_EVENT,event=TRIAL_END,trial_index={trial},"
                     "pid=1,tgid=1")
    output = "\n".join(lines) + "\n"
    assert verify_event_structure(output, campaign, good_sites) == []
    dropped = "\n".join(line for line in lines
                        if "SITE_RESTORED" not in line or
                        "t000" not in line) + "\n"
    assert verify_event_structure(dropped, campaign, good_sites)
    # the first before= in the stream belongs to the first SITE_FLIPPED;
    # corrupting it must trip the event-vs-CSV agreement check
    wrong_byte = output.replace(",before=85,", ",before=1,", 1)
    if ",before=1," in wrong_byte:
        assert verify_event_structure(wrong_byte, campaign, good_sites)

    # --- gate-side plumbing: a tiny run_build snapshot feeds the sampler
    with tempfile.TemporaryDirectory() as tmp:
        page = 2 << 20
        v_base = 0x700000000000
        table = Path(tmp) / "table"
        table.mkdir()
        (table / "gddr_seed_table.csv").write_text(
            "page_index,page_base,lambda,channel_root,channel_size,"
            "super_tail,n_shoulder,n_deep,n_low,n_nodes,n_bank_classes,"
            "classified\n"
            "0,0x2000000,1000,0x2000000,1,0,0,0,10,1,1,1\n")
        (table / "bank_classes.csv").write_text(
            "page_index,page_base,offset,pa,bank_class,bank_size,row_class,"
            "row_size,n_deep,n_low,n_shoulder\n")
        (table / "page_anchors.csv").write_text(
            "page_base,candidate,seen,votes,valid\n"
            "0x2000000,0x1f9dc0,3,3,true\n")
        areg = Path(tmp) / "allocations.csv"
        areg.write_text(",".join(ALLOC_FIELDS) + "\n"
                        f"r-x,a,0,{v_base:#x},{page},512,o,ph,lt,1,"
                        "TENSOR:data\n")
        amap = Path(tmp) / "map.csv"
        amap.write_text(
            "run_id,g1_5_run_id,device,gpu_uuid,allocation_id,"
            "allocation_api,va_page_base,va_page_end_exclusive,"
            "fb_pa_page_base,page_size,aperture,pte_valid,"
            "covered_allocation_bytes,physical_coverage_status\n"
            f"m,r-x,0,u,a,api,{v_base:#x},{v_base + page:#x},0x2000000,"
            f"{page},VIDEO,true,4096,LOCAL_VIDEO_COMPLETE\n")
        out = Path(tmp) / "snap"
        built = run_build(areg, amap, 0, table, out, lambda t="": None)
        assert not built.problems, built.problems
        assert resident_bytes_of(built.rows) == page
        tiny_anchors = fault_model.load_anchors(table)
        tiny = fault_model.sample_campaign("L1", 2, built.rows,
                                           tiny_anchors, seed=3,
                                           resident_bytes_expected=None)
        assert [len(sites) for sites in tiny] == [2, 2]
        assert verify_work_model(fault_model.level_by_name("L1"), tiny) == []

    print("g5 campaign self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.level is None:
        raise SystemExit("run_g5_campaign.py: error: --level is required "
                         "for a campaign (one frozen BER level per run)")
    if os.geteuid() != 0:
        raise PermissionError(
            "run through sudo; the CUDA child is dropped to the invoking user")
    return run_once(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(f"GPU_M2D_G5_CAMPAIGN_ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
