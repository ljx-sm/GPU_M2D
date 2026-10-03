#!/usr/bin/env python3
"""GPU_M2D G5: dual-addressing fault-injection campaign (one BER level).

One campaign = one BER level = one or more runner PROCESSES (segments).
The flow extends the G4-T2 gated skeleton (observer attaches pre-context;
gate-time registry; ONLINE per-run snapshot; byte-exact restore; strict
closing ledger) with the G5 trial loop of docs/G5_FAULT_MODEL.md §6:

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

RESTART PROTOCOL (user decision 2026-09-28, EfficientNet-B0 and every
later full-surface workload): a flip in TRT create_execution_context-
phase CONTROL state kills the runner PROCESS with a CUDA illegal memory
access mid-trial -- a process-fatal reliability event, not an
output-observable fault. The runner flushes the three result CSVs after
every COMPLETED trial (a trial is on disk iff it reached TRIAL_END), so
nothing completed is lost; the orchestrator then verifies the dead
segment's completed-trial prefix, counts the dying trial as PROCESS_FATAL
(excluded from the accuracy mean exactly like a DUE, reported separately
as the crash-rate-vs-BER reliability curve), and relaunches a FRESH
gated segment (new observer, snapshot, and sampling of the remaining
trial slots -- trials are independent draws of the same frozen model, so
segments pool statistically; segment k seeds from --seed + k and its
trial numbering is shifted to its global execution-order slots at merge
time). Only a mid-trial death carrying a known fatal CUDA signature (see
RECOVERABLE_CUDA_SIGNATURES) is recoverable; every other death, and any
prefix-verification failure, fails the whole level closed (exit 2). A
--bootstrap run is exempt: single flat run, measure-only.

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
sys.path.insert(0, str(PROJECT / "tools/g8_cache"))

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
import cache_model  # noqa: E402
import residency_map  # noqa: E402

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
# G8-T2 L2 cache mode (--cache-ber): the residency pass runs between the
# clean pass and the gate; cache flips are applied/removed per image.
G8_SKELETON_AFTER = "CLEAN_PASS_END"
G8_SKELETON_EVENTS = ["L2_PROBE_PASS_BEGIN", "L2_PROBE_PASS_END"]
G8_PROBE_ARGS = ["--l2-probe-threshold", "440", "--l2-probe-per-sm", "16",
                 "--l2-probe-unit", "32", "--l2-probe-every", "1",
                 "--l2-probe-stride", "auto", "--l2-probe-alternate", "1",
                 "--l2-probe-map", "1", "--l2-probe-passes", "1"]
CACHE_RESULT_FIELDS = [
    "run_id", "device", "trial_index", "cache_index", "target_id",
    "allocation_id", "byte_offset", "bit_in_byte", "xor_mask", "gpu_va",
    "expected_gpu_va", "cache_class", "start_image", "last_image", "applied",
    "apply_image", "before", "after", "removal", "removal_image",
    "remove_before", "remove_after",
]
CAMPAIGN_VARIABLE_EVENTS = {"ALLOCATED", "FREE", "TRIAL_BEGIN",
                            "CACHE_FLIPPED", "CACHE_REMOVED",
                            "SITE_FLIPPED", "TRIAL_INJECTED_END",
                            "SITE_RESTORED", "TRIAL_SANITY_BEGIN",
                            "TRIAL_SANITY_END", "TRIAL_END"}
TRIAL_EVENT_NAMES = {"TRIAL_BEGIN", "SITE_FLIPPED", "TRIAL_INJECTED_END",
                     "SITE_RESTORED", "TRIAL_SANITY_BEGIN",
                     "TRIAL_SANITY_END", "TRIAL_END"}

OUTCOMES = ("BENIGN", "SDC_TOP1", "SDC_NUMERIC", "DUE_INVALID_OUTPUT")
# Restart protocol: the trial the runner process DIED on (a flip in TRT
# runtime control state surfacing as a fatal CUDA error). Like a DUE it
# is excluded from the accuracy mean; it is reported separately as the
# crash-rate reliability curve (summary.process_fatal_* fields).
PROCESS_FATAL = "PROCESS_FATAL"
# the dead-process output must contain the harness fail marker plus at
# least ONE known fatal CUDA signature. Two verified so far -- the same
# phenomenon (a flip in TRT runtime control state killing the process at
# the dying trial's first injected inference) surfaced by the driver as
# different strings; both corpses had the identical recoverable shape:
#   - "illegal memory access" (mobilenet 2026-09-27, effnet/swin L1-L4)
#   - "operation not supported on global/shared address space"
#     (swin L5 2026-09-28; TRT teardown adds Myelin Error 717 destroying
#     streams + ScopedCudaEvent destructor errors around it)
MANDATORY_DEATH_NEEDLES = ("GPU_M2D_G1_5_FAIL",)
RECOVERABLE_CUDA_SIGNATURES = ("illegal memory access",
                               "operation not supported on "
                               "global/shared address space")
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


def expected_skeleton_for(hold_seconds: int, cache_mode: bool) -> list[str]:
    """The fixed lifecycle skeleton of one campaign process; in G8 cache
    mode the residency pass sits between the clean pass and the gate."""
    skeleton = [name for name in CAMPAIGN_SKELETON
                if not (name in ("HOLD_BEGIN", "HOLD_END")
                        and hold_seconds <= 0)]
    if cache_mode:
        at = skeleton.index(G8_SKELETON_AFTER) + 1
        skeleton[at:at] = G8_SKELETON_EVENTS
    return skeleton


def verify_cache_rows(cache_campaign: list[list[dict]], rows: list[dict],
                      trial_rows: list[dict]) -> list[str]:
    """Independent re-verification of the G8 cache result rows of every
    COMPLETED trial (plan §5) against the sampled cache work:
      - one row per sampled site, identical identity (target, allocation,
        byte, bit, expected VA, start/last image, class);
      - a site whose start image the trial reached was applied at exactly
        its start image, at the expected VA, with after == before ^ mask;
      - removal at its last image -- or at the trial's last evaluated image
        when a DUE aborted the pass first (and only then);
      - read-only: re_xor back to exactly the pre-cache value;
        engine-written: restored (back to the pre-cache value) or
        overwritten (the engine rewrote the byte);
      - a site the aborted trial never reached: not applied.
    Several flips may share one byte (different bits, overlapping
    lifetimes): the per-row checks are single-bit XORs, and the byte-level
    values are checked by replay_cache_bytes in the runner's order."""
    failures: list[str] = []
    by_trial: dict[int, list[dict]] = {}
    for row in rows:
        by_trial.setdefault(int(row["trial_index"]), []).append(row)
    trials = {int(t["trial_index"]): t for t in trial_rows}
    for t in sorted(set(by_trial) - set(trials)):
        failures.append(f"cache rows for trial {t} without a trial row")
    for t, trow in sorted(trials.items()):
        sites = cache_campaign[t] if t < len(cache_campaign) else []
        got = sorted(by_trial.get(t, []), key=lambda r: int(r["cache_index"]))
        if len(got) != len(sites):
            failures.append(f"trial {t}: {len(got)} cache rows != "
                            f"{len(sites)} sampled sites")
            continue
        last_eval = int(trow["images_evaluated"]) - 1
        due = trow["injected_outcome"] == "DUE_INVALID_OUTPUT"
        replay: list[tuple[dict, dict]] = []
        for site, row in zip(sites, got):
            tag = f"trial {t} cache {site['cache_index']}"
            mask = 1 << site["bit_in_byte"]
            identity = (
                row["target_id"] == site["target_id"]
                and row["allocation_id"] == site["allocation_id"]
                and int(row["byte_offset"]) == site["byte_offset"]
                and int(row["bit_in_byte"]) == site["bit_in_byte"]
                and int(row["expected_gpu_va"], 16) == site["expected_gpu_va"]
                and int(row["start_image"]) == site["start_image"]
                and int(row["last_image"]) == site["last_image"]
                and row["cache_class"] == site["cache_class"])
            if not identity:
                failures.append(f"{tag}: row disagrees with the sampled work")
                continue
            if site["start_image"] > last_eval:
                if not (due and row["applied"] == "0"
                        and row["removal"] == "not_applied"):
                    failures.append(f"{tag}: unreached site must be "
                                    "not_applied (and only after a DUE)")
                continue
            before, after = int(row["before"]), int(row["after"])
            if row["applied"] != "1" or \
                    int(row["apply_image"]) != site["start_image"] or \
                    int(row["gpu_va"], 16) != site["expected_gpu_va"] or \
                    int(row["xor_mask"]) != mask or after != before ^ mask:
                failures.append(f"{tag}: apply verification failed")
                continue
            want_removal_image = site["last_image"]
            if site["last_image"] > last_eval:
                if not due:
                    failures.append(f"{tag}: lifetime past the pass without "
                                    "a DUE abort")
                want_removal_image = last_eval
            if int(row["removal_image"]) != want_removal_image:
                failures.append(f"{tag}: removed at image "
                                f"{row['removal_image']} != "
                                f"{want_removal_image}")
            r_before, r_after = int(row["remove_before"]), \
                int(row["remove_after"])
            if site["cache_class"] == cache_model.READ_ONLY:
                if row["removal"] != "re_xor" or r_after != r_before ^ mask:
                    failures.append(f"{tag}: read-only removal must re-XOR")
                    continue
            elif row["removal"] == "restored":
                if r_after != r_before ^ mask:
                    failures.append(f"{tag}: conditional restore inconsistent")
                    continue
            elif row["removal"] == "overwritten":
                if r_after != r_before:
                    failures.append(f"{tag}: 'overwritten' must leave the "
                                    "byte untouched")
                    continue
            else:
                failures.append(f"{tag}: bad removal {row['removal']!r}")
                continue
            replay.append((site, row))
        failures += replay_cache_bytes(t, replay)
    return failures


def replay_cache_bytes(trial: int, applied: list[tuple[dict, dict]]
                       ) -> list[str]:
    """Replay one trial's applied cache flips per byte in the runner's
    order: at each image, applies (cache_index order) precede the
    inference and removals follow it in application order (apply image,
    then cache_index). A byte with active flips must read pristine ^
    active_mask (pristine = its value before the first active flip):
    every apply 'before' and every removal 'remove_before' equals the
    replayed value -- except on an engine-written byte the engine rewrote
    ('overwritten', after which the byte is no longer tracked) -- and a
    read-only byte returns to pristine when its last flip is removed. With
    one flip per byte this is exactly r_before == after, r_after == before."""
    failures: list[str] = []
    ops = []
    for site, row in applied:
        ops.append(((site["start_image"], 0, site["cache_index"]),
                    "apply", site, row))
        ops.append(((int(row["removal_image"]), 1, site["start_image"],
                     site["cache_index"]), "remove", site, row))
    ops.sort(key=lambda op: op[0])
    state: dict[tuple[str, int], dict] = {}
    for _, kind, site, row in ops:
        key = (site["allocation_id"], site["byte_offset"])
        read_only = site["cache_class"] == cache_model.READ_ONLY
        tag = f"trial {trial} cache {site['cache_index']}"
        byte = state.get(key)
        if kind == "apply":
            before = int(row["before"])
            if byte is None:
                byte = state[key] = {"pristine": before, "value": before,
                                     "active": 0, "overwritten": False}
            elif not byte["overwritten"] and before != byte["value"]:
                if read_only:
                    failures.append(f"{tag}: read-only byte changed before "
                                    "this flip was applied")
                byte["overwritten"] = True
            byte["value"] = int(row["after"])
            byte["active"] += 1
            continue
        if byte is None:
            failures.append(f"{tag}: removed without an active flip")
            continue
        r_before, r_after = int(row["remove_before"]), int(row["remove_after"])
        intact = not byte["overwritten"] and r_before == byte["value"]
        if read_only and not intact:
            failures.append(f"{tag}: read-only byte changed during its "
                            "lifetime")
        elif not read_only and (row["removal"] == "restored") != intact:
            failures.append(f"{tag}: removal {row['removal']!r} disagrees "
                            "with the byte's replayed value")
        if row["removal"] == "overwritten":
            byte["overwritten"] = True
        byte["value"] = r_after
        byte["active"] -= 1
        if byte["active"] == 0:
            if read_only and r_after != byte["pristine"]:
                failures.append(f"{tag}: read-only byte not back to its "
                                "pre-cache value")
            del state[key]
    return failures


def verify_cache_events(output: str, rows: list[dict]) -> list[str]:
    """CACHE_FLIPPED / CACHE_REMOVED events must match the applied rows of
    the completed trials one-to-one (counts and before/after values)."""
    failures: list[str] = []
    applied = {(row["trial_index"], row["cache_index"]): row
               for row in rows if row["applied"] == "1"}
    trials = {row["trial_index"] for row in rows}
    events = [(n, f) for n, f in parse_all_events(output)
              if n in ("CACHE_FLIPPED", "CACHE_REMOVED")
              and f.get("trial_index") in trials]
    for kind, before_key, after_key in (("CACHE_FLIPPED", "before", "after"),
                                        ("CACHE_REMOVED", "remove_before",
                                         "remove_after")):
        seen = [f for n, f in events if n == kind]
        if len(seen) != len(applied):
            failures.append(f"{kind}: {len(seen)} events != "
                            f"{len(applied)} applied cache rows")
            continue
        for f in seen:
            row = applied.get((f.get("trial_index"), f.get("cache_index")))
            if row is None or f.get("before") != row[before_key] or \
                    f.get("after") != row[after_key]:
                failures.append(f"{kind} {f.get('trial_index')}/"
                                f"{f.get('cache_index')}: disagrees with "
                                "the cache result CSV")
    return failures


def analyze_process_death(output: str, campaign: list[list[dict]]
                          ) -> tuple[dict | None, list[str]]:
    """Classify a runner process that died mid-campaign (restart
    protocol). RECOVERABLE -- and the only restartable class -- is the
    process-fatal fault: the output carries the harness fail marker plus
    one of the known fatal CUDA signatures (RECOVERABLE_CUDA_SIGNATURES),
    every trial before the dying one reached TRIAL_END, and the dying
    trial (the LAST TRIAL_BEGIN, exactly one open) has its complete
    SITE_FLIPPED set with nothing after it (the death hits the trial's
    first injected inference). Returns ({completed, dying}, []) or
    (None, reasons); the caller fail-closes on any other death shape."""
    reasons: list[str] = []
    for needle in MANDATORY_DEATH_NEEDLES:
        if needle not in output:
            reasons.append(f"failure signature {needle!r} absent")
    if not any(sig in output for sig in RECOVERABLE_CUDA_SIGNATURES):
        reasons.append("no recoverable CUDA death signature ("
                       + " / ".join(RECOVERABLE_CUDA_SIGNATURES)
                       + ") in output")
    trial_events = [(name, fields) for name, fields in
                    parse_all_events(output) if name in TRIAL_EVENT_NAMES]
    begins = [fields["trial_index"] for name, fields in trial_events
              if name == "TRIAL_BEGIN"]
    ended = {fields["trial_index"] for name, fields in trial_events
             if name == "TRIAL_END"}
    if begins != [str(index) for index in range(len(begins))]:
        reasons.append("TRIAL_BEGIN stream not contiguous from 0 "
                       f"(work-file contract): {begins[:8]}")
        return None, reasons
    open_trials = [trial for trial in begins if trial not in ended]
    if len(open_trials) != 1 or open_trials[0] != begins[-1]:
        reasons.append(f"expected exactly one open (dying) trial that is "
                       f"the last begun; open={open_trials}")
        return None, reasons
    completed = len(begins) - 1
    if not 0 <= completed < len(campaign):
        reasons.append(f"dying position {completed} outside the sampled "
                       f"{len(campaign)} trials")
        return None, reasons
    dying_at = len(trial_events) - 1
    while trial_events[dying_at][0] != "TRIAL_BEGIN":
        dying_at -= 1
    tail = [name for name, _ in trial_events[dying_at + 1:]]
    expected_tail = ["SITE_FLIPPED"] * len(campaign[completed])
    if tail != expected_tail:
        reasons.append(f"dying trial's event tail {tail[:8]}... != its "
                       f"complete {len(campaign[completed])}-site flip set "
                       "(the death must hit the first injected inference)")
        return None, reasons
    if reasons:
        return None, reasons
    return {"completed": completed, "dying": int(begins[-1])}, []


# --------------------------------------------------------------------------
# restart-protocol merge (offline-testable)
# --------------------------------------------------------------------------

# canonical result file -> (header fields, trial_index column, target_id
# column or None); the trial column is renumbered to global slots and a
# target_id's t%03d prefix along with it
MERGE_CSV_SPECS: dict[str, tuple[list[str], int, int | None]] = {
    "g1_5_g5_site_result.csv": (SITE_RESULT_FIELDS, 2, 5),
    "g1_5_g5_trial_result.csv": (TRIAL_RESULT_FIELDS, 2, None),
    "g1_5_g5_image_detail.csv": (IMAGE_DETAIL_FIELDS, 2, None),
    "work.csv": (fault_model.WORK_FIELDS, 0, 3),
}


# G8 cache-mode result files (present only with --cache-ber): merged like
# the G5 files; a segment's cache rows exist only for its completed
# trials, its cache work for every slot it was assigned.
G8_MERGE_CSV_SPECS: dict[str, tuple[list[str], int, int | None]] = {
    "g1_5_g8_cache_site_result.csv": (CACHE_RESULT_FIELDS, 2, 4),
    "cache_work.csv": (cache_model.CACHE_WORK_FIELDS, 0, 2),
}


def shift_trial_fields(fields: list[str], delta: int, trial_col: int,
                       target_col: int | None = None) -> list[str]:
    """One CSV row with its trial_index (and target_id's t%03d prefix)
    shifted to the trial's GLOBAL execution-order slot."""
    shifted = list(fields)
    global_index = int(shifted[trial_col]) + delta
    shifted[trial_col] = str(global_index)
    if target_col is not None:
        head, sep, rest = shifted[target_col].partition("-")
        # t%03d-...: DRAM sites; c%03d-...: G8 cache sites
        if head[:1] not in ("t", "c") or not sep:
            raise RuntimeError(f"malformed target_id "
                               f"{shifted[target_col]!r}")
        shifted[target_col] = f"{head[0]}{global_index:03d}-{rest}"
    return shifted


def read_result_csv(path: Path, fields: list[str]) -> list[list[str]]:
    """Raw rows (list of field lists) of a result CSV, '#' comments and
    header validated against the expected field list."""
    if not path.is_file():
        raise RuntimeError(f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as source:
        lines = [line for line in source if not line.startswith("#")]
    reader = csv.reader(lines)
    header = next(reader, None)
    if header != fields:
        raise RuntimeError(f"unexpected header in {path}: {header}")
    return [row for row in reader if row]


def write_merged_csv(path: Path, fields: list[str], rows: list[list[str]],
                     comment: str | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as sink:
        if comment is not None:
            sink.write(f"# {comment}\n")
        writer = csv.writer(sink, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def sha256_if_present(path: Path) -> str:
    """Hash of an OPTIONAL level artifact: a level whose every segment
    died mid-trial has no final va-pa map at all (the closing ledger
    only runs on a clean exit), and the summary must record that as an
    empty hash instead of dying -- the per-segment evidence lives in
    segment_details regardless."""
    return sha256(path) if path.is_file() else ""


def merge_campaign_outputs(level_dir: Path, segments: list[dict],
                           args: argparse.Namespace, images_total: int
                           ) -> dict:
    """Merge the verified per-segment result CSVs into the level run's
    CANONICAL files at level_dir root, renumbering every trial to its
    global execution-order slot (segment-local trials are contiguous
    from 0 -- the runner's work-file contract; segment k starts at slot
    completed_0..k-1 + one slot per earlier PROCESS_FATAL). A segment's
    dying trial keeps its slot but contributes no result rows. A crashed
    segment's work file still lists every slot it was ASSIGNED (the
    process died mid-way); only the trials it consumed -- completed plus
    the dying one -- map to global slots, the never-run tail was
    re-sampled by a later segment and is dropped here. A single clean
    segment is hardlinked (delta 0, byte-identical). Fail-closed on any
    count/shape disagreement."""
    if not segments:
        raise RuntimeError("merge: no segments")
    single = (len(segments) == 1 and segments[0]["crashed_trial"] is None)
    standard_names = list(MERGE_CSV_SPECS) + [
        "g1_5_g5_clean_pass.csv", "g1_5_allocations.csv",
        "g1_5_allocations_gate.csv", "gpu_va_pa_map.csv",
        "gpu_va_pa_map_gate.csv", "events.csv",
    ]
    g8_names = [name for name in G8_MERGE_CSV_SPECS
                if (segments[0]["dir"] / name).is_file()]
    if single:
        for name in standard_names + g8_names + ["work_detail.json",
                                                 "harness.log"]:
            source = segments[0]["dir"] / name
            if source.is_file():
                os.link(source, level_dir / name)
        trial_rows = read_result_csv(level_dir / "g1_5_g5_trial_result.csv",
                                     TRIAL_RESULT_FIELDS)
        return {"mode": "hardlink",
                "trial_rows": [dict(zip(TRIAL_RESULT_FIELDS, row))
                               for row in trial_rows],
                "site_rows": None, "image_rows": None}

    def consumed_of(segment: dict) -> int:
        return segment["completed"] + \
            (1 if segment["crashed_trial"] is not None else 0)

    merged: dict[str, list[list[str]]] = {name: [] for name in MERGE_CSV_SPECS}
    for segment in segments:
        delta = segment["first_trial"]
        consumed = consumed_of(segment)
        for name, (fields, trial_col, target_col) in MERGE_CSV_SPECS.items():
            rows = read_result_csv(segment["dir"] / name, fields)
            merged[name].extend(
                shift_trial_fields(row, delta, trial_col, target_col)
                for row in rows if int(row[trial_col]) < consumed)
    for name in g8_names:
        fields, trial_col, target_col = G8_MERGE_CSV_SPECS[name]
        rows_out: list[list[str]] = []
        for segment in segments:
            consumed = consumed_of(segment)
            completed = segment["completed"]
            keep = consumed if name == "cache_work.csv" else completed
            rows = read_result_csv(segment["dir"] / name, fields)
            rows_out.extend(
                shift_trial_fields(row, segment["first_trial"], trial_col,
                                   target_col)
                for row in rows if int(row[trial_col]) < keep)
        write_merged_csv(level_dir / name, fields, rows_out)
    for name, (fields, _, _) in MERGE_CSV_SPECS.items():
        write_merged_csv(level_dir / name, fields, merged[name],
                         comment=("gpu-m2d g5 campaign merged over restart "
                                  "segments; trials renumbered to global "
                                  "execution-order slots"
                                  if name == "work.csv" else None))

    # count sanity, fail-closed
    completed_total = sum(seg["completed"] for seg in segments)
    sites_expected = sum(len(sites) for seg in segments
                         for sites in seg["campaign"][:seg["completed"]])
    bits = segments[0]["extra"]["level"]["bits"]
    if len(merged["g1_5_g5_trial_result.csv"]) != completed_total:
        raise RuntimeError(f"merged trial rows "
                           f"{len(merged['g1_5_g5_trial_result.csv'])} != "
                           f"completed trials {completed_total}")
    if len(merged["g1_5_g5_site_result.csv"]) != sites_expected:
        raise RuntimeError(f"merged site rows "
                           f"{len(merged['g1_5_g5_site_result.csv'])} != "
                           f"completed sites {sites_expected}")
    if len(merged["g1_5_g5_image_detail.csv"]) != \
            completed_total * images_total:
        raise RuntimeError(f"merged image rows "
                           f"{len(merged['g1_5_g5_image_detail.csv'])} != "
                           f"{completed_total} trials x {images_total}")
    if len(merged["work.csv"]) != args.trials * bits:
        raise RuntimeError(f"merged work rows {len(merged['work.csv'])} != "
                           f"{args.trials} trials x {bits} bits -- the "
                           "consumed-slot partition is inconsistent")
    seen = [row[2] for row in merged["g1_5_g5_trial_result.csv"]]
    if len(set(seen)) != len(seen):
        raise RuntimeError("merged trial_index values are not unique")

    # shared-evidence files come from the FIRST segment that has each
    # file (fresh process each segment: its own registry/maps/clean
    # pass; the summary's segment_details carries every segment's own
    # paths and hashes). A crashed segment never wrote its FINAL va-pa
    # map -- the process died before the closing ledger ran -- so the
    # final map comes from the first cleanly-finished segment, while
    # the gate map / clean pass / registry, which every segment writes
    # before the campaign gate, still come from segment 0.
    for name in standard_names[len(MERGE_CSV_SPECS):]:
        for segment in segments:
            source = segment["dir"] / name
            if source.is_file():
                os.link(source, level_dir / name)
                break
    with (level_dir / "harness.log").open("w", encoding="utf-8") as sink:
        for segment in segments:
            sink.write(f"===== {segment['dir'].name} "
                       f"(seed {segment['seed']}, trials "
                       f"{segment['first_trial']}.."
                       f"{segment['first_trial'] + segment['trials'] - 1}"
                       f" of {args.trials}) =====\n")
            log = segment["dir"] / "harness.log"
            if log.is_file():
                sink.write(log.read_text(encoding="utf-8",
                                         errors="replace"))

    # merged work detail: the level's frozen echo + per-segment provenance
    # + the CONSUMED trials' sites renumbered to their global slots
    # (exactly args.trials x bits after the partition check above)
    sites_global: list[dict] = []
    for segment in segments:
        delta = segment["first_trial"]
        for sites in segment["campaign"][:consumed_of(segment)]:
            for site in sites:
                shifted = dict(site)
                shifted["trial_index"] = site["trial_index"] + delta
                head, sep, rest = site["target_id"].partition("-")
                shifted["target_id"] = (f"t{shifted['trial_index']:03d}-"
                                        + rest if sep else site["target_id"])
                sites_global.append(shifted)
    work_detail = {
        "schema": "gpu-m2d.g5.campaign.work.v1",
        "run_id": level_dir.name,
        "device": args.device,
        "level": segments[0]["extra"]["level"],
        "seed": args.seed,
        "trial_seed_rule": "random.Random(seed * 100003 + trial_index); "
                           "segment k samples with seed + k",
        "trials": args.trials,
        "segments": [{
            "segment": seg["segment"], "dir": seg["dir"].name,
            "seed": seg["seed"], "trials": seg["trials"],
            "first_trial": seg["first_trial"],
            "completed": seg["completed"],
            "crashed_trial": seg["crashed_trial"],
        } for seg in segments],
        "sites": sites_global,
    }
    (level_dir / "work_detail.json").write_text(json.dumps(
        work_detail, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"mode": "merged",
            "trial_rows": [dict(zip(TRIAL_RESULT_FIELDS, row))
                           for row in merged["g1_5_g5_trial_result.csv"]],
            "site_rows": merged["g1_5_g5_site_result.csv"],
            "image_rows": merged["g1_5_g5_image_detail.csv"]}


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
                           site_rows: list[dict],
                           partial_trial: list[dict] | None = None
                           ) -> list[str]:
    """The observed trial event stream must be exactly the sequence the
    work file prescribes (flip order, restore order, per-trial bracket),
    and every SITE_FLIPPED must agree with the site result CSV.

    partial_trial (restart protocol): the trial the runner process DIED
    on. The expected stream then additionally ends with that trial's
    TRIAL_BEGIN plus its COMPLETE SITE_FLIPPED set and nothing after --
    the process-fatal death hits the trial's first injected
    inference, before TRIAL_INJECTED_END or any restore. The dying
    trial's site rows do not exist (never flushed) so the CSV-agreement
    check skips them."""
    failures: list[str] = []
    observed = [(name, fields) for name, fields in
                parse_all_events(output) if name in TRIAL_EVENT_NAMES]
    expected = expected_trial_sequence(campaign)
    if partial_trial is not None:
        dying = str(partial_trial[0]["trial_index"])
        expected.append(("TRIAL_BEGIN", {"trial_index": dying}))
        for site in partial_trial:
            expected.append(("SITE_FLIPPED", {"target_id":
                                              site["target_id"]}))
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


class SegmentFatal(RuntimeError):
    """A fail-closed segment failure: anything except the recoverable
    process-fatal fault (a known fatal CUDA signature mid-trial). The
    LEVEL aborts with status FAIL_CLOSED -- no restart."""


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
    parser.add_argument("--workload", default=fault_model.DEFAULT_WORKLOAD,
                        choices=sorted(fault_model.WORKLOADS),
                        help="frozen fault-model workload (its level table "
                             "and nominal R guard)")
    parser.add_argument("--level", default=None,
                        help="frozen BER level of the selected workload; "
                             "required unless --bootstrap")
    parser.add_argument("--trials", type=int, default=100,
                        help="trials of this campaign (default 100)")
    parser.add_argument("--seed", type=int, default=7,
                        help="campaign RNG seed; trial seeds derive as "
                             "seed*100003 + trial_index")
    parser.add_argument("--hold-seconds", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0,
                        help="wall-clock budget for the runner (the full "
                             "campaign runs inside it)")
    parser.add_argument("--prelude-seconds", type=float, default=600.0,
                        help="budget for the runner's CPU prelude (host "
                             "preprocessing of every evaluation image) "
                             "before the pre-allocation gate; the G7 10K "
                             "ImageNet pass needs more than the G5 default")
    parser.add_argument("--cache-faults", action="store_true",
                        help="G8 L2 cache faults with the FROZEN model "
                             "(G8-T3): BER_cache = rho x the level's DRAM "
                             "BER, rho = fault_model.CACHE_RHO (1.0)")
    parser.add_argument("--cache-ber", type=float, default=None,
                        help="G8 L2 cache faults: BER_cache per trial "
                             "(n_cache = round(BER_cache x R_eff_bits) from "
                             "this process's residency map). Enables the "
                             "in-process residency pass, the map "
                             "self-checks and per-image cache flips. Until "
                             "the cache level table is frozen (G8-T3) the "
                             "value is an explicit, recorded parameter")
    parser.add_argument("--bootstrap", action="store_true",
                        help="measure-only run: attach the observer, build "
                             "the per-run snapshot, record the live "
                             "resident-byte total R (for freezing a new "
                             "workload's level table), then stop -- no "
                             "trials, no flips")
    # Runner passthrough (G7 workload parameterization; unset flags are
    # omitted so the runner defaults keep the G5 behavior byte-identical).
    parser.add_argument("--class-count", type=int, default=None)
    parser.add_argument("--preprocess", choices=("legacy", "canonical"),
                        default=None)
    parser.add_argument("--resize-scale", type=int, default=None)
    parser.add_argument("--interp", choices=("bicubic", "bilinear"),
                        default=None)
    parser.add_argument("--mean", default=None,
                        help="R,G,B canonical-mode mean (model_meta.json)")
    parser.add_argument("--std", default=None,
                        help="R,G,B canonical-mode std (model_meta.json)")
    parser.add_argument("--image-cache-dir", type=Path, default=None,
                        help="host preprocessed-image cache directory shared "
                             "by every segment/restart of every workload "
                             "with the same sample CSV + preprocessing spec "
                             "(default: <output-root>/../image_cache). The "
                             "runner key-checks the cache fail-closed and "
                             "falls back to fresh preprocessing on any "
                             "mismatch; the first run populates it")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT / "artifacts/g5/campaign")
    args = parser.parse_args()
    # G8 cache mode: --cache-faults (frozen, BER_cache = rho x level BER)
    # or --cache-ber X (explicit override, recorded as unfrozen).
    args.cache_ber_frozen = False
    if args.cache_faults:
        if args.cache_ber is not None:
            parser.error("--cache-faults and --cache-ber are mutually "
                         "exclusive")
        if args.bootstrap or args.level is None:
            parser.error("--cache-faults needs a --level (not --bootstrap)")
        try:
            level = fault_model.level_by_name(args.level, args.workload)
        except fault_model.ModelError as exc:
            parser.error(str(exc))
        args.cache_ber = fault_model.cache_ber_for(level)
        args.cache_ber_frozen = True
    if args.cache_ber is not None:
        if args.bootstrap:
            parser.error("--cache-ber cannot be combined with --bootstrap")
        if args.cache_ber < 0:
            parser.error("--cache-ber must be >= 0")
    return args


def finalize(failures: list[str], run_dir: Path, test_log: Path,
             event_output: Path, observer_info: dict[str, int],
             args: argparse.Namespace, run_id: str, device_uuid: str,
             started_ns: int, extra: dict[str, Any], uid: int, gid: int,
             status_override: str | None = None,
             ) -> int:
    status = status_override or ("FAIL_CLOSED" if failures
                                 else "G5_CAMPAIGN_VERIFIED")
    summary_output = run_dir / "summary.json"
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
        "target_tgid": observer_info.get("target_tgid", -1),
        "workload": args.workload,
        "level": args.level,
        "trials_requested": args.trials,
        "seed": args.seed,
        "table_dir": str(args.table),
        "engine_path": str(args.engine),
        "engine_sha256": sha256(args.engine) if args.engine.is_file() else "",
        "runner_path": str(args.runner),
        "runner_sha256": sha256(args.runner) if args.runner.is_file() else "",
        "event_count": observer_info.get("event_count", 0),
        "lost_event_count": observer_info.get("lost_event_count", 0),
        "event_output": str(event_output),
        "event_output_sha256": (sha256(event_output)
                                if Path(event_output).is_file() else ""),
        "test_log": str(test_log),
        "test_log_sha256": (sha256(test_log)
                            if Path(test_log).is_file() else ""),
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
    if args.bootstrap:
        outputs.append(run_dir / "bootstrap.json")
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
    print(f"device={args.device} workload={args.workload} "
          f"level={args.level or 'bootstrap'} status={status} "
          f"run={run_id}")
    print(f"summary={summary_output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 2 if failures else 0


def execute_segment(args: argparse.Namespace, contract: dict,
                    contract_hash: str, level: dict | None,
                    level_dir: Path, segment_index: int, trials: int,
                    seed: int, first_trial: int, uid: int, gid: int,
                    cotenancy: str, device_uuid: str) -> dict:
    """One gated runner process of a level = segment k of the restart
    protocol (a --bootstrap run is segment 0 and FLAT: its files live
    directly in the level dir, preserving the run_bootstrap_* layout the
    freeze tooling globs). The segment owns its observer, gates,
    snapshot, sampled work, and (per-trial flushed) result CSVs inside
    segment_dir, and returns a result dict:

      {"fatal": [...]}                    -> level fails closed, no restart
      {"bootstrap": True}                 -> measure-only run complete
      {"crashed_trial": <global slot>}    -> process-fatal trial; restart
      {"passed": True}                    -> segment finished cleanly

    Everything inside raises nothing past RuntimeError; the except below
    converts every failure into a fatal result so the level driver owns
    the fail-closed decision."""
    if args.bootstrap:
        segment_dir = level_dir
    else:
        segment_dir = level_dir / f"segment_{segment_index:03d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        os.chown(segment_dir, uid, gid)
    segment_name = (level_dir.name if args.bootstrap
                    else f"{level_dir.name}/{segment_dir.name}")
    event_output = segment_dir / "events.csv"
    test_log = segment_dir / "harness.log"
    registry_output = segment_dir / "g1_5_allocations.csv"
    registry_gate_copy = segment_dir / "g1_5_allocations_gate.csv"
    map_output = segment_dir / "gpu_va_pa_map.csv"
    map_gate_output = segment_dir / "gpu_va_pa_map_gate.csv"
    work_output = segment_dir / "work.csv"
    work_detail_output = segment_dir / "work_detail.json"
    site_result_output = segment_dir / "g1_5_g5_site_result.csv"
    trial_result_output = segment_dir / "g1_5_g5_trial_result.csv"
    image_detail_output = segment_dir / "g1_5_g5_image_detail.csv"
    clean_pass_output = segment_dir / "g1_5_g5_clean_pass.csv"
    cache_mode = getattr(args, "cache_ber", None) is not None
    cache_work_output = segment_dir / "cache_work.csv"
    cache_result_output = segment_dir / "g1_5_g8_cache_site_result.csv"
    residency_prefix = segment_dir / "g1_5_l2"
    gate = segment_dir / f".gate_{os.getpid()}_{time.time_ns()}"
    release = segment_dir / f".release_{os.getpid()}_{time.time_ns()}"
    run_id = segment_name
    workload_entry = fault_model.workload_by_name(args.workload)
    r_nominal = workload_entry["resident_bytes_nominal"]

    trt_runtime_dir = Path(os.environ.get(
        "GPU_M2D_REMU_ROOT", "/data1/luojx/REMU")) / \
        ".local/deps/tensorrt-8.6.1/tensorrt_libs"
    opencv_lib_dir = Path(os.environ.get(
        "GPU_M2D_REMU_ROOT", "/data1/luojx/REMU")) / ".local/deps/conda/lib"
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        part for part in (str(trt_runtime_dir), str(opencv_lib_dir),
                          os.environ.get("LD_LIBRARY_PATH", "")) if part)

    # G7 workload parameterization: forward only the flags the operator set
    # so an unset --workload default keeps the G5 runner argv byte-identical
    runner_passthrough: list[str] = []
    if args.class_count is not None:
        runner_passthrough += ["--class-count", str(args.class_count)]
    if args.preprocess is not None:
        runner_passthrough += ["--preprocess", args.preprocess]
    if args.resize_scale is not None:
        runner_passthrough += ["--resize-scale", str(args.resize_scale)]
    if args.interp is not None:
        runner_passthrough += ["--interp", args.interp]
    if args.mean is not None:
        runner_passthrough += ["--mean", args.mean]
    if args.std is not None:
        runner_passthrough += ["--std", args.std]
    if args.image_cache_dir is not None:
        runner_passthrough += ["--image-cache-dir", str(args.image_cache_dir)]
    g8_runner_args: list[str] = []
    if cache_mode:
        g8_runner_args = [
            "--l2-probe-out", str(residency_prefix.resolve()),
            *G8_PROBE_ARGS,
            "--l2-probe-expect-surface-bytes", str(r_nominal),
            "--campaign-cache-work", str(cache_work_output.resolve()),
        ]
        excludes_g8 = workload_entry.get("surface_excludes", ())
        if excludes_g8:
            g8_runner_args += ["--l2-probe-exclude", ",".join(excludes_g8)]

    observer: G2Observer | None = None
    process: subprocess.Popen[bytes] | None = None
    started_ns = time.time_ns()
    captured = bytearray()
    failures: list[str] = []
    extra: dict[str, Any] = {"cotenancy_at_start": cotenancy,
                             "runner_passthrough": runner_passthrough,
                             "level": level,
                             "segment": segment_index,
                             "segment_dir": segment_name,
                             "segment_seed": seed,
                             "segment_trials": trials,
                             "segment_first_trial": first_trial}

    def observer_info() -> dict[str, int]:
        if observer is None:
            return {"target_tgid": -1, "event_count": 0,
                    "lost_event_count": 0}
        return {"target_tgid": observer.target_tgid,
                "event_count": len(observer.rows),
                "lost_event_count": observer.lost_event_count}

    def segment_result(**over: Any) -> dict:
        result = {"segment": segment_index, "dir": segment_dir,
                  "name": segment_name, "seed": seed, "trials": trials,
                  "first_trial": first_trial, "passed": False,
                  "completed": 0, "crashed_trial": None, "campaign": [],
                  "extra": extra, "test_log": test_log,
                  "event_output": event_output,
                  "observer_info": observer_info(), "bootstrap": False}
        result.update(over)
        return result

    observer = G2Observer(contract, contract_hash, 0, event_output)
    try:
        process = subprocess.Popen(
            [
                str(args.runner.resolve()),
                "--engine", str(args.engine.resolve()),
                "--sample-csv", str(args.sample_csv.resolve()),
                "--sample-index", str(args.sample_index),
                "--device", str(args.device),
                "--output-prefix", str((segment_dir / "g1_5").resolve()),
                "--observer-gate", str(gate),
                "--hold-seconds", str(args.hold_seconds),
                "--gate-timeout-seconds", str(max(5, int(args.timeout_seconds))),
                "--campaign-work", str(work_output.resolve()),
                "--campaign-release", str(release.resolve()),
                "--campaign-gate-timeout-seconds",
                str(max(600, int(args.timeout_seconds))),
                *runner_passthrough,
                *g8_runner_args,
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
        # BEFORE this marker (CPU-only), so the wait budget is minutes
        # (the G7 10K ImageNet prelude needs more than the G5 default).
        prelude, gate_ready = read_until_marker(
            process, b"GPU_M2D_EVENT,event=WAIT_PRE_ALLOC_GATE,",
            min(args.timeout_seconds, args.prelude_seconds))
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

        snapshot_dir = segment_dir / "snapshot"
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
        # For a workload with surface_excludes (runtime CONTROL state
        # proven process-fatal, not output-observable) the frozen R is
        # the EXCLUSION-FILTERED total, so the guard below and the
        # sampler both work on the injection surface, never the raw
        # residency; bootstrap records keep the FULL measured R.
        resident_bytes = resident_bytes_of(result.rows)
        surface_rows = fault_model.surface_rows_for(args.workload,
                                                    result.rows)
        surface_bytes = resident_bytes_of(surface_rows)
        excludes = workload_entry.get("surface_excludes", ())
        if excludes:
            extra.update({
                "surface_excludes": list(excludes),
                "surface_resident_bytes": surface_bytes,
                "excluded_resident_bytes": resident_bytes - surface_bytes,
            })
            print(f"fault surface: excluding {', '.join(excludes)} "
                  f"({resident_bytes - surface_bytes:,} B of "
                  f"{resident_bytes:,} B); injection surface "
                  f"{surface_bytes:,} B")

        if args.bootstrap:
            # measure-only: record this engine's live R so its workload's
            # level table can be derived and frozen; no trials, no flips.
            # The runner sits blocked at the campaign gate having already
            # written the clean-pass CSV (the baseline-validation
            # artifact) -- terminate it now.
            bootstrap_output = segment_dir / "bootstrap.json"
            bootstrap_output.write_text(json.dumps({
                "schema_version": "gpu-m2d.g5.bootstrap.v1",
                "workload": args.workload,
                "resident_bytes": resident_bytes,
                "resident_bits": resident_bytes * 8,
                "device": args.device,
                "device_uuid": device_uuid,
                "engine_path": str(args.engine),
                "engine_sha256": sha256(args.engine),
                "runner_path": str(args.runner),
                "runner_sha256": sha256(args.runner),
                "runner_passthrough": runner_passthrough,
                "snapshot_manifest": result.manifest,
                "snapshot_rows": len(result.rows),
                "allocation_count": len(registry),
                "started_wall_time_ns": started_ns,
                "measured_wall_time_ns": time.time_ns(),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(bootstrap_output, 0o644)
            print(f"bootstrap: workload={args.workload} "
                  f"resident_bytes={resident_bytes} "
                  f"R_bits={resident_bytes * 8}")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            for _ in range(20):
                observer.poll(50)
            drain_pipe(process, captured)
            extra.update({
                "bootstrap_output": str(bootstrap_output),
                "bootstrap_sha256": sha256(bootstrap_output),
                "resident_bytes": resident_bytes,
                "snapshot_dir": str(snapshot_dir),
                "snapshot_sha256": result.manifest.get("snapshot_sha256",
                                                       ""),
                "snapshot_manifest": result.manifest,
                "g1_5_run_id": g1_5_run_id,
                "registry_allocation_count": len(registry),
            })
            return segment_result(bootstrap=True, completed=0,
                                  campaign=[])

        if surface_bytes != r_nominal:
            raise RuntimeError(
                f"snapshot injection-surface residency {surface_bytes} bytes "
                f"!= frozen R {r_nominal} of workload {args.workload} -- the "
                "level table must be re-derived (docs/G5_FAULT_MODEL.md §5)")

        anchors = fault_model.load_anchors(args.table)
        anchor_pages = len(anchors)
        print(f"[{run_id}] anchors: {anchor_pages} pages with valid "
              "consensus masks")
        campaign = fault_model.sample_campaign(
            args.level, trials, surface_rows, anchors, seed,
            resident_bytes_expected=r_nominal, workload=args.workload)
        model_failures = verify_work_model(level, campaign)
        if model_failures:
            raise RuntimeError("sampled campaign violates the frozen model: "
                               + "; ".join(model_failures[:10]))
        total_sites = sum(len(sites) for sites in campaign)
        print(f"[{run_id}] sampler: level {args.level} BER {level['ber']:g} "
              f"B={level['bits']} (s,d,t)=({level['s']},{level['d']},"
              f"{level['t']}), {trials} trials, {total_sites} sites")
        fault_model.write_work_csv(work_output, campaign)
        work_detail = {
            "schema": "gpu-m2d.g5.campaign.work.v1",
            "run_id": run_id,
            "device": args.device,
            "level": level,
            "seed": seed,
            "trial_seed_rule": "random.Random(seed * 100003 + trial_index)",
            "trials": trials,
            "resident_bytes": resident_bytes,
            "anchor_pages": anchor_pages,
            "frozen_composition_echo": [level["s"], level["d"],
                                        level["t"]],
            "snapshot_manifest": result.manifest,
            "sites": [site for sites in campaign for site in sites],
        }
        if excludes:
            work_detail["surface_excludes"] = list(excludes)
            work_detail["surface_resident_bytes"] = surface_bytes
        work_detail_output.write_text(json.dumps(
            work_detail, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(work_output, 0o644)
        os.chmod(work_detail_output, 0o644)

        # core provenance both the crash and clean exits need (the merge
        # and the level summary read them from segment extras)
        extra.update({
            "g1_5_run_id": g1_5_run_id,
            "registry_allocation_count": len(registry),
            "snapshot_dir": str(snapshot_dir),
            "snapshot_sha256": result.manifest.get("snapshot_sha256", ""),
            "snapshot_manifest": result.manifest,
            "resident_bytes": resident_bytes,
            "anchor_pages": anchor_pages,
            "total_sites": total_sites,
        })

        # ---- G8-T2 cache sites from THIS process's residency map
        cache_campaign: list[list[dict]] = []
        if cache_mode:
            classes = workload_entry.get("cache_alloc_classes")
            if classes is None:
                raise RuntimeError(f"workload {args.workload} has no "
                                   "cache_alloc_classes (G8-T2c)")
            try:
                rmap = residency_map.load_map(residency_prefix)
            except residency_map.MapError as exc:
                raise RuntimeError(f"residency map refused: {exc}") from exc
            live_bases = {row["allocation_id"]: int(row["gpu_va"], 16)
                          for row in registry}
            images_at_gate = len(read_csv_rows(clean_pass_output,
                                               CLEAN_PASS_FIELDS))
            checks = cache_model.map_self_checks(
                str(residency_prefix), rmap, images=images_at_gate,
                frozen_r=r_nominal, live_bases=live_bases)
            print(f"[{run_id}] G8 residency map: R_eff "
                  f"{rmap.r_eff / 8e6:.3f} MB of {r_nominal / 1e6:.3f} MB, "
                  f"stride {checks['values']['stride']}, in-gap "
                  f"{checks['values']['in_gap_share']:.4%}, locked "
                  f"{checks['values']['locked_share']:.3%}")
            if checks["failures"]:
                raise RuntimeError("G8 residency-map self-check failed "
                                   "(fail-closed): "
                                   + "; ".join(checks["failures"]))
            try:
                n_cache, cache_campaign = cache_model.sample_cache_campaign(
                    rmap, classes, args.cache_ber, trials, seed, live_bases)
            except cache_model.CacheModelError as exc:
                raise RuntimeError(f"cache sampling refused: {exc}") from exc
            cache_failures = cache_model.verify_cache_work(
                rmap, classes, n_cache, cache_campaign)
            if cache_failures:
                raise RuntimeError("sampled cache sites violate the model: "
                                   + "; ".join(cache_failures[:10]))
            cache_model.write_cache_work(cache_work_output, cache_campaign)
            os.chmod(cache_work_output, 0o644)
            print(f"[{run_id}] G8 cache sampler: BER_cache {args.cache_ber:g} "
                  f"({'frozen rho=' + str(fault_model.CACHE_RHO) if args.cache_ber_frozen else 'explicit'}) "
                  f"x R_eff_bits {rmap.r_eff:.0f} -> n_cache {n_cache} per "
                  f"trial, {n_cache * trials} cache sites")
            n_expected = args.cache_ber * rmap.r_eff
            if n_cache == 0:
                # round() keeps the cache count fixed per process like the
                # DRAM B (user decision 2026-10-03); a zero is legitimate
                # (e.g. the 1e-8 rung under a heavy co-tenant) but must be
                # visible, never silent.
                print(f"[{run_id}] WARNING G8 n_cache = 0: expected "
                      f"{n_expected:.3f} cache flips per trial rounds to 0 "
                      "(this process runs DRAM faults only)")
            extra["g8_cache"] = {
                "cache_ber": args.cache_ber,
                "cache_ber_frozen": args.cache_ber_frozen,
                "cache_rho": (fault_model.CACHE_RHO
                              if args.cache_ber_frozen else None),
                "n_cache": n_cache,
                "n_cache_expected": n_expected,
                "r_eff_bits": rmap.r_eff,
                "surface_bits": rmap.meta["surface_bits"],
                "residency_map_json": str(residency_prefix) + "_residency.json",
                "residency_map_bin_sha256": rmap.meta["bin_sha256"],
                "self_checks": checks["values"],
                "cache_work_output": str(cache_work_output),
                "cache_work_sha256": sha256(cache_work_output),
                "cache_sites_total": n_cache * trials,
            }

        release.write_text(f"campaign_ready target_tgid={process.pid}\n",
                           encoding="utf-8")
        os.chmod(release, 0o644)

        finished = drain_until_marker(process, observer,
                                      PASS_MARKER.encode(), captured,
                                      args.timeout_seconds)
        if not finished:
            if process.poll() is None:
                process.terminate()
                raise RuntimeError("runner did not report " + PASS_MARKER +
                                   " (timeout)")
            # The runner PROCESS died mid-campaign. The single recoverable
            # class is the process-fatal fault (restart protocol, user
            # decision 2026-09-28): a flip in TRT create_execution_context
            # control state surfaces as one of the known fatal CUDA
            # signatures (RECOVERABLE_CUDA_SIGNATURES); the runner flushed
            # every COMPLETED trial's rows, so the prefix on disk is
            # verified below and the dying trial becomes PROCESS_FATAL.
            # Every other death shape fails the level.
            for _ in range(20):
                observer.poll(50)
            drain_pipe(process, captured)
            output = captured.decode("utf-8", errors="replace")
            death, reasons = analyze_process_death(output, campaign)
            if death is None:
                raise RuntimeError(
                    "runner died with an unrecoverable failure shape: "
                    + "; ".join(reasons))
            signature = next(sig for sig in RECOVERABLE_CUDA_SIGNATURES
                             if sig in output)
            completed_now = death["completed"]
            prefix = campaign[:completed_now]
            print(f"[{run_id}] PROCESS_FATAL: dying trial "
                  f"{first_trial + death['dying']} ({signature}); "
                  f"verifying the {completed_now} completed trials on disk")
            site_rows = read_csv_rows(site_result_output, SITE_RESULT_FIELDS)
            failures.extend(verify_site_rows(prefix, site_rows, registry))
            trial_rows = read_csv_rows(trial_result_output,
                                       TRIAL_RESULT_FIELDS)
            if cache_mode:
                cache_rows = read_csv_rows(cache_result_output,
                                           CACHE_RESULT_FIELDS)
                failures.extend(verify_cache_rows(cache_campaign, cache_rows,
                                                  trial_rows))
                failures.extend(verify_cache_events(output, cache_rows))
            clean_rows = read_csv_rows(clean_pass_output, CLEAN_PASS_FIELDS)
            images_total = len(clean_rows)
            if images_total == 0:
                failures.append("clean pass CSV is empty")
            else:
                failures.extend(verify_trial_rows(trial_rows, prefix,
                                                  images_total))
                image_rows = read_csv_rows(image_detail_output,
                                           IMAGE_DETAIL_FIELDS)
                failures.extend(verify_image_rows(image_rows, trial_rows,
                                                  clean_rows))
            failures.extend(verify_event_structure(
                output, prefix, site_rows,
                partial_trial=campaign[completed_now]))
            # the dying trial's flips were mid-pass and its GPU state died
            # with the process; the closing ledger/skeleton checks only
            # apply to a clean exit. Observer-level checks still do.
            if registry_output.read_text() != registry_gate_copy.read_text():
                failures.append("allocation registry changed between the "
                                "campaign gate and the death of the run")
            kernel_uuids = sorted({str(row["gpu_uuid"]) for row in
                                   observer.rows
                                   if row["event_type"] == "PTE_HEADER"
                                   and row["gpu_uuid"]})
            if len(kernel_uuids) > 1:
                failures.append(f"multiple kernel GPU UUIDs: {kernel_uuids}")
            elif kernel_uuids and normalize_gpu_uuid(kernel_uuids[0]) != \
                    normalize_gpu_uuid(device_uuid):
                failures.append(f"kernel GPU UUID {kernel_uuids[0]} != "
                                f"device UUID {device_uuid}")
            if len({str(row["address_space_id"]) for row in observer.rows
                    if row["event_type"] in ("MAP_RETURN", "PTE_HEADER")}) > 1:
                failures.append("multiple UVM address-space IDs observed")
            if observer.lost_event_count:
                failures.append(f"lost BPF events: {observer.lost_event_count}")
            if failures:
                raise RuntimeError("crashed-segment prefix verification "
                                   "failed: " + "; ".join(failures[:10]))
            return segment_result(
                completed=completed_now,
                crashed_trial=first_trial + death["dying"],
                campaign=campaign)
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
        expected_skeleton = expected_skeleton_for(args.hold_seconds,
                                                  cache_mode)
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
        if cache_mode:
            cache_rows = read_csv_rows(cache_result_output, CACHE_RESULT_FIELDS)
            failures.extend(verify_cache_rows(cache_campaign, cache_rows,
                                              trial_rows))
            failures.extend(verify_cache_events(output, cache_rows))
            removal_counts: dict[str, int] = {}
            for row in cache_rows:
                removal_counts[row["removal"]] = \
                    removal_counts.get(row["removal"], 0) + 1
            extra.setdefault("g8_cache", {}).update({
                "cache_result_output": str(cache_result_output),
                "cache_result_sha256": sha256(cache_result_output),
                "cache_rows": len(cache_rows),
                "removal_counts": removal_counts,
            })

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
        if len(trial_rows) != trials:
            failures.append(f"trial count {len(trial_rows)} != requested "
                            f"{trials}")

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
        if failures:
            return {**segment_result(passed=False, completed=trials,
                                     campaign=campaign), "fatal": failures}
        return segment_result(passed=True, completed=trials,
                              campaign=campaign)
    except RuntimeError as error:
        return {**segment_result(), "fatal": [f"segment failed closed: "
                                              f"{error}"]}
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


def _segment_details(segments: list[dict]) -> list[dict]:
    return [{
        "segment": seg["segment"],
        "dir": seg["name"],
        "seed": seg["seed"],
        "trials": seg["trials"],
        "first_trial": seg["first_trial"],
        "completed": seg["completed"],
        "crashed_trial": seg["crashed_trial"],
        "g1_5_run_id": seg["extra"].get("g1_5_run_id"),
        "event_output": str(seg["event_output"]),
        "event_output_sha256": (sha256(seg["event_output"])
                                if Path(seg["event_output"]).is_file()
                                else ""),
        "harness_log": str(seg["test_log"]),
        "harness_log_sha256": (sha256(seg["test_log"])
                               if Path(seg["test_log"]).is_file() else ""),
        "snapshot_dir": seg["extra"].get("snapshot_dir"),
        "snapshot_sha256": seg["extra"].get("snapshot_sha256", ""),
        "total_sites": seg["extra"].get("total_sites"),
        "campaign_window_ns": seg["extra"].get("campaign_window_ns"),
        # G8 cache mode: this segment's own residency map, self-check
        # values and n_cache (per process -- may differ between segments)
        "g8_cache": seg["extra"].get("g8_cache"),
    } for seg in segments]


def run_once(args: argparse.Namespace) -> int:
    uid, gid = drop_to_invoking_user()
    contract, contract_hash = load_and_validate_contract(
        args.contract.resolve(), DEFAULT_OPEN_SOURCE_REPO)
    if not args.runner.is_file() or not os.access(args.runner, os.X_OK):
        raise RuntimeError(f"runner missing or not executable: {args.runner}")
    if args.trials < 1:
        raise RuntimeError("--trials must be >= 1")
    level: dict | None = None
    if not args.bootstrap:
        # a real campaign runs fail-closed against the frozen level table;
        # the bootstrap run exists precisely to MEASURE R for a workload
        # whose table is not frozen yet (empty levels, R placeholder 0)
        fault_model.assert_frozen_levels(args.workload)
        level = fault_model.level_by_name(args.level, args.workload)
    trt_runtime_dir = Path(os.environ.get(
        "GPU_M2D_REMU_ROOT", "/data1/luojx/REMU")) / \
        ".local/deps/tensorrt-8.6.1/tensorrt_libs"
    for required in (args.engine, args.sample_csv,
                     trt_runtime_dir / "libnvinfer.so.8",
                     args.table / "gddr_seed_table.csv",
                     args.table / "bank_classes.csv",
                     args.table / "page_anchors.csv"):
        if not required.is_file():
            raise RuntimeError(f"missing required asset: {required}")

    cotenancy = cotenancy_snapshot(args.device)
    device_uuid = query_device_uuid(args.device)
    # shared across levels, workloads (same CSV + preprocessing spec), and
    # every restart segment: one 5.6 GiB file populates once, then each
    # relaunched segment loads it in seconds instead of re-preprocessing
    # 10K images on the host (~3 min CPU) before the pre-allocation gate
    if args.image_cache_dir is None:
        args.image_cache_dir = args.output_root.parent / "image_cache"
    print(f"image cache dir: {args.image_cache_dir}")
    level_tag = args.level if args.level else "bootstrap"
    level_dir = args.output_root / \
        f"run_{level_tag}_gpu{args.device}_{time.time_ns()}"
    level_dir.mkdir(parents=True, exist_ok=True)
    os.chown(level_dir, uid, gid)
    started_ns = time.time_ns()

    if args.bootstrap:
        result = execute_segment(args, contract, contract_hash, level,
                                 level_dir, 0, args.trials, args.seed, 0,
                                 uid, gid, cotenancy, device_uuid)
        return finalize(result.get("fatal", []), level_dir,
                        result["test_log"], result["event_output"],
                        result["observer_info"], args, level_dir.name,
                        device_uuid, started_ns, result["extra"], uid, gid,
                        status_override=("G5_BOOTSTRAP_MEASURED"
                                         if not result.get("fatal")
                                         else None))

    # ---- campaign: the restart-protocol segment loop. Every iteration
    # runs one full gated segment (fresh observer + snapshot + sampling)
    # for the REMAINING trial slots; a process-fatal trial consumes its
    # slot as PROCESS_FATAL and the loop relaunches. A crash can never
    # loop forever: each dying trial consumes exactly one slot, and any
    # death without a dying trial is fatal.
    segments: list[dict] = []
    completed = 0
    crashes = 0
    while completed + crashes < args.trials:
        segment_index = len(segments)
        if segment_index > args.trials + 8:
            return finalize(
                ["restart budget exhausted: more segments than trials + 8; "
                 "the crash rate exceeds what the protocol can absorb"],
                level_dir, level_dir / "harness.log",
                segments[-1]["event_output"] if segments else
                level_dir / "events.csv",
                segments[-1]["observer_info"] if segments else {},
                args, level_dir.name, device_uuid, started_ns,
                {"segments": len(segments),
                 "segment_details": _segment_details(segments)},
                uid, gid)
        seg = execute_segment(args, contract, contract_hash, level,
                              level_dir, segment_index,
                              args.trials - completed - crashes,
                              args.seed + segment_index,
                              completed + crashes, uid, gid, cotenancy,
                              device_uuid)
        if seg.get("fatal"):
            return finalize(seg["fatal"], level_dir, seg["test_log"],
                            seg["event_output"], seg["observer_info"],
                            args, level_dir.name, device_uuid, started_ns,
                            {"segments": len(segments) + 1,
                             "segment_details": _segment_details(
                                 segments + [seg]),
                             "restart_protocol_used": bool(segments)},
                            uid, gid)
        segments.append(seg)
        completed += seg["completed"]
        if seg["crashed_trial"] is not None:
            crashes += 1
            print(f"restart: {args.trials - completed - crashes} trial "
                  f"slot(s) remain ({completed} completed + {crashes} "
                  "PROCESS_FATAL)")

    images_total = len(read_csv_rows(segments[0]["dir"] /
                                     "g1_5_g5_clean_pass.csv",
                                     CLEAN_PASS_FIELDS))
    merge = merge_campaign_outputs(level_dir, segments, args, images_total)
    trial_rows = merge["trial_rows"]
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
    fatal_trials = [seg["crashed_trial"] for seg in segments
                    if seg["crashed_trial"] is not None]

    first = segments[0]["extra"]
    site_output = level_dir / "g1_5_g5_site_result.csv"
    trial_output = level_dir / "g1_5_g5_trial_result.csv"
    image_output = level_dir / "g1_5_g5_image_detail.csv"
    work_output = level_dir / "work.csv"
    clean_output = level_dir / "g1_5_g5_clean_pass.csv"
    extra: dict[str, Any] = {
        "segments": len(segments),
        "restart_protocol_used": bool(fatal_trials),
        "segment_details": _segment_details(segments),
        "process_fatal_trials": fatal_trials,
        "process_fatal_count": len(fatal_trials),
        "trials_completed": completed,
        "images_per_trial": images_total,
        "trial_outcome_histogram": outcome_histogram,
        "image_totals": image_totals,
        "restore_totals": restore_totals,
        "g1_5_run_id": first.get("g1_5_run_id"),
        "registry_allocation_count": first.get("registry_allocation_count"),
        "snapshot_dir": first.get("snapshot_dir"),
        "snapshot_sha256": first.get("snapshot_sha256", ""),
        "snapshot_manifest": first.get("snapshot_manifest"),
        "resident_bytes": first.get("resident_bytes"),
        "anchor_pages": first.get("anchor_pages"),
        # sites of the level's trials exactly (trials x bits); a crashed
        # segment's never-run tail was re-sampled later and is NOT added
        "total_sites": args.trials * level["bits"],
        "surface_excludes": first.get("surface_excludes", []),
        # G8 cache mode summary (per-segment detail in segment_details)
        "g8_cache_ber": getattr(args, "cache_ber", None),
        "g8_cache_ber_frozen": getattr(args, "cache_ber_frozen", False)
        if getattr(args, "cache_ber", None) is not None else None,
        "g8_cache_rho": fault_model.CACHE_RHO
        if getattr(args, "cache_ber_frozen", False) else None,
        "g8_n_cache_per_segment": [
            (seg["extra"].get("g8_cache") or {}).get("n_cache")
            for seg in segments] if getattr(args, "cache_ber", None)
        is not None else None,
        "surface_resident_bytes": first.get("surface_resident_bytes"),
        "work_output": str(work_output),
        "work_sha256": sha256(work_output),
        "site_result_output": str(site_output),
        "site_result_sha256": sha256(site_output),
        "trial_result_output": str(trial_output),
        "trial_result_sha256": sha256(trial_output),
        "image_detail_output": str(image_output),
        "image_detail_sha256": sha256(image_output),
        "clean_pass_output": str(clean_output),
        "clean_pass_sha256": sha256(clean_output),
        "map_output": str(level_dir / "gpu_va_pa_map.csv"),
        "map_output_sha256": sha256_if_present(level_dir / "gpu_va_pa_map.csv"),
        "map_gate_output": str(level_dir / "gpu_va_pa_map_gate.csv"),
        "map_gate_output_sha256": sha256_if_present(
            level_dir / "gpu_va_pa_map_gate.csv"),
        "contract_sha256": contract_hash,
        "nvidia_module_sha256": contract["nvidia_module_sha256"],
        "nvidia_uvm_module_sha256": contract["nvidia_uvm_module_sha256"],
    }
    extra.update({key: value for key, value in first.items()
                  if key in ("cotenancy_at_start", "runner_passthrough")})
    return finalize([], level_dir, level_dir / "harness.log",
                    segments[0]["event_output"],
                    segments[0]["observer_info"], args, level_dir.name,
                    device_uuid, started_ns, extra, uid, gid)


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

    # --- restart protocol: process-death classification
    def trial_event_lines(sites) -> str:
        out = []
        trial = str(sites[0]["trial_index"])
        out.append(f"GPU_M2D_EVENT,event=TRIAL_BEGIN,trial_index={trial},"
                   "pid=1,tgid=1")
        for site in sites:
            out.append(f"GPU_M2D_EVENT,event=SITE_FLIPPED,trial_index="
                       f"{trial},target_id={site['target_id']},pid=1,"
                       "tgid=1")
        out.append(f"GPU_M2D_EVENT,event=TRIAL_INJECTED_END,trial_index="
                   f"{trial},pid=1,tgid=1")
        for site in reversed(sites):
            out.append(f"GPU_M2D_EVENT,event=SITE_RESTORED,trial_index="
                       f"{trial},target_id={site['target_id']},pid=1,"
                       "tgid=1")
        out.append(f"GPU_M2D_EVENT,event=TRIAL_SANITY_BEGIN,trial_index="
                   f"{trial},pid=1,tgid=1")
        out.append(f"GPU_M2D_EVENT,event=TRIAL_SANITY_END,trial_index="
                   f"{trial},pid=1,tgid=1")
        out.append(f"GPU_M2D_EVENT,event=TRIAL_END,trial_index={trial},"
                   "pid=1,tgid=1")
        return "\n".join(out) + "\n"

    ima_tail = ("GPU_M2D_G1_5_FAIL: synchronize inference stream failed: "
                "an illegal memory access was encountered\n")

    def dying_trial_lines(trial: int, sites) -> str:
        return ("GPU_M2D_EVENT,event=TRIAL_BEGIN,"
                f"trial_index={trial},pid=1,tgid=1\n"
                + "".join(f"GPU_M2D_EVENT,event=SITE_FLIPPED,"
                          f"trial_index={trial},"
                          f"target_id={site['target_id']},pid=1,tgid=1\n"
                          for site in sites))

    crashed_output = (trial_event_lines(campaign[0])
                      + trial_event_lines(campaign[1])
                      + dying_trial_lines(2, campaign[2])
                      + ima_tail)
    death, reasons = analyze_process_death(crashed_output, campaign)
    assert death == {"completed": 2, "dying": 2}, (death, reasons)
    # the swin-L5 death class (2026-09-28): a DIFFERENT fatal CUDA
    # string with the same recoverable shape is restartable too
    addr_tail = ("GPU_M2D_G1_5_FAIL: synchronize inference stream "
                 "failed: operation not supported on global/shared "
                 "address space\n")
    death, reasons = analyze_process_death(
        trial_event_lines(campaign[0]) + trial_event_lines(campaign[1])
        + dying_trial_lines(2, campaign[2]) + addr_tail, campaign)
    assert death == {"completed": 2, "dying": 2}, (death, reasons)
    # no known CUDA death signature -> unrecoverable
    death, reasons = analyze_process_death(
        crashed_output.replace("illegal memory access", "boom"), campaign)
    assert death is None and any("signature" in r for r in reasons)
    # anything after the dying trial's flips (it survived the injected
    # pass) -> unrecoverable
    last_flip = (f"target_id={campaign[2][-1]['target_id']},pid=1,tgid=1\n")
    survivor = crashed_output.replace(
        last_flip + ima_tail,
        last_flip + "GPU_M2D_EVENT,event=TRIAL_INJECTED_END,trial_index=2,"
        "pid=1,tgid=1\n" + ima_tail)
    death, reasons = analyze_process_death(survivor, campaign)
    assert death is None and any("flip set" in r for r in reasons), reasons
    # incomplete flip set before death -> unrecoverable
    partial_flips = (trial_event_lines(campaign[0])
                     + trial_event_lines(campaign[1])
                     + dying_trial_lines(2, campaign[2][:4])
                     + ima_tail)
    death, reasons = analyze_process_death(partial_flips, campaign)
    assert death is None, (death, reasons)
    # the FIRST trial dying (zero completed) is recoverable
    death, reasons = analyze_process_death(
        dying_trial_lines(0, campaign[0]) + ima_tail, campaign)
    assert death == {"completed": 0, "dying": 0}, (death, reasons)
    # a stream with no open trial (clean end) has no dying trial
    death, reasons = analyze_process_death(
        "".join(trial_event_lines(sites) for sites in campaign) + ima_tail,
        campaign)
    assert death is None, death

    # --- restart protocol: event structure with the dying trial's
    # partial sequence appended. `lines` carries the full CSV-agreeing
    # stream of the 3-trial campaign; its first two trials are the dead
    # segment's completed prefix, and only their site rows exist on disk
    # (the dying trial's were never flushed, so the agreement check must
    # skip them).
    per_trial_lines = 2 * len(campaign[0]) + 5  # flips + restores + 5 brackets
    prefix_site_rows = good_sites[:2 * len(campaign[0])]
    dying_stream = ("\n".join(lines[:2 * per_trial_lines]) + "\n"
                    + dying_trial_lines(2, campaign[2]))
    assert verify_event_structure(dying_stream, campaign[:2],
                                   prefix_site_rows,
                                   partial_trial=campaign[2]) == []
    swapped = list(campaign[2])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert verify_event_structure(dying_stream, campaign[:2],
                                  prefix_site_rows, partial_trial=swapped)

    # --- restart protocol: global-slot renumbering
    row = ["r-x", "0", "4", "0", "1", "t004-e000-s01", "trt-internal-0",
           "label", "12", "3", "8", "0x10", "0x10", "85", "93", "1", "1",
           "1", "exact"]
    shifted = shift_trial_fields(row, 78, 2, 5)
    assert shifted[2] == "82" and shifted[5] == "t082-e000-s01"
    assert row[2] == "4"  # the input row is not mutated

    # --- restart protocol: two-segment merge (one crash) + single
    # clean segment hardlink
    campaign4 = fault_model.sample_campaign("L2", 4, rows_fixture,
                                            anchors_fixture, seed=13)
    campaign5 = fault_model.sample_campaign("L2", 5, rows_fixture,
                                            anchors_fixture, seed=17)
    assert verify_work_model(level, campaign4) == []
    assert verify_work_model(level, campaign5) == []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        images_total = 2

        def site_csv_row(trial, site):
            return ["r-x", "0", str(trial), "0", "0", site["target_id"],
                    "trt-internal-0", "TENSORRT_INTERNAL_UNKNOWN", "0",
                    "0", "1", "0x10", "0x10", "85", "86", "1", "1", "1",
                    "exact"]

        def trial_csv_row(trial):
            return ["r-x", "0", str(trial), str(level["bits"]),
                    str(level["s"] + level["d"] + level["t"]),
                    str(images_total), str(images_total),
                    str(images_total), "0", "0", "0", "BENIGN", "7",
                    "0.5", "1", "1", "0", "0", "0"]

        def image_csv_row(trial, image):
            return ["r-x", "0", str(trial), str(image), "1", "7", "7",
                    "0.5", "0.5", "IMAGE_BENIGN"]

        def work_csv_row(trial, site):
            return [str(trial), "0", "0", site["target_id"],
                    "trt-internal-0", "0", "0", "0x10"]

        def fill_segment(seg_dir, sampled, completed, sites_per_trial):
            # completed trials' rows only; work.csv carries every SAMPLED
            # trial's sites, the dying one included
            seg_dir.mkdir(parents=True)
            site_rows, trial_rows, image_rows = [], [], []
            for trial in range(completed):
                site_rows += [site_csv_row(trial, site)
                              for site in sites_per_trial[trial]]
                trial_rows.append(trial_csv_row(trial))
                image_rows += [image_csv_row(trial, image)
                               for image in range(images_total)]
            write_merged_csv(seg_dir / "g1_5_g5_site_result.csv",
                             SITE_RESULT_FIELDS, site_rows)
            write_merged_csv(seg_dir / "g1_5_g5_trial_result.csv",
                             TRIAL_RESULT_FIELDS, trial_rows)
            write_merged_csv(seg_dir / "g1_5_g5_image_detail.csv",
                             IMAGE_DETAIL_FIELDS, image_rows)
            write_merged_csv(seg_dir / "work.csv", fault_model.WORK_FIELDS,
                             [work_csv_row(trial, site)
                              for trial in range(sampled)
                              for site in sites_per_trial[trial]],
                             comment="work")

        level_dir = root / "run_L2_gpu0_test"
        level_dir.mkdir()
        seg0_dir = level_dir / "segment_000"
        seg1_dir = level_dir / "segment_001"
        # segment 0 was ASSIGNED 5 slots (work.csv lists all 5) but died
        # on local trial 2: consumed = 2 completed + 1 dying, locals 3-4
        # were re-sampled by segment 1 and must be dropped at merge
        fill_segment(seg0_dir, 5, 2, campaign5)
        fill_segment(seg1_dir, 4, 4, campaign4)
        for seg_dir in (seg0_dir, seg1_dir):
            write_merged_csv(seg_dir / "g1_5_g5_clean_pass.csv",
                             CLEAN_PASS_FIELDS,
                             [["0", "a.jpg", "7", "7", "0.5"],
                              ["1", "b.jpg", "7", "7", "0.5"]])
        # a crashed segment wrote only its GATE map (the FINAL map needs
        # a clean exit -- the closing ledger never ran); the clean
        # segment wrote both. Regression for the 2026-09-28 L2 chain
        # abort: the level's final map must come from the first CLEAN
        # segment, and hashing the level's maps for the summary extras
        # must never raise on the crashed-first-segment shape.
        (seg0_dir / "gpu_va_pa_map_gate.csv").write_text("gate-map-0\n")
        (seg1_dir / "gpu_va_pa_map_gate.csv").write_text("gate-map-1\n")
        (seg1_dir / "gpu_va_pa_map.csv").write_text("final-map-1\n")
        merge_args = argparse.Namespace(trials=7, device=0, seed=7)
        segments = [
            {"segment": 0, "dir": seg0_dir, "name": "s0", "seed": 7,
             "trials": 5, "first_trial": 0, "completed": 2,
             "crashed_trial": 2, "campaign": campaign5,
             "extra": {"level": level}},
            {"segment": 1, "dir": seg1_dir, "name": "s1", "seed": 8,
             "trials": 4, "first_trial": 3, "completed": 4,
             "crashed_trial": None, "campaign": campaign4,
             "extra": {"level": level}},
        ]
        merge = merge_campaign_outputs(level_dir, segments, merge_args,
                                       images_total)
        assert merge["mode"] == "merged"
        assert len(merge["trial_rows"]) == 6  # 2 + 4 completed
        trial_ids = sorted(int(row["trial_index"])
                           for row in merge["trial_rows"])
        assert trial_ids == [0, 1, 3, 4, 5, 6]  # slot 2 = PROCESS_FATAL
        work_rows = read_result_csv(level_dir / "work.csv",
                                    fault_model.WORK_FIELDS)
        assert len(work_rows) == 7 * level["bits"]
        assert {int(row[0]) for row in work_rows} == set(range(7))
        site_rows = read_result_csv(level_dir / "g1_5_g5_site_result.csv",
                                    SITE_RESULT_FIELDS)
        assert {row[5].split("-")[0] for row in site_rows} == \
            {f"t{i:03d}" for i in [0, 1, 3, 4, 5, 6]}
        assert (level_dir / "gpu_va_pa_map.csv").read_text() == \
            "final-map-1\n"  # donor = first segment that HAS the file
        assert (level_dir / "gpu_va_pa_map_gate.csv").read_text() == \
            "gate-map-0\n"  # every segment has the gate map -> segment 0
        assert sha256_if_present(level_dir / "gpu_va_pa_map.csv") == \
            sha256(seg1_dir / "gpu_va_pa_map.csv")
        assert sha256_if_present(level_dir / "never-written.csv") == ""
        detail = json.loads((level_dir / "work_detail.json").read_text())
        assert detail["trials"] == 7 and len(detail["segments"]) == 2
        assert len(detail["sites"]) == 7 * level["bits"]

        # single clean segment -> hardlink mode, identical content
        solo_dir = root / "run_L2_gpu0_solo"
        solo_dir.mkdir()
        solo_seg_dir = solo_dir / "segment_000"
        fill_segment(solo_seg_dir, 3, 3, campaign)
        write_merged_csv(solo_seg_dir / "g1_5_g5_clean_pass.csv",
                         CLEAN_PASS_FIELDS,
                         [["0", "a.jpg", "7", "7", "0.5"],
                          ["1", "b.jpg", "7", "7", "0.5"]])
        solo = [{"segment": 0, "dir": solo_seg_dir, "name": "s0",
                 "seed": 7, "trials": 3, "first_trial": 0, "completed": 3,
                 "crashed_trial": None, "campaign": campaign,
                 "extra": {"level": level}}]
        merge = merge_campaign_outputs(solo_dir, solo, merge_args,
                                       images_total)
        assert merge["mode"] == "hardlink"
        assert len(merge["trial_rows"]) == 3
        assert (solo_dir / "g1_5_g5_site_result.csv").read_text() == \
            (solo_seg_dir / "g1_5_g5_site_result.csv").read_text()

    # ---- G8-T2 cache verifiers -------------------------------------------
    def cache_site(trial, k, cls, start, last, byte=10, bit=3):
        return {"trial_index": trial, "cache_index": k,
                "target_id": f"c{trial:03d}-{k}",
                "allocation_id": "trt-internal-0", "byte_offset": byte,
                "bit_in_byte": bit, "expected_gpu_va": 0x1000 + byte,
                "start_image": start, "last_image": last, "cache_class": cls}

    def cache_row(site, applied=True, before=0x40, removal=None,
                  removal_image=None, remove_before=None, remove_after=None):
        mask = 1 << site["bit_in_byte"]
        after = before ^ mask
        if removal is None:
            removal = "re_xor" if site["cache_class"] == "read_only" \
                else "restored"
        return {
            "trial_index": str(site["trial_index"]),
            "cache_index": str(site["cache_index"]),
            "target_id": site["target_id"],
            "allocation_id": site["allocation_id"],
            "byte_offset": str(site["byte_offset"]),
            "bit_in_byte": str(site["bit_in_byte"]),
            "xor_mask": str(mask),
            "gpu_va": hex(site["expected_gpu_va"]) if applied else "NA",
            "expected_gpu_va": hex(site["expected_gpu_va"]),
            "cache_class": site["cache_class"],
            "start_image": str(site["start_image"]),
            "last_image": str(site["last_image"]),
            "applied": "1" if applied else "0",
            "apply_image": str(site["start_image"] if applied else 0),
            "before": str(before if applied else 0),
            "after": str(after if applied else 0),
            "removal": removal if applied else "not_applied",
            "removal_image": str(site["last_image"] if removal_image is None
                                 else removal_image),
            "remove_before": str(after if remove_before is None
                                 else remove_before),
            "remove_after": str(before if remove_after is None
                                else remove_after)}

    ro = cache_site(0, 0, "read_only", 2, 9)
    ew = cache_site(0, 1, "engine_written", 4, 4, byte=11)
    ew_over = cache_site(0, 2, "engine_written", 5, 5, byte=12)
    cache_camp = [[ro, ew, ew_over]]
    good_rows = [cache_row(ro), cache_row(ew),
                 cache_row(ew_over, removal="overwritten", remove_before=7,
                           remove_after=7)]
    full_trial = [{"trial_index": "0", "images_evaluated": "10",
                   "injected_outcome": "SDC_NUMERIC"}]
    assert verify_cache_rows(cache_camp, good_rows, full_trial) == []
    # DUE at image 4: the read-only flip is removed at 4, the one starting
    # at 5 was never applied
    due_trial = [{"trial_index": "0", "images_evaluated": "5",
                  "injected_outcome": "DUE_INVALID_OUTPUT"}]
    due_rows = [cache_row(ro, removal_image=4), cache_row(ew),
                cache_row(ew_over, applied=False)]
    assert verify_cache_rows(cache_camp, due_rows, due_trial) == []
    # the same early removal without a DUE is refused
    assert verify_cache_rows(cache_camp, due_rows, full_trial)
    # read-only removal not back to the pre-cache value is refused
    bad = [cache_row(ro, remove_after=0x41), cache_row(ew),
           good_rows[2]]
    assert any("read-only" in f for f in
               verify_cache_rows(cache_camp, bad, full_trial))
    # 'overwritten' while the byte still held the flipped value is refused
    bad = [good_rows[0], good_rows[1],
           cache_row(ew_over, removal="overwritten")]
    assert any("overwritten" in f for f in
               verify_cache_rows(cache_camp, bad, full_trial))
    # a missing row is refused
    assert verify_cache_rows(cache_camp, good_rows[:2], full_trial)
    # G8-T4 fix: two flips on different bits of ONE byte with overlapping
    # lifetimes (the L7 trial-18 case: bit 4 from image 2, bit 2 from
    # image 5, both removed at 9 in application order, i.e. FIFO)
    sa = cache_site(0, 0, "read_only", 2, 9, byte=20, bit=4)
    sb = cache_site(0, 1, "read_only", 5, 9, byte=20, bit=2)
    pair = [[sa, sb]]
    fifo = [cache_row(sa, before=238, remove_before=250, remove_after=234),
            cache_row(sb, before=254, remove_before=234, remove_after=238)]
    assert verify_cache_rows(pair, fifo, full_trial) == []
    # LIFO (b removed first, at 7) is equally valid
    sb7 = dict(sb, last_image=7)
    lifo = [cache_row(sa, before=238, remove_before=254, remove_after=238),
            cache_row(sb7, before=254, remove_before=250, remove_after=254)]
    assert verify_cache_rows([[sa, sb7]], lifo, full_trial) == []
    # a corrupted shared byte is still refused
    bad = [fifo[0], cache_row(sb, before=254, remove_before=235,
                              remove_after=239)]
    assert any("read-only" in f for f in
               verify_cache_rows(pair, bad, full_trial))
    # ... and so is a second flip whose 'before' ignores the first one
    bad = [fifo[0], cache_row(sb, before=238, remove_before=234,
                              remove_after=238 ^ 4)]
    assert verify_cache_rows(pair, bad, full_trial)
    # engine-written pair in one image, both conditionally restored
    ea = cache_site(0, 0, "engine_written", 4, 4, byte=30, bit=1)
    eb = cache_site(0, 1, "engine_written", 4, 4, byte=30, bit=5)
    e_rows = [cache_row(ea, before=0, remove_before=34, remove_after=32),
              cache_row(eb, before=2, remove_before=32, remove_after=0)]
    assert verify_cache_rows([[ea, eb]], e_rows, full_trial) == []
    # the engine rewrote that byte: both 'overwritten' is valid, a restore
    # after the rewrite is refused
    e_over = [cache_row(ea, before=0, removal="overwritten",
                        remove_before=99, remove_after=99),
              cache_row(eb, before=2, removal="overwritten",
                        remove_before=99, remove_after=99)]
    assert verify_cache_rows([[ea, eb]], e_over, full_trial) == []
    e_bad = [e_over[0], cache_row(eb, before=2, remove_before=99,
                                  remove_after=99 ^ 32)]
    assert verify_cache_rows([[ea, eb]], e_bad, full_trial)
    # events must match the applied rows one-to-one
    events = "".join(
        f"GPU_M2D_EVENT,event=CACHE_FLIPPED,trial_index=0,cache_index="
        f"{r['cache_index']},before={r['before']},after={r['after']}\n"
        f"GPU_M2D_EVENT,event=CACHE_REMOVED,trial_index=0,cache_index="
        f"{r['cache_index']},before={r['remove_before']},"
        f"after={r['remove_after']}\n" for r in good_rows)
    assert verify_cache_events(events, good_rows) == []
    assert verify_cache_events(events.replace("before=64", "before=65", 1),
                               good_rows)
    # skeleton: the residency pass sits between clean pass and gate
    sk = expected_skeleton_for(1, True)
    at = sk.index("CLEAN_PASS_END")
    assert sk[at + 1:at + 3] == G8_SKELETON_EVENTS
    assert expected_skeleton_for(1, False) == CAMPAIGN_SKELETON
    # restart merge renumbers cache target ids like DRAM ones
    assert shift_trial_fields(["2", "c002-7"], 5, 0, 1) == ["7", "c007-7"]

    print("g5 campaign self-test: PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.level is None and not args.bootstrap:
        raise SystemExit("run_g5_campaign.py: error: --level is required "
                         "for a campaign (one frozen BER level per run); "
                         "--bootstrap is the only level-less mode")
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
