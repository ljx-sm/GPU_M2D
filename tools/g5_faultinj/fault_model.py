#!/usr/bin/env python3
"""GPU_M2D G5 fault model: frozen parameter table + campaign site sampler.

Implements docs/G5_FAULT_MODEL.md (G5-T1, user-confirmed 2026-09-21):

  - event mix SBU 60% / MCU 40% (shares of flip EVENTS), MCU 2-bit:3-bit
    = 1:1 by event count;
  - no strict adjacency: horizontal = same-row inside ONE 256 B PA block
    (measured column region: in-page PA bits 0-7 never leave bank/row);
    vertical = same bank / different row via the G3 table-v4 per-page
    anchor consensus masks (measured; mask < 2 MiB so the mate stays on
    the same resident PA page); L = same-row pair + one anchor mate; the
    four L orientations share ONE sampling distribution, the orientation
    is kept as a label;
  - BER levels 1e-8 .. 1e-4 with B = round(BER * R) and frozen
    compositions (s, d, t) = exhaustive least-squares fit to the 60/20/20
    event shares (ties toward more SBU, then more 2-bit). L1-L5 are the
    user-confirmed 2026-09-21 core; L6-L9 (5e-6 .. 1e-4) are the
    post-campaign extension ladder, same rule, run single-card per the
    verified L1-L5 no-card-effect result (docs/G5_FAULT_MODEL.md §5);
  - within a level B and (s, d, t) are frozen; only positions randomize
    across the 100 trials (SBU byte/bit, MCU base, anchor choice,
    intra-block offsets, orientation label, per-site bits).

Every site carries provenance labels; nothing below 256 B granularity is
claimed (column relation inside a block is unknown; row distance is
unknown). Sites must land inside byte-residency intervals of the run's
G4 snapshot (non-resident cells are unreachable -> BENIGN by construction,
so they are never sampled). Fail-closed: a site that cannot be placed
after bounded retries refuses the trial (SamplingError).

Pure stdlib; no bcc import (offline analysis module, self-testable).
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# frozen level tables, one per workload (docs/G5_FAULT_MODEL.md §5 for the
# G5 table; each G7 workload's R is MEASURED by a bootstrap run of that
# engine and only then frozen here -- the campaign residency guard refuses
# any live snapshot that disagrees with the frozen R)
# ---------------------------------------------------------------------------

WORKLOADS = {
    # G5: ResNet-50/RESISC45 implicit-engine campaign (machine-verified
    # 2026-09-21)
    "g5_resisc45_resnet50": {
        "resident_bytes_nominal": 26_428_428,
        "levels": [
            # core ladder (user-confirmed 2026-09-21; docs/G5_FAULT_MODEL.md §5)
            {"level": "L1", "ber": 1e-8, "bits": 2, "s": 2, "d": 0, "t": 0},
            {"level": "L2", "ber": 5e-8, "bits": 11, "s": 4, "d": 2, "t": 1},
            {"level": "L3", "ber": 1e-7, "bits": 21, "s": 8, "d": 2, "t": 3},
            {"level": "L4", "ber": 5e-7, "bits": 106, "s": 41, "d": 13,
             "t": 13},
            {"level": "L5", "ber": 1e-6, "bits": 211, "s": 79, "d": 27,
             "t": 26},
            # extension ladder (added 2026-09-21 after the L1-L5 campaign;
            # same derivation rule, literals produced by the EXHAUSTIVE
            # solver; run single-card per the verified L1-L5 no-card-effect
            # result)
            {"level": "L6", "ber": 5e-6, "bits": 1057, "s": 397, "d": 132,
             "t": 132},
            {"level": "L7", "ber": 1e-5, "bits": 2114, "s": 794, "d": 264,
             "t": 264},
            {"level": "L8", "ber": 5e-5, "bits": 10571, "s": 3964, "d": 1322,
             "t": 1321},
            {"level": "L9", "ber": 1e-4, "bits": 21143, "s": 7928, "d": 2643,
             "t": 2643},
        ],
    },
    # G7: ResNet-50/ImageNet-1K explicit-engine campaign workload. R was
    # MEASURED by the bootstrap run
    # artifacts/g7/campaign/run_bootstrap_gpu0_1790260933358547193 (2026-09-24,
    # GPU 0, engine sha256 0368bffd..., clean pass 7850/10000 = 78.50% --
    # bit-identical to the python INT8 eval) and the five user-selected BER
    # levels (1e-7 .. 1e-5, 2026-09-24) derived from it; all B < 3000 so the
    # compositions are exhaustive-solver literals.
    "g7_imagenet1k_resnet50": {
        "resident_bytes_nominal": 34_959_884,
        "levels": [
            {"level": "L1", "ber": 1e-7, "bits": 28, "s": 11, "d": 4,
             "t": 3},
            {"level": "L2", "ber": 5e-7, "bits": 140, "s": 53, "d": 18,
             "t": 17},
            {"level": "L3", "ber": 1e-6, "bits": 280, "s": 105, "d": 35,
             "t": 35},
            {"level": "L4", "ber": 5e-6, "bits": 1398, "s": 523, "d": 175,
             "t": 175},
            {"level": "L5", "ber": 1e-5, "bits": 2797, "s": 1049, "d": 349,
             "t": 350},
        ],
    },
    # G7-v2: ResNet-50/ImageNet-1K on the HEAD-QUANTIZED v2 engine
    # (user decision 2026-09-24: every weighted op INT8 per-channel,
    # including the classifier-head Gemm; quantizer wrapper
    # tools/g7_prep/quantize_g7_qdq_head.py). The BER ladder is the
    # G5 core ladder for a directly comparable curve.
    # R was MEASURED by the bootstrap run
    # artifacts/g7/campaign/run_bootstrap_gpu0_1790310201539047669
    # (engine sha256 cc516d3afcda...,
    # snapshot allocations=7, pa_pages=16, rows=20);
    # L1-L5 frozen by tools/g7_prep/freeze_g7v2_levels.py.
    # EXTENDED 2026-09-25 (user decision) with L6 3e-6, L7 5e-6,
    # L8 7e-6, L9 1e-5 by tools/g7_prep/extend_g7v2_levels.py,
    # derived from the SAME R (engine and bootstrap above still
    # match -- no re-bootstrap). All B < 3000 so every literal is
    # an exhaustive-solver literal.
    "g7v2_imagenet1k_resnet50": {
        "resident_bytes_nominal": 28_832_268,
        "levels": [
            {"level": "L1", "ber": 1e-08, "bits": 2, "s": 2, "d": 0, "t": 0},
            {"level": "L2", "ber": 5e-08, "bits": 12, "s": 5, "d": 2, "t": 1},
            {"level": "L3", "ber": 1e-07, "bits": 23, "s": 8, "d": 3, "t": 3},
            {"level": "L4", "ber": 5e-07, "bits": 115, "s": 43, "d": 15,
             "t": 14},
            {"level": "L5", "ber": 1e-06, "bits": 231, "s": 86, "d": 29,
             "t": 29},
            {"level": "L6", "ber": 3e-06, "bits": 692, "s": 260, "d": 87,
             "t": 86},
            {"level": "L7", "ber": 5e-06, "bits": 1153, "s": 433, "d": 144,
             "t": 144},
            {"level": "L8", "ber": 7e-06, "bits": 1615, "s": 605, "d": 202,
             "t": 202},
            {"level": "L9", "ber": 1e-05, "bits": 2307, "s": 865, "d": 289,
             "t": 288},
        ],
    },
}

DEFAULT_WORKLOAD = "g5_resisc45_resnet50"

# Back-compat aliases: offline analysis imports these module attributes.
RESIDENT_BYTES_NOMINAL = WORKLOADS[DEFAULT_WORKLOAD]["resident_bytes_nominal"]
R_BITS = RESIDENT_BYTES_NOMINAL * 8  # 211,427,424 bits
LEVELS = WORKLOADS[DEFAULT_WORKLOAD]["levels"]

EVENT_SHARE_TARGET = (0.60, 0.20, 0.20)  # SBU / 2-bit MCU / 3-bit MCU

# Composition solving: exhaustive below EXHAUSTIVE_BITS_LIMIT, windowed
# above it (the O(B^2) exhaustive grid costs ~41 s at B=21143 and runs at
# every campaign start through assert_frozen_levels). The window centers
# on the analytic 60/20/20 optimum (s = 0.375B, d = t = 0.125B). No
# unsoundness is possible: the frozen literals for L8/L9 were produced by
# the EXHAUSTIVE solver, so a windowed/exhaustive disagreement at assert
# time raises ModelError and refuses the campaign (fail-closed).
# self_test cross-checks windowed == exhaustive on sampled bit counts.
EXHAUSTIVE_BITS_LIMIT = 3000
COMPOSITION_WINDOW = 64
PATTERN_LABELS_2BIT = ("2H", "2V")
PATTERN_LABELS_3BIT = ("3H", "3V", "L-up-left", "L-up-right",
                       "L-down-left", "L-down-right")
MAX_SITE_ATTEMPTS = 64

# provenance label sets per pattern leg (docs/G5_FAULT_MODEL.md §3)
LABELS_SAME_ROW = ("same-row:measured-column-region", "column:unknown",
                   "column-stride-sub256B:unknown")
LABELS_SAME_BANK = ("same-bank:measured-kernel-mask", "row-distance:unknown",
                    "column:unknown")


class SamplingError(RuntimeError):
    """A site could not be placed inside the residency after retries."""


class ModelError(RuntimeError):
    """The frozen level table no longer matches its derivation rule."""


def resolve_composition(bits: int,
                        exhaustive_limit: int = EXHAUSTIVE_BITS_LIMIT,
                        ) -> tuple[int, int, int]:
    """(s, d, t) with s + 2d + 3t = bits minimizing the squared deviation
    of the event shares from (60%, 20%, 20%); ties -> more SBU, then more
    2-bit. Exhaustive over all feasible tuples for bits <= exhaustive_limit;
    above that, a +-COMPOSITION_WINDOW box around the analytic optimum
    (same objective and tie rule; equivalence cross-checked in self_test,
    and the frozen literals pin the real levels -- see the comment at
    EXHAUSTIVE_BITS_LIMIT)."""
    if bits <= exhaustive_limit:
        s_lo, s_hi = 0, bits
        d_lo, d_hi = 0, bits // 2
    else:
        events_analytic = bits / 1.6
        s_lo = max(0, int(0.6 * events_analytic) - COMPOSITION_WINDOW)
        s_hi = min(bits, int(0.6 * events_analytic) + COMPOSITION_WINDOW)
        d_lo = max(0, int(0.2 * events_analytic) - COMPOSITION_WINDOW)
        d_hi = min(bits // 2,
                   int(0.2 * events_analytic) + COMPOSITION_WINDOW)
    best: tuple[tuple[float, int, int], tuple[int, int, int]] | None = None
    for s in range(s_lo, s_hi + 1):
        for d in range(d_lo, d_hi + 1):
            rest = bits - s - 2 * d
            if rest < 0 or rest % 3:
                continue
            t = rest // 3
            events = s + d + t
            if events == 0:
                continue
            shares = (s / events, d / events, t / events)
            dist = sum((a - b) ** 2 for a, b in zip(shares, EVENT_SHARE_TARGET))
            key = (dist, -s, -d)
            if best is None or key < best[0]:
                best = (key, (s, d, t))
    if best is None:
        raise ModelError(f"no composition sums to {bits} bits")
    return best[1]


def workload_by_name(name: str) -> dict:
    try:
        return WORKLOADS[name]
    except KeyError:
        raise ModelError(
            f"unknown workload {name!r}; expected one of "
            f"{sorted(WORKLOADS)}") from None


def assert_frozen_levels(workload: str = DEFAULT_WORKLOAD) -> None:
    """Re-derive the workload's table from its BERs and R; refuse on drift."""
    entry = workload_by_name(workload)
    r_bits = entry["resident_bytes_nominal"] * 8
    for row in entry["levels"]:
        bits = round(row["ber"] * r_bits)
        composition = resolve_composition(bits)
        if bits != row["bits"] or composition != (row["s"], row["d"],
                                                  row["t"]):
            raise ModelError(
                f"{workload}/{row['level']}: derived ({bits}, {composition}) "
                f"!= frozen ({row['bits']}, "
                f"({row['s']}, {row['d']}, {row['t']})); the level table "
                "must be re-derived and re-confirmed")


def level_by_name(name: str, workload: str = DEFAULT_WORKLOAD) -> dict:
    levels = workload_by_name(workload)["levels"]
    for entry in levels:
        if entry["level"] == name:
            return entry
    raise ModelError(f"unknown level {name} in workload {workload}; expected "
                     f"one of {[e['level'] for e in levels]}")


# ---------------------------------------------------------------------------
# inputs: G3 anchor consensus + G4 snapshot rows
# ---------------------------------------------------------------------------

def load_anchors(table_dir: Path) -> dict[int, list[int]]:
    """page_base -> sorted valid anchor masks (table_v4 page_anchors.csv,
    per-page strict-majority consensus; every page of the universe has at
    least the two universal kernel masks)."""
    anchors: dict[int, list[int]] = {}
    path = Path(table_dir) / "page_anchors.csv"
    with path.open(encoding="utf-8", newline="") as source:
        lines = [line for line in source if not line.startswith("#")]
        for row in csv.reader(lines):
            if not row or row[0] == "page_base" or row[4] != "true":
                continue
            anchors.setdefault(int(row[0], 16), []).append(int(row[1], 16))
    for masks in anchors.values():
        masks.sort()
    return anchors


class ResidencyIndex:
    """Byte-residency of one run's G4 snapshot, indexed by PA page.

    rows: snapshot_pages.csv dicts with allocation_id, semantic_label,
    allocation_va_base, allocation_size_bytes, va_page_base,
    byte_start_in_page, byte_end_in_page, fb_pa_page_base. A PA page may
    host several allocations as disjoint byte ranges; sites map back to
    (allocation, byte_offset, gpu_va) through the covering row.
    """

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.pages: dict[int, list[dict]] = {}
        for row in rows:
            pa_page = int(str(row["fb_pa_page_base"]), 16)
            self.pages.setdefault(pa_page, []).append(row)
        self.total_bytes = 0
        self.page_weights: list[tuple[int, int, int]] = []  # (pa, start, cum)
        for pa_page in sorted(self.pages):
            span = sum(self._length(row) for row in self.pages[pa_page])
            self.total_bytes += span
            self.page_weights.append((pa_page, self.total_bytes, span))

    @staticmethod
    def _length(row: dict) -> int:
        return (int(str(row["byte_end_in_page"]), 16)
                - int(str(row["byte_start_in_page"]), 16))

    def random_resident_byte(self, rng: random.Random,
                             ) -> tuple[dict, int, int]:
        """Uniform over resident BYTES -> (snapshot row, in-page offset,
        offset inside the row's interval)."""
        target = rng.randrange(self.total_bytes)
        low, high = 0, self.total_bytes
        pa_page = 0
        for pa, cumulative, _span in self.page_weights:
            if target < cumulative:
                pa_page = pa
                break
            low = cumulative
        local = target - low
        for row in self.pages[pa_page]:
            length = self._length(row)
            if local < length:
                start = int(str(row["byte_start_in_page"]), 16)
                return row, start + local, local
            local -= length
        raise SamplingError("residency index walk failed (corrupt index)")

    def covering_row(self, pa_page: int, in_page: int) -> dict | None:
        for row in self.pages.get(pa_page, ()):
            start = int(str(row["byte_start_in_page"]), 16)
            end = int(str(row["byte_end_in_page"]), 16)
            if start <= in_page < end:
                return row
        return None


# ---------------------------------------------------------------------------
# site construction
# ---------------------------------------------------------------------------

def _make_site(row: dict, in_page: int, bit: int,
               event_index: int, site_index: int, pattern: str,
               labels: tuple[str, ...], anchor_mask: int | None,
               trial_index: int) -> dict:
    # the snapshot maps one VA page to one PA page 1:1, so a byte at PA
    # in-page offset X sits at VA va_page + X inside the covering row
    va_page = int(str(row["va_page_base"]), 16)
    base = int(str(row["allocation_va_base"]), 16)
    byte_offset = va_page + in_page - base
    pa_page = int(str(row["fb_pa_page_base"]), 16)
    return {
        "trial_index": trial_index,
        "event_index": event_index,
        "site_index": site_index,
        "target_id": f"t{trial_index:03d}-e{event_index:03d}-s{site_index:02d}",
        "pattern": pattern,
        "allocation_id": row["allocation_id"],
        "semantic_label": row["semantic_label"],
        "byte_offset": byte_offset,
        "bit": bit,
        "expected_gpu_va": base + byte_offset,
        "provenance": list(labels),
        "chain": {
            "fb_pa_page_base": f"{pa_page:#x}",
            "pa_in_page_offset": f"{in_page:#x}",
            "va_page_base": row["va_page_base"],
            "anchor_mask": f"{anchor_mask:#x}" if anchor_mask else "",
        },
    }


def _site_pa(site: dict) -> tuple[int, int]:
    return (int(site["chain"]["fb_pa_page_base"], 16),
            int(site["chain"]["pa_in_page_offset"], 16))


def _draw_bit(rng: random.Random, used: set[tuple[int, int, int]],
              pa: int, in_page: int) -> int:
    for _ in range(MAX_SITE_ATTEMPTS):
        bit = rng.randrange(8)
        if (pa, in_page, bit) not in used:
            used.add((pa, in_page, bit))
            return bit
    raise SamplingError(f"all 8 bits of PA byte {pa:#x}+{in_page:#x} are "
                        "already used in this trial")


def _block_mate(rng: random.Random, index: ResidencyIndex,
                row: dict, in_page: int, taken: set[int]) -> tuple[dict, int]:
    """Another byte of the SAME 256 B PA block (same row: measured column
    region covers in-page bits 0-7), resident and not already taken."""
    block = in_page & ~0xFF
    for _ in range(MAX_SITE_ATTEMPTS):
        mate = block + rng.randrange(256)
        if mate in taken:
            continue
        mate_row = index.covering_row(int(str(row["fb_pa_page_base"]), 16),
                                      mate)
        if mate_row is not None:
            return mate_row, mate
    raise SamplingError("no resident block mate after retries "
                        f"(page {row['fb_pa_page_base']} block {block:#x})")


def _anchor_mate(rng: random.Random, index: ResidencyIndex, row: dict,
                 in_page: int, anchors: dict[int, list[int]],
                 taken: set[int], exclude_masks: set[int],
                 ) -> tuple[dict, int, int]:
    """The same-bank/different-row mate: in_page XOR a valid anchor mask of
    the SAME page (measured kernel masks; mask < 2 MiB keeps the mate on
    the page). Returns (mate_row, mate_in_page, mask)."""
    pa_page = int(str(row["fb_pa_page_base"]), 16)
    masks = [m for m in anchors.get(pa_page, ()) if m not in exclude_masks]
    if not masks:
        raise SamplingError(f"page {pa_page:#x} has no usable anchor")
    for _ in range(MAX_SITE_ATTEMPTS):
        mask = rng.choice(masks)
        mate = in_page ^ mask
        if mate in taken:
            continue
        mate_row = index.covering_row(pa_page, mate)
        if mate_row is not None:
            return mate_row, mate, mask
    raise SamplingError("no resident anchor mate after retries "
                        f"(page {pa_page:#x})")


def sample_event(rng: random.Random, index: ResidencyIndex,
                 anchors: dict[int, list[int]], pattern: str,
                 event_index: int, trial_index: int,
                 used: set[tuple[int, int, int]]) -> list[dict]:
    """One flip event of the requested pattern -> its site list (1-3 sites,
    all distinct at (PA byte, bit) granularity)."""
    sites: list[dict] = []

    def add(row: dict, in_page: int, labels: tuple[str, ...],
            anchor_mask: int | None = None) -> None:
        pa = int(str(row["fb_pa_page_base"]), 16)
        bit = _draw_bit(rng, used, pa, in_page)
        sites.append(_make_site(row, in_page, bit, event_index,
                                len(sites), pattern, labels, anchor_mask,
                                trial_index))

    for _attempt in range(MAX_SITE_ATTEMPTS):
        sites = []
        try:
            row, in_page, _ = index.random_resident_byte(rng)
            taken = {in_page}
            if pattern == "SBU":
                add(row, in_page, ("chain:measured",))
            elif pattern in ("2H", "3H"):
                add(row, in_page, LABELS_SAME_ROW)
                mates = 1 if pattern == "2H" else 2
                for _ in range(mates):
                    mate_row, mate = _block_mate(rng, index, row, in_page,
                                                 taken)
                    taken.add(mate)
                    add(mate_row, mate, LABELS_SAME_ROW)
            elif pattern in ("2V", "3V"):
                add(row, in_page, LABELS_SAME_BANK)
                mates = 1 if pattern == "2V" else 2
                exclude: set[int] = set()
                for _ in range(mates):
                    mate_row, mate, mask = _anchor_mate(
                        rng, index, row, in_page, anchors, taken, exclude)
                    taken.add(mate)
                    exclude.add(mask)
                    add(mate_row, mate, LABELS_SAME_BANK, mask)
            elif pattern.startswith("L-"):
                # same-row pair + one anchor mate (orientation is a label:
                # the anchor landing is random, so the four orientations
                # share one distribution -- docs/G5_FAULT_MODEL.md §3)
                add(row, in_page, LABELS_SAME_ROW)
                mate_row, mate = _block_mate(rng, index, row, in_page, taken)
                taken.add(mate)
                add(mate_row, mate, LABELS_SAME_ROW)
                v_row, v_mate, mask = _anchor_mate(
                    rng, index, row, in_page, anchors, taken, set())
                add(v_row, v_mate, LABELS_SAME_BANK, mask)
            else:
                raise ModelError(f"unknown pattern {pattern}")
            return sites
        except SamplingError:
            # rollback the bits this attempt reserved and retry from a
            # fresh base; bounded by the outer loop
            for site in sites:
                pa, off = _site_pa(site)
                used.discard((pa, off, site["bit"]))
            continue
    raise SamplingError(f"event {pattern} could not be placed after "
                        f"{MAX_SITE_ATTEMPTS} attempts")


def sample_trial(level: dict, trial_index: int, index: ResidencyIndex,
                 anchors: dict[int, list[int]], seed: int) -> list[dict]:
    """All events of one trial, event order shuffled, bits and positions
    fresh (the frozen (s, d, t) echo travels with the campaign manifest)."""
    rng = random.Random(seed * 100_003 + trial_index)
    patterns: list[str] = []
    patterns += ["SBU"] * level["s"]
    patterns += [rng.choice(PATTERN_LABELS_2BIT) for _ in range(level["d"])]
    patterns += [rng.choice(PATTERN_LABELS_3BIT) for _ in range(level["t"])]
    rng.shuffle(patterns)
    used: set[tuple[int, int, int]] = set()
    sites: list[dict] = []
    for event_index, pattern in enumerate(patterns):
        sites.extend(sample_event(rng, index, anchors, pattern, event_index,
                                  trial_index, used))
    if len(sites) != level["bits"]:
        raise ModelError(f"trial {trial_index}: {len(sites)} sites != frozen "
                         f"B={level['bits']}")
    return sites


def sample_campaign(level_name: str, trials: int, snapshot_rows: list[dict],
                    anchors: dict[int, list[int]], seed: int,
                    resident_bytes_expected: int | None = None,
                    workload: str = DEFAULT_WORKLOAD,
                    ) -> list[list[dict]]:
    """The full work list for one campaign (level, N trials, one seed)."""
    assert_frozen_levels(workload)
    level = level_by_name(level_name, workload)
    index = ResidencyIndex(snapshot_rows)
    if resident_bytes_expected is not None and \
            index.total_bytes != resident_bytes_expected:
        raise ModelError(
            f"snapshot residency {index.total_bytes} bytes != nominal "
            f"{resident_bytes_expected}: R changed, the frozen level table "
            "must be re-derived (docs/G5_FAULT_MODEL.md §5)")
    campaign: list[list[dict]] = []
    for trial_index in range(trials):
        campaign.append(sample_trial(level, trial_index, index, anchors,
                                     seed))
    return campaign


# ---------------------------------------------------------------------------
# work file (runner contract; model metadata lives in work_detail.json)
# ---------------------------------------------------------------------------

WORK_FIELDS = ["trial_index", "event_index", "site_index", "target_id",
               "allocation_id", "byte_offset", "bit_in_byte",
               "expected_gpu_va"]


def write_work_csv(path: Path, campaign: list[list[dict]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as sink:
        sink.write("# gpu-m2d g5 campaign work file (frozen level table; "
                   "sites sampled from this run's dual-addressing snapshot)\n")
        writer = csv.DictWriter(sink, fieldnames=WORK_FIELDS)
        writer.writeheader()
        for sites in campaign:
            for site in sites:
                writer.writerow({
                    "trial_index": site["trial_index"],
                    "event_index": site["event_index"],
                    "site_index": site["site_index"],
                    "target_id": site["target_id"],
                    "allocation_id": site["allocation_id"],
                    "byte_offset": site["byte_offset"],
                    "bit_in_byte": site["bit"],
                    "expected_gpu_va": f"{site['expected_gpu_va']:#x}",
                })


# ---------------------------------------------------------------------------
# self-test (offline, no GPU, no bcc)
# ---------------------------------------------------------------------------

def _fixture() -> tuple[list[dict], dict[int, list[int]]]:
    page = 2 << 20

    def row(alloc, label, va_page, start, end, pa, base=None):
        base = base if base is not None else va_page
        return {
            "allocation_id": alloc, "semantic_label": label,
            "allocation_va_base": f"{base:#x}",
            "allocation_size_bytes": end - start,
            "va_page_base": f"{va_page:#x}",
            "byte_start_in_page": f"{start:#x}",
            "byte_end_in_page": f"{end:#x}",
            "fb_pa_page_base": f"{pa:#x}", "page_size": page,
            "in_universe": 1, "bank_linked": 1, "channel_root": f"{pa:#x}",
            "channel_size": 2, "row_class_count": 0,
            "same_row_site_nodes": 0, "valid_anchor_masks": "0x1f9dc0",
        }

    rows = [
        # a weights-like page fully resident
        row("trt-internal-0", "TENSORRT_INTERNAL_UNKNOWN",
            0x700000000000, 0, page, 0x20000000),
        # an input-binding page, 602112 resident bytes in the middle
        row("trt-binding-data-gpu-0", "TENSOR:data",
            0x700100000000, 0x100000, 0x100000 + 602112, 0x22000000),
        # a shared small-allocation page (2 KiB of residency)
        row("trt-internal-1", "TENSORRT_INTERNAL_UNKNOWN",
            0x700200000000, 0, 2048, 0x24000000),
    ]
    anchors = {
        0x20000000: [0x1f9dc0, 0x1fdc80, 0x119980],
        0x22000000: [0x1f9dc0, 0x1fdc80],
        0x24000000: [0x1f9dc0, 0x1fdc80],
    }
    return rows, anchors


def self_test() -> int:
    import tempfile

    # every workload's frozen table matches its derivation rule
    for workload_name in WORKLOADS:
        assert_frozen_levels(workload_name)
    assert resolve_composition(8) == (3, 1, 1)  # exact 60/20/20 block
    assert resolve_composition(2) == (2, 0, 0)
    # windowed large-B path == exhaustive (forced via a low limit)
    for bits in (173, 347, 1057, 1444, 2114):
        assert resolve_composition(bits, exhaustive_limit=100) == \
            resolve_composition(bits), bits

    rows, anchors = _fixture()
    index = ResidencyIndex(rows)
    page = 2 << 20
    assert index.total_bytes == page + 602112 + 2048

    level = {"level": "L3", "ber": 1e-7, "bits": 21, "s": 8, "d": 2, "t": 3}
    sites = sample_trial(level, 0, index, anchors, seed=7)
    assert len(sites) == 21
    used = set()
    for site in sites:
        pa = int(site["chain"]["fb_pa_page_base"], 16)
        off = int(site["chain"]["pa_in_page_offset"], 16)
        key = (pa, off, site["bit"])
        assert key not in used and 0 <= site["bit"] <= 7
        used.add(key)
        # every site lands inside a residency interval of its page
        covering = index.covering_row(pa, off)
        assert covering is not None and \
            covering["allocation_id"] == site["allocation_id"]
        # forward chain arithmetic: VA == allocation base + byte offset,
        # and byte_offset == va_page + PA in-page offset - allocation base
        base = int(covering["allocation_va_base"], 16)
        va_page = int(covering["va_page_base"], 16)
        assert site["expected_gpu_va"] == base + site["byte_offset"]
        assert site["byte_offset"] == va_page + off - base
    # determinism: same seed -> identical sites; different seed -> differs
    again = sample_trial(level, 0, index, anchors, seed=7)
    assert [s["target_id"] for s in sites] == [s["target_id"] for s in again]
    assert all(a["expected_gpu_va"] == b["expected_gpu_va"]
               for a, b in zip(sites, again))
    other = sample_trial(level, 0, index, anchors, seed=8)
    assert any(a["expected_gpu_va"] != b["expected_gpu_va"]
               for a, b in zip(sites, other))

    # pattern geometry on a synthetic run of every pattern
    rng = random.Random(1)
    for pattern in ("SBU", "2H", "3H", "2V", "3V", "L-up-left"):
        used = set()
        sites = sample_event(rng, index, anchors, pattern, 0, 0, used)
        assert all(s["pattern"] == pattern for s in sites)
        offsets = [int(s["chain"]["pa_in_page_offset"], 16) for s in sites]
        pages = {int(s["chain"]["fb_pa_page_base"], 16) for s in sites}
        assert len(pages) == 1, "one MCU stays on one PA page"
        base_block = offsets[0] & ~0xFF
        if pattern in ("2H", "3H"):
            assert {off & ~0xFF for off in offsets} == {base_block}
            assert len(set(offsets)) == len(offsets)
            assert len(sites) == (2 if pattern == "2H" else 3)
            assert all(any("measured-column-region" in lab
                           for lab in s["provenance"]) for s in sites)
        elif pattern in ("2V", "3V"):
            base = offsets[0]
            masks = [int(s["chain"]["anchor_mask"], 16) for s in sites[1:]]
            page_base = next(iter(pages))
            assert all(m in anchors[page_base] for m in masks)
            for off, mask in zip(offsets[1:], masks):
                assert off == base ^ mask
            if pattern == "3V":
                assert masks[0] != masks[1]
            assert len(sites) == (2 if pattern == "2V" else 3)
            assert all(any("measured-kernel-mask" in lab
                           for lab in s["provenance"]) for s in sites)
        elif pattern.startswith("L-"):
            assert len(sites) == 3
            block_mates = [o for o in offsets[1:] if (o & ~0xFF) == base_block]
            assert len(block_mates) == 1
            anchor_mates = [s for s in sites[1:] if s["chain"]["anchor_mask"]]
            assert len(anchor_mates) == 1

    # campaign-level: counts, per-trial distinctness, work CSV round-trip
    campaign = sample_campaign("L2", 5, rows, anchors, seed=42,
                               resident_bytes_expected=index.total_bytes)
    assert [len(sites) for sites in campaign] == [11] * 5
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work.csv"
        write_work_csv(work, campaign)
        lines = [line for line in work.read_text().splitlines()
                 if line and not line.startswith("#")]
        assert lines[0] == ",".join(WORK_FIELDS)
        assert len(lines) == 1 + 5 * 11
        fields = lines[1].split(",")
        assert fields[0] == "0" and fields[3] == "t000-e000-s00"
        assert fields[7].startswith("0x")

    # residency guard: wrong nominal R refuses fail-closed
    try:
        sample_campaign("L2", 1, rows, anchors, seed=1,
                        resident_bytes_expected=index.total_bytes + 1)
        raise AssertionError("residency mismatch must refuse")
    except ModelError:
        pass

    print("g5 fault model self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    parser.error("this module is a library + self-test; see "
                 "run_g5_campaign.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
