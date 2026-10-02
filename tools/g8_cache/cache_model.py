#!/usr/bin/env python3
"""G8-T2 L2 cache-fault sampler + residency-map self-checks
(docs/G8_CACHE_FAULT_PLAN.md §3.2-§3.6, §4; T2 plan).

Given a verified residency map of THIS campaign process
(residency_map.load_map) and the workload's cache classes
(fault_model.WORKLOADS[...]["cache_alloc_classes"]):

  n_cache = round(BER_cache * R_eff_bits)                          (§3.2)

per trial, each of the n_cache single-bit upsets:
  1. picks a unit (32-B sector) with probability
     bits_l * T_l / sum(bits * T)                                  (§3.4)
  2. picks its start image uniformly IN TIME within the unit's residency
     periods (an image's chance is proportional to its inference time)
  3. picks a byte uniformly among the unit's resident bytes, and a bit
  4. lasts until the end of the period containing the start image for a
     READ-ONLY allocation, only the start image for an ENGINE-WRITTEN one
     (the engine overwrites the data itself)
Two cache flips on the same (allocation, byte, bit) with overlapping
lifetimes are redrawn (XOR-ing one cell twice would cancel); a cache flip
MAY coincide with a DRAM-flipped bit (removal re-XORs, so the DRAM fault
survives -- §3.6).

Class rule: bindings (semantic label TENSOR:*) are engine-written; a
TRT-internal allocation must appear in the workload's cache_alloc_classes
(fail-closed otherwise).

Self-checks (T2-b, per process, fail-closed; thresholds below):
  - in-gap share: the share of in-situ probe latencies inside the T0
    calibration gap -- a blurred hit/miss separation (e.g. co-tenant
    queueing) makes the threshold unreliable;
  - order effect: hit rate of units probed in the first half of their
    sweep minus the second half (runner-counted, map json
    `order_effect`). The sweep's own fills evict not-yet-probed sectors,
    so the artifact reads late < early; with alternating direction every
    unit is sometimes early and sometimes late, so genuine residency
    changes cancel. (The per-unit direction-locked share is kept as an
    informational diagnostic only: under a bursty co-tenant, genuinely
    flickering units with few observations fake it.)
  - the map must describe this process: same allocations at the same VAs
    as the gate registry, the campaign's image count, the frozen surface.

Pure stdlib. `--self-test` covers sampling statistics, lifetimes, the
overlap rule, the class rule and every self-check.
"""

from __future__ import annotations

import bisect
import collections
import csv
import itertools
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import residency_map as rm  # noqa: E402

READ_ONLY = "read_only"
ENGINE_WRITTEN = "engine_written"
MAX_OVERLAP_REDRAWS = 64

# T2-b self-check thresholds (fail-closed).
CALIBRATION_GAP = (400, 480)   # T0/T1 single-lane calibration, cycles
MAX_IN_GAP_SHARE = 0.005       # idle GPU 0 measured <= 0.2 %
MAX_LOCKED_SHARE = 0.02        # informational (see module docstring)
# |early - late| probe hit rate. Calibration (GPU 0): idle ViT-B stride
# 16 (known sweep artifact) -3.87 pp; every production configuration
# idle or under a controlled co-tenant within +-2.3 pp (the shared cases
# are real-time eviction during the sweep, bounded the same way).
MAX_ORDER_EFFECT = 0.03

CACHE_WORK_FIELDS = ["trial_index", "cache_index", "target_id",
                     "allocation_id", "byte_offset", "bit_in_byte",
                     "expected_gpu_va", "start_image", "last_image",
                     "cache_class"]


class CacheModelError(RuntimeError):
    pass


def allocation_class(range_meta: dict, classes: dict) -> str:
    alloc_id = range_meta["allocation_id"]
    if range_meta.get("semantic_label", "").startswith("TENSOR:"):
        return ENGINE_WRITTEN
    if alloc_id in classes.get(READ_ONLY, ()):
        return READ_ONLY
    if alloc_id in classes.get(ENGINE_WRITTEN, ()):
        return ENGINE_WRITTEN
    raise CacheModelError(f"{alloc_id}: no cache lifetime class "
                          "(fault_model cache_alloc_classes)")


class CacheSampler:
    """Precomputed sampling structures over one residency map."""

    def __init__(self, m: rm.ResidencyMap, classes: dict):
        self.m = m
        self.unit_bytes = m.meta["unit_bytes"]
        self.range_by_unit = []  # (first_unit, range_meta, class, base)
        for r in sorted(m.ranges, key=lambda x: x["first_unit"]):
            self.range_by_unit.append(
                (r["first_unit"], r, allocation_class(r, classes),
                 int(r["gpu_va"], 16)))
        self._firsts = [x[0] for x in self.range_by_unit]
        weights = m.exposure_weights()
        self.cumulative = list(itertools.accumulate(weights))
        self.total_weight = self.cumulative[-1] if self.cumulative else 0.0
        if self.total_weight <= 0:
            raise CacheModelError("residency map has zero exposure")
        self._prefix = m._prefix

    def range_of(self, unit: int):
        return self.range_by_unit[bisect.bisect_right(self._firsts, unit) - 1]

    def draw_unit(self, rng: random.Random) -> int:
        x = rng.random() * self.total_weight
        return bisect.bisect_right(self.cumulative, x)

    def draw_start(self, rng: random.Random, unit: int) -> tuple[int, int, int]:
        """(start image, period start, period end), uniform in time."""
        pre = self._prefix
        periods = self.m.periods(unit)
        spans = [pre[e + 1] - pre[s] for s, e in periods]
        tau = rng.random() * sum(spans)
        for (s, e), span in zip(periods, spans):
            if tau < span or (s, e) == periods[-1]:
                t = pre[s] + min(tau, span)
                image = bisect.bisect_right(pre, t) - 1
                return min(max(image, s), e), s, e
            tau -= span
        raise CacheModelError(f"unit {unit}: no period to start in")

    def draw_byte(self, rng: random.Random, unit: int) -> tuple[dict, str, int, int]:
        first, r, cls, base = self.range_of(unit)
        k = unit - first
        aligned = base & ~(self.unit_bytes - 1)
        lo = max(aligned + k * self.unit_bytes, base)
        hi = min(aligned + (k + 1) * self.unit_bytes, base + r["size_bytes"])
        if hi <= lo:
            raise CacheModelError(f"unit {unit}: empty byte window")
        return r, cls, base, lo - base + rng.randrange(hi - lo)


def n_cache_for(m: rm.ResidencyMap, ber: float) -> int:
    if ber < 0:
        raise CacheModelError("BER_cache must be >= 0")
    return round(ber * m.r_eff)


def sample_trial(sampler: CacheSampler, n: int, trial_index: int,
                 seed: int, live_bases: dict[str, int]) -> list[dict]:
    rng = random.Random(f"g8-cache-{seed}-{trial_index}")
    busy = collections.defaultdict(list)  # (alloc, byte, bit) -> [(a, b)]
    sites = []
    for k in range(n):
        for _ in range(MAX_OVERLAP_REDRAWS):
            unit = sampler.draw_unit(rng)
            start, p_start, p_end = sampler.draw_start(rng, unit)
            r, cls, base, offset = sampler.draw_byte(rng, unit)
            bit = rng.randrange(8)
            last = p_end if cls == READ_ONLY else start
            key = (r["allocation_id"], offset, bit)
            if all(last < a or start > b for a, b in busy[key]):
                break
        else:
            raise CacheModelError(f"trial {trial_index}: could not place "
                                  f"cache site {k} without overlap")
        busy[key].append((start, last))
        live = live_bases.get(r["allocation_id"])
        if live is None or live != base:
            raise CacheModelError(f"{r['allocation_id']}: map VA "
                                  f"{base:#x} != live registry VA "
                                  f"{live if live is None else hex(live)}")
        sites.append({
            "trial_index": trial_index, "cache_index": k,
            "target_id": f"c{trial_index:03d}-{k}",
            "allocation_id": r["allocation_id"], "byte_offset": offset,
            "bit_in_byte": bit, "expected_gpu_va": base + offset,
            "start_image": start, "last_image": last, "cache_class": cls,
            "unit": unit, "period": (p_start, p_end),
        })
    return sites


def sample_cache_campaign(m: rm.ResidencyMap, classes: dict, ber: float,
                          trials: int, seed: int,
                          live_bases: dict[str, int]) -> tuple[int, list]:
    sampler = CacheSampler(m, classes)
    n = n_cache_for(m, ber)
    return n, [sample_trial(sampler, n, t, seed, live_bases)
               for t in range(trials)]


def verify_cache_work(m: rm.ResidencyMap, classes: dict, n: int,
                      campaign: list[list[dict]]) -> list[str]:
    """Independent re-check of a sampled campaign against the map."""
    failures = []
    ranges = {r["allocation_id"]: r for r in m.ranges}
    for t, sites in enumerate(campaign):
        if len(sites) != n:
            failures.append(f"trial {t}: {len(sites)} cache sites != n {n}")
        busy = collections.defaultdict(list)
        for s in sites:
            r = ranges.get(s["allocation_id"])
            if r is None:
                failures.append(f"{s['target_id']}: allocation not in map")
                continue
            cls = allocation_class(r, classes)
            periods = m.periods(s["unit"])
            if (s["period"][0], s["period"][1]) not in periods or not \
                    s["period"][0] <= s["start_image"] <= s["period"][1]:
                failures.append(f"{s['target_id']}: start outside a residency "
                                "period of its unit")
            want_last = s["period"][1] if cls == READ_ONLY else s["start_image"]
            if s["last_image"] != want_last or s["cache_class"] != cls:
                failures.append(f"{s['target_id']}: lifetime/class mismatch")
            if not 0 <= s["byte_offset"] < r["size_bytes"] or \
                    not 0 <= s["bit_in_byte"] < 8:
                failures.append(f"{s['target_id']}: byte/bit out of range")
            key = (s["allocation_id"], s["byte_offset"], s["bit_in_byte"])
            for a, b in busy[key]:
                if not (s["last_image"] < a or s["start_image"] > b):
                    failures.append(f"{s['target_id']}: overlaps another "
                                    "cache flip on the same bit")
            busy[key].append((s["start_image"], s["last_image"]))
    return failures


def write_cache_work(path: Path, campaign: list[list[dict]]) -> None:
    with path.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(CACHE_WORK_FIELDS)
        for sites in campaign:
            for s in sites:
                w.writerow([s["trial_index"], s["cache_index"], s["target_id"],
                            s["allocation_id"], s["byte_offset"],
                            s["bit_in_byte"], f"{s['expected_gpu_va']:#x}",
                            s["start_image"], s["last_image"],
                            s["cache_class"]])


def in_gap_share(hist_csv: Path, gap=CALIBRATION_GAP) -> float:
    total = inside = 0
    with hist_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            count = int(row["count"])
            total += count
            if gap[0] <= int(row["bin_lo_cycles"]) < gap[1]:
                inside += count
    if total == 0:
        raise CacheModelError(f"{hist_csv}: empty latency histogram")
    return inside / total


def locked_share(m: rm.ResidencyMap) -> float:
    """Share of the surface's BYTES in direction-locked units."""
    if not m.meta.get("alternate"):
        return 0.0
    d = rm.direction_lock(m)
    every, stride = m.meta["probe_every"], m.meta["stride"]
    locked_bytes = 0
    for r in m.ranges:
        for u in range(r["first_unit"], r["first_unit"] + r["units"]):
            lo, hi = m.offsets[u], m.offsets[u + 1]
            if hi - lo < d["min_periods"]:
                continue
            parities = {((m.flat[2 * p] // every) // stride) % 2
                        for p in range(lo, hi) if m.flat[2 * p] > 0}
            if len(parities) == 1:
                locked_bytes += m.unit_resident_bytes[u]
    return locked_bytes / (m.meta["surface_bits"] / 8)


def order_effect(m: rm.ResidencyMap) -> dict:
    """Early-minus-late probe hit rate of the pass (map json)."""
    oe = m.meta.get("order_effect")
    if not oe or not oe.get("early_n") or not oe.get("late_n"):
        raise CacheModelError("map has no order_effect counts")
    p_early = oe["early_hits"] / oe["early_n"]
    p_late = oe["late_hits"] / oe["late_n"]
    se = (p_early * (1 - p_early) / oe["early_n"]
          + p_late * (1 - p_late) / oe["late_n"]) ** 0.5
    return {"early_hit_rate": p_early, "late_hit_rate": p_late,
            "delta": p_early - p_late, "se": se}


def map_self_checks(prefix: str, m: rm.ResidencyMap, *, images: int,
                    frozen_r: int, live_bases: dict[str, int]) -> dict:
    """T2-b per-process checks; returns {"values": ..., "failures": [...]}."""
    failures = []
    gap_share = in_gap_share(Path(prefix + "_hist.csv"))
    lock = locked_share(m)   # informational only (module docstring)
    if gap_share > MAX_IN_GAP_SHARE:
        failures.append(f"in-gap share {gap_share:.4%} > {MAX_IN_GAP_SHARE:.2%}"
                        " (hit/miss separation blurred)")
    try:
        oe = order_effect(m)
    except CacheModelError as exc:
        oe = None
        failures.append(str(exc))
    if oe is not None and abs(oe["delta"]) > MAX_ORDER_EFFECT:
        failures.append(f"order effect {oe['delta']:+.2%} beyond "
                        f"+-{MAX_ORDER_EFFECT:.0%} (sweep-order artifact)")
    if m.images != images:
        failures.append(f"map has {m.images} images, campaign {images}")
    if m.meta["surface_bits"] != 8 * frozen_r:
        failures.append(f"map surface {m.meta['surface_bits'] // 8} B != "
                        f"frozen R {frozen_r} B")
    for r in m.ranges:
        live = live_bases.get(r["allocation_id"])
        if live != int(r["gpu_va"], 16):
            failures.append(f"{r['allocation_id']}: map VA {r['gpu_va']} != "
                            f"live registry VA {live}")
    return {"values": {"in_gap_share": gap_share, "locked_share": lock,
                       "order_effect": oe,
                       "stride": m.meta["stride"],
                       "stride_rule": m.meta.get("stride_rule"),
                       "r_eff_bits": m.r_eff,
                       "surface_bits": m.meta["surface_bits"]},
            "failures": failures}


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _fixture(tmp: Path, periods_per_unit, images=10, times=None,
             alternate=True, stride=1):
    p = str(tmp / "m")
    ranges = [
        {"allocation_id": "trt-internal-0", "first_unit": 0, "units": 3,
         "size_bytes": 96, "gpu_va": "0x1000",
         "semantic_label": "TENSORRT_INTERNAL_UNKNOWN"},
        {"allocation_id": "trt-binding-data-gpu-0", "first_unit": 3,
         "units": 1, "size_bytes": 20, "gpu_va": "0x2000",
         "semantic_label": "TENSOR:data"},
    ]
    rm._write_synthetic(p, 32, times or [1.0] * images, [32, 32, 32, 20],
                        periods_per_unit, ranges)
    import json
    meta = json.loads(Path(p + "_residency.json").read_text())
    meta.update({"alternate": alternate, "stride": stride,
                 "observation_window_images": stride,
                 "order_effect": {"early_hits": 500, "early_n": 1000,
                                  "late_hits": 495, "late_n": 1000}})
    Path(p + "_residency.json").write_text(json.dumps(meta))
    with open(p + "_hist.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["allocation_id", "bin_lo_cycles", "count"])
        w.writerow(["trt-internal-0", 304, 990])
        w.writerow(["trt-internal-0", 720, 10])
    return p, rm.load_map(p)


def self_test() -> int:
    classes = {READ_ONLY: ("trt-internal-0",), ENGINE_WRITTEN: ()}
    live = {"trt-internal-0": 0x1000, "trt-binding-data-gpu-0": 0x2000}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # unit 0 always resident, unit 1 never, unit 2 only images 6..9,
        # binding unit always
        p, m = _fixture(tmp, [[(0, 9)], [], [(6, 9)], [(0, 9)]])
        # bits: 256,256,256,160 ; T: 10,0,4,10 -> R_eff = (2560+1024+1600)/10
        assert abs(m.r_eff - 518.4) < 1e-9, m.r_eff
        assert n_cache_for(m, 0.5) == 259
        # realistic density (n << cells) so the overlap redraw does not
        # distort the exposure shares: 21 flips/trial over 50 trials
        n, camp = sample_cache_campaign(m, classes, 0.04, 50, 7, live)
        assert n == 21
        assert verify_cache_work(m, classes, n, camp) == []
        hits = collections.Counter()
        for sites in camp:
            for s in sites:
                hits[s["unit"]] += 1
                if s["unit"] == 2:
                    assert 6 <= s["start_image"] <= 9 and s["last_image"] == 9
                if s["unit"] == 0:
                    assert s["last_image"] == 9          # read-only: to period end
                if s["unit"] == 3:
                    assert s["last_image"] == s["start_image"]  # binding
                    assert s["cache_class"] == ENGINE_WRITTEN
                    assert 0 <= s["byte_offset"] < 20
                assert s["expected_gpu_va"] == live[s["allocation_id"]] + \
                    s["byte_offset"]
        assert hits[1] == 0
        # exposure shares 2560 : 1024 : 1600 of 5184 over 1050 draws
        # (binomial sd <= 0.016)
        total = sum(hits.values())
        for unit, share in ((0, 2560 / 5184), (2, 1024 / 5184),
                            (3, 1600 / 5184)):
            assert abs(hits[unit] / total - share) < 0.05, (unit, hits)
        # deterministic per (seed, trial)
        _, again = sample_cache_campaign(m, classes, 0.04, 50, 7, live)
        assert again == camp
        # verify catches a tampered lifetime
        bad = [list(t) for t in camp]
        site = next(s for s in bad[0] if s["unit"] == 0)
        bad[0][bad[0].index(site)] = dict(site, last_image=5)
        assert verify_cache_work(m, classes, n, bad)
        # time-uniform start: image 0 lasts 91 of 100 time units
        (tmp / "t").mkdir()
        _, m2 = _fixture(tmp / "t", [[(0, 9)], [], [], [(0, 9)]],
                         times=[91.0] + [1.0] * 9)
        sampler = CacheSampler(m2, classes)
        rng = random.Random(1)
        starts = collections.Counter(sampler.draw_start(rng, 0)[0]
                                     for _ in range(4000))
        assert 0.88 < starts[0] / 4000 < 0.94, starts[0]
        # class rule: an unclassified internal allocation is refused
        try:
            CacheSampler(m, {READ_ONLY: (), ENGINE_WRITTEN: ()})
        except CacheModelError:
            pass
        else:
            raise AssertionError("unclassified allocation must be refused")
        # live VA mismatch is refused
        try:
            sample_cache_campaign(m, classes, 0.5, 1, 7,
                                  dict(live, **{"trt-internal-0": 0x9999}))
        except CacheModelError:
            pass
        else:
            raise AssertionError("VA mismatch must be refused")
        # self-checks: clean map passes; frozen R / VA / images mismatches
        chk = map_self_checks(p, m, images=10, frozen_r=116, live_bases=live)
        assert chk["failures"] == [], chk
        assert abs(chk["values"]["in_gap_share"]) < 1e-12
        chk = map_self_checks(p, m, images=11, frozen_r=117,
                              live_bases={"trt-internal-0": 1})
        assert len(chk["failures"]) == 4, chk["failures"]
        # in-gap share above the limit fails
        with open(p + "_hist.csv", "a", newline="") as handle:
            csv.writer(handle).writerow(["trt-internal-0", 432, 20])
        chk = map_self_checks(p, m, images=10, frozen_r=116, live_bases=live)
        assert any("in-gap" in f for f in chk["failures"]), chk
        # an order effect beyond +-3 pp fails; within passes
        import json as _json
        meta = _json.loads(Path(p + "_residency.json").read_text())
        meta["order_effect"] = {"early_hits": 600, "early_n": 1000,
                                "late_hits": 550, "late_n": 1000}
        Path(p + "_residency.json").write_text(_json.dumps(meta))
        bad_map = rm.load_map(p)
        assert abs(order_effect(bad_map)["delta"] - 0.05) < 1e-12
        with open(p + "_hist.csv", "w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["allocation_id", "bin_lo_cycles", "count"])
            w.writerow(["trt-internal-0", 304, 1000])
        chk = map_self_checks(p, bad_map, images=10, frozen_r=116,
                              live_bases=live)
        assert any("order effect" in f for f in chk["failures"]), chk
        # direction-locked units are counted by bytes (informational)
        p3dir = tmp / "lock"
        p3dir.mkdir()
        osc = [(i, i) for i in range(0, 40, 2)]   # resident on even sweeps
        _, m3 = _fixture(p3dir, [osc, [(0, 39)], [(0, 39)], [(0, 39)]],
                         images=40)
        assert abs(locked_share(m3) - 32 / 116) < 1e-12, locked_share(m3)
    print("cache_model self-test: PASS")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    print(__doc__)
    sys.exit(1)
