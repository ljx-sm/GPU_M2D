#!/usr/bin/env python3
"""G8 L2 residency map reader / verifier / comparator (plan §3.2, §3.4, §4).

A residency map is the pair PREFIX_residency.json + PREFIX_residency.bin
written by the runner's L2 probe pass with --l2-probe-map 1
(apps/resnet50_int8_g1_5.cpp, gpu_m2d::ResidencyMapBuilder).

Binary layout (little-endian):
    8 B  magic "G8RMAP01"
    u32  version (1)            u32  unit_bytes (32 or 128)
    u64  units                  u64  images            u64  periods
    f64  image_time_ms[images]  (GPU inference time per image)
    u16  unit_resident_bytes[units]
    u64  offsets[units + 1]     (periods of unit u: offsets[u] .. offsets[u+1])
    u32  (start, end)[periods]  (inclusive image indices, sorted per unit)

Loading is FAIL-CLOSED: sha256 must match the JSON, every structural
invariant must hold (sizes, monotone offsets, periods inside [0, images),
start <= end, strictly separated and sorted within a unit), and the
independently recomputed T_total and R_eff_bits must equal the runner's
values. A map whose probe pass was not neutral (mismatches != 0) is
refused.

    R_eff_bits = sum_l(8 * unit_resident_bytes_l * T_l) / T_total     (§3.2)

Pure stdlib; importable by the G8-T2 orchestrator (load_map, unit
weights for residency-weighted sampling). CLI:

    residency_map.py summary PREFIX [PREFIX ...]
    residency_map.py compare PREFIX_A PREFIX_B
    residency_map.py --self-test

Exit codes: 0 ok, 2 verification failure, 1 usage/input error.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

MAGIC = b"G8RMAP01"
HEADER = struct.Struct("<8sIIQQQ")
REL_TOL = 1e-9


class MapError(RuntimeError):
    pass


def _read_array(blob: bytes, offset: int, typecode: str, count: int):
    item = array.array(typecode)
    end = offset + item.itemsize * count
    if end > len(blob):
        raise MapError("residency map truncated")
    item.frombytes(blob[offset:end])
    if sys.byteorder != "little":
        item.byteswap()
    return item, end


class ResidencyMap:
    """A verified residency map. Units are in global order; ranges map
    allocation ids to [first_unit, first_unit + units)."""

    def __init__(self, meta: dict, image_time, unit_bytes_arr, offsets, flat):
        self.meta = meta
        self.image_time = image_time          # array('d'), ms per image
        self.unit_resident_bytes = unit_bytes_arr  # array('H')
        self.offsets = offsets                # array('Q'), units + 1
        self.flat = flat                      # array('I'), 2 * periods
        self.units = len(unit_bytes_arr)
        self.images = len(image_time)
        self.ranges = meta["ranges"]
        prefix = [0.0]
        for t in image_time:
            prefix.append(prefix[-1] + t)
        self._prefix = prefix
        self.t_total = prefix[-1]

    def periods(self, unit: int) -> list[tuple[int, int]]:
        lo, hi = self.offsets[unit], self.offsets[unit + 1]
        return [(self.flat[2 * p], self.flat[2 * p + 1]) for p in range(lo, hi)]

    def resident_time(self, unit: int) -> float:
        pre = self._prefix
        t = 0.0
        for p in range(self.offsets[unit], self.offsets[unit + 1]):
            t += pre[self.flat[2 * p + 1] + 1] - pre[self.flat[2 * p]]
        return t

    def resident_times(self) -> list[float]:
        return [self.resident_time(u) for u in range(self.units)]

    def exposure_weights(self) -> list[float]:
        """bits_l * T_l per unit: the plan §3.4 placement weight (also the
        numerator terms of R_eff_bits)."""
        return [8.0 * self.unit_resident_bytes[u] * self.resident_time(u)
                for u in range(self.units)]

    def r_eff_bits(self) -> float:
        return sum(self.exposure_weights()) / self.t_total

    def range_of(self, unit: int) -> dict:
        for r in self.ranges:
            if r["first_unit"] <= unit < r["first_unit"] + r["units"]:
                return r
        raise MapError(f"unit {unit} outside every range")


def load_map(prefix: Path | str, require_neutral: bool = True) -> ResidencyMap:
    prefix = str(prefix)
    json_path = Path(prefix + "_residency.json")
    bin_path = Path(prefix + "_residency.bin")
    try:
        meta = json.loads(json_path.read_text())
        blob = bin_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise MapError(f"{prefix}: cannot read map: {exc}") from exc
    if meta.get("schema") != "gpu-m2d.g8.residency-map.v1":
        raise MapError(f"{prefix}: unknown schema {meta.get('schema')!r}")
    digest = hashlib.sha256(blob).hexdigest()
    if digest != meta.get("bin_sha256"):
        raise MapError(f"{prefix}: bin sha256 {digest} != json "
                       f"{meta.get('bin_sha256')}")
    if require_neutral and meta.get("neutral_mismatches") != 0:
        raise MapError(f"{prefix}: probe pass was not neutral "
                       f"({meta.get('neutral_mismatches')} mismatches)")
    if len(blob) < HEADER.size:
        raise MapError(f"{prefix}: truncated header")
    magic, version, unit_bytes, units, images, periods = HEADER.unpack_from(blob)
    if magic != MAGIC or version != 1:
        raise MapError(f"{prefix}: bad magic/version")
    if (unit_bytes, units, images, periods) != (
            meta["unit_bytes"], meta["units"], meta["images"], meta["periods"]):
        raise MapError(f"{prefix}: header disagrees with json")
    off = HEADER.size
    image_time, off = _read_array(blob, off, "d", images)
    unit_bytes_arr, off = _read_array(blob, off, "H", units)
    offsets, off = _read_array(blob, off, "Q", units + 1)
    flat, off = _read_array(blob, off, "I", 2 * periods)
    if off != len(blob):
        raise MapError(f"{prefix}: {len(blob) - off} trailing bytes")

    # structural invariants
    if offsets[0] != 0 or offsets[units] != periods:
        raise MapError(f"{prefix}: offsets do not span the periods")
    for u in range(units):
        lo, hi = offsets[u], offsets[u + 1]
        if hi < lo:
            raise MapError(f"{prefix}: offsets not monotone at unit {u}")
        prev_end = -2
        for p in range(lo, hi):
            s, e = flat[2 * p], flat[2 * p + 1]
            if not (0 <= s <= e < images):
                raise MapError(f"{prefix}: unit {u} period ({s},{e}) out of range")
            if s <= prev_end + 1:
                raise MapError(f"{prefix}: unit {u} periods overlap/adjacent")
            prev_end = e
        if not 0 < unit_bytes_arr[u] <= unit_bytes:
            raise MapError(f"{prefix}: unit {u} resident bytes "
                           f"{unit_bytes_arr[u]} outside (0, {unit_bytes}]")
    covered = sorted((r["first_unit"], r["units"]) for r in meta["ranges"])
    nxt = 0
    for first, n in covered:
        if first != nxt:
            raise MapError(f"{prefix}: ranges do not tile the units")
        nxt = first + n
    if nxt != units:
        raise MapError(f"{prefix}: ranges cover {nxt} of {units} units")
    if sum(8 * b for b in unit_bytes_arr) != meta["surface_bits"]:
        raise MapError(f"{prefix}: surface_bits disagrees with unit bytes")

    m = ResidencyMap(meta, image_time, unit_bytes_arr, offsets, flat)
    if not math.isclose(m.t_total, meta["t_total_ms"], rel_tol=REL_TOL):
        raise MapError(f"{prefix}: T_total {m.t_total} != json {meta['t_total_ms']}")
    r_eff = m.r_eff_bits()
    if not math.isclose(r_eff, meta["r_eff_bits"], rel_tol=REL_TOL):
        raise MapError(f"{prefix}: recomputed R_eff_bits {r_eff} != json "
                       f"{meta['r_eff_bits']}")
    m.r_eff = r_eff
    return m


def summarize(m: ResidencyMap) -> dict:
    t_total = m.t_total
    per_range = []
    for r in sorted(m.ranges, key=lambda x: -x["units"]):
        first, n = r["first_unit"], r["units"]
        always = never = 0
        weighted = 0.0
        bits = 0
        periods = 0
        for u in range(first, first + n):
            t = m.resident_time(u)
            b = 8 * m.unit_resident_bytes[u]
            weighted += b * t
            bits += b
            k = m.offsets[u + 1] - m.offsets[u]
            periods += k
            if k == 1 and m.flat[2 * m.offsets[u]] == 0 and \
                    m.flat[2 * m.offsets[u] + 1] == m.images - 1:
                always += 1
            elif k == 0:
                never += 1
        per_range.append({
            "allocation_id": r["allocation_id"], "units": n,
            "bytes": bits // 8,
            "always": always / n, "never": never / n,
            "partial": (n - always - never) / n,
            "resident_time_fraction": weighted / (bits * t_total),
            "periods_per_unit": periods / n,
        })
    return {"r_eff_bits": m.r_eff, "surface_bits": m.meta["surface_bits"],
            "t_total_ms": t_total, "periods": m.meta["periods"],
            "per_range": per_range}


def print_summary(prefix: str, m: ResidencyMap) -> None:
    s = summarize(m)
    meta = m.meta
    print(f"== {prefix}")
    print(f"  unit {meta['unit_bytes']} B, units {m.units}, images {m.images}, "
          f"stride {meta['stride']}, every {meta['probe_every']}, "
          f"alternate {meta['alternate']}, threshold {meta['threshold_cycles']}, "
          f"window {meta['observation_window_images']} images")
    print(f"  T_total {s['t_total_ms']:.3f} ms ({s['t_total_ms'] / m.images:.4f} "
          f"ms/image), periods {s['periods']}")
    print(f"  R_eff_bits {s['r_eff_bits']:.0f} = {s['r_eff_bits'] / 8 / 1e6:.3f} MB "
          f"of surface {s['surface_bits'] / 8 / 1e6:.3f} MB "
          f"({100 * s['r_eff_bits'] / s['surface_bits']:.2f} %)  "
          "[recomputed == runner]")
    print("  allocation               bytes      units  resident-time  always "
          " never  partial  periods/unit")
    for r in s["per_range"]:
        print(f"  {r['allocation_id']:22s} {r['bytes']:10d} {r['units']:9d}"
              f"   {r['resident_time_fraction']:8.4f}    {r['always']:6.3f} "
              f"{r['never']:6.3f}  {r['partial']:6.3f}   {r['periods_per_unit']:8.3f}")


def compare(a: ResidencyMap, b: ResidencyMap) -> dict:
    """Unit-by-unit comparison keyed by (allocation id, local unit index),
    so maps from different processes (different VAs/PAs) are comparable."""
    if a.meta["unit_bytes"] != b.meta["unit_bytes"]:
        raise MapError("maps use different probe units")
    rb = {r["allocation_id"]: r for r in b.ranges}
    abs_diffs = []
    same_class = 0
    total = 0
    for r in a.ranges:
        other = rb.get(r["allocation_id"])
        if other is None or other["units"] != r["units"]:
            raise MapError(f"allocation {r['allocation_id']} differs between maps")
        for k in range(r["units"]):
            fa = a.resident_time(r["first_unit"] + k) / a.t_total
            fb = b.resident_time(other["first_unit"] + k) / b.t_total
            abs_diffs.append(abs(fa - fb))
            ca = 0 if fa == 0 else (2 if fa == 1 else 1)
            cb = 0 if fb == 0 else (2 if fb == 1 else 1)
            same_class += ca == cb
            total += 1
    abs_diffs.sort()
    return {
        "units": total,
        "mean_abs_diff": sum(abs_diffs) / total,
        "p99_abs_diff": abs_diffs[min(total - 1, int(0.99 * total))],
        "max_abs_diff": abs_diffs[-1],
        "same_class": same_class / total,
        "r_eff_a": a.r_eff, "r_eff_b": b.r_eff,
        "r_eff_rel_diff": abs(a.r_eff - b.r_eff) / max(a.r_eff, b.r_eff),
    }


def _write_synthetic(prefix: str, unit_bytes: int, image_time, unit_bytes_arr,
                     periods_per_unit, ranges, neutral=0, tamper=False):
    flat = []
    offsets = [0]
    for plist in periods_per_unit:
        for s, e in plist:
            flat += [s, e]
        offsets.append(len(flat) // 2)
    blob = HEADER.pack(MAGIC, 1, unit_bytes, len(unit_bytes_arr),
                       len(image_time), len(flat) // 2)
    blob += array.array("d", image_time).tobytes()
    blob += array.array("H", unit_bytes_arr).tobytes()
    blob += array.array("Q", offsets).tobytes()
    blob += array.array("I", flat).tobytes()
    Path(prefix + "_residency.bin").write_bytes(blob)
    t_total = sum(image_time)
    pre = [0.0]
    for t in image_time:
        pre.append(pre[-1] + t)
    r_eff = sum(8 * unit_bytes_arr[u] * sum(pre[e + 1] - pre[s] for s, e in pl)
                for u, pl in enumerate(periods_per_unit)) / t_total
    meta = {
        "schema": "gpu-m2d.g8.residency-map.v1",
        "bin_sha256": hashlib.sha256(blob).hexdigest(),
        "unit_bytes": unit_bytes, "units": len(unit_bytes_arr),
        "images": len(image_time), "periods": len(flat) // 2,
        "stride": 1, "probe_every": 1, "alternate": True,
        "threshold_cycles": 440, "observation_window_images": 1,
        "neutral_mismatches": neutral,
        "surface_bits": sum(8 * b for b in unit_bytes_arr),
        "t_total_ms": t_total, "r_eff_bits": r_eff * (1.01 if tamper else 1.0),
        "ranges": ranges,
    }
    Path(prefix + "_residency.json").write_text(json.dumps(meta))


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        p = str(Path(tmp) / "m")
        ranges = [{"allocation_id": "w", "first_unit": 0, "units": 3},
                  {"allocation_id": "s", "first_unit": 3, "units": 1}]
        times = [1.0] * 9 + [11.0]  # total 20
        periods = [[(0, 9)], [], [(0, 2), (7, 9)], [(4, 9)]]
        _write_synthetic(p, 32, times, [32, 32, 32, 8], periods, ranges)
        m = load_map(p)
        assert m.t_total == 20.0
        assert m.resident_time(0) == 20.0 and m.resident_time(1) == 0.0
        assert m.resident_time(2) == 3.0 + 13.0
        # (256*20 + 256*16 + 64*16) / 20 = 10240 / 20
        assert math.isclose(m.r_eff, 512.0), m.r_eff
        s = summarize(m)
        w = next(r for r in s["per_range"] if r["allocation_id"] == "w")
        assert math.isclose(w["always"], 1 / 3) and math.isclose(w["never"], 1 / 3)
        # identical maps compare perfectly
        c = compare(m, load_map(p))
        assert c["mean_abs_diff"] == 0 and c["same_class"] == 1.0
        # fail-closed paths
        for kind in ("tamper", "neutral", "overlap", "sha"):
            q = str(Path(tmp) / kind)
            bad_periods = [[(0, 4), (5, 9)]] + periods[1:] if kind == "overlap" \
                else periods
            _write_synthetic(q, 32, times, [32, 32, 32, 8], bad_periods, ranges,
                             neutral=1 if kind == "neutral" else 0,
                             tamper=kind == "tamper")
            if kind == "sha":
                blob = bytearray(Path(q + "_residency.bin").read_bytes())
                blob[-1] ^= 1
                Path(q + "_residency.bin").write_bytes(bytes(blob))
            try:
                load_map(q)
            except MapError:
                pass
            else:
                raise AssertionError(f"{kind} map must be refused")
    print("residency_map self-test: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("summary", "compare"))
    parser.add_argument("prefix", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.command is None or not args.prefix:
        parser.error("give a command and map prefix(es)")
    try:
        maps = [load_map(p) for p in args.prefix]
    except MapError as exc:
        print(f"G8_RESIDENCY_MAP_FAIL {exc}")
        return 2
    if args.command == "summary":
        for prefix, m in zip(args.prefix, maps):
            print_summary(prefix, m)
        print(f"G8_RESIDENCY_MAP_VERIFIED maps={len(maps)}")
        return 0
    if len(maps) != 2:
        parser.error("compare needs exactly two prefixes")
    try:
        c = compare(*maps)
    except MapError as exc:
        print(f"G8_RESIDENCY_MAP_FAIL {exc}")
        return 2
    print(f"compare {args.prefix[0]}  vs  {args.prefix[1]}")
    print(f"  units {c['units']}; per-unit resident-time fraction |diff| mean "
          f"{c['mean_abs_diff']:.5f}, p99 {c['p99_abs_diff']:.4f}, max "
          f"{c['max_abs_diff']:.4f}; same class (always/never/partial) "
          f"{100 * c['same_class']:.3f} %")
    print(f"  R_eff_bits {c['r_eff_a']:.0f} vs {c['r_eff_b']:.0f} "
          f"(rel diff {100 * c['r_eff_rel_diff']:.4f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
