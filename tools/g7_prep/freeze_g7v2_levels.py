#!/usr/bin/env python3
"""GPU_M2D G7-v2: freeze the head-quantized v2 ResNet-50 workload's level
table from its bootstrap run's MEASURED residency R (user decision
2026-09-24: v2 engine, BER ladder 1e-8 .. 1e-6 = the G5 core ladder).

Run AFTER the v2 bootstrap campaign (sudo scripts/run_g2_observer_probe.sh
--api g5campaign --bootstrap --workload g7v2_imagenet1k_resnet50 ...) and
BEFORE any --level campaign of that workload:

  /usr/bin/python3 tools/g7_prep/freeze_g7v2_levels.py \
      [--bootstrap-json PATH] [--check]

What it does (all fail-closed):
  1. locates the newest artifacts/g7/campaign/run_bootstrap_gpu0_*/
     bootstrap.json (or takes an explicit --bootstrap-json);
  2. verifies it is a v2-bootstrap record: schema gpu-m2d.g5.bootstrap.v1,
     workload g7v2_imagenet1k_resnet50, and its engine_sha256 equals the
     sha256 of the CURRENT /data1/luojx/g7_models/resnet50/clean.engine
     (guards against freezing the v1 engine's R by accident);
  3. regression-checks the derivation pipeline itself: re-derives the v1
     workload (g7_imagenet1k_resnet50) from ITS frozen R + BERs through
     the same code path and requires exact agreement with ITS frozen
     literals (the pipeline is provably the one that produced v1);
  4. derives bits = round(ber * R * 8) and (s, d, t) =
     fault_model.resolve_composition(bits) for the five v2 BER levels and
     rewrites the g7v2_imagenet1k_resnet50 entry in
     tools/g5_faultinj/fault_model.py in place (placeholder R=0/literals 0
     -> measured R + exhaustive-solver literals, comment citing the
     bootstrap run);
  5. reloads the rewritten module and requires
     fault_model.assert_frozen_levels("g7v2_imagenet1k_resnet50") and
     fault_model.self_test() to pass.

Idempotent: if the entry is already frozen with the SAME measured R the
script verifies only and exits 0 (no rewrite). --check never writes.

stdlib-only (runs under /usr/bin/python3).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FAULT_MODEL_PY = PROJECT / "tools/g5_faultinj/fault_model.py"
CAMPAIGN_ROOT = PROJECT / "artifacts/g7/campaign"
ENGINE = Path("/data1/luojx/g7_models/resnet50/clean.engine")

WORKLOAD = "g7v2_imagenet1k_resnet50"
V1_WORKLOAD = "g7_imagenet1k_resnet50"
SCHEMA = "gpu-m2d.g5.bootstrap.v1"
V2_BERS = [("L1", 1e-8), ("L2", 5e-8), ("L3", 1e-7), ("L4", 5e-7),
           ("L5", 1e-6)]


def die(message: str) -> None:
    print(f"freeze_g7v2_levels: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fault_model():
    sys.path.insert(0, str(FAULT_MODEL_PY.parent))
    if "fault_model" in sys.modules:
        del sys.modules["fault_model"]
    return importlib.import_module("fault_model")


def newest_bootstrap() -> Path:
    candidates = sorted(CAMPAIGN_ROOT.glob("run_bootstrap_gpu0_*/bootstrap.json"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        die(f"no run_bootstrap_gpu0_*/bootstrap.json under {CAMPAIGN_ROOT}")
    return candidates[-1]


def derive_levels(r_bytes: int, fm) -> list[dict]:
    r_bits = r_bytes * 8
    levels = []
    for name, ber in V2_BERS:
        bits = round(ber * r_bits)
        try:
            s, d, t = fm.resolve_composition(bits)
        except fm.ModelError as exc:
            die(f"R={r_bytes:,} B gives {name} bits={bits} "
                f"({exc}) -- the residency is too small for this BER "
                "ladder; re-examine the bootstrap measurement")
        assert s + 2 * d + 3 * t == bits, (s, d, t, bits)
        levels.append({"level": name, "ber": ber, "bits": bits,
                       "s": s, "d": d, "t": t})
    return levels


def v1_regression(fm) -> None:
    """The same derivation path must reproduce the frozen v1 literals."""
    entry = fm.WORKLOADS[V1_WORKLOAD]
    r_bits = entry["resident_bytes_nominal"] * 8
    for row in entry["levels"]:
        bits = round(row["ber"] * r_bits)
        comp = fm.resolve_composition(bits)
        if bits != row["bits"] or comp != (row["s"], row["d"], row["t"]):
            die(f"derivation regression: v1 {row['level']} derived "
                f"({bits}, {comp}) != frozen ({row['bits']}, "
                f"({row['s']}, {row['d']}, {row['t']}))")
    print(f"regression PASS: pipeline reproduces all {len(entry['levels'])} "
          f"v1 literals")


def entry_block(source: str) -> tuple[int, int]:
    """(start, end) line span of the g7v2 entry inside WORKLOADS. `end` is
    the entry's closing '},'; `start` additionally swallows any run of
    contiguous 4-space '# ' comment lines immediately above the key line,
    so a rewrite replaces the placeholder's comment block too. The brace
    scan starts AT the key line (comment lines carry no braces and would
    otherwise read as depth 0)."""
    lines = source.splitlines(keepends=True)
    key = None
    for i, line in enumerate(lines):
        if re.match(rf'^    "{WORKLOAD}": \{{\s*$', line):
            key = i
            break
    if key is None:
        die(f"entry \"{WORKLOAD}\" not found in {FAULT_MODEL_PY}")
    start = key
    while start > 0 and lines[start - 1].startswith("    # "):
        start -= 1
    depth = 0
    for j in range(key, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if j > key and depth == 0:
            return start, j
    die(f"unbalanced braces after entry key line {key + 1}")


def render_entry(r_bytes: int, levels: list[dict], run_id: str,
                 engine_sha: str, note_extra: str) -> str:
    out = [
        f'    # G7-v2: ResNet-50/ImageNet-1K on the HEAD-QUANTIZED v2 engine',
        f'    # (user decision 2026-09-24: every weighted op INT8 per-channel,',
        f'    # including the classifier-head Gemm; quantizer wrapper',
        f'    # tools/g7_prep/quantize_g7_qdq_head.py). The BER ladder is the',
        f'    # G5 core ladder (1e-8 .. 1e-6) for a directly comparable curve.',
        '    # R was MEASURED by the bootstrap run',
        f'    # artifacts/g7/campaign/{run_id}',
        f'    # (engine sha256 {engine_sha[:12]}...,',
        f'    #{note_extra})',
        '    # and the five user-selected BER levels (2026-09-24) derived',
        '    # from it; all B < 3000 so the compositions are exhaustive-',
        '    # solver literals. Frozen in place by tools/g7_prep/',
        '    # freeze_g7v2_levels.py.',
        f'    "{WORKLOAD}": {{',
        f'        "resident_bytes_nominal": {r_bytes:,}'.replace(",", "_") + ",",
        f'        "levels": [',
    ]
    for row in levels:
        head = ('            {"level": "' + row["level"] + '", "ber": '
                + format(row["ber"], "g") + ', "bits": ' + str(row["bits"])
                + ', "s": ' + str(row["s"]) + ', "d": ' + str(row["d"]) + ",")
        tail = ' "t": ' + str(row["t"]) + "},"
        if len(head) + len(tail) <= 79:
            out.append(head + tail)
        else:
            out.append(head)
            out.append("             " + tail[1:])
    out.append("        ],")
    out.append("    },")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-json", type=Path, default=None)
    parser.add_argument("--check", action="store_true",
                        help="verify only; never write")
    args = parser.parse_args()

    fm = load_fault_model()
    v1_regression(fm)

    boot_path = args.bootstrap_json or newest_bootstrap()
    try:
        boot = json.loads(boot_path.read_text())
    except json.JSONDecodeError as exc:
        die(f"bootstrap json {boot_path} is unreadable/truncated: {exc}")
    if boot.get("schema_version") != SCHEMA:
        die(f"bootstrap schema {boot.get('schema_version')!r} != {SCHEMA!r}")
    if boot.get("workload") != WORKLOAD:
        die(f"bootstrap workload {boot.get('workload')!r} != {WORKLOAD!r} "
            "(point me at the v2 bootstrap run, or rerun --bootstrap)")
    engine_sha = sha256(ENGINE)
    if boot.get("engine_sha256") != engine_sha:
        die(f"bootstrap engine_sha256 {str(boot.get('engine_sha256'))[:12]}... "
            f"!= current {ENGINE} sha256 {engine_sha[:12]}... -- the engine "
            "file changed since the bootstrap run")
    r_bytes = boot.get("resident_bytes", 0)
    if not isinstance(r_bytes, int) or r_bytes <= 0:
        die(f"bootstrap resident_bytes {r_bytes!r} is not a positive integer")

    run_id = boot_path.parent.name
    snap = boot.get("snapshot_manifest", {}).get("counts", {})
    note = (f" snapshot allocations={snap.get('allocations')}, "
            f"pa_pages={snap.get('pa_pages')}, rows={snap.get('rows')}")

    levels = derive_levels(r_bytes, fm)

    frozen_now = fm.WORKLOADS[WORKLOAD]
    if frozen_now["resident_bytes_nominal"] == r_bytes:
        # already frozen with this R: verify agreement, rewrite nothing
        frozen_levels = frozen_now["levels"]
        if (len(frozen_levels) != len(levels) or
                [e["level"] for e in frozen_levels] != [e["level"] for e in levels] or
                [e["ber"] for e in frozen_levels] != [e["ber"] for e in levels]):
            die("already-frozen level set (names/BERs/count) differs from "
                f"the required ladder {[(n, b) for n, b in V2_BERS]} -- "
                "re-derive and re-confirm manually")
        for row, now in zip(levels, frozen_levels):
            if (row["bits"], row["s"], row["d"], row["t"]) != \
                    (now["bits"], now["s"], now["d"], now["t"]):
                die(f"already-frozen {row['level']} disagrees with a fresh "
                    f"derivation from the SAME R: derived {row} vs frozen "
                    f"{now} -- re-derive and re-confirm manually")
        print(f"already frozen at R={r_bytes:,} B; literals agree; no rewrite")
    else:
        if frozen_now["resident_bytes_nominal"] != 0:
            die(f"entry frozen at R="
                f"{frozen_now['resident_bytes_nominal']:,} but bootstrap "
                f"measured {r_bytes:,} -- refusing to silently re-freeze")
        if args.check:
            print("--check: entry still a placeholder (R=0); would freeze "
                  f"R={r_bytes:,} B, levels:")
            for row in levels:
                print(f'  {row["level"]}: ber={row["ber"]:g} bits={row["bits"]}'
                      f' s={row["s"]} d={row["d"]} t={row["t"]}')
            return 3  # distinct exit: inspected-but-NOT-frozen
        source = FAULT_MODEL_PY.read_text()
        start, end = entry_block(source)
        block = render_entry(r_bytes, levels, run_id, engine_sha, note)
        lines = source.splitlines(keepends=True)
        new_source = "".join(lines[:start]) + block + "".join(lines[end + 1:])
        # atomic replace: a crash mid-write can never leave a truncated
        # fault_model.py behind
        tmp = FAULT_MODEL_PY.with_suffix(".py.freeze_tmp")
        tmp.write_text(new_source)
        os.replace(tmp, FAULT_MODEL_PY)
        print(f"rewrote {FAULT_MODEL_PY} entry {WORKLOAD} "
              f"(lines {start + 1}-{end + 1}) with R={r_bytes:,} B")

    # final verification against the (possibly rewritten) file
    fm2 = load_fault_model()
    fm2.assert_frozen_levels(WORKLOAD)
    if fm2.self_test() != 0:
        die("fault_model.self_test() failed after the rewrite")
    entry = fm2.WORKLOADS[WORKLOAD]
    if entry["resident_bytes_nominal"] != r_bytes:
        die("frozen R does not match the bootstrap measurement")
    print(f"freeze VERIFIED: workload={WORKLOAD} R={r_bytes:,} B "
          f"({r_bytes * 8:,} bits)")
    for row in entry["levels"]:
        print(f'  {row["level"]}: ber={row["ber"]:g} bits={row["bits"]} '
              f's={row["s"]} d={row["d"]} t={row["t"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
