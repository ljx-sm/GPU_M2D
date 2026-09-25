#!/usr/bin/env python3
"""GPU_M2D G7-v2: APPEND levels L6-L9 (BER 3e-6, 5e-6, 7e-6, 1e-5 -- user
decision 2026-09-25) to the already-frozen g7v2_imagenet1k_resnet50 entry
of tools/g5_faultinj/fault_model.py, derived from the SAME measured R.

The original five levels (L1 1e-8 .. L5 1e-6) were frozen from the v2
bootstrap run by freeze_g7v2_levels.py; this tool never touches them. It
runs AFTER that freeze and BEFORE any --level L6..L9 campaign:

  /usr/bin/python3 tools/g7_prep/extend_g7v2_levels.py [--check]

Fail-closed chain (mirrors the freeze tool):
  1. regression-checks the derivation pipeline against the v1 workload's
     frozen literals (same code path must reproduce them exactly);
  2. locates the newest v2 bootstrap.json and verifies schema, workload,
     engine_sha256 == sha256 of the CURRENT v2 engine, and that its
     measured R equals the entry's frozen resident_bytes_nominal -- the
     extension must ride on the exact residency the campaigns ran with;
  3. re-derives the FIVE existing levels and requires exact agreement
     (the base table must be sound before anything is appended);
  4. derives the four new levels from the same R; requires every new
     bits < EXHAUSTIVE_BITS_LIMIT so the appended literals stay
     exhaustive-solver provenance, and requires no level-name/BER
     collision with the existing five;
  5. rewrites the entry in place (comment records both the original
     freeze provenance and the extension decision), atomically via
     os.replace;
  6. reloads the module and requires assert_frozen_levels + self_test
     to pass on the nine-level table.

Idempotent: if all four appended levels already exist with agreeing
literals, verifies only and exits 0. --check never writes (exit 3 while
the extension is still pending).

stdlib-only (runs under /usr/bin/python3).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_g7v2_levels import (  # noqa: E402  (shared fail-closed helpers)
    ENGINE, SCHEMA, WORKLOAD, die, entry_block, load_fault_model,
    newest_bootstrap, sha256, v1_regression,
)

FAULT_MODEL_PY = Path(__file__).resolve().parents[2] / "tools/g5_faultinj/fault_model.py"

# The original freeze ladder (must already be present, untouched).
CORE_BERS = [("L1", 1e-8), ("L2", 5e-8), ("L3", 1e-7), ("L4", 5e-7),
             ("L5", 1e-6)]
# The 2026-09-25 extension ladder.
EXT_BERS = [("L6", 3e-6), ("L7", 5e-6), ("L8", 7e-6), ("L9", 1e-5)]


def derive(r_bytes: int, fm, ladder: list[tuple[str, float]]) -> list[dict]:
    r_bits = r_bytes * 8
    out = []
    for name, ber in ladder:
        bits = round(ber * r_bits)
        if bits >= fm.EXHAUSTIVE_BITS_LIMIT:
            die(f"{name} bits={bits} >= EXHAUSTIVE_BITS_LIMIT "
                f"{fm.EXHAUSTIVE_BITS_LIMIT} -- the extension ladder must "
                "stay in exhaustive-solver territory")
        try:
            s, d, t = fm.resolve_composition(bits)
        except fm.ModelError as exc:
            die(f"R={r_bytes:,} B gives {name} bits={bits} ({exc})")
        assert s + 2 * d + 3 * t == bits, (s, d, t, bits)
        out.append({"level": name, "ber": ber, "bits": bits,
                    "s": s, "d": d, "t": t})
    return out


def render_levels(levels: list[dict]) -> list[str]:
    lines = []
    for row in levels:
        head = ('            {"level": "' + row["level"] + '", "ber": '
                + format(row["ber"], "g") + ', "bits": ' + str(row["bits"])
                + ', "s": ' + str(row["s"]) + ', "d": ' + str(row["d"]) + ",")
        tail = ' "t": ' + str(row["t"]) + "},"
        if len(head) + len(tail) <= 79:
            lines.append(head + tail)
        else:
            lines.append(head)
            lines.append("             " + tail[1:])
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify only; never write")
    args = parser.parse_args()

    fm = load_fault_model()
    v1_regression(fm)

    boot_path = newest_bootstrap()
    try:
        boot = json.loads(boot_path.read_text())
    except json.JSONDecodeError as exc:
        die(f"bootstrap json {boot_path} is unreadable/truncated: {exc}")
    if boot.get("schema_version") != SCHEMA:
        die(f"bootstrap schema {boot.get('schema_version')!r} != {SCHEMA!r}")
    if boot.get("workload") != WORKLOAD:
        die(f"bootstrap workload {boot.get('workload')!r} != {WORKLOAD!r} "
            "(the newest bootstrap run is not the v2 one)")
    engine_sha = sha256(ENGINE)
    if boot.get("engine_sha256") != engine_sha:
        die(f"bootstrap engine_sha256 {str(boot.get('engine_sha256'))[:12]}... "
            f"!= current {ENGINE} sha256 {engine_sha[:12]}... -- the engine "
            "file changed since the bootstrap run")
    r_bytes = boot.get("resident_bytes", 0)
    if not isinstance(r_bytes, int) or r_bytes <= 0:
        die(f"bootstrap resident_bytes {r_bytes!r} is not a positive integer")

    entry = fm.WORKLOADS[WORKLOAD]
    if entry["resident_bytes_nominal"] != r_bytes:
        die(f"frozen R={entry['resident_bytes_nominal']:,} != bootstrap "
            f"R={r_bytes:,} -- the entry no longer rides on the measured "
            "residency; re-examine before extending")

    # the five core levels must still re-derive exactly (sound base)
    core = derive(r_bytes, fm, CORE_BERS)
    frozen_levels = entry["levels"]
    if len(frozen_levels) < len(CORE_BERS):
        die(f"frozen table has {len(frozen_levels)} levels; the five core "
            "levels must exist before an extension")
    for row, now in zip(core, frozen_levels[:len(CORE_BERS)]):
        if (row["level"], row["ber"], row["bits"], row["s"], row["d"],
                row["t"]) != (now["level"], now["ber"], now["bits"],
                              now["s"], now["d"], now["t"]):
            die(f"core level {row['level']} drifted from its derivation: "
                f"derived {row} vs frozen {now}")

    ext = derive(r_bytes, fm, EXT_BERS)
    frozen_names = [e["level"] for e in frozen_levels]
    clashes = [row["level"] for row in ext if row["level"] in frozen_names]
    if len(clashes) != len(EXT_BERS) and clashes:
        die(f"levels {clashes} already present but not all four extension "
            "levels -- inconsistent partial extension, re-derive manually")

    if len(clashes) == len(EXT_BERS):
        # already extended: verify agreement, rewrite nothing
        frozen_by_name = {e["level"]: e for e in frozen_levels}
        for row in ext:
            now = frozen_by_name[row["level"]]
            if (row["ber"], row["bits"], row["s"], row["d"],
                    row["t"]) != (now["ber"], now["bits"], now["s"],
                                  now["d"], now["t"]):
                die(f"appended level {row['level']} disagrees with a fresh "
                    f"derivation from the SAME R: derived {row} vs frozen "
                    f"{now}")
        print(f"already extended to {len(frozen_levels)} levels; literals "
              "agree; no rewrite")
    else:
        if args.check:
            print("--check: would append to "
                  f"{WORKLOAD} (R={r_bytes:,} B unchanged):")
            for row in ext:
                print(f'  {row["level"]}: ber={row["ber"]:g} bits={row["bits"]}'
                      f' s={row["s"]} d={row["d"]} t={row["t"]}')
            return 3  # distinct exit: inspected-but-NOT-extended
        if len(frozen_levels) != len(CORE_BERS):
            die(f"frozen table has {len(frozen_levels)} levels, expected "
                f"exactly {len(CORE_BERS)} before a first extension")
        run_id = boot_path.parent.name
        snap = boot.get("snapshot_manifest", {}).get("counts", {})
        out = [
            '    # G7-v2: ResNet-50/ImageNet-1K on the HEAD-QUANTIZED v2 engine',
            '    # (user decision 2026-09-24: every weighted op INT8 per-channel,',
            '    # including the classifier-head Gemm; quantizer wrapper',
            '    # tools/g7_prep/quantize_g7_qdq_head.py). The BER ladder is the',
            '    # G5 core ladder for a directly comparable curve.',
            '    # R was MEASURED by the bootstrap run',
            f'    # artifacts/g7/campaign/{run_id}',
            f'    # (engine sha256 {engine_sha[:12]}...,',
            f'    # snapshot allocations={snap.get("allocations")}, '
            f'pa_pages={snap.get("pa_pages")}, rows={snap.get("rows")});',
            '    # L1-L5 frozen by tools/g7_prep/freeze_g7v2_levels.py.',
            '    # EXTENDED 2026-09-25 (user decision) with L6 3e-6, L7 5e-6,',
            '    # L8 7e-6, L9 1e-5 by tools/g7_prep/extend_g7v2_levels.py,',
            '    # derived from the SAME R (engine and bootstrap above still',
            '    # match -- no re-bootstrap). All B < 3000 so every literal is',
            '    # an exhaustive-solver literal.',
            f'    "{WORKLOAD}": {{',
            f'        "resident_bytes_nominal": '
            + f'{r_bytes:,}'.replace(",", "_") + ",",
            '        "levels": [',
        ]
        out += render_levels(core + ext)
        out += ["        ],", "    },"]

        source = FAULT_MODEL_PY.read_text()
        start, end = entry_block(source)
        lines = source.splitlines(keepends=True)
        block = "\n".join(out) + "\n"
        new_source = "".join(lines[:start]) + block + "".join(lines[end + 1:])
        # syntax-check BEFORE the atomic replace: a render regression can
        # never leave a fault_model.py that no longer parses
        compile(new_source, str(FAULT_MODEL_PY), "exec")
        tmp = FAULT_MODEL_PY.with_suffix(".py.ext_tmp")
        tmp.write_text(new_source)
        os.replace(tmp, FAULT_MODEL_PY)
        print(f"rewrote {FAULT_MODEL_PY} entry {WORKLOAD} "
              f"(lines {start + 1}-{end + 1}): 5 -> 9 levels, R unchanged "
              f"at {r_bytes:,} B")

    # final verification against the (possibly rewritten) file
    fm2 = load_fault_model()
    fm2.assert_frozen_levels(WORKLOAD)
    if fm2.self_test() != 0:
        die("fault_model.self_test() failed after the rewrite")
    final = fm2.WORKLOADS[WORKLOAD]
    if final["resident_bytes_nominal"] != r_bytes:
        die("frozen R changed during the extension")
    print(f"extension VERIFIED: workload={WORKLOAD} "
          f"{len(final['levels'])} levels R={r_bytes:,} B")
    for row in final["levels"][len(CORE_BERS):]:
        print(f'  {row["level"]}: ber={row["ber"]:g} bits={row["bits"]} '
              f's={row["s"]} d={row["d"]} t={row["t"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
