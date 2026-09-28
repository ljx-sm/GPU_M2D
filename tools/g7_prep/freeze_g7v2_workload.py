#!/usr/bin/env python3
"""GPU_M2D G7-v2 five-model extension: freeze ONE new workload's
seven-level table from ITS bootstrap run's MEASURED residency R (user
decision 2026-09-27: the SAME v2 head-quantized INT8 engine family,
protocol, and settings as the ResNet-50 campaign; BER ladder L1..L7 =
1e-7, 5e-7, 1e-6, 3e-6, 5e-6, 7e-6, 1e-5 -- the ResNet-50 curve's
L3-L9 BERs).

Run AFTER that model's bootstrap campaign and BEFORE any --level
campaign of that workload:

  sudo scripts/run_g2_observer_probe.sh --api g5campaign --device 0 \\
      --bootstrap --workload g7v2_imagenet1k_<model> \\
      --engine /data1/luojx/g7_models/<model>/clean.engine ...   (user, tmux)

  /usr/bin/python3 tools/g7_prep/freeze_g7v2_workload.py \\
      --workload g7v2_imagenet1k_<model> [--check]

What it does (all fail-closed):
  1. locates the newest artifacts/g7/campaign/run_bootstrap_gpu0_*/
     bootstrap.json WHOSE workload field matches -- the campaign root now
     holds several models' bootstraps, so newest-overall (the
     freeze_g7v2_levels.py rule) would be wrong here (or takes an
     explicit --bootstrap-json);
  2. verifies it is a bootstrap record of the expected schema whose
     engine_sha256 equals the sha256 of that model's CURRENT
     /data1/luojx/g7_models/<model>/clean.engine (guards against freezing
     the wrong engine's R by accident);
  3. for a workload in the EXCLUDES table (fault-surface scoping, see
     below): reads the bootstrap run's own g1_5_allocations.csv, requires
     the per-allocation sizes to reconcile with the measured R, and
     derives the INJECTION SURFACE R_eff = R - excluded allocations --
     otherwise R_eff is the full measured R;
  4. regression-checks the derivation pipeline itself: re-derives the
     FROZEN g7v2_imagenet1k_resnet50 table (all 9 levels) from ITS frozen
     R through the same code path and requires exact agreement (the
     pipeline is provably the one that produced the ResNet-50 campaign);
  5. derives bits = round(ber * R_eff * 8) and (s, d, t) for the seven
     levels with the EXHAUSTIVE solver FORCED regardless of size (a
     ViT-scale R pushes L7 past the 3000-bit windowed threshold), and
     additionally requires the DEFAULT path (windowed above 3000 bits) to
     agree, so the campaign-start assert_frozen_levels can never disagree
     with the frozen literals;
  6. rewrites the workload's PLACEHOLDER entry (R=0, all-zero literals)
     in tools/g5_faultinj/fault_model.py in place (a scoped workload's
     entry also carries its surface_excludes field);
  7. reloads the rewritten module and requires
     fault_model.assert_frozen_levels(workload) and
     fault_model.self_test() to pass.

Idempotent: if the entry is already frozen with the SAME measured R the
script verifies only and exits 0 (no rewrite). --check never writes.
Refuses to re-freeze an entry frozen at a DIFFERENT R.

stdlib-only (runs under /usr/bin/python3).
"""

from __future__ import annotations

import argparse
import csv
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
G7_MODELS_ROOT = Path("/data1/luojx/g7_models")

# the five extension workloads and their (already-built, already
# verified) v2 head-quantized engines
MODELS = {
    "g7v2_imagenet1k_mobilenetv3_large_100":
        ("mobilenetv3_large_100", "MobileNetV3-Large-100"),
    "g7v2_imagenet1k_efficientnet_b0":
        ("efficientnet_b0", "EfficientNet-B0"),
    "g7v2_imagenet1k_vit_base_patch16_224":
        ("vit_base_patch16_224", "ViT-Base/16"),
    "g7v2_imagenet1k_deit_small_patch16_224":
        ("deit_small_patch16_224", "DeiT-Small/16"),
    "g7v2_imagenet1k_swin_tiny_patch4_window7_224":
        ("swin_tiny_patch4_window7_224", "Swin-Tiny"),
}

REGRESSION_WORKLOAD = "g7v2_imagenet1k_resnet50"
SCHEMA = "gpu-m2d.g5.bootstrap.v1"
# user decision 2026-09-27: seven levels, the ResNet-50 curve's L3-L9 BERs
LADDER = [("L1", 1e-7), ("L2", 5e-7), ("L3", 1e-6), ("L4", 3e-6),
          ("L5", 5e-6), ("L6", 7e-6), ("L7", 1e-5)]

# G7-v2 fault-surface scoping (user decision 2026-09-27, MobileNetV3
# ONLY): the TensorRT create_execution_context-phase PRIVATE buffers are
# runtime CONTROL state, not model data. The 2026-09-27 three-seed
# diagnostic L1 runs (artifacts/g7/campaign/run_L1_gpu0_*) proved that a
# flip at an address-bearing offset of trt-internal-5 (246,272 B) kills
# the runner PROCESS with a CUDA illegal memory access on the FIRST
# injected inference -- a process-fatal reliability event, not an
# output-observable fault -- while every flip in the weights, scratch,
# deserialize constants, and input binding survived and restored (even
# benign-offset internal-5 hits completed their trials). trt-internal-6
# (2,048 B, same allocation class) is excluded with it. ResNet-50 and
# the other extension models keep FULL-surface injection until they
# individually show the same failure; the full-surface protocol stays
# available as the control experiment (paper appendix).
EXCLUDES: dict[str, tuple[str, ...]] = {
    "g7v2_imagenet1k_mobilenetv3_large_100":
        ("trt-internal-5", "trt-internal-6"),
}


def die(message: str) -> None:
    print(f"freeze_g7v2_workload: FAIL: {message}", file=sys.stderr)
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


def read_bootstrap(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        die(f"bootstrap json {path} is unreadable/truncated: {exc}")


def newest_bootstrap(workload: str) -> Path:
    matches: list[tuple[float, Path]] = []
    seen: dict[str, int] = {}
    for path in sorted(CAMPAIGN_ROOT.glob("run_bootstrap_gpu0_*/bootstrap.json")):
        doc = read_bootstrap(path)
        wl = doc.get("workload", "?")
        seen[wl] = seen.get(wl, 0) + 1
        if wl == workload and doc.get("schema_version") == SCHEMA:
            matches.append((path.stat().st_mtime, path))
    if not matches:
        others = ", ".join(f"{w} ({n})" for w, n in sorted(seen.items()))
        die(f"no bootstrap.json of workload {workload!r} under "
            f"{CAMPAIGN_ROOT} (bootstrap runs present: {others or 'none'}) "
            "-- run that model's --bootstrap campaign first")
    return max(matches)[1]


def effective_r(boot: dict, boot_path: Path,
                excludes: tuple[str, ...]) -> tuple[int, int, int]:
    """(R_eff, R_full, excluded_bytes): the workload's INJECTION-SURFACE
    residency -- the bootstrap's FULL measured R minus the excluded
    allocations' sizes, read from the bootstrap run's own
    g1_5_allocations.csv so the subtraction uses the same measurement the
    bootstrap recorded. Fail-closed: the registry must exist, its sizes
    must sum to the measured R, and every excluded id must be present."""
    r_full = boot["resident_bytes"]
    if not excludes:
        return r_full, r_full, 0
    registry = boot_path.parent / "g1_5_allocations.csv"
    if not registry.is_file():
        die(f"surface scoping needs the bootstrap run's allocation "
            f"registry {registry}, which is missing")
    sizes: dict[str, int] = {}
    with open(registry, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            sizes[row["allocation_id"]] = int(row["size_bytes"])
    total = sum(sizes.values())
    if total != r_full:
        die(f"allocation registry {registry} sums to {total:,} B but the "
            f"bootstrap measured {r_full:,} B -- per-allocation sizes "
            "cannot be subtracted fail-closed")
    missing = [alloc for alloc in excludes if alloc not in sizes]
    if missing:
        die(f"excluded allocation(s) {missing} absent from {registry}")
    excluded = sum(sizes[alloc] for alloc in excludes)
    if excluded <= 0 or r_full - excluded <= 0:
        die(f"exclusions ({excluded:,} B) leave no positive residency")
    return r_full - excluded, r_full, excluded


def derive_levels(r_bytes: int, fm) -> list[dict]:
    r_bits = r_bytes * 8
    levels = []
    for name, ber in LADDER:
        bits = round(ber * r_bits)
        try:
            # force the EXHAUSTIVE solver regardless of size: the frozen
            # literals must be the true optimum even past the 3000-bit
            # windowed threshold (one-time cost at freeze time)
            s, d, t = fm.resolve_composition(bits, exhaustive_limit=10 ** 9)
        except fm.ModelError as exc:
            die(f"R={r_bytes:,} B gives {name} bits={bits} ({exc}) -- the "
                "residency is too small for this BER ladder; re-examine "
                "the bootstrap measurement")
        assert s + 2 * d + 3 * t == bits, (s, d, t, bits)
        if fm.resolve_composition(bits) != (s, d, t):
            die(f"{name} bits={bits}: exhaustive ({s}, {d}, {t}) disagrees "
                "with the default (windowed above 3000 bits) derivation -- "
                "the campaign-start assert would refuse this literal; "
                "widen COMPOSITION_WINDOW and re-derive by hand")
        levels.append({"level": name, "ber": ber, "bits": bits,
                       "s": s, "d": d, "t": t})
    return levels


def regression(fm) -> None:
    """The same derivation path must reproduce the frozen ResNet-50 v2
    literals (the campaign that already ran end-to-end on this stack)."""
    entry = fm.WORKLOADS[REGRESSION_WORKLOAD]
    r_bits = entry["resident_bytes_nominal"] * 8
    for row in entry["levels"]:
        bits = round(row["ber"] * r_bits)
        comp = fm.resolve_composition(bits, exhaustive_limit=10 ** 9)
        if bits != row["bits"] or comp != (row["s"], row["d"], row["t"]):
            die(f"derivation regression: {REGRESSION_WORKLOAD} {row['level']} "
                f"derived ({bits}, {comp}) != frozen ({row['bits']}, "
                f"({row['s']}, {row['d']}, {row['t']}))")
    print(f"regression PASS: pipeline reproduces all {len(entry['levels'])} "
          f"{REGRESSION_WORKLOAD} literals")


def entry_block(source: str, workload: str) -> tuple[int, int]:
    """(start, end) line span of the workload's entry inside WORKLOADS.
    `end` is the entry's closing '},'; `start` additionally swallows any
    run of contiguous 4-space '# ' comment lines immediately above the
    key line, so a rewrite replaces the placeholder's comment block too.
    The brace scan starts AT the key line (comment lines carry no braces
    and would otherwise read as depth 0)."""
    lines = source.splitlines(keepends=True)
    key = None
    for i, line in enumerate(lines):
        if re.match(rf'^    "{re.escape(workload)}": \{{\s*$', line):
            key = i
            break
    if key is None:
        die(f"entry \"{workload}\" not found in {FAULT_MODEL_PY}")
    start = key
    while start > 0 and lines[start - 1].startswith("    # "):
        start -= 1
    depth = 0
    for j in range(key, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if j > key and depth == 0:
            return start, j
    die(f"unbalanced braces after entry key line {key + 1}")


def render_entry(workload: str, display: str, r_bytes: int, levels: list[dict],
                 run_id: str, engine_path: Path, engine_sha: str,
                 note_extra: str, excludes: tuple[str, ...] = (),
                 r_full: int = 0, excluded_bytes: int = 0) -> str:
    out = [
        f'    # G7-v2 five-model extension (user decision 2026-09-27):',
        f'    # {display}/ImageNet-1K on the SAME head-quantized v2 INT8',
        f'    # engine family as g7v2_imagenet1k_resnet50 (every weighted',
        f'    # op INT8 per-channel, classifier head included); seven-level',
        f'    # ladder L1..L7 = 1e-7, 5e-7, 1e-6, 3e-6, 5e-6, 7e-6, 1e-5',
        f"    # (the ResNet-50 curve's L3-L9 BERs).",
    ]
    if excludes:
        out += [
            f'    # The bootstrap run',
            f'    # artifacts/g7/campaign/{run_id}',
            f'    # (engine {engine_path},',
            f'    # sha256 {engine_sha[:12]}...,{note_extra})',
            f'    # measured the FULL residency R={r_full:,} B.',
            f'    # FAULT-SURFACE SCOPING (user decision 2026-09-27, this',
            f'    # workload only): the create_execution_context-phase TRT',
            f'    # private control-state allocation(s)',
            f'    # {", ".join(excludes)} ({excluded_bytes:,} B total) are',
            f'    # EXCLUDED from the injection surface -- the 2026-09-27',
            f'    # three-seed diagnostic L1 runs proved flips at',
            f'    # address-bearing offsets there kill the runner process',
            f'    # (CUDA illegal memory access on the first injected',
            f'    # inference: a process-fatal reliability event, not an',
            f'    # output-observable fault), while flips in the weights,',
            f'    # scratch, deserialize constants, and input binding all',
            f'    # survived and restored. The frozen R below is the',
            f'    # INJECTION SURFACE {r_full:,} - {excluded_bytes:,} =',
            f'    # {r_bytes:,} B; full-surface injection remains the',
            f'    # control experiment (paper appendix), and every other',
            f'    # workload keeps full-surface injection. The seven levels',
            f'    # are derived from this surface R; every literal is an',
            f'    # EXHAUSTIVE-solver literal. Frozen in place by',
            f'    # tools/g7_prep/freeze_g7v2_workload.py.',
        ]
    else:
        out += [
            f'    # R was MEASURED by the',
            f'    # bootstrap run',
            f'    # artifacts/g7/campaign/{run_id}',
            f'    # (engine {engine_path},',
            f'    # sha256 {engine_sha[:12]}...,',
            f'    #{note_extra})',
            f'    # and the seven levels derived from it; every literal is an',
            f'    # EXHAUSTIVE-solver literal (bits past the 3000-bit windowed',
            f'    # threshold were forced exhaustive at freeze time and',
            f'    # cross-checked against the windowed path). Frozen in place',
            f'    # by tools/g7_prep/freeze_g7v2_workload.py.',
        ]
    out += [
        f'    "{workload}": {{',
        f'        "resident_bytes_nominal": {r_bytes:,}'.replace(",", "_") + ",",
    ]
    if excludes:
        out.append('        "surface_excludes": ('
                   + ", ".join(f'"{alloc}"' for alloc in excludes) + "),")
    out.append('        "levels": [')
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
    parser.add_argument("--workload", required=True,
                        choices=sorted(MODELS),
                        help="which extension workload to freeze")
    parser.add_argument("--bootstrap-json", type=Path, default=None)
    parser.add_argument("--check", action="store_true",
                        help="verify only; never write")
    args = parser.parse_args()
    workload = args.workload
    model_dir, display = MODELS[workload]
    engine = G7_MODELS_ROOT / model_dir / "clean.engine"
    if not engine.is_file():
        die(f"engine {engine} is missing")

    fm = load_fault_model()
    regression(fm)

    boot_path = args.bootstrap_json or newest_bootstrap(workload)
    boot = read_bootstrap(boot_path)
    if boot.get("schema_version") != SCHEMA:
        die(f"bootstrap schema {boot.get('schema_version')!r} != {SCHEMA!r}")
    if boot.get("workload") != workload:
        die(f"bootstrap workload {boot.get('workload')!r} != {workload!r}")
    if Path(boot.get("engine_path", "")).name:
        if Path(boot["engine_path"]).resolve() != engine.resolve():
            die(f"bootstrap engine_path {boot['engine_path']} is not this "
                f"model's engine {engine}")
    engine_sha = sha256(engine)
    if boot.get("engine_sha256") != engine_sha:
        die(f"bootstrap engine_sha256 {str(boot.get('engine_sha256'))[:12]}... "
            f"!= current {engine} sha256 {engine_sha[:12]}... -- the engine "
            "file changed since the bootstrap run")
    r_bytes = boot.get("resident_bytes", 0)
    if not isinstance(r_bytes, int) or r_bytes <= 0:
        die(f"bootstrap resident_bytes {r_bytes!r} is not a positive integer")

    run_id = boot_path.parent.name
    snap = boot.get("snapshot_manifest", {}).get("counts", {})
    note = (f" snapshot allocations={snap.get('allocations')}, "
            f"pa_pages={snap.get('pa_pages')}, rows={snap.get('rows')}")

    excludes = EXCLUDES.get(workload, ())
    r_eff, r_full, excluded_bytes = effective_r(boot, boot_path, excludes)
    if excludes:
        print(f"fault-surface scoping: full R={r_full:,} B minus "
              f"{excluded_bytes:,} B ({', '.join(excludes)}) -> injection "
              f"surface R_eff={r_eff:,} B")

    levels = derive_levels(r_eff, fm)

    frozen_now = fm.WORKLOADS[workload]
    if frozen_now["resident_bytes_nominal"] == r_eff:
        # already frozen with this R: verify agreement, rewrite nothing
        if tuple(frozen_now.get("surface_excludes", ())) != excludes:
            die(f"already-frozen surface_excludes "
                f"{frozen_now.get('surface_excludes', ())} != required "
                f"{excludes} -- re-derive and re-confirm manually")
        frozen_levels = frozen_now["levels"]
        if (len(frozen_levels) != len(levels) or
                [e["level"] for e in frozen_levels] != [e["level"] for e in levels] or
                [e["ber"] for e in frozen_levels] != [e["ber"] for e in levels]):
            die("already-frozen level set (names/BERs/count) differs from "
                f"the required ladder {[(n, b) for n, b in LADDER]} -- "
                "re-derive and re-confirm manually")
        for row, now in zip(levels, frozen_levels):
            if (row["bits"], row["s"], row["d"], row["t"]) != \
                    (now["bits"], now["s"], now["d"], now["t"]):
                die(f"already-frozen {row['level']} disagrees with a fresh "
                    f"derivation from the SAME R: derived {row} vs frozen "
                    f"{now} -- re-derive and re-confirm manually")
        print(f"already frozen at R={r_eff:,} B; literals agree; no rewrite")
    else:
        if frozen_now["resident_bytes_nominal"] != 0:
            die(f"entry frozen at R="
                f"{frozen_now['resident_bytes_nominal']:,} but this "
                f"bootstrap + exclusions give {r_eff:,} -- refusing to "
                "silently re-freeze (reset the entry to its PLACEHOLDER "
                "form first if the change is deliberate)")
        if args.check:
            print(f"--check: entry still a placeholder (R=0); would freeze "
                  f"R={r_eff:,} B"
                  + (f" (injection surface: full {r_full:,} B minus "
                     f"{excluded_bytes:,} B excluded {list(excludes)})"
                     if excludes else "") + ", levels:")
            for row in levels:
                print(f'  {row["level"]}: ber={row["ber"]:g} bits={row["bits"]}'
                      f' s={row["s"]} d={row["d"]} t={row["t"]}')
            return 3  # distinct exit: inspected-but-NOT-frozen
        source = FAULT_MODEL_PY.read_text()
        start, end = entry_block(source, workload)
        block = render_entry(workload, display, r_eff, levels, run_id,
                             engine, engine_sha, note, excludes, r_full,
                             excluded_bytes)
        lines = source.splitlines(keepends=True)
        new_source = "".join(lines[:start]) + block + "".join(lines[end + 1:])
        # atomic replace: a crash mid-write can never leave a truncated
        # fault_model.py behind
        tmp = FAULT_MODEL_PY.with_suffix(".py.freeze_tmp")
        tmp.write_text(new_source)
        os.replace(tmp, FAULT_MODEL_PY)
        print(f"rewrote {FAULT_MODEL_PY} entry {workload} "
              f"(lines {start + 1}-{end + 1}) with R={r_eff:,} B")

    # final verification against the (possibly rewritten) file
    fm2 = load_fault_model()
    fm2.assert_frozen_levels(workload)
    if fm2.self_test() != 0:
        die("fault_model.self_test() failed after the rewrite")
    entry = fm2.WORKLOADS[workload]
    if entry["resident_bytes_nominal"] != r_eff:
        die("frozen R does not match the bootstrap measurement")
    if tuple(entry.get("surface_excludes", ())) != excludes:
        die("frozen surface_excludes does not match the EXCLUDES table")
    print(f"freeze VERIFIED: workload={workload} R={r_eff:,} B "
          f"({r_eff * 8:,} bits)"
          + (f" -- injection surface of the full {r_full:,} B residency "
             f"(excluded {excluded_bytes:,} B: {', '.join(excludes)})"
             if excludes else ""))
    for row in entry["levels"]:
        print(f'  {row["level"]}: ber={row["ber"]:g} bits={row["bits"]} '
              f's={row["s"]} d={row["d"]} t={row["t"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
