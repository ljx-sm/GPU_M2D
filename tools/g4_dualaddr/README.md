# G4 dual-addressing — snapshot builder

`build_snapshot.py` joins the three validated legs into the per-run
TensorBit↔GDDR chain G5 consumes:

```text
G1.5 allocation registry   (allocation_id, byte, bit) <-> GPU VA
G2  eBPF observer (per run) (allocation_id, VA page) <-> fb PA page
G3  EMT table v4 + API      PA page -> bank component / row class / anchors
```

## Usage

```bash
python3 tools/g4_dualaddr/build_snapshot.py --self-test
python3 tools/g4_dualaddr/build_snapshot.py \
    --allocations artifacts/g2/observer/trt/<run>/g1_5_allocations.csv \
    --va-pa artifacts/g2/observer/trt/<run>/gpu_va_pa_map.csv \
    --device 0 --table artifacts/g3/table_v4 \
    --out artifacts/g4/snapshot_trt_gpu0
python3 tools/g4_dualaddr/build_snapshot.py \
    --snapshot artifacts/g4/snapshot_trt_gpu0 --lookup-va 0x77279c000000
python3 tools/g4_dualaddr/build_snapshot.py \
    --snapshot artifacts/g4/snapshot_trt_gpu0 --lookup-pa 0x1f000000
```

**Pair the registry with its own run.** The top-level
`artifacts/g1_5/gpu{N}_allocations.csv` and the aggregated
`artifacts/g2/gpu_va_pa_map.csv` come from DIFFERENT process executions
(the `g1_5_run_id` string is a logical label, not unique per execution),
so their VAs do not intersect. Always build from one run directory's
`g1_5_allocations.csv` + `gpu_va_pa_map.csv`.

## Semantics (all fail-closed, pinned by `--self-test`)

- Every ACTIVE allocation's full VA range must be covered by observed
  pages (`pte_valid`, `aperture=VIDEO`, `LOCAL_VIDEO_COMPLETE`, 2 MiB);
  a gap refuses the snapshot (exit 2, nothing written).
- Every resident PA page must be inside the G3 table universe — outside
  means the driver handed out memory the table never measured; refused.
- Two different VA pages may not alias one PA page (refused; deliberate
  revisit required if G5 ever wants aliasing).
- Byte residency comes from the registry VA bounds: several small
  allocations may share one 2 MiB page as DISJOINT byte ranges (the TRT
  workload does — `trt-internal-1/2/3` + `prob` + `index` share one page);
  overlapping ranges refuse.
- Map rows whose allocation has no ACTIVE registry entry refuse at load.
- Unlinked pages (bank UNKNOWN) and pages without anchors are carried as
  honest columns + warnings — SEU sites stay valid there, relation models
  degrade with labels, nothing is guessed.

Outputs: `snapshot_pages.csv` (one row per allocation×VA page with both
addresses, byte residency, and the GDDR classes) and `manifest.json`
(input/table sha256, counts, warnings, and the mapping checksum =
sha256 of the snapshot CSV).

## G4-T1 result (2026-09-20, offline on the captured TRT runs)

Built for the three latest per-card TRT runs (back to back, shared
server): each 18 rows / 14 distinct PA pages / 26,428,428 resident bytes
over 7 allocations (4 TRT-internal owned-unknown + 3 semantic bindings),
all pages inside the table universe; bank linked 14/14 (GPU0, GPU1) and
12/14 (GPU2 — two unlinked pages warned); anchors 14/14 on every card;
forward/reverse lookups round-trip.

Two findings that shape G4-T2:

- **The three cards' PA page sets are all different** (0x1f0.., 0x205..,
  0x26e.. segments): co-tenant memory state moves where allocations land.
  The per-run snapshot + fail-closed universe check absorbs this by
  design; a run only refuses if pushed fully outside the 22 GiB universe.
- **No resident page hosts a measured same-row class** (the 5 row-class
  pages do not intersect the workload residency): G5 same-row MCU needs
  placement steering (copy the data under test onto row-class pages) or a
  row-mining top-up round — G4-T2 produces the decision data.

## Validity boundary

- The snapshot is run-scoped: never reused across runs (registry + map
  are per-execution; only the PA→GDDR table is per-model).
- G2 observes at map time; mid-run remap is outside the observation
  window and is caught by the injector's value-level verification, not
  here.
- TRT-internal regions carry ownership + physical coordinates only
  (label `TENSORRT_INTERNAL_UNKNOWN`); no layer/weight semantic claim.
