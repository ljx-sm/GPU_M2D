# G4 Dual-Addressing Integration

Status: **PASS** on 2026-09-20 (T1 offline on all three cards' captured
runs; T2 live gated runs on all three cards).

## Scope

G4 joins the three validated legs into the per-run chain G5 consumes:

```text
(allocation_id, byte offset, bit)   <- G1.5 AllocationRegistry
        <-> GPU VA page             <- G2 eBPF observer (per run)
        <-> framebuffer PA page     <- G3 EMT table v4 (per model)
        <-> bank component / row class / anchors
```

Nothing new is measured in G4; the stage proves the join is correct,
fail-closed, and — in T2 — that a physical site chosen from the GDDR side
can be driven back to a live TensorRT byte and actually flipped.

## T1: offline snapshot builder (`tools/g4_dualaddr/build_snapshot.py`)

Folds ONE captured G1.5+G2 run for ONE device into `snapshot_pages.csv` +
`manifest.json` (mapping checksum = sha256 of the CSV). Fail-closed rules
(pinned by `--self-test`): every ACTIVE allocation's full VA range must
carry observed valid VIDEO COMPLETE 2 MiB pages; every resident PA page
must be inside the G3 table universe; no PA aliasing across VA pages;
disjoint byte residency on shared pages; dangling map rows refuse;
unlinked/no-anchor pages ride as honest warnings.

Three-card result (per-card latest TRT runs): 18 rows / 14 PA pages /
26.4 MiB over 7 allocations each; bank linked 14/14 (GPU0/GPU1), 12/14
(GPU2, warned); anchors 14/14 everywhere; `--lookup-va/--lookup-pa`
round-trip.

Lessons/findings carried into T2:

- **Pair a registry with its own run.** The top-level
  `artifacts/g1_5/` CSVs and the aggregated map come from different
  executions (`g1_5_run_id` is a logical label, not unique).
- **Co-tenant memory state moves where allocations land** — the three
  cards' PA page sets were disjoint (0x1f0../0x205../0x26e..). The
  per-run snapshot + universe check absorbs this by design.
- **Workload residency did not intersect the 5 measured row-class
  pages** — G5 same-row MCU needs placement steering or a row-mining
  top-up (decision data for G5).

## T2: live gated XOR through the chain (`tools/g4_dualaddr/run_g4_t2_injection.py`)

One run, live, per card:

1. the G2 observer attaches before any CUDA context (runner at the
   pre-allocation gate);
2. the runner builds the full G1.5 workload, runs a clean inference,
   writes its **gate-time allocation registry**, and blocks at the
   **injection gate**;
3. the orchestrator builds the per-allocation PTE ledger live (complete
   valid local-VIDEO coverage, exactly one pending-free notice each),
   writes `gpu_va_pa_map_gate.csv`, and folds registry + map + G3 table
   v4 into the run's snapshot (same `run_build` path as T1);
4. **reverse-chain sites** are selected from the snapshot —
   `t1-binding`: bank-linked page preferred, then the `data` input
   binding; `t2-internal`: bank-linked page preferred, then the largest
   TRT-internal allocation; byte = midpoint of the resident range; bits
   are fixed policy constants (5 / 2);
5. the work file releases the runner, which per target: checks the LIVE
   registry forward VA against the orchestrator's expected VA, snapshots
   the whole allocation, XORs the bit on device, verifies
   `after == before ^ mask`, verifies every other byte unchanged,
   reverse-maps the flipped VA; then runs the injected inference (an
   invalid numeric output is recorded as DUE, not a tool failure),
   restores every fault (byte + whole-allocation compare), and finishes
   with a sanity inference that must reproduce the clean output;
6. post-teardown the orchestrator re-runs the strict ledger (frees inside
   the teardown window), requires every mapping alive across the whole
   injection window, requires gate == final registry (no realloc), diffs
   the gate vs final map (mid-run remap detector), and re-verifies every
   result row independently of the runner.

Runner mode: `gpu_m2d_resnet50_int8_g1_5 --injection-work PATH
--injection-release PATH` (additive; the legacy element/bit self-test
path is unchanged and still passes). Orchestration entry (the only sudo
surface): `sudo scripts/run_g2_observer_probe.sh --api g4t2 [--device N]`
(without `--device` it loops over all GPUs **sequentially** — one eBPF
observer at a time).

### T2 result (2026-09-20, three cards, back-to-back, 3 co-tenant
processes present and recorded — not a refusal, post-timing policy)

| card | t1 binding target | t2 internal target | checks |
|------|-------------------|--------------------|--------|
| GPU0 | `data` byte 301056 bit 5, 50→18, PA 0x29be00000 (comp 0x15c7) | `trt-internal-0` byte 1 MiB bit 2, 23→19, PA 0x29a200000 (comp 0x14d1) | all pass |
| GPU1 | `data` byte 301056 bit 5, 50→18, PA 0x143a00000 (comp 0xf9) | `trt-internal-0` byte 3 MiB bit 2, 43→47, PA 0x142000000 (comp 0xc4f) | all pass |
| GPU2 | `data` byte 301056 bit 5, 50→18, PA 0x244200000 (comp 0x839) | `trt-internal-0` byte 1 MiB bit 2, 23→19, PA 0x242600000 (comp 0x1da9) | all pass |

Each run: 7 allocations / 18 map rows / 119 eBPF events / 0 lost; every
target XOR-verified (`after == before ^ mask`), guard bytes unchanged,
reverse map exact, restore byte-exact, post-restore sanity inference
equal to clean; **all six injected inferences were SDC_NUMERIC** — every
fault changed the numeric output, so the flips were semantically real,
not just memory-level. The three cards' PA segments were disjoint
(0x29a../0x142../0x242.. — co-tenant driven) yet all inside the 22 GiB
table universe; the universal anchor masks 0x1f9dc0/0x1fdc80 from G3
appear on the selected pages across all three cards.

## Validity boundary

- The snapshot is run-scoped; only the PA→GDDR table is per-model.
  Never reuse a Tensor→PA mapping across runs.
- G2 observes at map time; a mid-run remap is outside the observation
  window — T2 detects it after the fact (gate-vs-final map diff) and the
  injector's value-level checks would catch it live.
- TRT-internal regions carry ownership + physical coordinates only
  (`TENSORRT_INTERNAL_UNKNOWN`); no layer/weight semantic claim.
- Co-tenancy is recorded, never refused; a run only refuses when
  allocations land outside the table universe (VRAM pressure), which is
  the designed protection.
