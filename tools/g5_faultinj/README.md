# G5 fault-injection campaign tooling

G5 turns the G4 dual-addressing chain into a full radiation
fault-injection campaign over the ResNet-50 INT8 / TensorRT workload
(docs/G5_FAULT_MODEL.md): one frozen BER level per campaign, N trials
(default 100), 1000 evaluation images per trial, every site XOR-flipped
through the live (allocation, byte, bit) ↔ VA ↔ PA ↔ GDDR chain of THIS
runner process.

Components:

- `fault_model.py` — the frozen level table and the site sampler
  (pure stdlib, offline). `--self-test`.
- `run_g5_campaign.py` — the orchestrator. Root-attached eBPF observer +
  gated CUDA child (dropped to the invoking user), gate-time ledger/map/
  snapshot, sampling from that snapshot, work file, post-run independent
  re-verification of every result row. `--self-test` (offline).
- Entry point (visudo-whitelisted wrapper):

  ```bash
  sudo scripts/run_g2_observer_probe.sh --api g5campaign \
       --device 0 --level L3 [--trials 100] [--seed 7]
  ```

  Without `--device` it loops over all GPUs sequentially (one observer at
  a time). One level per invocation. `--table`, `--sample-index`,
  `--timeout-seconds` are honored as in `--api g4t2`.

  Levels L1–L9 live in `fault_model.py` / docs/G5_FAULT_MODEL.md §5.
  L1–L5 (1e-8..1e-6) are the confirmed core, run three-card in the
  2026-09-21 campaign; L6–L9 (5e-6..1e-4) are the post-campaign
  extension ladder, run SINGLE-card (`--device N` explicit) per the
  verified L1–L5 no-card-effect result. Same derivation rule,
  semantics, seed and trial protocol for all nine.

## Flow (what is checked where)

1. Observer attaches before any CUDA context (pre-allocation gate). In
   campaign mode the runner has already preprocessed all 1000 images on
   the host CPU before the gate — the gated window stays CUDA-only.
2. Runner allocates, runs the strict CLEAN pass (every image; recorded to
   `g1_5_g5_clean_pass.csv`), writes the gate registry, blocks at
   `CAMPAIGN_GATE_WAIT`.
3. Orchestrator: ALLOCATED↔registry agreement, PTE preledger, gate map,
   `build_snapshot.run_build` (fail-closed join), then:
   - residency guard: the snapshot's resident-byte total must equal the
     frozen R = 26,428,428 or the campaign refuses (the level table is
     derived from R; docs/G5_FAULT_MODEL.md §5);
   - `fault_model.sample_campaign(level, trials, rows, anchors, seed)`
     samples every trial from THAT snapshot;
   - `verify_work_model` re-checks each trial against the frozen
     (s, d, t)/B and (PA byte, bit) distinctness;
   - writes `work.csv` (runner contract) + `work_detail.json`
     (per-site chain/provenance metadata, snapshot manifest).
4. Release gate; per trial the runner: flips ALL sites (each verified
   after == before ^ mask, reverse chain, allocation-level guard compare =
   pristine ^ masks before any enqueue) → full 1000-image pass with the
   faults held (input faults re-applied after each per-image copy, output
   faults after each enqueue, TRT-internal never re-applied) → per-image
   classification vs clean records → DUE abort on first invalid output
   (rest of the trial marked evaluated=0) → reverse-order restore →
   per-class restore verification → strict single-image sanity inference.
5. After teardown: closing ledger, gate-vs-final map diff (mid-run remap
   detector), mappings-alive-across-window, registry stability, campaign
   lifecycle skeleton, EXACT trial event stream match against the work
   file, and independent re-verification of all four result CSVs
   (xor math, VA equality, classification re-derivation from the logged
   numbers, counter/outcome precedence, DUE block shape, event↔CSV
   agreement), UUID/address-space/lost-event checks.

## Result files (per run directory, `artifacts/g5/campaign/run_<level>_gpu<N>_<ns>/`)

- `g1_5_g5_site_result.csv` — one row per flipped site.
- `g1_5_g5_trial_result.csv` — one row per trial (counts, outcome,
  sanity, restore stats).
- `g1_5_g5_image_detail.csv` — one row per (trial, image).
- `g1_5_g5_clean_pass.csv` — the campaign's clean reference pass.
- `work.csv` / `work_detail.json`, `snapshot/`, gate/final maps and
  registries, `events.csv`, `harness.log`, `summary.json`.

## Honesty notes (read before interpreting the data)

- `restored_byte_ok` means "the cell was not rewritten between flip and
  restore" (`unflip.after == flip.before`). It is INFORMATIONAL and is
  legitimately false for cells the engine owns and rewrites during the
  pass (output bindings, TRT scratch) and for input-binding cells whose
  bytes the pass legitimately re-stages per image. It is not a failure
  signal.
- `restore_check` per allocation class:
  - `exact` — full allocation compared byte-exact (input binding is
    fail-closed on this; TRT-internal exact is the normal weights case);
  - `mismatch:N` — informational TRT-internal mismatch: the engine
    rewrote N scratch bytes during the pass (soft-upset semantics,
    recorded, not fatal);
  - `skipped:engine-owned-output` — output binding: the engine rewrites
    the cell on every enqueue so no stable baseline exists; the flip was
    proven by the pre-pass allocation guard compare.
- The per-trial sanity inference is single-image (`--sample-index`, the
  G4-T2 convention), not the full 1000-image pass; INT8 execution is
  deterministic, so a strictly reproduced output plus the byte-level
  restore proofs are the no-residue evidence.
- L1 (B=2) has composition (2, 0, 0): 2 SBUs, no MCU — a BER small
  enough that 2-bit MCU events cannot reach the 20% event-share target;
  documented in docs/G5_FAULT_MODEL.md §5.
- Faults in TRT-internal regions are never re-applied during the pass:
  weights persist (a flipped weight bit stays flipped for the whole
  trial); engine scratch is honest soft-upset (the engine itself
  rewrites it). This asymmetry is the intended semantics, not a gap.

## Analysis

`fault_model.py` and the CSV re-verifiers in `run_g5_campaign.py` are
importable stdlib-only modules; aggregate analysis scripts must not
import bcc (run those under any python, run the orchestrator only under
`/usr/bin/python3`).

- `analyze_campaign.py` — per-run and pooled-per-level tables,
  recomputed from the four result CSVs (`--min-trials 100` for formal
  campaigns only).
- `plot_accuracy_curve.py` — the paper figure (accuracy vs BER, log x,
  nine levels, trial-level 95% CI error bars). Needs matplotlib, so run
  it under an env that has it, e.g.
  `/data1/luojx/miniforge3/envs/vit_fault/bin/python
  tools/g5_faultinj/plot_accuracy_curve.py`; every plotted number is
  re-derived from the CSVs at run time and the plotted points are also
  dumped to `fig_accuracy_vs_ber_points.csv`.
