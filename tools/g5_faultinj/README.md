# G5 fault-injection campaign tooling

This directory turns the G4 dual-addressing chain into a full
fault-injection campaign. Each campaign runs one frozen BER level. It has
N trials (default 100), and every trial runs a full evaluation pass. Every
site is XOR-flipped through the live (allocation, byte, bit) ↔ VA ↔ PA ↔
GDDR chain of THIS runner process
(fault model: [docs/G5_FAULT_MODEL.md](../../docs/G5_FAULT_MODEL.md)).

The same stack serves two campaign families, selected by `--workload`:

- **G5** (`g5_resisc45_resnet50`, the default): ResNet-50 INT8 on
  RESISC45, 1000 images per trial, nine levels 1e-8 … 1e-4 (complete
  2026-09-22; analysis in [docs/G6_ANALYSIS.md](../../docs/G6_ANALYSIS.md)).
- **G7-v2** (`g7v2_imagenet1k_*`): six ImageNet-1K INT8 TensorRT models,
  10,000 images per trial (complete 2026-09-29; results in
  [docs/G7V2_RESULTS.md](../../docs/G7V2_RESULTS.md), engines in
  [tools/g7_prep/README.md](../g7_prep/README.md)). ResNet-50 v2 runs
  nine levels (1e-8 … 1e-5); the other five models run seven levels
  (1e-7 … 1e-5, the same BERs as ResNet-50's L3–L9).
  `g7_imagenet1k_resnet50` is the superseded v1 (FP32-head) workload and
  is kept only for its preliminary runs.

## Components

- `fault_model.py`: the `WORKLOADS` table and the site sampler (pure stdlib,
  offline). Each workload entry has its own frozen R
  (`resident_bytes_nominal`) and its own level ladder. It may also carry an
  optional `surface_excludes`, a list of allocations held out of the
  injection surface. `surface_rows_for()` is the single filter point.
  `assert_frozen_levels()` re-derives every ladder from its BERs and R at
  each campaign start and refuses on drift. `--self-test`.
- `run_g5_campaign.py` is the orchestrator:
  - root-attached eBPF observer plus a gated CUDA child (dropped to the
    invoking user);
  - builds the gate-time ledger, map and snapshot, and samples from that
    snapshot;
  - writes the work file;
  - runs the segment loop and restart protocol (below);
  - after the run, re-verifies every result row independently.

  `--self-test` runs offline.
- Entry point (a visudo-whitelisted wrapper):

  ```bash
  # G5 (defaults = RESISC45 engine/split, 1000 images)
  sudo scripts/run_g2_observer_probe.sh --api g5campaign \
       --device 0 --level L3 [--trials 100] [--seed 7]

  # G7-v2: pass the workload plus the runner passthrough flags that match
  # the model's model_meta.json (example: ViT-B, as recorded in its
  # bootstrap.json / summary.json runner_passthrough)
  sudo scripts/run_g2_observer_probe.sh --api g5campaign --device 0 \
       --workload g7v2_imagenet1k_vit_base_patch16_224 --level L1 \
       --engine /data1/luojx/g7_models/vit_base_patch16_224/clean.engine \
       --sample-csv /data1/luojx/datasets/imagenet1k/splits/g7_eval_10000_perclass10.csv \
       --output-root artifacts/g7/campaign \
       --class-count 1000 --preprocess canonical --resize-scale 248 \
       --interp bicubic --mean 0.5,0.5,0.5 --std 0.5,0.5,0.5 \
       --image-cache-dir artifacts/g7/image_cache \
       [--prelude-seconds N] [--timeout-seconds N]
  ```

  Without `--device` the wrapper loops over all GPUs sequentially (one
  observer at a time). Each invocation runs one level. `--table`,
  `--sample-index` and `--timeout-seconds` behave as in `--api g4t2`.
  Unset passthrough flags keep the G5 runner defaults byte-identical.

## Adding a workload (G7 procedure)

1. **Bootstrap**: `--bootstrap` instead of `--level` runs the observer,
   the per-run snapshot and the clean pass. It records the engine's live
   resident-byte total R into `bootstrap.json` and then stops, with no
   trials and no flips.
2. **Freeze**: add the workload to `fault_model.WORKLOADS` as a placeholder
   (R = 0). A placeholder refuses any campaign but accepts `--bootstrap`.
   Then freeze its ladder from the newest matching bootstrap:
   - `tools/g7_prep/freeze_g7v2_workload.py` for the five-model seven-level
     ladder;
   - `freeze_g7v2_levels.py` plus `extend_g7v2_levels.py` for the ResNet-50
     v2 nine-level ladder.

   Both tools pin the engine sha256 and force the exhaustive composition
   solver, cross-checked against the windowed one. They regression-check
   the derivation against the frozen ResNet-50 table and rewrite
   `fault_model.py` atomically. A re-freeze requires an explicit
   placeholder reset.
3. **Run** the levels. The residency guard refuses any campaign whose live
   injection-surface total differs from the frozen R.

## Flow (what is checked where)

1. The observer attaches before any CUDA context exists (pre-allocation
   gate). In campaign mode the runner has already preprocessed every
   evaluation image on the host CPU before the gate, so the gated window
   stays CUDA-only. It reloads that image buffer from the host image cache
   when one is present (below).
2. The runner allocates the workload and runs the strict CLEAN pass over
   every image (recorded to `g1_5_g5_clean_pass.csv`). It then writes the
   gate registry and blocks at `CAMPAIGN_GATE_WAIT`.
3. The orchestrator:
   - checks ALLOCATED↔registry agreement and builds the PTE preledger and
     gate map;
   - builds the snapshot with `build_snapshot.run_build` (the fail-closed
     join);
   - applies the **residency guard**: the snapshot's injection-surface
     total (`surface_rows_for`, i.e. full residency minus
     `surface_excludes`) must equal the workload's frozen R, or the
     campaign refuses;
   - samples every trial from THAT snapshot with
     `fault_model.sample_campaign(level, trials, rows, anchors, seed,
     resident_bytes_expected, workload)`;
   - `verify_work_model` re-checks each trial against the frozen
     (s, d, t)/B and for (PA byte, bit) distinctness;
   - writes `work.csv` (the runner contract) and `work_detail.json`
     (per-site chain/provenance metadata and the snapshot manifest).
4. The orchestrator releases the gate. Per trial the runner:
   1. flips ALL sites. Each flip is verified (after == before ^ mask,
      reverse chain), and an allocation-level guard compares against
      pristine ^ masks before any enqueue;
   2. runs a full evaluation pass with the faults held. Input faults are
      re-applied after each per-image copy and output faults after each
      enqueue; TRT-internal faults are never re-applied;
   3. classifies each image against the clean records. The first invalid
      output aborts the trial as DUE, and the rest of its images are marked
      evaluated=0;
   4. restores the sites in reverse order, verifies the restore per
      allocation class, and runs a strict single-image sanity inference;
   5. flushes the three result CSVs. A trial is on disk iff it reached
      TRIAL_END.
5. After teardown the orchestrator checks:
   - the closing ledger;
   - the gate-vs-final map diff (mid-run remap detector) and that mappings
     stayed alive across the window;
   - registry stability and the campaign lifecycle skeleton;
   - that the trial event stream EXACTLY matches the work file;
   - all four result CSVs, re-verified independently: xor math, VA
     equality, classification re-derived from the logged numbers,
     counter/outcome precedence, DUE block shape, and event↔CSV agreement;
   - UUID, address-space and lost-event checks.

## Restart protocol and PROCESS_FATAL (2026-09-28)

**Why.** A flip in TensorRT's `create_execution_context`-phase private
control state can kill the runner process at the dying trial's first
injected inference. That buffer holds device-side kernel parameters. The
share of R it covers varies by engine: ResNet-50 0.08 %, MobileNetV3 2.6 %,
Swin-T 28 %, EfficientNet-B0 58 %. Such a death is a process-fatal
reliability event, not an output-observable fault.

**How.** One BER level is a segment loop (`execute_segment`).

- **Recoverable death.** `analyze_process_death` accepts a death as
  recoverable ONLY when all of these hold:
  - the harness fail marker `GPU_M2D_G1_5_FAIL` is present;
  - one of the `RECOVERABLE_CUDA_SIGNATURES` is present: `illegal memory
    access`, or `operation not supported on global/shared address space`.
    The second string was first seen on Swin L5; it is the same mechanism
    with a different driver string;
  - every earlier trial reached TRIAL_END;
  - exactly one trial is open, and its complete flip set is the tail of
    the event stream.
- **Handling.** The dead segment's completed prefix is fully re-verified.
  The dying trial becomes **PROCESS_FATAL** on its global execution-order
  slot. A fresh gated segment (new observer, snapshot and sampling; seed
  `--seed + k`) then relaunches for the remaining trial slots.
- **Any other death shape** fails the level closed (exit 2).
- **Merge.** `merge_campaign_outputs` renumbers segment-local trials to
  global slots; a single clean segment is a byte-identical hardlink. It
  drops a crashed segment's never-run work tail, because those slots were
  re-sampled. Each shared-evidence file is taken from the first segment
  that has it: a crashed segment never writes its final VA-PA map.
- **Directory layout.** A restarted level keeps `segment_000/`,
  `segment_001/`, … under the level directory next to the merged files.
  `summary.json` carries `restart_protocol_used`, `segments`,
  `segment_details`, `process_fatal_trials` and `process_fatal_count`.
- `--bootstrap` runs are exempt: a single flat, measure-only run.

**Scoping alternative.** MobileNetV3 instead excludes its fatal pools
(`surface_excludes = trt-internal-5/6`, 2.66 % of R, user decision
2026-09-27). A crash there was near-certain per trial at high BER, so
restarting was not viable. Every other G7-v2 workload injects over its FULL
surface under the restart protocol. Over the whole G7-v2 campaign 61
crashes were absorbed with zero lost or fabricated trials.

**Host image cache.** Relaunch cost is kept down by caching the
preprocessed CHW float32 buffer. The cache lives at
`<output-root>/../image_cache/g5img_<key>.bin` (≈5.6 GiB for 10K images).
The key is SHA-256 over the sample CSV, the full preprocessing spec, the
image count, the tensor size and the runner binary. Key, header and blob
checksum are verified on load, and any mismatch falls back to fresh
preprocessing. A relaunch takes ~1.5 min instead of ~4.5.

## Result files

Each level directory is
`<output-root>/run_<level>_gpu<N>_<ns>/`, where the output root is
`artifacts/g5/campaign/` for G5 and `artifacts/g7/campaign/` for G7.
It contains:

- `g1_5_g5_site_result.csv`: one row per flipped site.
- `g1_5_g5_trial_result.csv`: one row per trial (counts, outcome, sanity,
  restore stats).
- `g1_5_g5_image_detail.csv`: one row per (trial, image).
- `g1_5_g5_clean_pass.csv`: the campaign's clean reference pass.
- `work.csv` / `work_detail.json`, `snapshot/`, gate/final maps and
  registries, `events.csv`, `harness.log`, `summary.json`, and
  `segment_NNN/` for restarted levels.

Bootstrap runs are `run_bootstrap_gpu<N>_<ns>/` with `bootstrap.json`
(measured R, engine sha256, snapshot manifest, runner passthrough).

## Honesty notes (read before interpreting the data)

- `restored_byte_ok` means "the cell was not rewritten between flip and
  restore" (`unflip.after == flip.before`). It is INFORMATIONAL and is
  legitimately false in three cases: cells the engine owns and rewrites
  during the pass (output bindings, TRT scratch), and input-binding cells,
  whose bytes the pass re-stages per image. It is not a failure signal.
- `restore_check` per allocation class:
  - `exact`: the full allocation compared byte-exact. The input binding
    fails closed on this; TRT-internal `exact` is the normal case for
    weights;
  - `mismatch:N`: an informational TRT-internal mismatch. The engine
    rewrote N scratch bytes during the pass (soft-upset semantics,
    recorded, not fatal);
  - `skipped:engine-owned-output`: an output binding. The engine rewrites
    the cell on every enqueue, so no stable baseline exists; the flip was
    proven by the pre-pass allocation guard compare.
- The per-trial sanity inference is single-image (`--sample-index`, the
  G4-T2 convention), not the full pass. INT8 execution is deterministic, so
  a strictly reproduced output plus the byte-level restore proofs are the
  no-residue evidence.
- At low BER a level can be too small to express the 60/40 mix. G5 L1 and
  G7-v2 ResNet-50 L1 (B = 2) are 2×SBU with no MCU; see
  docs/G5_FAULT_MODEL.md §5.
- Faults in TRT-internal regions are never re-applied during the pass.
  Weights persist: a flipped weight bit stays flipped for the whole trial.
  Engine scratch is honest soft-upset: the engine itself rewrites it. This
  asymmetry is the intended semantics, not a gap.
- **Accuracy convention (user decision 2026-09-25)**: the primary injected
  accuracy averages ONLY trials that completed normally. Trials with a DUE
  image, and PROCESS_FATAL trials, are excluded from the mean and reported
  as separate rates. G5 had zero DUE, so its numbers are unchanged by this
  rule.

## Analysis

`fault_model.py` and the CSV re-verifiers in `run_g5_campaign.py` are
importable, stdlib-only modules. Aggregate analysis scripts must not
import bcc: run those under any python, and run the orchestrator only
under `/usr/bin/python3`.

- `analyze_campaign.py` prints per-run and pooled-per-level tables,
  recomputed from the four result CSVs:
  - `--min-trials 100` keeps formal campaigns only;
  - `--root` selects the campaign root;
  - `--workload` is REQUIRED (fail-closed) when a root holds several
    workloads' runs, because v1/v2 level names collide at different BERs;
  - columns include DUE%, `pf` (PROCESS_FATAL) and pooled P(trial crash).
- `analyze_g6_attribution.py` produces the G6-T0 attribution tables for the
  G5 campaign (docs/G6_ANALYSIS.md):
  - flip-integrity recount;
  - per-allocation residency/site/restore_check attribution;
  - weight corrosion per trial;
  - perturbation-vs-margin percentiles.

  Allocation ids are normalized across cards; binding ids carry the device
  index.
- `plot_accuracy_curve.py` plots one workload's accuracy vs BER on a log x
  axis with trial-level 95 % CI error bars (`--workload`, `--root`,
  `--levels`, `--outdir`). It uses the non-DUE accuracy convention. Levels
  with any PROCESS_FATAL get a second panel showing P(trial crash) and
  P(trial DUE). Formality is judged by attempted trials, not completed
  ones. Plotted points are dumped to a points CSV.
- `plot_six_models.py` draws the G7-v2 paper figure: six models on one
  panel. ResNet-50 contributes its L3–L9, matched by BER value. It writes
  to `artifacts/g7/campaign/fig_six_models/`, and `--points-csv`
  re-renders from the written CSV without the ~18 min recollect.

The plotting scripts need matplotlib, e.g.

```bash
python3 tools/g5_faultinj/analyze_campaign.py --min-trials 100
python3 tools/g5_faultinj/analyze_campaign.py --root artifacts/g7/campaign \
        --workload g7v2_imagenet1k_swin_tiny_patch4_window7_224 --min-trials 100
/data1/luojx/miniforge3/envs/vit_fault/bin/python \
        tools/g5_faultinj/plot_accuracy_curve.py
/data1/luojx/miniforge3/envs/vit_fault/bin/python \
        tools/g5_faultinj/plot_six_models.py
```

Every plotted number is re-derived from the CSVs at run time.
