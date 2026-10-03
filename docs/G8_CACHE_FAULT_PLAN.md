# G8 Plan — L2 Cache Fault Injection on Top of the GDDR Campaign

Status: **DESIGN CONFIRMED by the user, 2026-10-01. G8-T0 complete
2026-10-01 (GPU 0): probe calibration V6 PASS, hook neutrality V7 PASS
(§11). G8-T1 complete 2026-10-02 (GPU 0): residency maps for all six
G7-v2 models built, independently verified, and reproducible, V9 PASS
(§12). G8-T2 complete 2026-10-02 (GPU 0): the cache-fault chain runs
end to end, with smoke campaigns VERIFIED on all six models and a
controlled shared-GPU test (§13). G8-T3 complete 2026-10-03: the cache
upset rate is frozen at BER_cache = ρ × the level's DRAM BER with ρ = 1
(§3.3, §14). Self-check limits confirmed by the user 2026-10-03 (§9).
G8-T4 in progress: ResNet-50 v2 first (§6).** This file has been revised in place through the
2026-09-29 … 10-03 discussion (earlier versions are in git history; §10
records what changed and why).

Scope:

| Dimension | In scope | Out of scope |
| --- | --- | --- |
| Memory level | GDDR6X DRAM (existing G5/G7 path, unchanged) + **L2 cache** | L1, shared memory, register file, instruction/constant caches |
| Hardware | **GPU 0 only** | GPU 1/2 (left alone for now) |
| Cache fault type | **random single-bit upsets (SBU)** | cache MCU (would need the physical SRAM layout, which is unmeasurable) |
| Time resolution | **image (one inference) boundaries** | events inside a single inference |

## 1. Platform facts (measured 2026-09-29, GPU 0, driver 580.95.05)

| Item | Value | Source |
| --- | --- | --- |
| L2 cache | 75,497,472 B = **72 MiB** (603,979,776 bits), shared by all 128 SMs | `cudaGetDeviceProperties` |
| Persisting-L2 set-aside max | 49.5 MiB (not used by this design) | same |
| DRAM ECC | mode exists, **Disabled** (current and pending) | `nvidia-smi -q -d ECC` |
| SRAM error counters | N/A (no L2 parity/ECC reporting exposed) | same |
| L2 fill granularity | **32-B sector**: touching sector 0 of a line leaves sectors 1–3 absent (T0) | `tools/g8_cache/g8_l2_calib` |
| Effective L2 capacity | hit fraction after touching W twice: 1.000 up to 64 MiB, 0.903 at 68, 0.617 at 72, 0.06 at 80, 0 from 96 MiB (T0) | same |

Notes on these facts:

- **ECC is a setting.** The "no ECC" assumption of the fault model matches
  the actual configuration, but it is a setting, not a hardware absence.
  This corrects "none on GeForce" in `docs/G3_SURVEY.md` §4.
- **L2 is assumed unprotected**, the worst case, and the assumption is
  stated explicitly.

Workload surfaces (frozen R from G7-v2) against the 72 MiB L2:

| Model | Frozen R | Cache lines (R / 128 B) | Fits in L2? |
| --- | --- | --- | --- |
| MobileNetV3 | 9.1 MB | ~71 K | yes |
| EfficientNet-B0 | 17.1 MB | ~134 K | yes |
| DeiT-S | 26.0 MB | ~203 K | yes |
| ResNet-50 | 28.8 MB | ~225 K | yes |
| Swin-T | 43.8 MB | ~342 K | yes |
| ViT-B | 93.9 MB | ~733 K | **no** |

## 2. Principles

1. **DRAM faults are persistent.** GDDR has no ECC and no scrubbing, so
   upsets accumulate and stay. The existing campaign (select GDDR cells →
   G3 table / G2 map → byte/bit → GPU XOR, held for the whole trial) is
   physically right and **stays exactly as it is**. The XOR store modifies
   the L2 copy of the line (dirty); every SM reads that copy, and eviction
   writes it back to GDDR.
2. **Cache faults are transient.** An L2 line is a temporary copy. A flip
   in a cached copy lasts only while the line stays in L2 (for read-only
   data) or until the engine overwrites the data. The hard part of cache
   faults is **when and how long a flip exists**, and this design answers
   it with an in-process measurement of each line's residency (§4).
3. **A cache flip is applied through an address; no physical L2 location
   is needed.** No instruction addresses an L2 SRAM cell (slice/set/way/bit).
   A "cache line" in this design is **a 128-B-aligned block of the
   workload's memory**. XOR-ing a byte of that block while it is resident
   modifies exactly the cached copy that all SMs read. For SBU this is
   exact: with one bit per event, the physical SRAM neighbourhood never
   matters.
4. **Residency is a per-process behaviour, so it is measured in the same
   process.** L2 is a memory-side cache indexed by **physical** address:
   its set and slice come from the PA, and lines in the same set compete
   for its ways. Physical addresses change with every process (G4), and
   co-tenant load changes over time. A line's residency pattern is
   therefore valid only for the process it was measured in. The code
   never uses the PA. The consequence is only that every campaign process
   measures its own residency before its trials (§3.1, §4).

## 3. Confirmed fault logic

### 3.1 Per-process flow — mirrors the DRAM path

| Step | DRAM faults (existing) | Cache faults (new) |
| --- | --- | --- |
| 1. Start | the runner allocates, the observer records VA→PA, then the clean pass runs | same process, same clean pass |
| 2. Sampling basis | **snapshot**: this process's allocations × VA→PA map × G3 GDDR table (the table is measured once, long-lived hardware rule) | **residency map**: one 10K-image probe pass recording every line's residency periods and every image's inference time (§4), measured fresh in this process; from it the process's average exposure R_eff_bits (§3.2) |
| 3. Plan all trials up front | orchestrator draws every trial's DRAM sites (B frozen per level) → work file | orchestrator computes n_cache from R_eff_bits and draws every trial's cache sites → the same work file |
| 4. Trials | runner executes the pre-planned flips | runner executes the pre-planned flips and removals |

- **What varies between trials** is only the random draws: the DRAM cells,
  and the cache lines, start images, bytes and bits. The **100 trials**
  average out this randomness, exactly as for DRAM. Each trial draws its
  DRAM set and its cache set independently.
- **A restart segment re-measures.** After a PROCESS_FATAL crash the new
  process gets new physical addresses, so the new segment measures its
  own residency map before its trials.

### 3.2 Number of cache flips per trial

Unlike DRAM, where the whole model stays resident all the time, the cache
holds only part of the model at any moment, and that part keeps changing.
The count is therefore based on the model's **average exposure in L2**:
the time-averaged number of model bits resident in L2 during the 10K-image
inference.

```text
R_eff_bits = Σ_ℓ ( bits_ℓ × T_ℓ ) / T_total
n_cache    = round( BER_cache × R_eff_bits )

bits_ℓ  = the line's resident bits: 1,024 for a full 128-B line, fewer for a
          partial line at an allocation edge / on a shared page
T_ℓ     = the inference time during which line ℓ was resident
          (the sum of its residency periods, §4)
T_total = the total inference time of the 10K images (§4)
```

- **Same unit in numerator and denominator.** T_ℓ and T_total are both
  GPU inference time, measured per image, with probe time excluded (§4).
  Dividing by Σ T_ℓ instead would always give ~1,024 bits (one line),
  because the residency times cancel, so the denominator must be the total
  inference time.
- **Limit checks**:
  - every line resident throughout gives R_eff_bits = R_bits, the DRAM
    case;
  - every line resident half the time gives R_bits / 2;
  - for ViT-B (94 MB > 72 MiB), R_eff_bits is automatically ≤ the L2
    size, because no more than 72 MiB can be resident at any moment.
- **Per-process, not frozen per level.** R_eff_bits comes from each
  process's own residency measurement, so n_cache is computed per process
  (per restart segment). All trials of one process use the same n_cache.
  Each process's T_total, R_eff_bits and n_cache are recorded in the work
  file metadata and in `summary.json` (§5).
- This counts upsets in the model's own data only. Upsets in L2 lines that
  hold other processes' data, or nothing, are outside the count.
- The injection surface is the same as the DRAM sampler's
  (`fault_model.surface_rows_for`), so Σ_ℓ bits_ℓ = R_bits.

### 3.3 Cache upset rate BER_cache — frozen (user decision 2026-10-03)

```text
BER_cache = ρ × BER_DRAM(level),   ρ = λ_SRAM,per-bit / λ_DRAM,per-bit = 1
```

- **ρ = 1, a single value, no sweep.** The fault-tolerance analysis is
  set in a space-computing context. There, the measured per-bit ratio of
  SRAM cache to DRAM under heavy ions is about 0.1–2, and REMU's
  memory-agnostic rate is equivalent to ρ = 1. The literature survey
  (G8-T3, [reports/GPU SRAM vs DRAM error
  rates.md](../reports/GPU%20SRAM%20vs%20DRAM%20error%20rates.md)) found
  no measurement of L2 and GDDR on the same GPU, and none at all for
  Ada/4N/GDDR6X. Terrestrial-neutron evidence would instead put ρ near
  10³, which is outside this campaign's scope.
- **Equal-fluence trial (modelling assumption).** One trial is treated as
  one exposure window shared by GDDR and L2, so the DRAM flips of the
  level and the cache flips of the trial come from the same per-bit rate.
  The survey's time-window factor (DRAM flips accumulate and persist,
  while a cache flip lives only while its line stays resident) is
  deliberately not modelled (user decision 2026-10-03).
- **Fixed count, `round()`.** n_cache stays `round(BER_cache ×
  R_eff_bits)`, fixed per process like the DRAM `B = round(BER × R)`.
  Only placement varies between trials; Poisson sampling was considered
  and rejected. A process whose expected count rounds to 0 runs
  DRAM-only. The orchestrator prints a `WARNING G8 n_cache = 0` line and
  records `n_cache_expected` in `summary.json`. On an idle GPU this never
  happens. Under a heavy co-tenant it happens only at ResNet-50's L1
  (1e-8, expected 0.33–0.36).
- **Code.** `fault_model.CACHE_RHO = 1.0` and `cache_ber_for(level)`.
  The orchestrator's `--cache-faults` mode uses the frozen value.
  `--cache-ber X` remains as an explicit override, recorded as unfrozen.

### 3.4 Placement and timing of each flip

Radiation upsets are random in time and space: every cell present in L2
has the same small chance to flip at every moment. Hence, for each of the
n_cache flips:

1. **Line**: pick a cache line ℓ of the injection surface with
   probability **`bits_ℓ × T_ℓ / Σ (bits × T)`**, i.e. ∝ its measured
   exposure: its resident bits × its total resident inference time. This
   is the same quantity as the numerator of R_eff_bits (§3.2), so the
   count and the placement come from one exposure measure. For full lines
   this is simply ∝ T_ℓ: a line resident for 80 % of the inference time
   gets about 8× the flips of one resident for 10 %. A line never observed
   resident gets none.
2. **Start image**: pick t **uniformly in time within ℓ's residency
   periods**, i.e. among the images in which ℓ was resident, each with
   probability ∝ its measured inference time. With fixed-shape batch-1
   inference these times are nearly equal, so this is close to uniform over
   those images. A flip never starts while its line is absent.
3. **Byte and bit**: uniform byte within the line (0–127, restricted to
   resident bytes for lines at allocation edges or on shared pages) and a
   uniform bit (0–7).
4. **Duration**: this is **not** a separate random draw; it follows from t.

| Line class | Members | Duration |
| --- | --- | --- |
| **Read-only** | weights, deserialize constants: anything the engine never writes during inference | from t to the **end of the residency period containing t**, i.e. until the line's eviction, capped at the trial end. After eviction the line is refetched from GDDR and is correct again; a later residency period of the same line can host its own, separate flips. |
| **Engine-written** | activations/scratch, input binding, output bindings | **at most the image t**: the engine overwrites the data itself during the inference (even after a dirty write-back, the next write clears the flip) |

Further rules:

- Engine-written lines are still placed and timed by residency (steps
  1–2). Only their duration is capped by the overwrite.
- **Class assignment** comes from measured campaign evidence: an
  allocation is read-only only if every past trial restored it `exact`
  (the G5/G7 `restore_check`). Anything showing `mismatch` or rewrite
  behaviour is engine-written.
- Only the workload's **injection surface** is eligible, the same set the
  DRAM sampler uses (`fault_model.surface_rows_for`; e.g. MobileNetV3's
  excluded ctx pools stay excluded).
- Cache sites of a trial must not overlap: two flips may not occupy the
  same (byte, bit) during overlapping lifetimes. A cache flip **may**
  coincide with a DRAM-flipped bit; this is physically meaningful and is
  handled by §3.6.

### 3.5 Mapping and injection chain

```text
selected cache line ℓ (allocation a, 128-B-aligned line offset)   ← residency-time-weighted (§3.4)
  + byte in line + bit
  → VA = base(a) + line offset + byte                              ← offset arithmetic in the G1.5 registry
  → GPU XOR (the same device injector as DRAM sites)
```

- The PA (G2 map) is recorded per site for the log only.
- The G3 GDDR table is **not** used for cache sites.
- Flips are applied only at images where the measurement saw the line
  resident, so the XOR kernel's load hits L2 and its store changes the
  cached copy.

### 3.6 Removal at the end of a flip's lifetime

Removal models "line evicted, value fetched again from GDDR". GDDR may
itself hold a DRAM flip in that byte, so removal must return the byte to
its value **before the cache flip, DRAM flips included**, not to the
pristine original.

- **Read-only data**: **XOR again with the same mask.** This is exact
  because nothing else writes there. It also covers the rare case of a
  cache flip on a DRAM-flipped bit: the cached copy is temporarily
  correct, and the DRAM error returns at "eviction", which is the physical
  behaviour.
- **Engine-written data**: the engine overwrites the byte during the
  image, so removal is skipped. If needed it is conditional: only if the
  byte still holds the flipped value. **Input-binding** cache flips are
  applied after the per-image input copy, the same way the runner already
  re-applies input-binding DRAM faults.

### 3.7 Trial loop

```text
(per process, once) clean pass → residency measurement pass (§4) → all 100 trials planned

trial start:  apply the DRAM flips (existing path)                   ← held all trial
for image i = 0 .. 9999:
    stage input i
    apply cache flips whose start image == i                         ← XOR, verified
    re-apply input-binding flips (existing rule; cache ones included)
    inference → classify image i against its clean record
    remove cache flips whose last image == i                         ← re-XOR (read-only) / skip (engine-written)
trial end:    restore the DRAM flips (existing path); sanity inference must reproduce clean
```

The DUE abort, PROCESS_FATAL restart protocol (re-measuring per segment,
§3.1), gating, snapshot, residency guard and closing checks are all
inherited from G5/G7.

## 4. Residency measurement (inside every campaign process, GPU 0)

**Purpose**: give every cache line ℓ of the injection surface its
**residency timeline** over the 10,000 images: the list of residency
periods `[start, end]` in image indices, and its total resident inference
time T_ℓ. Also give the total inference time T_total and the process's
average exposure R_eff_bits. These feed §3.2 and §3.4.

**Where in the flow**: after the clean pass and before trial sampling, in
the same process and at the same physical addresses as the trials it
plans. It uses the same engine, the same image list and order, and the
same runner configuration.

**Method**: a time series of probes over a full 10,000-image inference
pass with no faults.

1. Every **k images**, at an image boundary, a probe kernel times one load
   for **every line** of the injection surface (~71 K lines for
   MobileNetV3 up to ~733 K for ViT-B). The latency separates an L2 hit
   from a GDDR fetch.
   - **Probe as calibrated in T0** (§11): one active lane per warp (a
     multi-lane warp times its slowest lane), 16 concurrent probes per SM,
     threshold **440 cycles** (hits 208–400, misses ≥ 464–480), compact
     output (§12).
   - **Production settings** (user decisions 2026-10-02):
     - **probe unit:** 32-B sector, because L2 fills per sector;
     - **probing:** every image (k = 1), with the direction alternating
       for each unit's successive observations;
     - **stride:** 1 for surfaces smaller than L2, and **64 for ViT-B**.
       ViT-B's surface is larger than L2, and a full sweep's own fills
       would evict what it probes last; each ViT-B unit's timeline then
       has a resolution of 64 images (§12).
2. A hit at boundary *i* marks ℓ as resident for the images
   `[i, i + k − 1]`. Consecutive resident blocks form a residency period;
   a miss ends the period.
3. **Inference time per image**: each image's inference is timed on the
   GPU with CUDA events around its enqueue, giving t_i. Probe kernels run
   *between* images and are **excluded**: the measurement pass is slower
   than a real pass, but only inference time enters the timeline.
   - T_ℓ is the sum of t_i over the images in ℓ's residency periods.
   - T_total is the sum of t_i over all 10,000 images.
   - Then R_eff_bits = Σ bits_ℓ × T_ℓ / T_total (§3.2).
4. **Choice of k (and stride)**: set from T0's measured costs.
   - The pure GPU inference time is **0.25 ms/image (ResNet-50)** and
     **0.65 ms/image (ViT-B)**. The 35–53 s per 10K pass quoted earlier
     includes host overhead.
   - A full probe sweep costs 0.034 ms for ResNet-50 (13 % of one
     inference). A stride-16 sweep costs 0.013 ms for ViT-B (2 %).
   - k = 1, probing at every image boundary, is therefore affordable.
   - Smaller k and stride mean finer timelines.
5. **Perturbation**: a probe that misses loads the line into L2. The
   engine reads every weight and constant line in every inference anyway,
   so this barely changes their behaviour. Lines the engine does not touch
   every inference can show slightly extended residency; this is recorded
   as a limitation.
6. **Resolution limit**: an eviction plus refetch that happens entirely
   between two probes is invisible. Residency periods are known to ±k
   images, so read-only flip lifetimes can be slightly overestimated.
7. **Neutrality check, built in**: the measurement pass's outputs must be
   bit-identical to the clean pass, image by image, which proves the
   probes do not change computation.

**Co-tenancy**: GPU 0 is shared. The co-tenant processes are recorded at
the start of the measurement and of the trials (the existing
`cotenancy_at_start` record). The trials assume L2 behaves as it did
during the measurement pass, which is the same workload in the same
process immediately before. A large change in co-tenant load during a long
campaign is a stated limitation.

**In-campaign settings (G8-T2).**

- **Stride:** `--l2-probe-stride auto`. A stride-64 pre-sweep measures
  the surface's miss fraction: ≤ 1 % → 1, ≤ 50 % → 64, > 50 % → 256.
- **Self-checks** before any cache site is sampled (fail-closed):
  - in-gap share ≤ 0.5 %;
  - **order effect**: the probe hit rate of units in the first half of
    their sweep minus the second half, |Δ| ≤ 3 pp;
  - the map matches this process (VAs, image count, frozen R).
- **Diagnostic only:** the direction-locked share. Under a bursty
  co-tenant, units that genuinely flicker fake it, so it does not gate
  (§13).

**Output**, per process (segment):

- `residency_map` file containing:
  - per line: the allocation, line offset, bits_ℓ, class, residency
    periods and T_ℓ;
  - per image: the inference time t_i;
  - per process: T_total and R_eff_bits;
  - the probe thresholds, k, and the co-tenancy record;
- its sha256 is recorded in the work-file metadata and in `summary.json`,
  together with T_total, R_eff_bits and the resulting n_cache.

## 5. Logging and verification

Each cache site records:

- fault class `L2-SBU`;
- allocation, line offset, byte, bit;
- line class (read-only / engine-written);
- the residency period it was drawn from;
- start image and last image;
- VA and PA (PA for reference);
- the before/after values of every apply and removal.

Expected volume is n_cache sites per trial, the same order as the DRAM B,
so every cache site is logged and verified in full, exactly like DRAM
sites.

Verification:

- every cache apply and removal is checked on device
  (`after == before ^ mask`);
- after a read-only removal the byte must equal its pre-cache value;
- the orchestrator re-verifies every cache row independently, as it does
  for DRAM rows;
- the trial event stream must match the work file, extended with
  per-image apply and remove events;
- the sampler re-check confirms that every cache site starts inside a
  measured residency period of its line, and that its last image is that
  period's end (read-only) or the start image (engine-written);
- the orchestrator **recomputes R_eff_bits and n_cache** from the
  residency map (Σ bits_ℓ × T_ℓ / T_total) and requires every trial of the
  process to carry exactly n_cache cache sites.

Reported metrics:

- trial accuracy (DRAM + cache) vs the existing DRAM-only curve;
- the **conditional error rate over images with at least one active cache
  flip**, i.e. how damaging a cache flip is when one occurs.

## 6. Phases

| Phase | Work | Output / gate |
| --- | --- | --- |
| **G8-T0** Feasibility (GPU 0) — **DONE 2026-10-01** | Calibrate the L2-hit vs GDDR-fetch latency of the probe kernel (thresholds, separation, stability); measure probe throughput for full-surface sweeps and choose k per model; runner hook for probe/apply/remove kernels at image boundaries. Done: probe calibration + runner probe pass (`--l2-probe-*`); apply/remove hooks are T2 work. | Gate: clean hit/miss separation; a hooked pass with no flips is bit-identical to the clean pass. **Both PASS** (§11). |
| **G8-T1** Residency pass — **DONE 2026-10-02** | Implement the in-process measurement pass (§4) and the residency-map format. Validation runs per model: two passes in one process (same-process stability) and runs in separate processes (how much the per-line pattern changes, reported as allocation-level statistics). Start with ResNet-50 v2 (default; the user may pick another first model). | Residency maps; stability report. |
| **G8-T2** Implementation — **DONE 2026-10-02** | Cache sampler (§3.2, §3.4) in `tools/g8_cache/cache_model.py` (classes in `fault_model.py`); runner per-image apply/remove; orchestrator flow (clean pass → residency pass → plan → trials, re-measure per restart segment) and independent re-verification; self-tests (apply/remove exactness, overlap with DRAM flips, engine-written skip, input re-staging, start-inside-residency check). | Self-tests PASS; a smoke campaign VERIFIED. |
| **G8-T3** Cache rate — **DONE 2026-10-03** | The user derives BER_cache from the DRAM BER via prior work; the cache level table is frozen alongside the DRAM levels. Done: literature survey; ρ = 1 frozen in `fault_model`; `--cache-faults` mode; confirmation smoke (§14). | The frozen table, documented as for G5-T1 (§14). |
| **G8-T4** Campaigns (GPU 0) — **in progress from 2026-10-03** | DRAM + L2 per model at the frozen levels, 100 trials × 10K images, compared against the existing DRAM-only runs. Levels: the seven BERs 1e-7 … 1e-5 of each model (ResNet-50 L3–L9, the others L1–L7), BER_cache = the same BER (ρ = 1), seed 7, the same engines, 10K split and preprocessing as G7-v2. One model at a time, ResNet-50 v2 first; the user reviews each model's results before the next. Driver: `scripts/run_g8_t4.sh`. | Accuracy curves (DRAM-only vs DRAM + L2) + the conditional cache-hit error rate. |

## 7. Validation additions

- **V6 Probe calibration**: L2 hits and GDDR fetches separate cleanly on
  GPU 0, and the thresholds are stable across repeats.
- **V7 Probe/hook neutrality**: every residency pass reproduces the clean
  pass bit-identically (built into §4).
- **V8 Apply/remove exactness**: every cache apply and remove satisfies
  `after == before ^ mask`. After a read-only removal the byte equals its
  pre-cache value, including the DRAM-overlap case.
- **V9 Residency stability**: in T1, two passes in the same process agree,
  and the cross-process variation is reported.
- **V10 Plan consistency**: every cache site starts inside a measured
  residency period, and its lifetime matches §3.4. R_eff_bits and n_cache
  recomputed from the residency map match the values the process used.
- The existing V4/V5 (per-site XOR checks, byte-exact trial restore,
  sanity inference) still apply.

## 8. Validity boundary

**Claimed**:

- L2 SBUs on the workload's data, with a per-trial count of
  `round(BER_cache × R_eff_bits)`, where R_eff_bits is the model's
  time-averaged resident bits in L2 measured in the same process;
- placement and start times weighted by each line's residency measured in
  the same process;
- lifetimes equal to the remaining residency period (read-only data) or
  ended by the engine's own overwrite (engine-written data);
- every site traced through the same allocation/VA chain as the DRAM
  faults.

**Not claimed / simplifications**:

- Image-level time resolution: a real flip in the middle of an inference
  affects only later layers, while ours is present for the whole
  inference (a slight overestimate).
- Probe resolution k: evictions shorter than k images are invisible.
- One residency measurement per process is assumed representative of that
  process's trials. Co-tenant load can drift.
- No L2 physical coordinates (slice/set/way/bit) and no cache MCU.
- No L2 tag/state-bit faults; no ECC; no L1/shared/register faults.
- BER_cache = ρ × BER_DRAM with ρ = 1 is a literature-based modelling
  choice for the space context, not measured on this hardware (§3.3).
- Equal-fluence trial: DRAM and L2 share one exposure window per trial,
  with no time-window factor between persistent DRAM flips and
  residency-limited cache flips (§3.3).

## 9. Open items

1. ~~BER_cache derivation~~: **resolved 2026-10-03**, ρ = 1 (§3.3,
   §14).
2. First model for T1/T4: ResNet-50 v2 by default.
3. ~~Probe unit~~: **resolved 2026-10-02**, 32-B sector.
4. ~~k and stride per model~~: **resolved 2026-10-02**. k = 1 with the
   direction alternating; stride 1 for surfaces smaller than L2 and 64 for
   ViT-B. T1 replaced the T0 suggestion of 16 because of the
   self-eviction artifact (§12).
5. Whether to report a "dedicated GPU" condition in addition to the
   shared one, if an idle window on GPU 0 becomes available. All T0
   measurements ran on an idle GPU 0.
6. ~~Provisional T2 self-check limits~~: **confirmed by the user
   2026-10-03**, unchanged:
   - in-gap share ≤ 0.5 %;
   - order effect |Δ| ≤ 3 pp;
   - auto-stride tiers: pre-sweep miss ≤ 1 % → stride 1, ≤ 50 % → 64,
     > 50 % → 256.

   Their calibration is in §13.

## 10. Design history (why the earlier options were dropped)

| Option (2026-09-29 … 10-01 discussion) | Why it was dropped |
| --- | --- |
| XOR "cache" bits once before inference and keep them all trial | Indistinguishable from a DRAM fault: the store goes through L2 and is written back, so the flip persists like DRAM, which contradicts the transient nature of cache. |
| Fresh cache flips at the full BER count for **every** image | The BER count is an accumulated exposure. Applying it per image multiplies cache exposure by ~10,000× per trial and would collapse accuracy for an unphysical reason. Replaced by one per-trial budget spread over time. |
| Each cache flip lives exactly one image | Ignores that read-only lines can stay resident across many images. Replaced by residency-derived lifetimes. |
| Allocation-level weighting (size × resident fraction), uniform line within the allocation, start uniform over the trial, duration drawn independently from a survival curve | Too coarse, and inconsistent: a flip could start while its line is absent, and its duration was decoupled from its start. Replaced by per-line residency-time weighting, start uniform within the line's residency periods, and duration = remaining period (§3.4). |
| Residency measured in separate runs (1K, then 10K paired probes) | Per-line residency depends on the process's physical addresses and on co-tenant load, so it does not transfer between processes. Replaced by an in-process 10K-image timeline before each campaign segment's trials (§4). |
| `n = BER × L2_bits × occupancy`, then `n = BER × min(R_bits, L2_bits)`, then `n = BER_cache × R_bits` | R_bits assumes the whole model is present all the time, which is true for DRAM but not for the cache. Replaced by `n = BER_cache × R_eff_bits` with `R_eff_bits = Σ bits_ℓ × T_ℓ / T_total`, the time-averaged resident bits (user decision 2026-10-01, §3.2). A denominator of Σ T_ℓ was rejected because it always reduces to ~1,024 bits (one line); the denominator is the total inference time of the 10K images. |
| Mid-inference injection via CUDA-graph capture; L1/shared via NVBit; three cards | More complex than needed. Scope set to L2 only, image-boundary resolution, GPU 0 only (user decision 2026-09-30). |
| A ρ sweep {1, 10, 100, 1000}; a time-window factor `T_win / T_acc`; Poisson-sampled n_cache (all proposed by the T3 survey) | A sweep multiplies the T4 campaign cost without changing the space-context conclusion; the time-window factor complicates the model; Poisson makes the cache count vary per trial while the DRAM count is fixed. Kept: ρ = 1, an equal-fluence trial, fixed `round()` with a visible n_cache = 0 warning (user decision 2026-10-03, §3.3). |

## 11. G8-T0 results (2026-10-01, GPU 0 idle)

Full tables are in `tools/g8_cache/README.md`. Artifacts (untracked) are
under `artifacts/g8/t0/`.

**V6 probe calibration: PASS.** One active lane per warp, 16 probes per
SM, 3 runs × 10 rounds:

- hits 208–400 cycles, misses ≥ 480 cycles, an empty gap of 80–112
  cycles;
- 100 % correct classification of cold, warm, reprobe, and the per-line
  mixed test (only the even lines resident);
- threshold 440 / 440 / 448 cycles, with **440** in production;
- the capacity cliff sits at the 72 MiB L2;
- L2 fills per 32-B sector.

**V7 probe-pass neutrality: PASS.** Every verified probe pass (27 passes
× 10,000 images, ResNet-50 v2 and ViT-B, every probe mode) reproduced the
clean pass bit-for-bit. In situ, ≤ 0.2 % of latencies fall inside the
calibration gap, so the threshold transfers to real TensorRT memory.

**Three probe-design findings**, all fixed in the shipped probe:

1. **The warp-max effect.** With all 32 lanes probing, every lane times
   the warp's slowest load. The mixed test then reads 0 % of the resident
   lines as hits, so this design measures per 32-line group, not per line.
   The probe therefore uses one active lane per warp.
2. **Self-eviction for surfaces larger than L2.** A full ViT-B sweep
   (94 MB) evicts whatever it probes last: forward vs reverse order gives
   weights 78 % vs 0 % resident. Staggered sub-sampled sweeps fix this. At
   stride 16 and 64, forward and reverse agree on weights within 0.4 pp
   and on R_eff within 0.8 pp.
3. **Concurrency limit.** Above about 16 probes per SM, queueing pushes
   hits into the miss range.

**Measured residency (idle GPU 0)**:

- **ResNet-50 v2 (28.8 MB)**: every unit of every allocation is
  L2-resident at all 10,000 image boundaries, which gives R_eff = R. This
  also holds with probes only every 100 images and per 32-B sector, so it
  is not created by the probe. Under §3.2/§3.4 that means
  `n_cache = round(BER_cache × R_bits)`, and every read-only cache flip
  lasts until the trial ends.
- **ViT-B (93.9 MB)**:
  - weights ~79 % resident;
  - scratch ~68–77 %;
  - the input binding 0 % at the boundaries (evicted by the inference's
    own streaming);
  - R_eff ≈ 73.5–74.2 MB, about the L2's effective capacity, as expected
    for a model larger than L2.

  Open: a residual ~8 pp forward/reverse difference on ViT-B scratch
  (~0.4 MB, ≈0.6 % of R_eff). It does not shrink with the stride, and it
  is recorded as a measurement uncertainty without a claimed cause.

**What T1 inherits**: the calibrated probe (440 cycles, 16/SM, 1 lane),
k = 1, stride 1 for surfaces below L2 and 16 above, and the open choices
in §9 (probe unit; first model).

## 12. G8-T1 results (2026-10-02, GPU 0 idle)

Full tables are in `tools/g8_cache/README.md` (T1 section). Artifacts
(untracked) are under `artifacts/g8/t1/`.

**What T1 built.**

- **Runner:** `--l2-probe-map 1` turns every probed sector's hit/miss
  into residency periods with `gpu_m2d::ResidencyMapBuilder`, which has a
  C++ unit test. It writes `_residency.bin` + `_residency.json`
  containing:
  - per-image GPU inference times;
  - per-unit resident bytes and periods;
  - T_total, R_eff_bits and the sha256.
- **Same-process passes:** `--l2-probe-passes N` runs N passes in one
  process.
- **Reader for T2:** `tools/g8_cache/residency_map.py`, stdlib and
  fail-closed. It verifies the sha256, the structure and the neutrality,
  and it independently recomputes T_total and R_eff_bits, which must
  equal the runner's values. Its `summary`, `compare` and `diagnose`
  commands produce the tables below.

**Final maps.** Each model ran 2 passes in one process plus a second
process, all neutral and verified.

| | ResNet-50 v2 (stride 1) | ViT-B (stride 64) |
| --- | --- | --- |
| R_eff | 28.832 MB = **100.00 %** of R | 74.212 MB = **79.06 %** of R (≈ L2 capacity) |
| Weights | 100 % of sectors always resident | 79.1 % always, 19.7 % never, 1.1 % partial |
| Scratch | 99.3 % always, 0.7 % partial | 71.2 % always, 22.5 % never, 6.2 % partial |
| Same class, same process / cross process | 99.98 % / 99.99 % | 99.77 % / 99.77 % |
| R_eff difference, same process / cross process | ≤ 0.0001 % | ≤ 0.0004 % |

**All six models.** The first T1 round measured only ResNet-50 v2 and
ViT-B. After a scoping fix, the other four were measured the same way:

- **The scoping fix.** `--l2-probe-exclude` and
  `--l2-probe-expect-surface-bytes` make the map cover exactly each
  workload's injection surface. MobileNetV3 excludes its ctx pools
  `trt-internal-5/6`. Every map's surface equals the frozen R.
- **Settings.** Stride 1 for the five models smaller than L2, 64 for
  ViT-B. Each model ran 2 passes in one process plus a second process: 18
  passes, all neutral.

| Model | Stride | Sectors | Frozen R (B) | Map surface = R | R_eff | R_eff / R | Weight sectors: always / never / partial | Same class: same proc / cross proc | Max R_eff diff | Direction-locked sectors | Mismatches (3 passes) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 v2 | 1 | 901,011 | 28,832,268 | yes | 28.832 MB | **100.00 %** | 100.0 / 0.0 / 0.0 % | 99.98 / 99.99 % | 0.00006 % | 2 (0.000 %) | 0 |
| MobileNetV3-L | 1 | 283,999 | 9,087,912 (scoped) | yes | 9.088 MB | **100.00 %** | 99.9 / 0.0 / 0.1 % | 99.95 / 99.95 % | 0.00001 % | 0 | 0 |
| EfficientNet-B0 | 1 | 535,759 | 17,144,232 | yes | 17.144 MB | **100.00 %** | 100.0 / 0.0 / 0.0 % | 99.96 / 99.97 % | 0.00001 % | 6 (0.001 %) | 0 |
| DeiT-S | 1 | 812,841 | 26,010,832 | yes | 26.011 MB | **100.00 %** | 100.0 / 0.0 / 0.0 % | 99.99 / 100.00 % | 0.00000 % | 0 | 0 |
| Swin-T | 1 | 1,368,740 | 43,799,616 | yes | 43.800 MB | **100.00 %** | 100.0 / 0.0 / 0.0 % | 99.99 / 99.99 % | 0.00000 % | 0 | 0 |
| ViT-B | 64 | 2,933,369 | 93,867,728 | yes | 74.212 MB | **79.06 %** | 79.1 / 19.7 / 1.1 % | 99.77 / 99.77 % | 0.00043 % | 30,022 (1.02 %) | 0 |

**V9 residency stability: PASS for all six.** Agreement is ≥ 99.77 % per
sector, and the R_eff difference is ≤ 0.0004 % within a process and
across processes. **Every model smaller than L2 is 100 % resident for the
whole run on an idle GPU**, so R_eff = R. Only ViT-B is partially
resident, at ≈ 74 MB, which is about the L2's capacity.

**Probe-footprint artifacts.** Measuring per sector exposed two artifacts
that T0's allocation-level counts had hidden:

1. **Scattered latency writes.** Each probe wrote its latency at
   `out[u]`, dirtying one L2 sector per probed unit: 5.9 MB per ViT-B
   sweep. **Fixed** with compact output. On ViT-B the reverse-locked
   weight sectors fell from 36,218 to 88, and R_eff rose from 69.05 to
   72.18 MB.
2. **The sweep's own miss fills.** These evict sectors the sweep has not
   reached yet, so with alternating direction those sectors oscillate
   with the probe direction. The effect is **bounded by the stride**:

   | ViT-B stride | Weights direction-locked | Scratch direction-locked | R_eff |
   | --- | --- | --- | --- |
   | 16 | 4.97 % | 12.7 % | 72.18 MB |
   | 64 | 0.77 % | 5.3 % | 74.21 MB |
   | 256 | 0.25 % | 0.79 % | 74.56 MB |

   Stride 64 was chosen: a 64-image resolution, with R_eff within 0.5 %
   of the stride-256 value. The remaining bias is in one direction: a
   miss can be an artifact but a hit cannot, so the true R_eff is at or
   slightly above the measured value. ResNet-50 is unaffected, because
   its whole surface is resident and its probes do not miss.

**Implications for T2.**

- **ResNet-50 v2** on an idle GPU 0:
  - R_eff = R, so `n_cache = round(BER_cache × R_bits)`;
  - placement is uniform over the surface;
  - every read-only cache flip lasts until the trial ends.
- **ViT-B**:
  - flips concentrate on the ~80 % of weight sectors that stay resident;
  - the ~20 % of weight sectors never resident at image boundaries
    receive none;
  - the input binding, streamed in and evicted within each inference, is
    never resident at boundaries and receives none. This is the
    image-level time-resolution limit stated in §8.

**For T2 to handle:**

- **The residency pass must run inside the campaign process**, before
  the campaign gate, so that the orchestrator can sample from the map.
- **The probe's output buffer is an unregistered `cudaMalloc`.** Under
  the observer it must either be registered in the allocation registry as
  a non-surface allocation, or be tolerated by the ledger.

## 13. G8-T2 results (2026-10-02, GPU 0)

Full tables are in `tools/g8_cache/README.md` (T2 section). Run
directories are under `artifacts/g8/t2/`.

**What T2 built.**

- **T2-a: in-campaign residency pass.**
  - The pass runs before the gate.
  - Its probe output buffer is a registered allocation
    (`G8_L2_PROBE_OUTPUT`), so it is ledger-covered and outside the
    surface.
  - It uses the auto stride.
- **T2-b: per-process self-checks** (§4).
- **T2-c: cache lifetime classes.** `derive_alloc_classes.py`, frozen in
  `fault_model.WORKLOADS`. A TRT-internal allocation is read-only iff
  every G7-v2 restore was `exact`; bindings are engine-written.
- **T2-d: sampler** (`cache_model.py`).
- **T2-e: runner per-image apply/remove** (§3.5–§3.7).
- **T2-f: orchestrator `--cache-ber`.** Sampling at the gate,
  independent re-verification of the cache rows and events, a
  cache-aware skeleton and restart merge, and the provenance in
  `summary.json`.

**Smoke campaigns: `G5_CAMPAIGN_VERIFIED` on all six models.** Each ran
DRAM BER 1e-7 + `--cache-ber 1e-7`, 2 trials, on an idle GPU 0.

| Model | Auto stride | R_eff | n_cache | Order effect | Cache removals: re-XOR / restored / overwritten |
| --- | --- | --- | --- | --- | --- |
| ResNet-50 v2 | 1 | 28.83 MB | 23 | 0.00 pp | 36 / 3 / 7 |
| MobileNetV3-L | 1 | 9.09 MB | 7 | 0.00 pp | 7 / 2 / 5 |
| EfficientNet-B0 | 1 | 17.14 MB | 14 | 0.00 pp | 15 / 2 / 11 |
| DeiT-S | 1 | 26.01 MB | 21 | 0.00 pp | 32 / 0 / 10 |
| Swin-T | 1 | 43.80 MB | 35 | 0.00 pp | 41 / 2 / 27 |
| ViT-B | 64 | 74.21 MB | 59 | −0.35 pp | 111 / 0 / 7 |

In every trial:

- read-only flips returned exactly to their pre-cache value;
- engine-written flips were restored or overwritten by the engine;
- the sanity inference reproduced clean.

**Controlled shared-GPU test.** The co-tenant was
`tools/g8_cache/g8_l2_thrash`, either heavy (64 MiB continuous) or
moderate (32 MiB, 50 % duty). The test used ResNet-50 v2 and ViT-B.

- **Neutrality and threshold hold.** 0 mismatches; in-gap share
  ≤ 0.18 % under load.
- **Residency follows the co-tenant.** Heavy load: ResNet-50 R_eff 28.8
  → 4.1–4.5 MB, ViT-B → 3–4 MB. Moderate load: ResNet-50 99.7 %, ViT-B
  42–55 MB, varying between processes.
- **The order effect is the gate statistic.** The known idle stride-16
  artifact reads −3.87 pp. Every production configuration, idle or
  shared, stays within ±2.3 pp. In the shared cases the residual is the
  co-tenant evicting during the sweep, bounding a map's bias at about
  1 pp.
- **The direction-locked share was demoted to informational.** It
  reached 5 % for moderate ViT-B even at stride 256, from genuine
  flicker.
- **Heavy contention needs stride 256.** Heavy ResNet-50 reads +2.18 pp
  at stride 64 and −0.12 pp at stride 256.

**Not yet exercised live:** a PROCESS_FATAL restart in cache mode. These
low-intensity smoke trials did not crash; the death analysis and merge
paths are covered by the orchestrator self-test.

**Next (T3):** BER_cache from prior work, and freezing the cache level
table. Done (§14).

## 14. G8-T3 results (2026-10-03)

**Literature survey.** [reports/GPU SRAM vs DRAM error
rates.md](../reports/GPU%20SRAM%20vs%20DRAM%20error%20rates.md), with
notes in `research_notes/GPU SRAM vs DRAM error rates/`. No prior work
measured per-bit upset rates of a GPU's L2 and its GDDR on the same
device, and none covers Ada, TSMC 4N or GDDR6X. The evidence for
ρ = λ_SRAM/λ_DRAM depends on the radiation environment:

| Environment | ρ | Basis |
| --- | --- | --- |
| Heavy ions (space) | ≈ 0.1–2 | 10 nm phone-chip L2 vs COTS DDR4 saturation cross sections |
| Titan K20X field (28 nm, ECC on, healthy cards) | ≈ 40–260 | same-device L2 vs GDDR5 SBE shares |
| Terrestrial neutrons | ≈ 10³ (10²–10⁴) | FinFET SRAM 2–20 FIT/Mb vs modern DRAM 0.001–0.0125 FIT/Mb |
| Protons | no number | no per-bit DRAM data |

**Decision (user, 2026-10-03).** ρ = 1 for the space-computing context,
no sweep. An equal-fluence trial, with no time-window factor. Fixed
`round()` (§3.3).

**Frozen cache level table.** BER_cache = the level's DRAM BER, for every
G7-v2 level. n_cache is computed per process from that process's R_eff
(§3.2). The values below use the idle-GPU T1/T2 R_eff, so they are
expectations, not frozen counts:

| Model | Levels → BER_cache (ρ = 1) → expected idle-GPU n_cache per trial |
| --- | --- |
| ResNet-50 v2 | L1 1e-8 → 2 · L2 5e-8 → 12 · L3 1e-7 → 23 · L4 5e-7 → 115 · L5 1e-6 → 231 · L6 3e-6 → 692 · L7 5e-6 → 1,153 · L8 7e-6 → 1,615 · L9 1e-5 → 2,307 |
| MobileNetV3-L | L1 1e-7 → 7 · L2 5e-7 → 36 · L3 1e-6 → 73 · L4 3e-6 → 218 · L5 5e-6 → 364 · L6 7e-6 → 509 · L7 1e-5 → 727 |
| EfficientNet-B0 | L1 1e-7 → 14 · L2 5e-7 → 69 · L3 1e-6 → 137 · L4 3e-6 → 411 · L5 5e-6 → 686 · L6 7e-6 → 960 · L7 1e-5 → 1,372 |
| DeiT-S | L1 1e-7 → 21 · L2 5e-7 → 104 · L3 1e-6 → 208 · L4 3e-6 → 624 · L5 5e-6 → 1,040 · L6 7e-6 → 1,457 · L7 1e-5 → 2,081 |
| Swin-T | L1 1e-7 → 35 · L2 5e-7 → 175 · L3 1e-6 → 350 · L4 3e-6 → 1,051 · L5 5e-6 → 1,752 · L6 7e-6 → 2,453 · L7 1e-5 → 3,504 |
| ViT-B | L1 1e-7 → 59 · L2 5e-7 → 297 · L3 1e-6 → 594 · L4 3e-6 → 1,781 · L5 5e-6 → 2,969 · L6 7e-6 → 4,156 · L7 1e-5 → 5,937 |

The only zero case observed: under the heavy `g8_l2_thrash` co-tenant
(§13), ResNet-50's R_eff drops to about 4.1–4.5 MB. At L1 (1e-8) the
expected count is then 0.33–0.36, which rounds to 0. That process runs
DRAM-only and prints `WARNING G8 n_cache = 0`.

**Confirmation smoke** (`--cache-faults`, 2 trials, GPU 0 idle, run
directories under `artifacts/g8/t3/campaign/`):

| Model | Level (BER) | Status | R_eff (stride) | n_cache (expected) | Cache sites | Removals: re-XOR / overwritten |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 v2 | L1 (1e-8) | VERIFIED | 28.832 MB (1) | 2 (2.307) | 4 | 3 / 1 |
| ViT-B | L1 (1e-7) | VERIFIED | 74.214 MB of 93.868 MB (64) | 59 (59.371) | 118 | 111 / 7 |

Both runs record `g8_cache_ber_frozen = true` and `g8_cache_rho = 1.0` in
`summary.json`, with `n_cache_expected` per segment. Every read-only flip
returned to its pre-cache value, and every trial's sanity inference
reproduced the clean output.

**Next (T4):** DRAM + L2 campaigns per model at the frozen levels (100
trials × 10K images, `--cache-faults`), compared against the existing
DRAM-only runs.
