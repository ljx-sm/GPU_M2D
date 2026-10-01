# G8 Plan — L2 Cache Fault Injection on Top of the GDDR Campaign

Status: **DESIGN CONFIRMED by the user, 2026-10-01. No code yet.** This
file has been revised in place through the 2026-09-29 … 10-01 discussion
(earlier versions are in git history; §10 records what changed and why).
One parameter is still open: the cache upset rate BER_cache (§3.3).

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
| 2. Sampling basis | **snapshot**: this process's allocations × VA→PA map × G3 GDDR table (the table is measured once, long-lived hardware rule) | **residency map**: one 10K-image probe pass recording every line's residency periods (§4), measured fresh in this process |
| 3. Plan all trials up front | orchestrator draws every trial's DRAM sites → work file | orchestrator draws every trial's cache sites → the same work file |
| 4. Trials | runner executes the pre-planned flips | runner executes the pre-planned flips and removals |

- **What varies between trials** is only the random draws: the DRAM cells,
  and the cache lines, start images, bytes and bits. The **100 trials**
  average out this randomness, exactly as for DRAM. Each trial draws its
  DRAM set and its cache set independently.
- **A restart segment re-measures.** After a PROCESS_FATAL crash the new
  process gets new physical addresses, so the new segment measures its
  own residency map before its trials.

### 3.2 Number of cache flips per trial

```text
n_cache = round(BER_cache × R_bits)
```

- R_bits is the workload's frozen injection surface in bits: the same R
  as the DRAM BER, `resident_bytes_nominal × 8` in `fault_model.WORKLOADS`.
- The count is **fixed** per level and trial, just like the DRAM B.
  Residency decides only *which lines* and *when* (§3.4), never *how many*.
- This counts upsets in the model's own data only. Upsets in L2 lines that
  hold other processes' data, or nothing, are outside the count.

### 3.3 Cache upset rate BER_cache — open (user, later)

BER_cache is a separate parameter from the DRAM BER. SRAM and GDDR cells
have different per-bit upset rates, and the relation will be derived
later from prior work (a literature step, like the G5-T0 survey). Every
other part of this design is independent of its value. The cache level
table is frozen only after that derivation (phase T3).

### 3.4 Placement and timing of each flip

Radiation upsets are random in time and space: every cell present in L2
has the same small chance to flip at every moment. Hence, for each of the
n_cache flips:

1. **Line**: pick a cache line ℓ of the injection surface with
   probability **∝ its total measured resident time** T_ℓ, the number of
   images it was resident across all its residency periods in the
   measurement pass. A line resident for 8,000 images gets about 8× the
   flips of one resident for 1,000. A line never observed resident gets
   none.
2. **Start image**: pick t **uniformly among the images in which ℓ was
   resident**. A flip never starts while its line is absent.
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
periods `[start, end]` in image indices, and the total resident time
T_ℓ. These feed §3.4.

**Where in the flow**: after the clean pass and before trial sampling, in
the same process and at the same physical addresses as the trials it
plans. It uses the same engine, the same image list and order, and the
same runner configuration.

**Method**: a time series of probes over a full 10,000-image inference
pass with no faults.

1. Every **k images**, at an image boundary, a probe kernel times one load
   for **every line** of the injection surface (~71 K lines for
   MobileNetV3 up to ~733 K for ViT-B). The latency separates an L2 hit
   from a GDDR fetch (the G3 technique; thresholds calibrated in T0).
2. A hit at boundary *i* marks ℓ as resident for the images
   `[i, i + k − 1]`. Consecutive resident blocks form a residency period;
   a miss ends the period. T_ℓ is the sum of its periods.
3. **Choice of k**: it is set from the probe throughput measured in T0,
   so the pass costs an acceptable time. The faultless inference itself
   takes ~35–53 s per 10K INT8 pass (G7-T0), and probes add per-boundary
   time. Smaller k means finer timelines.
4. **Perturbation**: a probe that misses loads the line into L2. The
   engine reads every weight and constant line in every inference anyway,
   so this barely changes their behaviour. Lines the engine does not touch
   every inference can show slightly extended residency; this is recorded
   as a limitation.
5. **Resolution limit**: an eviction plus refetch that happens entirely
   between two probes is invisible. Residency periods are known to ±k
   images, so read-only flip lifetimes can be slightly overestimated.
6. **Neutrality check, built in**: the measurement pass's outputs must be
   bit-identical to the clean pass, image by image, which proves the
   probes do not change computation.

**Co-tenancy**: GPU 0 is shared. The co-tenant processes are recorded at
the start of the measurement and of the trials (the existing
`cotenancy_at_start` record). The trials assume L2 behaves as it did
during the measurement pass, which is the same workload in the same
process immediately before. A large change in co-tenant load during a long
campaign is a stated limitation.

**Output**, per process (segment):

- `residency_map` file: per line, the allocation, line offset, class and
  residency periods, plus the probe thresholds, k, and the co-tenancy
  record;
- its sha256 is recorded in the work-file metadata and in `summary.json`.

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
  period's end (read-only) or the start image (engine-written).

Reported metrics:

- trial accuracy (DRAM + cache) vs the existing DRAM-only curve;
- the **conditional error rate over images with at least one active cache
  flip**, i.e. how damaging a cache flip is when one occurs.

## 6. Phases

| Phase | Work | Output / gate |
| --- | --- | --- |
| **G8-T0** Feasibility (GPU 0) | Calibrate the L2-hit vs GDDR-fetch latency of the probe kernel (thresholds, separation, stability); measure probe throughput for full-surface sweeps and choose k per model; runner hook for probe/apply/remove kernels at image boundaries. | Gate: clean hit/miss separation; a hooked pass with no flips is bit-identical to the clean pass. |
| **G8-T1** Residency pass | Implement the in-process measurement pass (§4) and the residency-map format. Validation runs per model: two passes in one process (same-process stability) and runs in separate processes (how much the per-line pattern changes, reported as allocation-level statistics). Start with ResNet-50 v2 (default; the user may pick another first model). | Residency maps; stability report. |
| **G8-T2** Implementation | Cache sampler (§3.2, §3.4) in `fault_model.py`; runner per-image apply/remove; orchestrator flow (clean pass → residency pass → plan → trials, re-measure per restart segment) and independent re-verification; self-tests (apply/remove exactness, overlap with DRAM flips, engine-written skip, input re-staging, start-inside-residency check). | Self-tests PASS; a smoke campaign VERIFIED. |
| **G8-T3** Cache rate | The user derives BER_cache from the DRAM BER via prior work; the cache level table is frozen alongside the DRAM levels. | The frozen table, documented as for G5-T1. |
| **G8-T4** Campaigns (GPU 0) | DRAM + L2 per model at the frozen levels, 100 trials × 10K images, compared against the existing DRAM-only runs. | Accuracy curves (DRAM-only vs DRAM + L2) + the conditional cache-hit error rate. |

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
  residency period, and its lifetime matches §3.4.
- The existing V4/V5 (per-site XOR checks, byte-exact trial restore,
  sanity inference) still apply.

## 8. Validity boundary

**Claimed**:

- L2 SBUs on the workload's data, with a per-trial count of
  `round(BER_cache × R_bits)`;
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
- BER_cache is a parameter derived from the literature, not measured on
  this hardware.

## 9. Open items

1. BER_cache derivation (§3.3), done by the user from prior work.
2. First model for T1/T4: ResNet-50 v2 by default.
3. The value of k per model, set in T0 from the probe throughput.
4. Whether to report a "dedicated GPU" condition in addition to the
   shared one, if an idle window on GPU 0 becomes available.

## 10. Design history (why the earlier options were dropped)

| Option (2026-09-29 … 10-01 discussion) | Why it was dropped |
| --- | --- |
| XOR "cache" bits once before inference and keep them all trial | Indistinguishable from a DRAM fault: the store goes through L2 and is written back, so the flip persists like DRAM, which contradicts the transient nature of cache. |
| Fresh cache flips at the full BER count for **every** image | The BER count is an accumulated exposure. Applying it per image multiplies cache exposure by ~10,000× per trial and would collapse accuracy for an unphysical reason. Replaced by one per-trial budget spread over time. |
| Each cache flip lives exactly one image | Ignores that read-only lines can stay resident across many images. Replaced by residency-derived lifetimes. |
| Allocation-level weighting (size × resident fraction), uniform line within the allocation, start uniform over the trial, duration drawn independently from a survival curve | Too coarse, and inconsistent: a flip could start while its line is absent, and its duration was decoupled from its start. Replaced by per-line residency-time weighting, start uniform within the line's residency periods, and duration = remaining period (§3.4). |
| Residency measured in separate runs (1K, then 10K paired probes) | Per-line residency depends on the process's physical addresses and on co-tenant load, so it does not transfer between processes. Replaced by an in-process 10K-image timeline before each campaign segment's trials (§4). |
| `n = BER × L2_bits × occupancy`, then `n = BER × min(R_bits, L2_bits)` | Replaced by `n = BER_cache × R_bits` (user decision 2026-10-01): the same R basis as the DRAM BER, with residency deciding only placement and timing. |
| Mid-inference injection via CUDA-graph capture; L1/shared via NVBit; three cards | More complex than needed. Scope set to L2 only, image-boundary resolution, GPU 0 only (user decision 2026-09-30). |
