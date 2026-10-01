# G8 Plan — L2 Cache Fault Injection on Top of the GDDR Campaign

Status: **DESIGN CONFIRMED by the user, 2026-09-30. No code yet.** This
file replaces the 2026-09-29 proposal in place (the earlier text is in git
history; §10 records what changed and why). One parameter is still open:
the cache upset rate (§3.3).

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

| Model | Frozen R | Fits in L2? |
| --- | --- | --- |
| MobileNetV3 | 9.1 MB | yes |
| EfficientNet-B0 | 17.1 MB | yes |
| DeiT-S | 26.0 MB | yes |
| ResNet-50 | 28.8 MB | yes |
| Swin-T | 43.8 MB | yes |
| ViT-B | 93.9 MB | **no** |

## 2. Principles

1. **DRAM faults are persistent.** GDDR has no ECC and no scrubbing, so
   upsets accumulate and stay. The existing campaign (select GDDR cells →
   G3 table / G2 map → byte/bit → GPU XOR, held for the whole trial) is
   physically right and **stays exactly as it is**. The XOR store modifies
   the L2 copy of the line (dirty); every SM reads that copy, and eviction
   writes it back to GDDR.
2. **Cache faults are transient.** An L2 line is a temporary copy. A flip
   in a cached copy lasts only as long as the line stays in L2 (for
   read-only data) or until the engine overwrites the data. The hard part
   of cache faults is therefore **how long a flip lasts**, and this design
   answers it with a measurement (§4), not an assumption.
3. **A cache flip is applied through an address.** No instruction
   addresses an L2 SRAM cell (slice/set/way/bit). A "cache line" in this
   design is **a 128-B-aligned block of the workload's memory that is
   resident in L2**. XOR-ing a byte of that block while it is resident
   modifies exactly the cached copy that all SMs read. For SBU this is
   exact: with one bit per event, the physical SRAM neighbourhood never
   matters, so the physical L2 mapping is not needed.

## 3. Confirmed fault logic

### 3.1 DRAM part — unchanged

Same frozen BER ladder, B = round(BER × R_bits), the same SBU/MCU fault
model (docs/G5_FAULT_MODEL.md), the same site sampling, and the same XOR,
verification and restore. Flips are applied at trial start and held
through all 10,000 images.

### 3.2 Number of cache flips per trial

```
n_cache = round(BER_cache × min(R_bits, L2_bits))      L2_bits = 603,979,776
```

- For models smaller than L2 (five of six), the cache-exposed bits are the
  model's own R_bits. For ViT-B the count is capped at the full L2.
- This counts only flips that land on the workload's data. Flips in L2
  lines holding other processes' data, or nothing, are benign and are not
  generated.
- n_cache is **fixed** by this rule. GPU 0 is shared, so how much of the
  model is actually resident is uncertain; that uncertainty affects only
  *where* the flips land and *how long* they last (§3.4–3.5), never *how
  many* there are.

### 3.3 Cache upset rate BER_cache — open (user, later)

BER_cache is a separate parameter from the DRAM BER. SRAM and GDDR cells
have different per-bit upset rates, and the relation will be derived
later from prior work (a literature step, like the G5-T0 survey). Every
other part of this design is independent of its value. The cache level
table is frozen only after that derivation (phase T3).

### 3.4 Where each flip lands — residency-weighted

Lines that stay resident longer are exposed longer, so they collect more
upsets. Each flip:

1. picks an allocation *a* with probability ∝ `size_a × f_a`, where `f_a`
   is the measured resident fraction of that allocation (§4);
2. picks a 128-B line uniformly within *a*;
3. picks a byte within the line uniformly (0–127) and a bit uniformly
   (0–7).

Further rules:

- Only the workload's **injection surface** is eligible, the same set the
  DRAM sampler uses (`fault_model.surface_rows_for`; e.g. MobileNetV3's
  excluded ctx pools stay excluded).
- All sites of a trial are distinct (byte, bit) pairs among cache sites.
  A cache flip **may** coincide with a DRAM-flipped bit; this is physically
  meaningful and handled by §3.7.

### 3.5 When it starts and how long it lasts

- **Start image**: uniform over the trial's 10,000 images.
- **Duration** depends on the allocation's class:

| Class | Members | Duration |
| --- | --- | --- |
| **Read-only** | weights, deserialize constants: anything the engine never writes during inference | drawn from the allocation's **measured survival curve** (§4), from 1 image up to the rest of the trial; capped at the trial end |
| **Engine-written** | activations/scratch, input binding, output bindings | **at most the one image** it starts in; the engine overwrites the data itself |

Class assignment is taken from measured campaign evidence: an allocation
is read-only only if every past trial restored it `exact` (the G5/G7
`restore_check`). Anything showing `mismatch` or rewrite behaviour is
engine-written.

### 3.6 Mapping and injection chain

```
selected cache line (allocation a, 128-B line offset)    ← residency-weighted sampling (§3.4)
  + byte in line (0–127) + bit (0–7)
  → VA = base(a) + line offset + byte                     ← offset arithmetic in the G1.5 registry
  → GPU XOR (the same device injector as DRAM sites)
```

- The PA (G2 map) is recorded per site for the log only.
- The G3 GDDR table is **not** used for cache sites.
- The line is resident in L2, so the XOR kernel's load hits L2 and its
  store changes exactly the cached copy.

### 3.7 Removal at the end of a flip's lifetime

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

### 3.8 Trial loop

```text
trial start:  apply the DRAM flips (existing path)                   ← held all trial
for image i = 0 .. 9999:
    stage input i
    apply cache flips whose start image == i                         ← XOR, verified
    re-apply input-binding flips (existing rule; cache ones included)
    inference → classify image i against its clean record
    remove cache flips whose last image == i                         ← re-XOR (read-only) / skip (engine-written)
trial end:    restore the DRAM flips (existing path); sanity inference must reproduce clean
```

The DUE abort, PROCESS_FATAL restart protocol, gating, snapshot,
residency guard and closing checks are all inherited unchanged from
G5/G7.

## 4. Residency measurement (per model, GPU 0)

**Purpose**: provide, per allocation,

- `f_a`, the **resident fraction**, used for where flips land;
- `S_a(d)`, the **survival curve**: P(line still in L2 after d images),
  used for how long read-only flips last.

**Protocol**: one full **10,000-image** run, the same engine, image list,
order and runner configuration as the campaign, after the normal warm-up
clean pass. Using 10K rather than a 1K shortcut lets survival be measured
up to the full trial length. That is exactly the case that matters for
models smaller than L2, whose weights may never be evicted.

1. Sample cache lines from every allocation. Split them into groups, one
   per delay d ∈ {1, 2, 4, …, 8192}.
2. **First probe** of a group at image boundary b₀: a timing kernel issues
   one load per line, and the latency separates an L2 hit from a GDDR
   fetch (the G3 technique; thresholds calibrated in T0).
   - The hit rate gives **f_a**.
   - After the probe every line of the group is in L2, because the probe
     loads it.
3. **Second probe** of the same group at b₀ + d: a hit means the line
   **survived d images**, which gives **S_a(d)**.
4. Each line is probed exactly twice; different groups cover different
   delays, so the probes do not contaminate each other. Weights are
   re-read by every inference anyway, so the extra probe access barely
   perturbs them.
5. Probes run only at image boundaries, never inside an inference.

**Cost**: ~35–53 s per 10K INT8 pass (G7-T0 measurement) plus negligible
probe time, so about a minute per model. Repeat **2–3 times** per model
to check reproducibility.

**Shared GPU**: GPU 0 is shared, and co-tenant jobs evict our lines, so
the measurement depends on the conditions at the time.

- Co-tenant processes are recorded at measurement time, as the campaigns
  already do.
- If the GPU's load changes substantially, re-measure right before the
  campaign.
- Measurements and campaigns are reported with their co-tenancy
  condition.

**Output**: `residency_<model>.json` (or CSV), holding `f_a`, `S_a(d)` per
allocation, the raw probe latencies, repeat ids, and the co-tenancy
record. It is consumed by the cache sampler; the workload freeze pins its
sha256.

## 5. Logging and verification

Each cache site records:

- fault class `L2-SBU`;
- allocation, line offset, byte, bit;
- start image and duration;
- the measured class (read-only / engine-written);
- VA and PA (PA for reference);
- the before/after values of every apply and removal.

Expected volume is about n_cache sites per trial, the same order as the
DRAM B, so every cache site is logged and verified in full, exactly like
DRAM sites.

Verification:

- every cache apply and removal is checked on device
  (`after == before ^ mask`);
- after a read-only removal the byte must equal its pre-cache value;
- the orchestrator re-verifies every cache row independently, as it does
  for DRAM rows;
- the trial event stream must match the work file, extended with
  per-image apply and remove events.

Reported metrics:

- trial accuracy (DRAM + cache) vs the existing DRAM-only curve;
- the **conditional error rate over images with at least one active cache
  flip**, i.e. how damaging a cache flip is when one occurs. This keeps
  the cache effect visible even when cache flips are rare.

## 6. Phases

| Phase | Work | Output / gate |
| --- | --- | --- |
| **G8-T0** Feasibility (GPU 0) | Calibrate the L2-hit vs GDDR-fetch latency of the probe kernel (thresholds, separation, stability); runner hook for probe/apply/remove kernels at image boundaries; check that a run with hooks but no flips reproduces the clean pass bit-identically. | Gate: clean separation of hit/miss; bit-identical clean pass. |
| **G8-T1** Residency measurement | Per model, a 10K-image paired-probe run, 2–3 repeats, co-tenancy recorded. Start with ResNet-50 v2 (default; the user may pick another first model). | `residency_<model>` files; repeat agreement reported. |
| **G8-T2** Implementation | Cache sampler (§3.2–3.5) in `fault_model.py`; runner per-image apply/remove; orchestrator work-file extension and independent re-verification; self-tests (apply/remove exactness, overlap with DRAM flips, engine-written skip, input re-staging). | Self-tests PASS; a smoke campaign VERIFIED. |
| **G8-T3** Cache rate | The user derives BER_cache from the DRAM BER via prior work; the cache level table is frozen alongside the DRAM levels. | The frozen table, documented as for G5-T1. |
| **G8-T4** Campaigns (GPU 0) | DRAM + L2 per model at the frozen levels, 100 trials × 10K images, compared against the existing DRAM-only runs. | Accuracy curves (DRAM-only vs DRAM + L2) + the conditional cache-hit error rate. |

## 7. Validation additions

- **V6 Probe calibration**: L2 hits and GDDR fetches separate cleanly on
  GPU 0, and the thresholds are stable across repeats.
- **V7 Hook neutrality**: a run with image-boundary hooks but no flips
  reproduces the clean pass bit-identically.
- **V8 Apply/remove exactness**: every cache apply and remove satisfies
  `after == before ^ mask`. After a read-only removal the byte equals its
  pre-cache value, including the DRAM-overlap case.
- **V9 Measurement reproducibility**: repeated residency runs agree under
  the recorded co-tenancy condition.
- The existing V4/V5 (per-site XOR checks, byte-exact trial restore,
  sanity inference) still apply.

## 8. Validity boundary

**Claimed**:

- L2 SBUs on the workload's data, with a per-trial count from BER_cache;
- placement weighted by measured residency;
- lifetimes from measured survival (read-only data) or the engine's own
  overwrite (engine-written data);
- every site traced through the same allocation/VA chain as the DRAM
  faults.

**Not claimed / simplifications**:

- Image-level time resolution: a real flip in the middle of an inference
  affects only later layers, while ours is present for the whole
  inference (a slight overestimate).
- A single representative residency measurement per model under the
  recorded condition. Residency on a shared GPU varies over time.
- No L2 physical coordinates (slice/set/way/bit) and no cache MCU.
- No L2 tag/state-bit faults; no ECC; no L1/shared/register faults.
- BER_cache is a parameter derived from the literature, not measured on
  this hardware.

## 9. Open items

1. BER_cache derivation (§3.3), done by the user from prior work.
2. First model for T1/T4: ResNet-50 v2 by default.
3. Whether to report a "dedicated GPU" condition in addition to the
   shared one, if an idle window on GPU 0 becomes available.

## 10. Design history (why the earlier options were dropped)

| Option (2026-09-29/30 discussion) | Why it was dropped |
| --- | --- |
| XOR "cache" bits once before inference and keep them all trial | Indistinguishable from a DRAM fault: the store goes through L2 and is written back, so the flip persists like DRAM, which contradicts the transient nature of cache. |
| Fresh cache flips at the full BER count for **every** image | The BER count is an accumulated exposure. Applying it per image multiplies cache exposure by ~10,000× per trial and would collapse accuracy for an unphysical reason. Replaced by one per-trial budget spread over time (§3.2, §3.5). |
| Each cache flip lives exactly one image | Ignores that read-only lines can stay resident across many images. Replaced by residency-weighted placement + measured survival (§3.4–3.5). |
| Count n from L2 bits × measured occupancy | The resident amount on a shared GPU is uncertain. Replaced by the fixed `min(R_bits, L2_bits)` rule (§3.2). |
| Residency measured with 1K images | Cannot resolve lifetimes up to the full trial. Replaced by full 10K-image runs (§4). |
| Mid-inference injection via CUDA-graph capture; L1/shared via NVBit; three cards | More complex than needed. Scope set to L2 only, image-boundary resolution, GPU 0 only (user decision 2026-09-30). |
