# G8 Plan — Cache-Aware Fault Injection (GDDR + L2, optionally L1/shared)

Status: **PROPOSAL, 2026-09-29.** This is an analysis and plan only. No code
has been written yet, and every "decision" item in §11 is open for the user.

Goal: extend the G5/G7 campaigns in two ways:

- **DRAM faults.** They keep the existing, validated dual-addressing path:
  GDDR cell → PA → VA → byte/bit → XOR.
- **Cache faults.** Upsets in on-chip SRAM caches (L2 first, L1/shared
  memory optionally) are emulated and combined with the DRAM faults during
  inference.

## TL;DR

1. **A cache fault cannot be produced by XOR-ing before inference.** Every
   CUDA store goes through L2 and, on eviction, reaches DRAM. So a "cache
   XOR" applied before the pass becomes the architectural value of that
   address. It is then indistinguishable from the DRAM faults we already
   inject. Three properties make a cache fault different:
   - **when it exists**: only while the line is resident, and for a clean
     line the fault vanishes at eviction;
   - **who sees it**: L2 is shared by all SMs, while an L1 copy is private
     to one SM;
   - **which data it can hit**: only what is resident at that moment.
2. The core of the plan is therefore **time-resolved injection inside one
   inference**. TensorRT is closed, so the proposed mechanism is **CUDA
   Graph capture of the TensorRT enqueue**. Small "fault-slot" kernels are
   inserted between the engine's kernels, and they apply and heal
   corruptions at kernel granularity. An L2 upset is then
   `XOR at slot k … heal at slot m`, where `[k, m)` is the line's residency
   interval.
3. The residency interval must be **measured, not assumed**. This follows
   the project's G3 discipline: an L2 residency census uses the same timing
   side-channel as G3, probing hit vs miss at chosen kernel boundaries.
4. Cache also changes the **DRAM** story. On this card L2 is **72 MiB**, and
   five of the six G7 models have a total resident surface R below that
   (only ViT-B, 94 MB, exceeds it). A real DRAM upset in a weight line that
   stays clean and resident in L2 is invisible until the line is evicted.
   The current model (faults visible from the start of the trial) may
   therefore **overestimate** DRAM-fault impact for small models on a
   dedicated GPU. This is a hypothesis that the census can settle (§6, T5).
5. **L1/shared memory** faults live inside a single kernel and on a single
   SM, so graph-level insertion cannot reach them. They need binary
   instrumentation (NVBit) at a large slowdown. The recommendation is L2
   first, with L1/shared as an optional, sub-sampled phase.

---

## 1. Platform facts (measured 2026-09-29, GPU 0, driver 580.95.05)

| Item | Value | Source |
| --- | --- | --- |
| L2 cache | 75,497,472 B = **72 MiB**, shared by all 128 SMs | `cudaGetDeviceProperties` |
| Persisting-L2 set-aside max | 51,904,512 B = **49.5 MiB** | same (`persistingL2CacheMaxSize`) |
| Access-policy window max | 134,213,632 B | same |
| Shared memory per SM | 100 KiB configurable (from a 128 KiB L1/shared array, Ada whitepaper) | same + whitepaper |
| Register file per SM | 64 K × 32-bit = 256 KiB (32 MiB chip-wide) | same |
| Global/local L1 caching | supported | device attributes |
| DRAM ECC | **mode exists, currently Disabled** (current and pending) | `nvidia-smi -q -d ECC` |
| SRAM error counters | N/A (no SRAM ECC/parity reporting exposed) | same |

Two consequences:

- **Correction to `docs/G3_SURVEY.md` §4.** That survey says "ECC none on
  GeForce". The RTX 4090 does expose a DRAM ECC mode, and it is disabled on
  all our runs. So the fault model's "no ECC" assumption matches the actual
  configuration, but it should be recorded as a configuration choice rather
  than a hardware limit.
- **Unknown SRAM protection.** Whether L2 or L1 SRAM has parity or ECC
  internally is not documented, and no counters are exposed. G8 assumes
  **unprotected SRAM**, which is the worst case, and must label it as an
  assumption.

Working-set context (frozen R from G7-v2 against the 72 MiB L2): MobileNetV3
9.3 MB, EfficientNet-B0 17.1 MB, DeiT-S 26.0 MB, ResNet-50 28.8 MB, Swin-T
43.8 MB, ViT-B 93.9 MB.

## 2. What "adding cache" means — two separate effects

| Effect | What happens physically | What our current model does | What G8 must add |
| --- | --- | --- | --- |
| **A. Cache as a fault site** | An SRAM cell in an L2/L1 line flips. Readers see the wrong value while the line is resident. A **clean** line then drops the error at eviction (the DRAM copy is correct). A **dirty** line writes the error back to DRAM. | Not modeled (plan §8 excluded it). | Time-bounded value corruption with a lifetime taken from residency. |
| **B. Cache as a filter for DRAM faults** | A GDDR cell flips. If the line is L2-resident and **clean**, readers keep hitting the correct L2 copy until eviction and refetch. If the line is **dirty**, write-back overwrites the DRAM upset (the fault is masked). | Every DRAM fault is visible from the start of the trial, because our XOR goes through L2. | A residency-aware visibility model (§6, T5). This may reduce DRAM-fault impact for models that fit in L2. |

Effect B is not what you asked for, but it is the same mechanism, and
ignoring it would make "DRAM + cache" less realistic than "DRAM only", not
more. The two effects share one measurement, the residency census.

## 3. Why a plain "cache XOR, then inference" is not a cache fault

The proposed order was: DRAM XOR → cache XOR → inference. Here is what the
hardware actually does with the second step.

- A CUDA kernel that does `load; xor; store` on a global address writes the
  value into its L2 line and marks it **dirty**. L2 is write-back and is the
  point of coherence for device memory. On eviction the dirty line is
  written to GDDR, so from then on the corrupted value is simply the
  memory's value.
- Before inference starts, the target line may not even be resident. If it
  is resident, the engine will not necessarily read it before it is evicted.
  Either way, the corruption outlives any real cache residency.
- The only software-visible handles on L2 are these:
  - normal loads and stores;
  - `discard.global.L2` (invalidate a line without write-back; used in G3);
  - load/store cache modifiers;
  - the persisting-L2 access-policy window.

  None of them addresses an L2 *cell*. There is no "write L2 only" store.

So at the architectural level a cache upset can only be emulated as **a
value corruption of the address whose line was hit, bounded in time by that
line's residency**. Emulated this way, the corruption has the right
properties:

- **L2 is shared by all SMs**, so every reader of the address sees the
  corrupted value. That matches the XOR-in-memory emulation exactly.
- **Its lifetime is controlled by us**: apply it at the upset time and heal
  it when the real line would have been evicted. We therefore need no L2
  write access, only (i) a way to act *during* inference and (ii) knowledge
  of residency.

L1 and shared memory are different. An L1 copy is private to one SM, and
shared memory is private to one thread block within one kernel. Corrupting
the global value would make the fault visible to all SMs, which is wrong.
These need instruction-level instrumentation instead (§5.3).

## 4. Proposed L2 fault semantics

For an L2 upset event at time *t* (a kernel boundary *k*) in a line holding
workload data:

| Line state at *t* | Real behavior | Emulation |
| --- | --- | --- |
| Not resident, or resident with other processes' data | No effect on this workload | BENIGN by construction (not sampled; it enters only through occupancy weighting, §7) |
| Resident and **clean** (weights, deserialize constants: never written during inference, as proven in G6 by exact weight restores) | Readers see the fault until eviction, then the DRAM copy is refetched correctly | XOR at slot *k*, **heal** (restore the original byte) at slot *m* = the first boundary after the measured eviction |
| Resident and **dirty** (activations/scratch written earlier in this pass; the input binding after its H2D copy) | The fault persists and is written back until the engine overwrites the data | XOR at slot *k*, **no heal**. The engine's own later writes clear it naturally, which is the same soft-upset behavior as G5 scratch faults |

Notes:

- **Heal must be conditional for anything the engine could write.** If the
  data was overwritten between *k* and *m*, an unconditional restore would
  clobber the engine's new value. Weights are read-only, so their heal is
  unconditional; for any other class the heal is "restore only if the byte
  still equals the faulted value".
- **Granularity is the kernel boundary.** An upset in the middle of a kernel
  is snapped to the next boundary. This is the main fidelity loss of the
  graph approach and must be stated in the paper.
- **Clean/dirty classification** comes from the allocation class (weights =
  clean). For finer granularity it can come from an optional write trace.

## 5. Mechanism — how to act inside a TensorRT inference

### 5.1 Primary: CUDA Graph capture + fault-slot kernels (recommended for L2)

1. Capture one `enqueueV2` inference of the engine into a CUDA graph with
   stream capture. TensorRT 8.6 supports this for static-shape engines, and
   all of ours are static batch-1.
2. Walk the graph and list the kernel nodes in topological order, with
   kernel names that allow layer attribution.
3. After every kernel node, insert one **fault-slot node**. This is a tiny
   generic kernel that reads a device-side schedule table. The table holds
   entries of the form "at slot i: XOR these (address, mask) / heal these
   (address, original, faulted, conditional)".
4. Instantiate once. For each trial or image, only the **device table** is
   rewritten (one `cudaMemcpy`), so no graph surgery or re-instantiation is
   needed per trial.
5. If a TensorRT aux stream makes the graph a DAG rather than a chain,
   slot *k* depends on kernel *k* and precedes kernel *k*'s successors.
   Lifetimes are then defined on the topological order, which is
   deterministic, instead of on wall-clock order across parallel branches.
   This must be reported.

Required equivalence gates (V6, §9):

- (a) graph replay with an **empty** table must give **bit-identical**
  outputs to the existing `enqueueV2` clean pass for all 10K images;
- (b) the allocation registry and the VA→PA map must be unchanged, because
  capture must not allocate;
- (c) the slot overhead must be measured. At a few µs per slot × ~100–300
  kernels it is a noticeable but acceptable fraction of a ~4 ms ResNet-50
  inference;
- (d) the slot kernels' own L2 footprint (one small table) must be shown
  not to change the residency census (§6).

### 5.2 Fallback: NVBit launch callbacks

If a model's engine cannot be captured, NVBit's `cuLaunchKernel` callback
can run the same slot logic before or after each engine kernel, without
instrumenting any instructions and therefore at low overhead. Before using
it, we must verify (i) a release that supports driver 580 / CUDA 12.4 /
sm_89, and (ii) its license terms, following the same study-before-use
discipline as the unlicensed G3 references.

### 5.3 L1 and shared memory: NVBit instruction instrumentation (optional phase)

L1 and shared-memory contents exist only inside one kernel on one SM. The
only way to emulate an upset there without source code is to instrument the
engine kernels' SASS: `LDG` (L1-cached global loads) and `LDS`/`STS`
(shared memory). A load's returned value is corrupted when (SM id, address,
time window) match the event. NVIDIA's NVBitFI is the reference design for
this; the licence must be checked before any reuse, and we would write our
own tool.

- **Cost**: a 10–100× slowdown is typical, so the full 10K-image × 100-trial
  protocol is infeasible. Plan on a fixed image subset (e.g. 500–1000
  images) and fewer trials, with CIs reported accordingly.
- **Semantics**: an L1 upset has a lifetime of at most the kernel. A
  shared-memory upset lives until the tile is overwritten.
- **Register file** (32 MiB chip-wide) and instruction/constant caches stay
  **out of scope**. They are closer to the classic NVBitFI "instruction
  output" fault class than to memory faults. Note that constant-bank
  corruption is likely process-fatal, like the G7 ctx-pool crashes.

### 5.4 Rejected as primary: cycle-level simulation

Accel-Sim/GPGPU-Sim could provide exact residency. However, they need
NVBit traces of closed TensorRT kernels, have no validated Ada/4090 L2
model (hashing, partitioning, replacement policy), and are orders of
magnitude slower. They would contradict the project's measured-not-assumed
rule. At most they are a cross-check.

## 6. Measurement: the L2 residency census (the G3 analogue for caches)

**Question.** For each workload region (allocation × line) and each kernel
boundary *k*, is the line L2-resident, and when does it leave?

**Method (timing, reusing G3 know-how).**

- A probe node is inserted at a single boundary *k*. It times one
  `ld.global` per sampled line: an L2 hit is fast, a miss pays the GDDR
  latency. The thresholds are calibrated the way G3's S1 calibrated DRAM.
- Probing perturbs the cache, because a probe load fetches the line. So
  **each run probes only one boundary**, and the state it measures is
  unperturbed. With ~1–5 ms per inference, sweeping all boundaries × a few
  thousand sampled lines costs minutes, not hours.
- **Across-image residency**: probe at the *start* of image *n+1* to learn
  whether weights survive from image *n*. This is decisive for both clean
  weight lifetimes and effect B.
- **Output**: `residency_census.csv` with (allocation, line offset,
  boundary k, hit/miss, repeat id), plus per-region survival curves
  P(resident at k+Δ | resident at k).

**Validation gates** (like G3 R-c/R-d): the same-card rerun must reproduce;
cross-card agreement on an idle card is expected but must be measured,
never presumed; an empty-slot graph and the plain `enqueueV2` must give the
same census.

**Co-tenancy caveat (important on this shared server).** L2 is shared with
every other process on the GPU. Right now all three cards run other users'
training jobs, which thrash L2. A census taken under co-tenancy measures
*that* environment. So:

- (i) the census and all cache campaigns need an **idle-GPU window**, as the
  G3 timing work did; or
- (ii) co-tenancy is recorded and reported as a separate "shared-GPU"
  condition.

The DRAM-only G5/G7 campaigns did not need this, because they did not
depend on cache state.

**Optional finer tool.** If the census shows complex patterns, an NVBit
memory-address trace of one inference gives the exact per-kernel access
sets. Those can fit a simple residency model, which is then validated
against the census rather than trusted on its own.

## 7. Fault model and rates for cache faults

### 7.1 Sampling an L2 event (single-bit)

An upset lands on a uniformly random L2 bit at a boundary *k*. It affects
the workload only if that bit holds workload data, which happens with
probability `occ(k) = resident_workload_bytes(k) / 72 MiB` from the census.
Conditioned on a hit, the victim byte is uniform over the workload bytes
resident at *k*, and the bit is uniform. So:

- **single-bit L2 events need no L2 set/way/slice mapping**, only the
  resident set;
- the lifetime `[k, m)` comes from the survival curve of the victim's
  region (§4).

### 7.2 Multi-bit cache upsets (MCU)

SRAM MCUs follow the physical array layout. That layout is typically
bit-interleaved, so one strike hits bits of *different* words or lines, and
it is not observable by timing. By the project's own rules (R2/R3), the
cache MCU spatial model can therefore only be an **assumed-tier** model.
Two examples of such a model:

- flips within one 32-B sector of a 128-B line;
- one bit in each of *n* consecutive lines of an L2 set, which would need
  the L2 set mapping via eviction sets, as GeForge used for page anchoring.

Every cache-MCU site must carry an `assumed-sram-layout` label, never a
measured one. Recommendation: start with **SBU only** for caches, and add a
labeled MCU variant later.

### 7.3 Rates — the time axis becomes first-class

This is the main modeling decision, because DRAM and cache faults
accumulate differently:

- **DRAM (no ECC, no scrubbing)**: upsets **accumulate** over mission time.
  Our per-trial BER, e.g. 1e-7, represents accumulated exposure (at the
  ≥ 1e-7 /bit/day rate quoted in the REMU paper's introduction, roughly one
  day).
- **Cache**: upsets exist only while a line is resident. Cache contents
  turn over, so they **do not accumulate** beyond residency.

Illustrative arithmetic, using the same per-bit rate for SRAM and DRAM:
72 MiB = 6.0e8 bits × 1e-7 /bit/day ≈ 60 L2 upsets per day. One ~4 ms
inference is therefore exposed to ~3e-6 events. Under physically consistent
rates, **cache faults matter only through long-lived resident lines**. On a
dedicated GPU these can be weights that stay resident across many
inferences, for minutes or hours, and the census decides whether that
happens.

Therefore two campaign modes are proposed:

- **Stress mode (sensitivity).** Inject a frozen number of L2 events per
  inference, as a ladder like the BER levels, with lifetimes from the
  census. This maps vulnerability vs event count, is directly comparable to
  G7, and makes no claim about real rates.
- **Mission mode (realism).** Use one exposure time *T* and per-bit upset
  rates λ_DRAM and λ_SRAM, whose ratio ρ = λ_SRAM/λ_DRAM comes from the
  user's literature survey, as the SBU/MCU ratios did in G5-T0. DRAM upsets
  accumulate over *T*, subject to the effect-B visibility of §2. L2 upsets
  arrive over *T* but contribute only while their line is resident. The
  result is an expected accuracy for "DRAM + cache" at a stated mission
  time.

### 7.4 Revised trial order

The proposed order (DRAM XOR → cache XOR → inference) becomes:

1. **Trial start**: apply the sampled DRAM faults (unchanged G5/G7 path,
   held for the trial). In mission mode, first apply the effect-B
   visibility filter: faults in lines that stay resident and clean are
   deferred until those lines' eviction boundary.
2. **Per image**: rewrite the fault-slot table with that image's L2 events.
   Each event is an XOR at slot *k*, plus a heal at slot *m* for clean
   lines.
3. **Replay the graph** for the image. Faults appear and disappear at their
   boundaries during the inference.
4. **Classify** each image against its clean record, exactly as now.
5. **Trial end**: restore DRAM faults. The heal entries already restored the
   L2 faults. The same byte-exact restore proofs and sanity inference apply.

## 8. Phases and deliverables

| Phase | Work | Output / gate |
| --- | --- | --- |
| **G8-T0** Feasibility (no campaigns) | (1) Capture each of the six engines into a CUDA graph: node count, kernel names, DAG shape, aux streams. (2) Empty-slot graph replay vs `enqueueV2` on the 10K clean pass. (3) L2 timing calibration (hit vs miss thresholds, stability). (4) Verify L2 write-back with a G3-style micro-test: store a changed value, `discard.global.L2`, re-read, and expect the old value. (5) NVBit version and licence check. (6) Record the ECC-mode fact. | Go/no-go per engine. **Gate V6**: bit-identical clean pass. |
| **G8-T1** Injection infrastructure | Runner graph mode, fault-slot kernel, device schedule table, conditional heal, per-slot verification in test mode (read back the target at slots *k−1*, *k* and *m*). Orchestrator: sample → table → verify, reusing the gate/snapshot/ledger skeleton. | **Gate V7**: in test mode the target value is faulted exactly on `[k, m)` and pristine outside it. **Gate V8**: a heal never overwrites engine-written data. Self-tests. |
| **G8-T2** L2 residency census | Per engine, on an idle card: per-boundary × sampled lines, plus across-image persistence. | `residency_census.csv` + survival curves. **Gate V9**: same-card reproduction and a cross-card check. |
| **G8-T3** Cache fault model | User decisions (§11): scope, ρ and lifetime policy, and whether the cache MCU variant is labeled assumed. Frozen like G5-T1. | `docs/G8_CACHE_FAULT_MODEL.md` (frozen). |
| **G8-T4** Campaigns | Stress-mode L2 ladder per model (ResNet-50 v2 first, as in G7), then DRAM + L2 combined at matched BER levels. | Curves: DRAM-only (existing) vs L2-only vs combined. Attribution by region and lifetime. |
| **G8-T5** DRAM visibility (effect B) | Apply the census to decide which DRAM faults would be masked or delayed by L2 residency. Re-weight or re-run the G7 levels in mission mode. | A statement on how much of the G5/G7 DRAM impact survives cache filtering, per model (small models vs ViT-B). |
| **G8-T6** (optional) L1/shared | NVBit instruction-level injector, image subset. | L1/shared vulnerability relative to L2, with CIs matching the reduced sample. |

Suggested order: T0 → T1 → T2 → T3 (user) → T4 → T5, with T6 only if L2
results justify it.

## 9. Validation additions (extending plan §9 V1–V5)

- **V6 Graph equivalence**: the empty-table graph replay gives
  bit-identical outputs to `enqueueV2` (all images), with an unchanged
  registry and map.
- **V7 Lifetime realization**: in instrumented test runs the target byte
  equals `orig ^ mask` exactly at boundaries `[k, m)` and `orig`
  elsewhere.
- **V8 Heal safety**: a conditional heal never overwrites a value the engine
  wrote. This is tested with synthetic scratch targets written between *k*
  and *m*.
- **V9 Census reproducibility**: a same-card rerun agrees (like R-c), a
  cross-card run agrees on idle cards (like R-d), and slot insertion does
  not change the census.
- The **existing V4/V5** stay: per-site `after == before ^ mask`, the
  byte-exact restore, and a sanity inference equal to clean after every
  trial.

## 10. Risks

| # | Risk | Mitigation |
| --- | --- | --- |
| C1 | TensorRT graph capture fails or differs for some engines (Myelin, aux streams) | T0 gate per engine. NVBit launch callbacks as the fallback (§5.2). |
| C2 | L2 microarchitecture is undocumented (slice hashing, partition duplication, replacement policy, whether dirty lines are written back early) | Measure residency (census) instead of modeling it. Micro-test the write-back behavior in T0. |
| C3 | Co-tenant processes pollute L2, so the census is not reproducible | Idle-GPU windows for census and cache campaigns; otherwise report shared-GPU as a separate condition. |
| C4 | Kernel-boundary granularity misses intra-kernel timing | State it as a fidelity boundary. L1/shared (T6) covers the intra-kernel regime for its own structures. |
| C5 | SRAM layout and protection unknown | SBU first. The MCU variant is labeled assumed. "Unprotected SRAM" is a stated worst-case assumption. |
| C6 | No reliable ρ (SRAM vs GDDR6X per-bit rate) for this process node | User literature survey (as in G5-T0) + a sensitivity band over ρ. The stress mode needs no ρ. |
| C7 | NVBit overhead or compatibility | Optional phase only, on an image subset. Pin the version and check the licence first. |
| C8 | Slot kernels perturb the cache | Tiny footprint; V9 checks the census with and without slots. |

## 11. Decisions needed from the user

1. **Scope**: L2 only for the first round (recommended), or L2 + L1/shared
   (NVBit, much slower)?
2. **Rate model**: stress mode only, mission mode only, or both
   (recommended)? If mission mode: who supplies ρ and the SRAM SBU/MCU
   ratios (literature survey, as in G5-T0)?
3. **Clean-line lifetime policy**: measured survival across images
   (recommended), or capped at the end of the current image (conservative,
   simpler)?
4. **Environment**: are dedicated idle-GPU windows available for the census
   and campaigns, or should shared-GPU runs be accepted as a reported
   condition?
5. **Effect B (T5)**: include the DRAM-visibility correction? It may revise
   how the G5/G7 DRAM results are interpreted for models that fit in L2.
6. **First model**: ResNet-50 v2 (recommended, as in G7), or a model whose
   surface exceeds L2 (ViT-B) to contrast the two regimes early?

## 12. Validity boundary (what G8 will and will not claim)

**Will claim:**

- L2 faults as time-bounded, SM-shared value corruptions at kernel
  granularity;
- lifetimes from measured residency;
- victims sampled from the measured resident set;
- every site traced through the same allocation/VA/PA chain as G5/G7.

**Will not claim:**

- cell-level SRAM physical locations or measured SRAM MCU geometry;
- intra-kernel timing for L2;
- register-file or instruction/constant-cache faults;
- ECC behavior;
- cycle-accurate propagation.

The paper statement in plan §8 changes as follows. It currently reads
"cache hierarchy outside the fault model". It becomes: "L2 upsets are
modeled at kernel-boundary granularity with measured residency; L1/shared
memory [optionally] by instruction-level instrumentation; the register file
remains out of scope."
