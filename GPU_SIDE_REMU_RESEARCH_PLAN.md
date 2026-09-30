# GPU-Side Memory-Aware Fault Injection Research Plan
## — Porting the REMU CPU-side method to GPU GDDR6X

> Target platform: NVIDIA RTX 4090 (GDDR6X)
> Target task: DNN inference (first phase focuses on controllable
> Tensor/Weight/Activation data)
> Core goal: build the bidirectional mapping `DNN Tensor Bit ↔ GPU VA ↔
> GPU PA ↔ GDDR Cell`, and perform controllable SEU/MCU fault injection in
> GPU device memory.
>
> Language note (2026-09-29): this document was originally written in
> Chinese and was translated to English on 2026-09-29 without changing
> its content. On the same date the per-phase status entries were
> regrouped under their own phase headings (the G5 entries had been
> filed under G4), and outcome notes were added to §10 and §11.

**Overall status (2026-09-29): G1–G6 closed; G7-v2 six-model ImageNet-1K
campaign complete.** See the status entries in §6.

---

## 1. Background and core problem

REMU's core idea is not simply "flip some DNN bit". It first establishes
the mapping between application data and real DRAM physical locations,
then chooses the physical storage cells that suffer SEU/MCU according to a
spatially correlated DRAM fault model, and finally maps them back to the
application data and performs the bit flip.

REMU's basic mapping chain is:

```text
Engine Byte
   ↕
CPU VA
   ↕
CPU PA
   ↕
DRAM Channel / Bank / Row / Column / DQ
```

GPU-side research must turn it into:

```text
Tensor / Element / Bit
        ↕
      GPU VA
        ↕
      GPU PA
        ↕
GDDR6X Channel / Bank / Row / Column / DQ
```

The final goal is:

```text
GDDR Cell
   ↕
GPU PA
   ↕
GPU VA
   ↕
Tensor / Element / Bit
```

so that when a fault occurs at a selected GDDR physical location, the
corresponding DNN data bit can be located precisely and the matching bit
flip performed in GPU device memory.

---

## 2. Scope

### 2.1 First-phase scope

The current phase studies only:

- GPU GDDR6X main memory;
- DNN data actually resident in GPU device memory;
- the mapping between DNN Tensor / Weight / Activation and GDDR physical
  locations;
- SEU;
- MCU, including row / column / DQ spatially correlated errors;
- GPU device-memory software bit flips;
- inference outcome classification: BENIGN / SDC / DUE.

Out of scope for now:

- GPU L1 / L2 cache faults;
- register file faults;
- shared memory faults;
- Tensor Core / SM pipeline faults;
- instruction faults;
- cycle-accurate radiation propagation;
- dynamic propagation of physical GDDR upsets through cache residency,
  eviction, and writeback.

The first phase is therefore explicitly positioned as:

> **GDDR-location-aware GPU device-memory fault injection**

and not as:

> **cycle-accurate physical GDDR-cell radiation propagation simulation**

---

## 3. What can be inherited directly from REMU

### 3.1 The dual-addressing idea

Keep:

```text
Forward:
DNN Data → VA → PA → Memory Cell

Reverse:
Memory Cell → PA → VA → DNN Data
```

but change the address space from CPU/LPDDR to GPU/GDDR6X.

### 3.2 Fault model

Can continue to use:

- BER;
- SEU;
- MCU multiplicity;
- row-adjacent MCU;
- column-adjacent MCU;
- DQ-adjacent MCU;
- fault count;
- fault probability;
- fault spatial correlation.

Probability parameters are later reconfigured from GDDR6X experimental
data or published radiation characterization literature.

### 3.3 Bitmap tree / physical adjacency search

REMU's bitmap-tree idea still applies:

```text
valid GPU physical addresses
        ↓
convert to GDDR coordinates
        ↓
build the set of valid GDDR cells
        ↓
search for SEU / MCU inside the valid region
```

The purpose is still to avoid picking random points over the whole VRAM
address space and mostly hitting invalid regions.

### 3.4 Fault statistics

Keep:

- injection ID;
- target tensor;
- physical location;
- bit index;
- BER;
- error pattern;
- inference result;
- BENIGN;
- SDC;
- DUE;
- retry;
- reproducibility seed.

---

## 4. What must be redone

## 4.1 Module A: DNN Semantic Mapper

### Goal

Establish:

```text
Tensor / Element / Bit
        ↕
      GPU VA
```

### What to record

For each target tensor, record at least:

```text
tensor_name
tensor_type
shape
dtype
layout
allocation_base_gpu_va
allocation_size
element_offset
bit_offset
lifetime
```

### First-phase advice

Do not try to cover every opaque TensorRT internal allocation at the
start.

Start from one fully controllable CUDA allocation, e.g.:

```text
the weight tensor of some layer
or
the activation buffer of some layer
```

It must be possible to do:

```text
layer3.conv1.weight
element = 12345
bit = 5

↕

GPU VA = 0xXXXXXXXX
```

### Acceptance criteria

- Given a tensor element / bit, the GPU VA can be computed;
- given a GPU VA, the tensor / element / bit can be looked up in reverse;
- the results can be verified with hand-built patterns.

---

## 4.2 Module B: GPU VA → GPU PA

### Goal

Establish:

```text
GPU Virtual Address
        ↕
GPU Physical / VRAM Address
```

This is the first core difficulty of a GPU REMU.

### REMU's original method

CPU:

```text
VA
 ↓
/proc/<pid>/pagemap
 ↓
PA
```

### The GPU-side problem

On the RTX 4090:

```text
cudaMalloc()
 ↓
GPU VA
 ↓
GPU MMU
 ↓
GPU PA
```

Ordinary CUDA APIs do not expose the GPU physical address directly.

GPU VA→PA acquisition therefore has to be researched separately.

### Requirements

In priority order:

1. prefer methods that do not modify the NVIDIA kernel module;
2. prefer existing verifiable tools / driver interfaces / GPU memory
   introspection;
3. if only driver instrumentation works, it must be isolated from the main
   experiment code;
4. never mistake a CPU `/proc/pid/pagemap` result for a GPU VRAM PA;
5. re-acquire the mapping after every allocation; never reuse a Tensor→PA
   mapping across runs.

### Acceptance criteria

Within one allocation:

```text
GPU VA A ↔ GPU PA X
```

must be stably and repeatably queryable.

After re-allocation:

```text
GPU VA / GPU PA
```

may change, but the mapping snapshot for this run must be rebuilt.

---

## 4.3 Module C: GPU PA → GDDR6X Coordinate

### Goal

Recover:

```text
GPU PA
 ↓
RTX 4090 Memory Controller Mapping
 ↓
Channel / Bank Group / Bank / Row / Column / DQ
```

i.e.:

\[
GDDRCoord = f_{4090MC}(GPU\ PA)
\]

This is the second core difficulty of a GPU REMU, and the most important
physical mapping problem of the whole project.

### Principle

REMU's:

```text
LPDDR4 default Ramulator mapping
```

must not be taken as the real RTX 4090 mapping.

One must distinguish:

```text
simulated DRAM mapping
```

from:

```text
real RTX4090 GDDR6X mapping
```

### Research content

Recover or verify:

- channel selection;
- bank-group selection;
- bank selection;
- row bits;
- column bits;
- burst offset;
- interleaving;
- XOR / hash function;
- DQ / bit-lane organization (if observable).

### Acceptance criteria

At minimum, verify:

- same-bank relationship;
- different-bank relationship;
- row-conflict relationship;
- row/column adjacency consistency;
- stability across repeated experiments;
- whether the mapping is identical across cards of the same model —
  validated separately as cross-GPU validation, never assumed transferable.

---

## 4.4 Module D: GDDR Reverse Index

Once Module B and Module C are complete, build for each workload:

```text
Tensor Bit
   ↕
GPU VA
   ↕
GPU PA
   ↕
GDDR Coordinate
```

Two extra indexes are recommended:

```text
GPU_PA → GDDR_COORD
GDDR_COORD → GPU_PA
```

and:

```text
GPU_VA → TensorBit
TensorBit → GPU_VA
```

giving in the end:

```text
GDDR_COORD → GPU_PA → GPU_VA → TensorBit
```

### Important principle

What is stored here is:

> **the mapping snapshot of the current allocation / current run**

not a permanent Tensor→GDDR mapping.

What may genuinely stay stable across runs is:

```text
GPU PA → GDDR Coordinate
```

i.e. the hardware memory-controller mapping rule.

---

## 4.5 Module E: GPU-side Fault Injector

### REMU's original method

CPU:

```cpp
*byteAddress ^= mask;
```

### GPU version

Changed to a GPU device-memory modification:

```text
target GPU VA
     ↓
CUDA fault kernel
     ↓
load
     ↓
XOR mask
     ↓
store
```

Recommended first version:

```text
finish Tensor allocation / initialization normally
        ↓
build the mapping
        ↓
select faults
        ↓
GPU kernel XOR
        ↓
cudaDeviceSynchronize()
        ↓
DNN inference
```

### Acceptance criteria

- dump the target value before injection;
- dump the target value after injection;
- only the specified bit may change;
- compare against fault-free inference;
- repeated runs are reproducible.

---

## 5. Full GPU-side REMU workflow

```text
[1] Load DNN / Prepare Tensor
        ↓
[2] GPU allocation complete
        ↓
[3] Record Tensor ↔ GPU VA
        ↓
[4] Acquire GPU VA ↔ GPU PA
        ↓
[5] GPU PA → GDDR6X coordinate
        ↓
[6] Build this run's TensorBit ↔ GDDRCell mapping
        ↓
[7] Build bitmap tree / reverse index
        ↓
[8] Select GDDR cells by the BER / SEU / MCU model
        ↓
[9] GDDR cell → GPU PA → GPU VA → Tensor bit
        ↓
[10] GPU-side XOR bit flip
        ↓
[11] cudaDeviceSynchronize()
        ↓
[12] Run DNN inference
        ↓
[13] BENIGN / SDC / DUE classification
        ↓
[14] Save mapping + fault + result log
```

---

## 6. Recommended phased implementation

### G1: Semantic Mapping

Status (2026-09-14): **PASS** on all three GPUs (G1 and the G1.5
ResNet-50 INT8 TensorRT integration); see `docs/G1_VALIDATION.md` and
`docs/G1_5_VALIDATION.md`.

Goal:

```text
Tensor Bit ↔ GPU VA
```

Tasks:

- build tensor metadata;
- obtain the GPU pointer;
- implement element offset;
- implement bit offset;
- reverse lookup.

Acceptance:

```text
Tensor element / bit
↔
GPU VA
```

fully verifiable.

---

### G2: GPU Virtual-to-Physical Mapping

Status (2026-09-15): **PASS** (all three cards pass all three validation
layers: scratch device/VMM, VMM alias double-mapping, and the full
TensorRT workload; `gpu_va_pa_map.csv` generated). See
`docs/G2_VALIDATION.md`.

Goal:

```text
GPU VA ↔ GPU PA
```

Tasks:

- survey the NVIDIA GPU MMU;
- find usable drivers/APIs/tools;
- validate RTX 4090 device allocations;
- validate mapping temporal stability;
- validate whether the mapping changes after re-allocation.

Output:

```text
gpu_va_pa_map.csv
```

---

### G3: GPU PA-to-GDDR Mapping

Status (2026-09-15): S0 due diligence complete — the reference tools
(GPUHammer/GDDRHammer, study only, no copying) and the verified
AD102/GDDR6X platform facts are in `docs/G3_SURVEY.md`.

Status (2026-09-16): S1 calibration **PASS** (GPU0) — the GDDR6X
row-conflict latency delta is ≈39 ns (98 cyc @2520 MHz), the conflict
cluster is tight (4.4 cyc), and the in-page offsets that conflict with
offset 0 match the published GA102/A6000 results; tool `tools/g3_probe/`.
Mapping-rule solving not yet started.

Status (2026-09-16): S2 **PASS** (GPU0) — PA-annotated timing pool: 64×8
MiB gated allocations; all 256 pages valid/VIDEO and PA-contiguous as one
512 MiB block (0x1ee00000..0x3ec00000). PAs are contiguous within a chunk
and step by 8 MiB per chunk in allocation order, but VA order is shuffled
(observed PTEs cannot be replaced by VA arithmetic). Timing
floor/baseline/conflict = 1023/1014/1140 cyc (≈47 ns); cross-page pairs at
1038–1098 cyc sit in an intermediate state. Tools `tools/g3_probe/` +
`tools/g2_observer/run_g3_pool_probe.py` (`--api g3pool`). S3 systematic
collection pending.

Status (2026-09-16): S3 toolchain ready — `--work-mode bit-scan`
(calibration triple + one single-bit differential pair group per PA bit;
4 base pages voting in-page; page level covered by the pool up to bit 31)
and `analyze_bit_scan.py` (low/mid/conflict three-state classification +
per-bit vote table + constraints.csv). Self-test passes, and a dry run on
the real S2 pool map validated it (512 MiB pool, 599 queries, bits 0..28
fully covered).

Status (2026-09-16): S3 collection complete (GPU0, 4 GiB pool, 791
queries, 788 usable constraints, 0 asymmetric, 0 integrity errors,
amplitude 124 cyc ≈ 46 ns). Findings: in-page bits 0–9 are always low
(column/burst-region candidates); bits 10–20 SPLIT across base pages, with
hard conflict votes on bits 10/11/16/19/20; page-level bits 21–31 are each
a three-state mix (conflict share 13–28%, far above the ~3% of a uniform
32-bank map) → a linear bit-slice mapping is ruled out, bank selection is
a non-linear (row-involved) hash, and the hash support spans the whole
measurable PA range. Baselines differ by up to ~30 cyc between pool
regions (suspected L2-slice proximity effect); the classification margin
is sufficient.
S3b toolchain ready — `--work-mode pair-scan` (anchored single-bit and
two-bit joint probes around the S1 conflict anchor 0xd0100 to separate
column bits from bank bits; includes an anchor-validity sweep and page-level
triple probes; dry run on the real 4 GiB pool map: 1587 queries) and
`analyze_pair_scan.py` (a linear toy decoder self-test validates every
interpretation path: bank_kept / bank_changed / two-bit cancellation /
anchor sweep).

Status (2026-09-16): S3b collection complete (GPU0, 4 GiB pool, 1587
queries, 0 asymmetric, 0 lost events). Whether anchor 0xd0100 keeps the
bank varies with position: only 1 of the 4 in-page base pages is valid
(the S1 page family 1144/1144); the page sweep finds 35/128 valid (32 mid /
61 low) → a fixed-support linear hash is directly ruled out by
measurement. In-page bits split on the valid base page: column bits
{0–7, 9, 11}, hash bits {8, 12–14, 16–18, 20} (8/16/17/18 drop deepest to
1022–1023), intermediate {10, 15, 19}. Two-bit joint probes: 840 pairs,
66 conflicts, shaped as "column bit × row carrier" with the carrier
rotating by base page (10@b2 / 11@b0 / 16@b3 / 19@b2); base page 0 has
pairwise-cancellation clusters → locally low-degree, seeded coefficients.
Page-level bit verdicts: row bits 25/27/28/29, bank bits 22/26,
column-like fold 21, mixed 23/30/31; the page sweep covers up to PA
bit 32. S4 solver inputs ready (constraints.csv 788 +
pair_constraints.csv 1587, mid as soft constraints).

Status (2026-09-16): S4 solver ready — `solve_mapping.py` (conflict →
same-bank homogeneous GF(2) equations; low ∧ xor touching row bits →
separation constraints; bank as a family of seeded polynomials of degree
≤ 2, enumerated exactly by conflict signature up to weight 3 + beam XOR
growth; row support re-solved with monotone clauses + unit propagation
with violation counting; two-phase alternation; mid used only for
scoring). Holdout ≥ 95% gate (a rehearsal of S5); synthetic seeded
ground-truth self-test: degree 2 gives zero residual, zero violations,
holdout ≥ 99% (the true functionals are recovered exactly at weight 2–3),
and degree 1 must report insufficiency. `--predict` classifies any PA
pair with the saved model.

Status (2026-09-16): S4 solve on the collected data complete — **model
class insufficient, gate FAIL (honest negative result)**. On the real
S3+S3b constraints (2375 pairs): degree 1 degenerates to the low base rate
0.782; degree 2 + linear row support 0.774; degree 2 + two seeded families
(bank and row both families of degree ≤ 2 GF(2) functionals, solved
alternately, with anchored completion search up to weight 4 on stalls and
a parsimony gate rejecting narrow coverage) 0.758 — all far below 95%.
Diagnostic chain: (1) 619/1424 (43%) lows fall inside the span of the
conflict features (216 of them exactly the anchor-mask bits {8,16,18,19}),
so the linear "xor touches a fixed row support" model is ruled out
directly by linear algebra — the row fold must be seeded; (2) labels are
reproducible (fresh S3b votes agree with S3 5/5, 6/6), and low latency
shows no clean drift with PA distance — it is not noise; (3) e16/e18/e19
each light up ~130 anchored conflicts but are blocked by ~190 anchor-invalid
lows, and row bits 25/27/28/29 each have ~17 conflicts blocked by 23–38
lows — the seeded structure separating them needs weight > 4 or degree
≥ 3 terms; (4) 556 mids (23%) are excluded; the shoulder band swallows
exactly the informative pairs near the threshold. The solver itself
recovers synthetic ground truth exactly (including weight-4 linear-kernel
bank bits and pure quadratic row-fold terms). S4b direction: per-region
local calibration triples (narrow the mid band) + targeted probes of
blocking pairs on anchor-invalid pages (decide "row fold degree ≥ 3" vs
"bank coverage gap"); S5 predictive validation is blocked on a model that
passes the gate.

Status (2026-09-16): S4b-0 complete (pure offline analysis, zero new
collection) — new `analyze_local_recal.py`: (1) the page-level latency
fingerprint λ is real (1011–1063, ~50 cyc ≈ 19 ns; in-page re-measurement
median differences 7–9 ≪ the global spread; no single-PA-bit effect →
hash-like positional behavior, consistent with channel/L2-slice path
differences); a pair value carries only one scalar, so ~12 discrete bands
cannot be resolved offline; (2) local recalibration is an asymmetric rule
(the low/mid boundary localizes to the page λ, the conflict gate stays
global — the symmetric version would demote 134/139 S3 conflicts to mid,
and measured conflict−amp falls at 981–1001, below the λ domain
1011–1063, i.e. the conflict penalty does not ride on the low-path
baseline): mid 556→257 (−54%), all 395 conflicts kept (134+139 of them
below their own local conflict threshold, flagged as S4b-1 targeted
re-probe candidates), 11 marginal lows corrected to mid; (3) the remaining
257 mids are saved as same-channel candidates
(same_channel_candidates.csv); the same-channel graph (conflict + residual
mid edges) has only 147 small components (largest 22 pages) — not
inconsistent with ~12 channels, but the edge density cannot merge ~170
pages per channel; the channel partition needs the S4b-1 per-page census;
(4) the solver on the recalibrated labels goes holdout 0.758→0.809 — a
real gain but still below the 0.95 gate: label quality is one bottleneck,
and the degree-2 / weight-4 model class being insufficient remains the
main one (consistent with the S4 diagnosis). S4b-1 (channel census +
targeted re-probe of blocking pairs) waits for an idle GPU0 window.

Status (2026-09-16): S4b-1 complete (GPU0, `--work-mode census`, 3164
queries, 0 lost events; the 4 GiB PA hole reproduced a third time; all
273 re-probe pairs + 4 pilot pages resolved, 0 dropped) — **the old
conflict class is proven bimodal, one concrete reason the S4 solve
failed**: (1) per-page single-access λ census (2048 pages, clean self
pairs): λ is real and spatial (flat within the self segment, no time
trend), but there are **no ~12 discrete bands** (gap-4 gives only 2
clusters, 83%/17%), both clusters are a fine-grained per-page layout (249
of 512 chunks mix fast and slow pages, no PA-bit effect > 0.02) → λ is not
a channel observable, and the λ-band channel hypothesis is falsified by
measurement (the S4b-0 pair-λ vs census-λ correlation is only r = 0.11;
the earlier estimate was a noise upper bound); (2) late in-run step: all
segments after the self segment are uniformly ~18 cyc lower (measured with
repeat blocks; the self segment is flat → a step, not a drift); the
analyzer applies an additive correction to self_second/reprobe/pilot;
(3) re-probing the 273 suspect conflicts: 228 fall into a **reproducible
shallow band** (corrected p10–p90 = 1104–1120 ≈ 0.70–0.85 of amplitude),
36 drop to low, 9 to mid, **0 deep conflicts**; back-applying a 0.90 gate
to old labels: **all** 139 S3 conflicts are shallow; S3b has 185 shallow +
71 deep (≥ 1139) — the two-band classifier merged a "partial-penalty
regime" with full row conflicts, and the S4 same-bank GF(2) constraints
were exactly this mixture; side finding: the ≥ 1200 deep tail (22 pairs)
clusters on pages with page number ≡ 7 mod 16 (a 32 MiB-periodic
super-conflict, material for S4b-2); (4) row equivalence-class pilot (only
deep conflicts merge banks, lows inside a bank merge rows; 0
contradictions): the transitive classes of the 2 deep anchor pages are
exactly {0, 0x200}/row and {M, M|0x200}/row — bit 9 is confirmed at the
hardware level as a column bit, the anchor flips the row, and non-anchor
hash bits leave the bank group; the anchor pairs of the 2 shallow pages
pay only the shoulder → S3b's "anchor validity 27%" is really the
deep/shallow split, not a property of a linear hash; (5) updated physical
model (S4b-2 solver input): low = different channel or same row;
shoulder ~1114 = same channel, different bank/bank group; deep ≥ 1139 =
same bank, different row. The shoulder gives same-channel candidates a
large positive set (228 re-probed + 185 S3b shallow + residual mids): the
channel partition is now a graph problem on shoulder edges, no longer a
λ problem.

Status (2026-09-16): S4b-2 complete (pure offline analysis, zero new
collection) — new `analyze_three_band.py` (three-band relabel of S3/S3b
using census λ; the PA hole is consistent so the page tables join 1:1,
missing pages refuse) and `analyze_channel_graph.py` (same-channel page
graph from cross-page shoulder + deep edges). (1) The retrospective split
agrees exactly with the census re-probe estimate: all 139 S3 conflicts are
shallow, S3b 185 shallow + 70 deep; census λ nearly erases the old mid band
(S3 241→5, S3b 315→124) — old mids were mostly λ smear and now correctly
count as low (1851 low of 2375 pairs); (2) the channel-partition signal
exists but is underdetermined: 380 cross-page same-channel edges (379
shoulder + 1 deep, including census re-probe shoulders) touch only 250
pages → 82 components (largest 13 pages, 5%); edge consistency is good
(cross-page lows landing inside a component: only 28/845 = 3.3%
contradiction rate, and those are single bad edges, not rule failures),
but far from merging ~170/2048 pages per channel; the two largest
components carry strong high-bit signatures (bits 27/28 100%, bit 31 92%,
bits 24–26 mutually exclusive) with mixed census λ inside a component →
the partition is address-structural, not λ-structural, closing the S4b-1
λ conclusion; (3) clean architectural fact: 69 of the 70 deep conflicts are
in-page (the bank hash consumes PA bits < 21), and page-level bit flips
produce only shoulder/low → channel selection is in the high bits, the
bank hash is in-page; (4) the solver still cannot fit the deep set on
three-band labels: all 18 degree-2 bank functionals vanish on the training
conflicts, but 47 of ~57 training conflicts have no explaining row
functional, holdout conflict recall 0/14; numerically 0.9557 would "pass"
the 95% gate, but that is just the all-low base rate (conflicts are only
3.6% of hard pairs) — `solve_mapping.py` now prints per-class recall and
flags DEGENERATE PASS; S5 remains blocked. Conclusion: with clean labels
the S4 verdict holds — the degree-2 / weight-4 model class cannot express
the seeded row fold, and 70 structurally similar deep conflicts cannot
open it up. Next: an in-pool empirical (bank, row) class table as the
fallback deliverable for G4/G5 + channel/bank targeted dense sampling
(S4b-3).

Status (2026-09-17): **Pivot — abandon closed-form hash reverse
engineering; switch to a GeForge-style empirical mapping table (EMT)**.
Rationale: (1) our S4/S4b-2 negative results and GeForge (S&P'26; the repo
has no license, used only as a methodology reference, see
`docs/G3_SURVEY.md` §1) footnote 1 corroborate each other — the PA→bank
function is "highly non-linear and mixes (nearly) all address bits"; they
did not reverse a closed form either, but built tables offline per model
(one file per bank, row→PA-chunk ownership table) + a row-stripe monotonic
row-ID model (App. B assumption, validated indirectly) + reuse across
cards of the same model; (2) our structural advantage: the G2 observer
gives the true PA of every pool page each run, so GeForge's page anchoring
problem (L2 fingerprint alignment, designed for an unprivileged attacker)
does not exist for us. The S4/S4b-2 solver line is sealed as an honest
negative result. New phase S5 = EMT: S5-T0 seed table (pure offline) →
S5-T1 anchor-expansion collection (`--work-mode table-build`) → S5-T2
validation gates (transitive consistency / class cardinality /
reproduction / cross-card G3-P6 / end-to-end prediction — the S5
predictive validation previously blocked by the gate is revived in table
form) → S5-T3 query API + G4 integration. Honest boundaries: row-ID
linearity is an inherited assumption (timing carries no row-distance
information; same standing as the literature); table coverage = the pool
PA range (G5 injection happens only inside the pool, so the loop closes);
a driver upgrade can shift PAs → table-build throughput must support
on-demand rebuilds; DQ stays unsupported (R3 unchanged).
S5-T0 complete (zero new collection) — new
`tools/g3_probe/build_bank_table.py`: merges edges from three sources,
deduplicated by priority reprobe > pilot > S3b > S3 (2455 edges: 54 deep /
320 shoulder / 1932 low; mid is never class evidence); bank class =
union-find over deep conflicts, row class = union-find over lows inside a
bank class; the channel component graph is recomputed and cross-checked
against the S4b-2 products; consistency gates C1 shoulder inside a bank
class / C2 deep inside a row class / C3 cross-page low inside a channel
component / C4 cross-page low inside a bank class / C5 cross-source label
disagreement. Two **corrections** to the S4b-2 record: (a) the "70" deep
conflicts are really 71 rows / 47 distinct PA pairs (S3b emits the same
pair under several probe types) + 3 pilot confirmations − 1 demoted by
reprobe = 54; (b) the channel components should be **74 / 220 pages**,
not 82 / 250 — the S4b-2 graph only appended reprobe shoulder edges and
did not demote band shoulder labels with the re-probe results; the 29
cross-page shoulder edges overturned by re-probing are removed here, and
the C3 contradiction rate drops to 20/807 (2.5%; the correction makes the
partition cleaner). Result (`artifacts/g3/table_v0/`:
gddr_seed_table.csv for 2048 pages + bank_classes.csv +
table_report.txt): 40 bank classes with ≥ 2 nodes (largest 12, pilot pages
as the backbone; 38 binary pairs), 4 row pairs, **0 bank classes spanning
pages** (the only cross-page deep edge in S4b-2 was the demoted one),
C1/C2/C4 zero contradictions; coverage honestly sparse — channel
components 220/2048 pages, classified nodes 38/2048 pages, singletons
2022/2114. This is the S5-T1 workload baseline.

Status (2026-09-17): **S5-T1 complete — whole-pool collection in one run,
and the measurements corrected the physical model**. New
`--work-mode table-build` six-segment workload (calibration 3 / self 2048
/ anchor_sweep 49152 = 24 masks × 2048 pages / classify 151478 = 2048
pages × 74 seed representatives / bank_map 392 = 8 pages × 25-offset grid
double probes / repeat 64; 203137 queries total, work segment 19 s ≈
10.8k q/s, 0 failures; anchor masks and representative pages taken from
`--seed-table artifacts/g3/table_v0`) +
`tools/g3_probe/analyze_table_build.py` (per-row PA check, fail-closed;
produces table_build_edges.csv / anchor_validity.csv / bank_map_pages.csv
/ channel_partition.csv) + `build_bank_table.py --t1` (T1 edges as the
highest-priority source) and `--channel-deep-only` (page graph unions
only cross-page deep edges).

**Model correction (measured in this run; replaces the S4b-1 cross-page
three-band model)**: each pair value is referenced to max(λ_a, λ_b), not a
global baseline. The inter-page λ spread (~120 cycles) exceeds the conflict
amplitude (110), so any global gate falls inside the λ spread and fakes
slow pages as "shoulder". With the λ reference the classify segment is
bimodal with an empty +30..+80 valley: low at d ≈ 0 (96%), deep at
d ≥ +80 (0.26% ≈ 1/384, matching the AD102 prior of 24 channels × 16
banks). The historical cross-page "shallow band" (S3b 185, census
re-probe 228) has the same d distribution for shoulder and low verdicts
under the λ reference — selection bias, not a physical regime. Corollary:
a cross-page low is compatible with "same channel, different bank", so the
old C3 rule is invalid; deep conflicts are the only cross-page class
evidence. Thresholds: low < 0.35·amp, deep ≥ 0.60·amp (the in-page valley
sits at +45..+70).

Result (run_pool_gpu0_1789656551689364094): 201022 pairs classified
{deep 11932 / low 187761 / mid 1329}; same-bank page graph with 387
cross-page deep edges → 60 components, 370/2048 pages (largest 11),
cross-page lows inside components (bad-deep-edge flag) **0**; anchor
validity **2048/2048 pages have a valid anchor** — 0x1fdc80 and 0x1f9dc0
are universal masks on every page (kernel masks of the bank hash),
0x119980/0x11e300/0xd0100 are partially valid (818/650/484 pages),
correcting S3b's "27% anchor validity, position-dependent" to
mask-dependent; the bank map shows a clean base-low/anchor-deep row split
on healthy pages (0x2ae00000 9/25: 0x200→row0, 0xd0300/0xd3880/0xd7b00
→rowM; 0x42e00000 6/25); the ≡ 7 mod 16 super-conflict pages
(0x1ee00000/0x32e00000/0x3ce00000) read deep against all 24 swept
candidates and 25/25 grid points are same-bank — the whole page's pair
baseline is raised (sweep d p50 ≈ +85 vs a typical +15), which is
super-conflict structure, not contamination (the 0x200 column probe still
reads low, λ is correct); cross-page same-bank set Jaccard p50 0.36 —
universal kernel + per-page seeded structure, consistent with the S4
verdict. Table v1 (`artifacts/g3/table_v1/`, `--t1` +
`--channel-deep-only`): 200183 deduplicated edges (T1 contributes 198129
as the highest-priority source; C5 records 321 disagreements, mostly
low↔shoulder flips — the expected λ correction), 39059 bank classes
(1741 with ≥ 2 nodes, largest 125, 60 spanning pages), 13444 row classes,
**C1/C2/C3/C4 zero contradictions**; classified-node coverage **2048/2048
pages** (T0 was 38/2048), same-bank page components 370/2048 pages — the
rest is honest slack (at a 1/384 hit rate most pages simply share no bank
with any representative page). Next: S5-T2 validation gates (R-a
transitive consistency / R-b class cardinality vs prior / R-c
reproduction / R-d cross-card / R-e end-to-end prediction).

Status (2026-09-17): **S5-T2 complete — all five gates pass (5/5 PASS);
the EMT table is qualified for G4/G5 consumption; row classes are
probabilistic (~1% error)**. New `tools/g3_probe/validate_table.py`
(self-test pins each gate's mechanism) + orchestrator
`--work-mode predict-check --pairs-csv` (the R-e workload: calibration 3 /
self on touched pages / predict per pair / repeat for drift; pairs leaving
the pool are dropped and reported). Per gate (bar and rationale written
into the module docstring): (1) **R-a PASS**, a1–a4 all 0 (including the
new count a3: a row class spanning two 2 MiB pages is physically
impossible — the in-page column field cannot absorb PA bits ≥ 21); (2)
**R-b PASS**, unbiased cross-page deep rate 774/302956 = 0.2555% vs the
prior 1/384 = 0.2604% (z = −0.53), implied bank count 391 (2σ 365..422) ⊇
384; the uniform-hash Monte-Carlo null is demoted to a diagnostic — the
representative pages were taken one per v0 channel component, and λ
correlates with bank → representatives cluster by bank (rep-rep deep edges
64 vs a null median of 6); components/coverage below the null interval
are selection bias, not a hash violation (the next build round should
sample representatives uniformly); (3) **R-c PASS**, same-card rerun
(run_pool_gpu0_1789662064404501578): 200830 common pairs agree 99.45%,
hard flips (deep↔low) **0**, deep|mid interval recall 10936/10936, median
d drift 4 cycles, anchor hard flips 0, same-bank partition co-membership
Jaccard 1.000 — deep↔mid counts as threshold wobble (each run's gate rides
its own single-query calibration amplitude, 110 vs 121); only deep↔low
across the empty valley falsifies; (4) **R-d PASS**, cross-card (GPU1
run_pool_gpu1_1789663447672819650: different UUID, identical pool PA
layout, first_pa 0x1ee00000): agreement 99.36%, hard flip rate 0/200830,
Jaccard 1.000, 387 deep edges → the same 60 components / 370 pages, bank
map offset sets and void pages identical, 0x1f9dc0 universal on both
cards (0x1fdc80 is 2047/2048 on GPU1); the median absolute λ drift of 25
cycles is informational, not gated (cross-card timing shifts as a whole:
amp 110 vs 111, calibration 1032/1038/1149) — exactly why the table
stores "classes", not "cycle counts"; the table holds **per model**, and
GeForge's same-model reuse claim gains measured evidence; (5) **R-e
PASS**, end-to-end prediction (215 unmeasured pairs: 128 deep + 87 low,
cross-page preferred, unknown row relations excluded): deep hard flips
0/128 = 100%, low hard flips 1/87 = 98.85%; the one hard falsification is
0x2aed3880/0x2aed7b00 — both addresses reproducibly read anchor-low
against 0x2ae00000's M (inferred same row), yet measure deep +97 against
each other: the double-probe row inference is not universal, row classes
are probabilistic at ~1%, and bank-level conclusions are solid (deep hard
flips 0 across every gate). Table v2 (`artifacts/g3/table_v2/`, two T1
runs merged + --channel-deep-only) is the version for G4/G5: bank classes
are consumed as measured facts; row-class collisions are treated as
strong evidence (future build rounds should double-read the low edges used
for row merges).

Status (2026-09-17): **User approved the T3.0 big-pool full-coverage table
build — three-card order: GPU0 big pool → GPU1/GPU2 replicas → user
confirmation → only then S5-T3 integration**. Rationale: the table is a
measured snapshot of "absolute PA → relation", not an offset formula (the
S4 reverse engineering failed, and the in-page grid's per-page seeded
Jaccard p50 0.36 proves the page base is mixed into the hash), so new PAs
can only be measured, not inferred; better to pay the coverage cost once
than hit missing pages mid-experiment. Engineering parameters: pool
704 × 32 MiB ≈ 22 GiB / 11264 pages (the card's 24 GiB leaves headroom);
representative pages switch to **uniform sampling** (new
`--rep-uniform N`, replacing the v0 seed table's one-representative-per-
component — the latter was confirmed by the R-b diagnosis to cluster by
bank); the count follows coverage math: with 384 banks, R uniform
representatives link a fraction 1−(1−1/384)^R of pages to a same-bank
representative (R = 384 → 63%, R = 1024 → 93%, R = 1536 → 98%); **R =
1024** chosen: classify 11264 × 1024 ≈ 11.5M queries, full workload
~11.8M ≈ 18-minute work segment (the earlier verbal estimate of "384
representatives, 7 minutes" is corrected by this math — 63% coverage is
not worth it, 93% is the sensible point). One-shot window requirement:
card idle + holding the whole 22 GiB pool; pages occupied by others cannot
enter the table but can be measured incrementally later (multiple `--t1`
merges; old entries stay valid forever — the hardware mapping does not
change). Expected benefit: from then on, any experiment pool ≤ 22 GiB
checks coverage before starting (a fail-closed coverage check goes into
the T3 API) and starts on a hit, structurally ruling out "building the
table mid-experiment"; and since R-d has shown same-model mappings are
identical card by card, one big table serves all three 4090s.

Status (2026-09-17): **T3.0 GPU0 big-pool table build complete — table v3
covers 96%, all five gates (R-a/R-b/R-c/R-d/R-e) pass**. Big-pool run
`run_pool_gpu0_1789670148751001580`: 704 × 32 MiB = 11264 pages,
11,815,371 queries in 513 s (23,028 q/s), 0 failures, amp 117; the PA hole
is stable (first_pa 0x1ee00000, the whole 22 GiB contiguous, the old 2048
pages a strict subset). Table v3 (`artifacts/g3/table_v3/`): universe of
11264 pages (2048 census + 9216 T1-only pool pages), 11.39M deduplicated
edges, C1–C4 zero contradictions, 372 same-bank page components covering
10852/11264 pages (**96.2%**, previously 370/2048 = 18%), classified nodes
covering all 11264/11264 pages. Anchor validity 11264/11264
(0x1f9dc0/0x1fdc80 nearly universal); the three ≡ 7 mod 16 super-conflict
pages agree with T1 exactly on 25/25 grid points; the three bank-map void
pages (0x22e/0x2ce/0x48e) are identified as stale labels from S3b mining
under the λ-reference model (low in all three runs), not reproduction
failures. Two **evidence-driven gate corrections** (rationale written into
the `validate_table.py` constant comments and the README T3.0 section):
(1) **R-b becomes an effective-class-count band [368, 400]** — over 11.5M
pairs the measured cross-page deep rate sits stably 2% below 1/384
(implied effective classes ≈ 392, 2σ 387..396, z = −3.4; consistent across
three runs), and a rate below 1/384 is impossible for any fixed
distribution over ≤ 384 buckets (non-uniformity only raises the collision
rate), so the direction rules out the "merged / fewer banks" corruption the
gate is meant to catch; a per-page degree breakdown proves σ is not
underestimated (non-rep page degree under-dispersed 2.20 vs 2.67, no page
above the null maximum degree — the apparent 27× over-dispersion is just
two populations, "ordinary pages ~2.7 / rep pages ~30"). The exact
uniform-384 null z-test and the MC null are kept as diagnostic output.
(2) **R-d Jaccard gates only same-shape runs** — co-membership between
different pools/representative sets compares pair coverage rather than
structure (big pool vs the old 2048 pages: Jaccard 0.199 but 0 hard flips
in both directions); on a shape mismatch it is printed as information,
while the hard-flip rate / anchor flips / interval recall gate as usual;
the three-card same-shape big-pool comparison of the T3.0 protocol still
gates at ≥ 0.99. R-c (vs the 2048-page rerun): 0 hard flips, interval
recall 10521/10521, median d drift 5, anchor hard flips 0; R-e: deep hard
flips 0/128, low 86/87 (the only hard falsification is still the
0x2aed3880/0x2aed7b00 row-inference limitation recorded in T2,
reproducible). `build_bank_table.py` was also corrected so the integrity
universe is each T1 run's pool_map (big-pool edges span 11264 pages; the
2048-page census universe was the wrong reference — the first v3 build
failed closed exactly here) and honestly leaves the λ column empty for
T1-only pages. Next: GPU1/GPU2 same-shape big-pool builds + big-vs-big
R-d; after all pass, report to the user and wait for confirmation to enter
S5-T3.

Status (2026-09-17): **T3.0 three cards complete — structure identical
card by card, all three big-vs-big R-d pairs pass; table v4 (five-source
merge) is the canonical G4/G5 table**. GPU1
`run_pool_gpu1_1789673269815802288` and GPU2
`run_pool_gpu2_1789673819559585717` have the same shape as GPU0
(704 × 32 MiB, `--rep-uniform 1024 --rep-seed 7`): cross-page deep edges
29441/29439/29441, identical component-size heads [42, 41, 40, 38, …],
identical bank-map grid tables and three void pages, 0x1f9dc0 nearly
universal on all three cards; calibration amplitudes differ per card
(117/121/101 cyc). The three big-vs-big R-d pairs: hard flips 1/43/635 per
pair ≤ 0.0054% (bar 0.1%), deep|mid interval recall
100%/99.95%/99.32%, same-bank page co-membership Jaccard all 1.000
(156286 pairs co-resident on both sides, 0 on one side only) — **the claim
that same-model mappings are identical now has complete three-card matrix
evidence**. Two merge/gate corrections (both evidence-driven; rationale in
code constants and the README): (1) **the R-d anchor-transfer gate
becomes a consumption-level criterion** — the anchor sweep's mid valley
is populated (3.5–7% mid cells per card, far above classify's 0.08%), and
per-card gate positions (amplitudes 117/121/101) compound two soft wobbles
of valley-band cells into per-cell deep↔low: 633 cells measured
GPU2 = deep / GPU1 = low, of which 601 are GPU0 = mid (one-directional,
209 pages), while every structural gate is 1.000 and **all 11264/11264
pages keep a common anchor in every pair** (per-page anchor-set Jaccard
p50 1.000) — so per-cell anchor hard flips are demoted to information and
the gate becomes "rate of pages without a common anchor ≤ 0.1%" (a truly
different bank hash would fail on almost every page); (2) **merge rule
completed** — mid never overturns a decided label (one deep measurement
is a same-bank fact); deep↔low competition across T1 runs is resolved by
majority vote (tie → mid, excluded, so a single card's band-edge deep read
cannot become a wrong same-bank edge; the three-card merge had 656
competitions: 622 ties, 34 low majorities). One attribution was also
corrected: the earlier note "GPU2 mids killed 26555 deep edges and lowered
coverage" was really C5 bookkeeping of duplicated sweep/classify rows
inside the GPU2 file; the real coverage gap was that the first v4 build
omitted the old 2048-page T1 run (the big pool's rep set never measured
those pairs; that independent evidence was not overturned) — coverage
recovered once it was folded in. Table v4 (`artifacts/g3/table_v4/`, five
T1 sources merged in time order): 11.39M deduplicated edges (91911 deep),
C1–C4 zero, **372 same-bank page components covering 10852/11264 pages
(96%)**, classified nodes covering all 11264/11264 pages. Honest slack:
412 pages unlinked (at a 1/384 hit rate they share no bank with any of the
1024 uniform reps), row classes ~1% probabilistic (the reproducible
falsifying pair 0x2aed3880/0x2aed7b00), three bank-map void pages (stale
S3b labels), λ column only for census pages. **Next: after user
confirmation, enter S5-T3 (query API + G4 integration, including a
fail-closed coverage check).**

Status (2026-09-17): **After the user confirmed three boundary decisions,
S5-T3 is complete — the EMT query API is in place
(`tools/g3_probe/query_table.py`, pure offline, self-test locked), and the
G3 phase is closed**. User-approved decisions: same/different bank and
same/different row are delivered as measured; row adjacency borrows from
GeForge (assumption + explicit label, not overcomplicated); adjacent
columns are dropped (timing cannot measure them in principle; the G5 fault
model is simplified: column adjacency folds into "same row, random
column"; DQ stays unsupported). API semantics: (1) **four provenance
tiers** — measured (direct measurement: anchor sweep cells, row-class low
edges, complete classify closure) / transitive (equivalence-class
closure, validated by R-e 0/128) / assumed (row adjacency: the GeForge App.
B monotonic row-stripe prior; timing has no row-distance information) /
unknown (never guessed); (2) **four reliability arguments written into the
docstring** — distinct page components ⇒ different bank (classify is
complete over page × representative, so same-bank pages must share a
component); cross-page within one component ⇒ different row (the a3
physical argument); **distinct in-page node bank classes ⇒ unknown, not
different bank** (sparse in-page sampling does not enjoy the completeness
argument); different row is declared only from direct deep evidence or the
cross-page physical argument (row inequality is not transitive through
unmeasured pairs); (3) **fail-closed coverage check** — PAs outside the
universe are refused (exit 2); unlinked pages warn by default and
`--require-linked` escalates to refusal; (4) **two model refusals** —
dq-adjacent (R3) and column-adjacent (folded, with the principled
explanation); (5) **falsifying-pair guard** — queries on
0x2aed3880/0x2aed7b00 are labeled `measured-contradicted` and removed by
the selector; (6) `--annotate-pool` joins a run's pool_map (VA page ↔ PA
page) with the GDDR classes into the snapshot CSV consumed by G4 (the G3
leg); (7) `--build-anchors` folds the anchor sweep cells of the three
big-pool runs into a per-page strict-majority consensus (per-cell flips
stay informational — exactly the valley-band wobble recorded by R-d).
Measured on table v4: anchor consensus 270,336 cells, **11264/11264 pages
keep a valid anchor**, universal masks 0x1f9dc0 + 0x1fdc80; big-pool
coverage check 11264/11264 PASS (412 unlinked-page warnings); measured
same-row sites = 113 pairs over 9 row classes (114 minus the removed
falsifying pair — the honest upper bound for same-row fault placement);
the S3b run is annotated 2048/2048 pages, bank known 1992/2048, 5 pages
with same-row sites. **Next, G4**: use the annotate-pool snapshot as the
G3 leg, combine with G1 (Tensor ↔ VA) and G2 (VA ↔ PA) to build the
TensorBit ↔ GDDR bidirectional chain and reverse index.

Goal:

```text
GPU PA ↔ GDDR6X coordinate
```

Suggested breakdown (as planned; see the status log above for what was
actually achieved — the closed-form route was replaced by the EMT):

#### G3-P1: GDDR6X topology
Confirm:

- memory size;
- channel;
- memory-chip organization;
- bank group;
- bank;
- row;
- column;
- burst.

#### G3-P2: Address-bit characterization
Study:

```text
GPU PA bit
→
channel/bank/row/column
```

#### G3-P3: Bank mapping
Recover:

```text
PA → Bank / Bank Group
```

#### G3-P4: Row mapping
Recover:

```text
PA → Row
```

#### G3-P5: Column / offset mapping
Recover:

```text
PA → Column / Burst / DQ
```

#### G3-P6: Cross-card validation
On 3 RTX 4090 cards of the same model, validate:

```text
f_MC_card0
f_MC_card1
f_MC_card2
```

Complete identity must not be presumed.

---

### G4: Dual Addressing Integration

Status (2026-09-20): **G4-T1 complete (pure offline, zero new collection) —
the three-leg join tool is in place and real snapshots are built on all
three cards**. New `tools/g4_dualaddr/build_snapshot.py`: G1.5
AllocationRegistry (allocation_id ↔ VA ↔ semantic label) × G2 per-run
`gpu_va_pa_map.csv` (join key = allocation_id + VA page, not VA
arithmetic) × G3 table v4 (PA page → bank component / row class / anchors)
→ a per-run dual-addressing snapshot (snapshot_pages.csv + manifest.json
with the mapping checksum = snapshot sha256). Fail-closed: every VA page of
an ACTIVE allocation must have a pte_valid + VIDEO + COMPLETE observation
row; resident PA pages must be inside the table universe; PA aliasing
across VA pages is refused; same-page byte residency must not overlap
(TRT measured: 5 small allocations share one page); map rows whose
allocation is dangling are refused; unlinked / anchor-less pages are
labeled honestly + warned. The self-test locks every path (including six
refusal kinds: gap / outside / invalid-PTE / alias / overlap / dangling).
One data lesson: the top-level `artifacts/g1_5/` and the aggregated map
come from different executions (g1_5_run_id is a logical label), so the
paired `g1_5_allocations.csv` inside the run directory must be used.
Measured on three cards (each card's latest TRT run): 18 rows / 14 PA
pages / 26.4 MiB resident / 7 allocations each, all inside the universe;
bank known 14/14 (GPU0/GPU1), 12/14 (GPU2, 2 unlinked-page warnings);
anchors 14/14 on all three cards; `--lookup-va/--lookup-pa` bidirectional
queries pass. Two findings that shape G4-T2: (1) the three cards' PA page
sets are all different (0x1f0.. / 0x205.. / 0x26e.. segments) — co-resident
VRAM state changes where allocations land; the per-run snapshot + universe
fail-closed absorbs this by design (the user-approved co-tenancy
conclusion is confirmed by measurement: observation and injection are not
affected by other processes; only VRAM pressure could push allocations out
of the universe, which refuses the run); (2) the intersection of workload
resident pages with the 5 measured same-row class pages is 0 — G5
same-row MCU needs placement planning (steering data under test onto
row-class pages) or a row-mining top-up; G4-T2 produces the decision data.

Status (2026-09-20): **G4-T2 complete — live gated runs on all three cards
PASS; the dual-address chain is closed (GDDR-side site selection → VA →
TensorRT byte → XOR bit flip → restore → zero residue)**. Deliverables:
`tools/g4_dualaddr/run_g4_t2_injection.py` (orchestrator + offline
self-test) + an additive runner T2 mode (`--injection-work /
--injection-release`; the legacy path is unchanged and passes regression)
+ `scripts/run_g2_observer_probe.sh --api g4t2` (the only sudo surface;
without `--device` it loops over the three cards sequentially, satisfying
the serial-eBPF constraint) + `docs/G4_VALIDATION.md`. Flow: the observer
attaches first → the runner builds all allocations + clean inference →
writes the gate-time registry and blocks → the orchestrator builds the PTE
ledger + gate map + snapshot **online** (reusing the same code path as
T1's `run_build`) → **reverse-chain site selection** from the snapshot (t1
binding: prefer a bank-linked page, then the data input binding; t2
internal: prefer a bank-linked page, then the largest TRT-internal
allocation; byte = midpoint of the resident range; bit = fixed policy
constant) → work file release → for each target the runner does a live
registry VA consistency check, whole-allocation snapshot, XOR,
`after == before ^ mask`, non-target bytes unchanged, reverse mapping;
then faulted inference (an illegal output is recorded as DUE, not a tool
failure), byte-by-byte restore + whole-allocation compare, and a
post-restore sanity inference that must reproduce the clean result;
closing strict ledger + liveness check within the injection window +
gate/final registry consistency + gate/final map diff (mid-run remap
detector) + independent orchestrator re-verification of every row. The
co-tenancy policy was exercised: all three card runs had 3 other
processes present (co-resident VRAM 10.2/4.7/8.8 GiB), recorded, not
refused; the three cards' PA segments were completely different
(0x29a.. / 0x142.. / 0x242..) but all inside the 22 GiB universe —
co-tenancy pressure absorbed by the per-run snapshot + universe
fail-closed by design. Measured (each: 7 allocations / 18 map rows / 119
eBPF events / 0 lost): t1 = data binding byte 301056 bit 5 (50→18),
t2 = trt-internal-0 (bit 2); every XOR verification / guard bytes
unchanged / reverse mapping / restore / sanity passed; **all six injected
inferences were SDC_NUMERIC** — the faults had real semantic effect, not
just a memory demo; the G3 universal anchors 0x1f9dc0/0x1fdc80 appear on
the selected pages on all three cards. G4 closed. **Next, G5**: GPU fault
injection (SEU / 2-bit / 3-bit MCU + row/column spatial correlation;
same-row MCU needs a placement-strategy decision first — workload
resident pages ∩ the 5 measured row-class pages = 0).

Combine:

```text
TensorBit ↔ GPU VA
GPU VA ↔ GPU PA
GPU PA ↔ GDDR
```

into:

```text
TensorBit ↔ GDDRCell
```

Build:

- mapping snapshot;
- reverse index;
- bitmap tree;
- mapping checksum;
- reproducibility log.

---

### G5: GPU Fault Injection

Status (2026-09-21): **G5-T0 (user literature survey) + G5-T1 (fault
model parameter table) complete — `docs/G5_FAULT_MODEL.md` finalized and
confirmed by the user**. T0: the user finished the survey on 9-21 and
fixed SBU 60% / MCU 40% (event shares), 2-bit : 3-bit = 1 : 1 within MCU
(event counts), spatial shapes from the literature figures (2-bit
same-row / same-column adjacent; 3-bit horizontal triple / vertical
triple / 4×L, uniform within a class), and decided **not to insist on
tight adjacency** — each shape lands on the strongest measurable relation.
T1 frozen parameters: horizontal = same row (random bytes inside the same
256 B block; S3/S3b measured in-page bits 0–7 are always column bits ⇒
same block means same row, same bank); vertical = same bank, different row
(table v4 per-page anchor consensus kernel-mask XOR, measured; masks
< 2 MiB always land on the same page); the four L shapes collapse into one
sampling distribution with the orientation kept as a label; the "assumed"
tier (GeForge row-stripe prior) is not consumed by G5 — every relation G5
injects is measured; placement planning and row-mining are both retired
(row-class pages only serve distant same-row pairs), so G5 needs zero new
GDDR collection. BER semantics = the fraction of resident bits flipped in
one trial (one faulted inference pass over 1000 images), R = 26,428,428 ×
8 = 211,427,424 bits; five levels 1e-8/5e-8/1e-7/5e-7/1e-6 → B =
2/11/21/106/211; (s,d,t) frozen by exhaustive least squares (L1 = 2×SBU,
a pure single-bit point — B = 2 cannot express 60/40, user-confirmed);
1 campaign per level (1 process, 1 bootstrap) × 100 trials; B and
composition frozen within a level, only positions random (SBU byte/bit,
MCU base / anchor choice / in-block offset / orientation label);
classification DUE > SDC_TOP1 > SDC_NUMERIC > BENIGN by per-image
comparison with the clean pass; the literature-source section is left for
the user to fill in. **Next, G5-T3**: campaign implementation (sampler +
runner campaign mode + orchestrator, reusing the T2 gating / restore /
closing skeleton).

Status (2026-09-21): **G5-T3 campaign implementation complete; L3 × 2
trial smoke G5_CAMPAIGN_VERIFIED**. Three parts:
`tools/g5_faultinj/fault_model.py` (frozen level table + site sampler,
pure stdlib, re-derivation self-check); runner campaign mode
(`apps/resnet50_int8_g1_5.cpp`: CPU preprocessing of all 1000 images
before the gate → strict clean pass recording each image → per trial all
sites flipped at once (per site after == before ^ mask, reverse chain,
allocation-level guard = pristine ^ masks) → 1000-image inference with
faults held (input faults re-applied per image, output faults re-applied
per enqueue, TRT-internal not re-applied = persistent weights / soft
scratch flips) → per-image classification against the clean record (the
first DUE aborts the rest of the trial) → reverse-order restore +
per-class restore verification (input binding compared exactly against
the last evaluated image, fail-closed; output bindings
skipped:engine-owned-output; TRT-internal informational mismatch:N) →
single-image sanity inference must reproduce clean);
`tools/g5_faultinj/run_g5_campaign.py` orchestrator (inside the gated
window snapshot → resident byte count must equal the frozen R = 26,428,428
or refuse → sampling → composition / PA-byte-bit distinctness re-check →
work.csv; closing ledger / map diff / registry stability / event stream
aligned entry by entry with work / independent re-check of the four result
CSVs (including re-deriving classification from recorded values, DUE
block shape, event ↔ CSV consistency)). One real ordering bug caught by
the verifier: SITE_RESTORED in reverse order (the T2 convention) did not
match the checker's forward expectation → the checker was fixed and the
self-test now builds the event stream independently. Entry point
`scripts/run_g2_observer_probe.sh --api g5campaign --level L1..L5
[--trials N] [--seed N]`. Smoke: 2 trials / 21 sites / image, whole
chain VERIFIED, trial ≈ 0.7 s → a 100-trial level ≈ 2 min. Remaining:
the full 5 × 100 campaign awaits user go-ahead.

Status (2026-09-21): **Full campaign complete — 5 levels × 3 cards = 15
campaigns, 1500 trials, 1,500,000 faulted image evaluations, all
G5_CAMPAIGN_VERIFIED (0 fail-closed, 0 lost BPF events, 1500/1500 sanity
reproduced clean)**. The user ran them serially in tmux (single-observer
constraint); analyzer `tools/g5_faultinj/analyze_campaign.py` (pure
stdlib, recomputes every metric from the four CSVs independently of the
runner). Pooled top-1 change rate (±95% CI, binomial): L1 0.140% ± 0.013
/ L2 0.242% ± 0.018 / L3 0.324% ± 0.020 / L4 0.773% ± 0.031 / L5 1.030% ±
0.036; numeric SDC rate 63.1% → 98.3% → 99.9% → ~100%; **DUE = 0 (all
1500 trials, 105,300 sites)** — constructive explanation: the output
bindings are only ~8 B of the 26.4 MiB residency (3e-7 share), uniform
sampling of 105,300 sites never hit them, and restore skipped = 0
confirms it; accuracy (clean → L5) 95.30% → 95.00%; P(trial has ≥ 1
top-1 change) 39.0% → 80.3% → 88.3% → 100% → 100%; r2w : w2r ≈ 2 : 1
(damage from flipped quantized weights is symmetric, slightly biased
correct → wrong). Cross-card consistency: L1 bit-identical (deterministic
allocation layout → identical weight sites; only 5 input SBU bits differ
and have zero output effect — INT8 quantization absorbs single input
element flips); L2–L5 cross-card differences 0.014–0.334 pp, explained by
over-dispersion of heavy-tailed trials (a single trial damaged at most
163/1000 images). Sampling property (recorded as is): SBU sites are
uniform over resident bytes (input share 2.18% ≈ residency share 2.28%),
but V/L-pattern anchor mates that fall outside a sparsely resident page
(the input page is 602 KiB / 2 MiB) are retried → the input page's share
among V/L sites drops to ~0.5–0.7% — a deterministic sampling property of
the frozen model, not a bias bug. Data: `artifacts/g5/campaign/` (15
formal + 2 smoke run directories + per-level console logs). **G5 data
collection closed; entering the G6 analysis phase.**

Status (2026-09-21): **Extension-level decision and implementation —
L6–L9 (5e-6/1e-5/5e-5/1e-4) added to the level table (user chose the
four-level plan)**. Rationale: a five-point power-law fit of L1–L5
(top-1 ~ B^0.44) predicts that adding only 5e-6/1e-5 would stop at
~94.7/94.5% with the curve still nearly straight; four levels reach an
extrapolated −2.2 pp (~93.1%), and at L9 the probability of the first DUE
is ~50% (expected output-binding hits 0.64 / campaign). Same derivation
rule: compositions (397,132,132) / (794,264,264) / (3964,1322,1321) /
(7928,2643,2643), literals produced by the exhaustive solver; for
B > 3000 the solver switches to a window (the re-derivation assertion at
every campaign start refuses on disagreement, fail-closed; the self-test
cross-checks window = exhaustive; startup overhead ~52 s → ~1 s).
Execution: single card (based on the L1–L5 three-card consistency
conclusion), seed 7, 100 trials, R guard and all checks unchanged, L1–L5
frozen levels kept as is. The V/L sparse-residency sampling property was
added to docs/G5_FAULT_MODEL.md §4, the extension rationale and
predictions to §5. The user to run the four L6–L9 campaigns in tmux.

Status (2026-09-22): **Extension campaign done — L6–L9 all
G5_CAMPAIGN_VERIFIED; the nine-level curve is closed**. L6/L7/L8 (GPU1)
passed first time; L9 was killed by timeout twice (69/100 and 88/100
trials); timestamp forensics located the cause in the orchestrator's drain
loop rescanning the whole accumulated buffer per 64 KiB chunk (O(n²) read
side → pipe back-pressure → the runner blocked on per-trial printing,
single-trial time growing linearly with index 24 → 298 s) — fixed to scan
only the new tail window (commit d817c70; equivalence self-test + g3/g4/g5
self-tests all pass); not a problem with the experiment itself; the two
failed run directories are kept (no summary.json, skipped by the analyzer
automatically). L9 rerun (GPU2): 100/100 trials, 2,114,300 sites, 0 lost
events, work segment 1524 s (25.4 min; before the fix 88 trials took
> 4 h). Nine-level pooled curve (clean 95.30%): L1–L5 accuracy
95.25/95.22/95.21/95.10/95.00% (plateau), L6 91.88%, L7 90.68%, L8
67.84%, **L9 36.09%**; top-1 change rate 0.14% → 1.03% (L1–L5) →
5.08/6.72/31.24/63.58%; DUE 0 at all nine levels (L9 expected
output-binding hits λ = 0.64, P(0) = e^-0.64 ≈ 53% — consistent with the
constructive model); r2w : w2r worsens from 2 : 1 to ~28–32 : 1. **The
pre-run power-law extrapolations (94.7/94.5/93.7/93.1%) were all broken
upward: the knee lies between BER ≈ 1e-6 and 5e-6, and damage above it is
strongly superlinear** — a more valuable paper finding than "adding two
points"; predicted vs measured was written back to
docs/G5_FAULT_MODEL.md §5. The whole G5 phase (9 levels, 19 campaigns,
1900 trials, 1.9 million faulted evaluations) completed data collection.

First phase does only:

- SEU;
- 2-bit MCU;
- 3-bit MCU;
- row-adjacent;
- column-adjacent;
- DQ-adjacent (if the mapping can be verified).

(As executed: column-adjacent folded into "same row, random column" and
DQ-adjacent unsupported — both unmeasurable by timing; see the G3 S5-T3
and G5-T1 entries.)

Execution:

```text
fault select
→ reverse map
→ GPU XOR
→ inference
```

---

### G6: DNN Reliability Evaluation

Status (2026-09-22): **G6-T0 complete (attribution analysis, zero new
collection) — tool `tools/g5_faultinj/analyze_g6_attribution.py` (pure
stdlib, every number recomputed live from the CSVs of the 19 VERIFIED
campaigns); full conclusions in `docs/G6_ANALYSIS.md`**. Answers to the
four questions (raised by the user on 9-22): (1) **the plateau is not
missed flips** — site count per level = B × trials exactly, 0
verification violations; and at L1 63.1% of images already show numerical
output shifts (~100% from L3) — the flips really propagate through all
layers to the logits, but the median |ΔP| (L5: 0.008) is two orders of
magnitude smaller than the decision margin (clean confidence p10/p50/p90 =
0.55/0.72/0.81) and is absorbed by it; half of the top-1 changes even
flip back to correct (r2w : w2r 2 : 1). (2) **91.8% of flips land in the
INT8 weight blob** (trt-internal-0, 24.0 MB ≈ 25.6M parameters; residency
share 90.79% ≈ site share → unbiased sampling); all 3.3 million
weight-region sites restore exact = the engine never rewrites them → the
1000 images of one trial are inferred with the same corrupted weights
(the carrier of systematic damage); flips in the scratch region
(internal-3) are rewritten by the engine = transient soft flips; the input
binding is under-sampled by the V/L anchoring effect (1.29% vs 2.28%) and
absorbed by INT8 quantization; the output bindings (8 B) were never hit.
(3) **knee mechanism = the perturbation distribution crossing the margin
distribution**: weight corruption L5 193 bytes (0.0008%) → L9 19,362
(0.081%), median |ΔP| 0.008 → 0.26; low-confidence images (< 0.5, 7.4%)
fall first, r2w : w2r 2 : 1 → 28 : 1 becomes one-directional, and at L9
the perturbation is on the order of the margins → 63.6% of images flip,
accuracy 36%; damage efficiency per byte 0.0016 → 0.003 pp (a second
superlinearity from accumulation across layers). (4) **DUE = 0 is
structural**: the only targets that can cause DUE are 8 B / 26.4 MiB (L9
expected hits 0.64, P(0) ≈ 53%, measured 0 is consistent); INT8 saturating
fixed-point arithmetic has no NaN/Inf propagation = quantization acts as
a DUE firewall; the XOR fault channel itself has no crash path (ECC-level
DUE is outside the fault model, §8). **Overall conclusion: a high
tolerance floor + cliff-type failure, with no linear graceful-degradation
band — low-BER behavior cannot be extrapolated (the pre-run power-law
extrapolation underestimated the L9 loss by 57 pp).** Optional follow-up:
G6-T1 site-level attribution (decompose top-1 changes by layer / byte
segment; needs work_detail.json per-site joined with per-image detail).
(Later note, 2026-09-29: the "DUE firewall" and "no crash path"
conclusions hold for this ResNet-50/RESISC45 engine only — see the G7
entries below and the note in `docs/G6_ANALYSIS.md` §4.)

Statistics:

```text
BER
fault count
fault location
tensor
layer
element
bit
SEU / MCU
BENIGN
SDC
DUE
accuracy drop
retry behavior
```

Final result:

```text
GDDR physical fault
→ DNN semantic impact
```

experiment database.

---

### G7: Multi-model / ImageNet-1K extension campaign

Status (2026-09-23): **G7-T0 base work complete — weights / ONNX / INT8
engines / FP32 + INT8 clean evaluation done for all six models in one go,
quantization-loss table on disk**. Downloads were bumpy but all resolved
(unauthenticated HF rate limiting: efficientnet revived once by resumable
download; ViT loaded directly as safetensors from a June HF cache snapshot,
bypassing timm's networked list_repo_files; DeiT-S / Swin-T pulled through
proxy SSL drops with an unbounded curl resume loop, no manual download by
the user needed). The Swin ms_in1k checkpoint contains 17 non-persistent
timm 1.0.26 buffers (attn_mask etc.); strict loading passes after removing
them. Engine builds (entropy calibration batch = 1, 1000 images, 4 GiB
workspace, zero-input smoke test after build) PASS for all six models:
resnet50 28.7 MB / 118 s, mobilenetv3 10.6 MB / 543 s, efficientnet
10.7 MB / 461 s, ViT-B 93.9 MB / 336 s, DeiT-S 33.2 MB / 271 s, Swin-T
46.5 MB / 464 s (all Transformer engines healthy, no PTQ collapse).
**Clean evaluation (same 10K eval set, same preprocessing, INT8 accepted by
two bit-identical passes) — quantization loss varies widely**:

| Model | FP32 top-1 | INT8 top-1 | Loss pp |
|---|---|---|---|
| ResNet-50 | 77.36% | 75.06% | 2.30 |
| MobileNetV3-L | 72.14% | 62.42% | **9.72** |
| EfficientNet-B0 | 74.90% | 69.15% | 5.75 |
| ViT-B/16 | 78.38% | 78.20% | **0.18** |
| DeiT-S | 78.05% | 71.91% | 6.14 |
| Swin-T | 79.41% | 78.24% | 1.17 |

(v1 numbers; superseded by the protocol-v2 table in the next entry.)
Transformers with large dense GEMMs (ViT-B / Swin-T) are nearly lossless
under entropy calibration; efficient CNNs (MobileNetV3 depthwise + SE +
hardswish, EfficientNet) lose 6–10 pp; DeiT-S's distilled weights are also
fairly INT8-sensitive — each BER curve is normalized to its own INT8 clean
baseline, and the loss is recorded honestly, not hidden (consistent
protocol: the same entropy-calibration rule for all six models, no
per-model calibrator swaps). Measured single pass over 10K images: FP32
66–160 s, INT8 35–53 s (G7 per-trial cost ≈ 10× G5's image count; the
9 levels × 100 trials schedule is scaled accordingly). Two engineering
lessons: a `pkill/pgrep -f` pattern matches the caller's own command line
(bit us twice: the wait loop never exits); `g7_models/incoming/` keeps the
original safetensors for provenance. **Next: per-workload
parameterization of the G5 runner** (preprocessing mean/std/interpolation,
engine path, eval split, per-model R table, evaluation of the ≈ 6 GB host
preprocessing buffer for 10K images), then a nine-level ResNet-50/ImageNet
campaign smoke test.

User's five decisions (frozen 2026-09-23): (1) six model variants:
ResNet-50 / MobileNetV3-Large / EfficientNet-B0 / ViT-B/16 / DeiT-S /
Swin-T, all INT8 PTQ (timm ImageNet-1k pretrained weights); (2)
calibration set = 1000 classes × 1 image from val = 1000 images; (3) eval
set = 1000 classes × 10 images = 10000 images, disjoint from the
calibration set (checked fail-closed); (4) strictly serial execution one
model at a time — first get satisfying ResNet-50/ImageNet results, then
the other five one by one; but the base work (download + quantization +
FP32/INT8 clean) is done all at once; (5) each of the six models runs FP32
and INT8 clean once (same 10K eval set), quantizing the quantization loss
up front. **Frozen and unchanged**: the fault model (SBU 60% / MCU 40%,
spatial shapes, composition rule), BER semantics (B = round(BER ×
R_bits), with R re-derived per workload and the level BER values identical
across models for comparison), the full gating / snapshot / residency
guard / flip / restore / classification / verification protocol, seed 7,
100 trials. Ready: ImageNet val (local /data1/luojx/datasets/imagenet1k,
all 50,000 images cross-checked against the official val_map, 0 errors);
the split builder `tools/g7_prep/build_imagenet_splits.py` (seed 7: calib
1000 + eval 10000, disjoint, manifest on disk); the six-model downloader
`tools/g7_prep/download_models.py` (including default_cfg preprocessing
metadata; the runner's preprocessing will be parameterized from it,
avoiding a second source of truth). Build chain found and in place: the
vit_fault env has `tensorrt_bindings` 8.6.1 installed (the earlier "no
TensorRT on this host" conclusion was wrong — only the `tensorrt` module
name had been searched), used with the tensorrt 8.6.1 runtime libraries +
cuDNN 8.9.7 under `/data1/luojx/REMU/.local/deps/` (the same
LD_LIBRARY_PATH wiring as stage13, wrapped in `build_g7_engines.sh` /
`eval_g7_clean.sh`). New tools: `export_g7_onnx.py` (three-binding
`data/prob/index` contract, opset 17, INT64_MAX slice-end sentinel
rewrite, every per-model fact from model_meta.json),
`build_g7_int8_engine.py` (`IInt8EntropyCalibrator2` batch = 1, the G7
1000-image calibration set, calibration cache keyed by a hash of
onnx + calib + mean/std + interpolation, zero-input smoke after build;
stage13 prior: all seven models' INT8 PTQ were healthy back then, vit_b16
95.6% with no Transformer collapse), `eval_g7_clean.py` (FP32 torch and
INT8 TRT with the same preprocessing on the same 10K set, INT8 accepted
by two bit-identical passes, quantization loss measured up front). To do:
the remaining five models (after download: ONNX export → INT8 build →
clean eval in one pass), and per-workload parameterization of the G5
runner (preprocessing mean/std, engine path, eval split, R table).

Status (2026-09-24 evening): **Protocol v2 finalized — canonical
preprocessing + explicit Q/DQ per-channel quantization, all six models'
quantization loss brought down to 0.46–2.11 pp** (the user approved two
protocol changes that day: preprocessing switches to timm canonical;
quantization switches to explicit Q/DQ). Two diagnoses settled it: (1)
v1's FP32 baseline was depressed by square stretching (canonical restores
paper-level accuracy); (2) the default explicit Q/DQ recipe hit three
independent problems on the swish-family architectures and Swin, each
located and fixed — **all damage is activation-side** (per-channel INT8
weights cost ~0 pp everywhere): EffNet's 16 SiLU-output tensors are
extremely sensitive to per-tensor symmetric INT8 (−22.3 of −22.65 pp
concentrated there), MobileNetV3's HardSwish outputs likewise
(−6.05 → −0.30 pp), `--use_zero_point` does not help; Swin hits two TRT
8.6.1 defects (Q/DQ in the window-attention region executes wrongly →
`--disable_mha_qdq`; ReduceMean axes have no valid form under opset 19 →
attribute form + downgrade the whole graph to opset 17). Final recipes
(the evidence chain is in the README): resnet/vit/deit default recipe;
mobile/effnet conv-only + swish-output bypass; swin conv-only +
disable_mha_qdq. Final table:

| Model | FP32 top-1 | INT8 top-1 | Loss pp |
| --- | --- | --- | --- |
| ResNet-50 | 80.61% | 78.50% | 2.11 |
| MobileNetV3-L | 75.64% | 75.15% | **0.49** |
| EfficientNet-B0 | 77.96% | 77.32% | **0.64** |
| ViT-B/16 | 79.41% | 78.22% | 1.19 |
| DeiT-S | 80.26% | 78.75% | 1.51 |
| Swin-T | 81.63% | 81.17% | **0.46** |

This table replaces the v1 table (the v1 implicit engines are kept as
fallback artifacts). (Its INT8 column was measured on the v1 FP32-head
engines; the v2 head-quantized engines the campaigns ran are re-measured
in the 2026-09-25 entry below.) Toolchain additions:
`dump_g7_calib_npy.py` (canonical 1000-image calibration npy),
`quantize_g7_qdq.sh` (ModelOpt explicit quantization + per-model recipe +
bypass/fix post-processing), `bypass_g7_qdq_activations.py` (swish-output
activation bypass), `fix_qdq_for_trt86.py` (TRT 8.6.1 graph
normalization), `build_g7_explicit_engine.py` (explicit engine build,
EXPLICIT_BATCH + kINT8 flag, Q/DQ census written to the summary).
ModelOpt 0.47 is installed in an isolated venv
(`/data1/luojx/REMU/.local/deps/modelopt-venv`, never touching vit_fault).

**Pre-registered hypotheses (for the experiment to decide)**: ① each
architecture's "tolerance floor" (knee position) and collapse slope differ
— Transformers with large dense GEMM weight blocks have a smaller share of
corrupted bytes per weight at the same BER, but small-parameter regions
such as LayerNorm / position embeddings may be fragile; ② MobileNetV3's
depthwise layers have very few bytes per kernel, so a single corrupted
byte has a relatively larger effect; ③ the DUE firewall of INT8 saturating
arithmetic also holds for Transformers (expected DUE all 0); ④ the
asymmetric evolution of r2w : w2r is either consistent or divergent across
architectures. Outputs: six accuracy-vs-BER curves in one figure (two
versions: each normalized to its clean baseline, and absolute), a knee
comparison table, and a second x-axis (share of corrupted weight bytes).

Status (2026-09-25): **G7-v2 engines finalized + ResNet-50/ImageNet-1K
nine-level campaign complete**. Two prerequisites: (a) the 9-24
fault-surface census found that ModelOpt 0.47's GEMV heuristic
(`enable_gemv_detection_for_trt`, default-on and without a CLI switch)
silently leaves the classifier-head Gemm in FP32 under batch-1 exports
(in all six models, verified by a single-flag counterfactual) — user
decision 9-24, v2 rebuild: `quantize_g7_qdq_head.py` turns the heuristic
off through the python API so the head is also per-channel INT8, and
`quantize_g7_qdq.sh` fails closed on any Gemm weight not behind a
DequantizeLinear; the v1 FP32-head engines are archived under
`<model>/v1_fp32head/` (only preliminary levels of
`g7_imagenet1k_resnet50`, R = 34,959,884, ever ran on them). Re-measured
v2 clean top-1: resnet50 78.42 / mobilenetv3 75.04 / effnet 77.36 / vit
78.21 / deit 78.73 / swin 81.30 (the in-campaign clean passes agree).
(b) The G5 runner is parameterized per workload (bit-exact port of
canonical preprocessing, class count, explicit-batch engines,
`--workload`, a measure-R-only `--bootstrap` mode); the G1–G6 protocol
itself is unchanged. ResNet-50 v2 bootstrap R = 28,832,268 B (6,127,616 B
less than v1 ≈ the head FP32 → INT8 difference, independently confirming
the head is quantized); nine levels 1e-8 / 5e-8 / 1e-7 / 5e-7 / 1e-6 /
3e-6 / 5e-6 / 7e-6 / 1e-5 (L6–L9 appended by the user on 9-25, derived
from the same R, all exhaustive-solver literals). Results (GPU0, 100
trials per level, all VERIFIED): clean 78.42% → 78.44 / 78.39 / 78.34 /
77.13 / 76.68 / 73.08 / 66.62 / 67.14 / 61.23%; top-1 change rate
0.46% → 28.15%; 2 DUE trials each at L8/L9 (the first DUE in G7).
**Statistical convention (user decision 9-25)**: injected accuracy
averages only normally completed trials; trials containing a DUE are
excluded from the mean and reported as a separate DUE rate (G5 had DUE = 0
throughout, so its outputs are byte-identical); `analyze_campaign.py`'s
`--workload` is enforced fail-closed under a multi-workload root (v1/v2
levels with the same name have different BERs).

Status (2026-09-27..28): **Five-model extension — flipping TRT runtime
control state can kill the process; decided model by model**. The five
models share a seven-level ladder L1–L7 = 1e-7 / 5e-7 / 1e-6 / 3e-6 /
5e-6 / 7e-6 / 1e-5 (i.e. ResNet-50 v2's L3–L9, aligned by BER value);
each model's R is measured by its own bootstrap and frozen by
`tools/g7_prep/freeze_g7v2_workload.py` (forced exhaustive solve +
windowed-solver consistency cross-check + derivation regression against
the ResNet-50 table). (1) **MobileNetV3**: the first L1 campaign and two
diagnostic reruns (seeds 7/8/10) died when a flip landed at an
address-bearing offset of `trt-internal-5` (TRT private control state from
the create_execution_context phase, 246,272 B, holding device-side
per-kernel execution parameters): CUDA illegal memory access at the first
injected inference → runner process death (reproduced point by point by
offline resampling; hits at scalar offsets survive). User decision 9-27:
**for this workload only**, move `trt-internal-5/6` (248,320 B = 2.66% of
R) out of the injection surface: frozen R = 9,087,912 B (full measured
9,336,232 B); full-surface injection is kept as the paper-appendix control
experiment. (2) **EfficientNet-B0**: the same failure class (died at L2
trial 83), but its ctx-phase pool is 58% of R with ~400× lower fatal
density (~1 per 4,228 pool bits). A 58% exclusion was briefly frozen
(b8bf6e5) and reverted the same day — it would make the injection surface
91% weights and change the experiment's meaning. User decision 9-28:
**full-surface injection + campaign restart protocol**. The runner
flushes the three result CSVs after each completed trial; when the process
dies mid-trial, and only when the structural evidence is complete (the
completed-trial prefix is contiguous and re-verified row by row, exactly
one trial is open, and its complete flip set is the tail of the event
stream) and the log carries a known fatal CUDA signature, that trial is
recorded as **PROCESS_FATAL** (excluded from the accuracy mean like DUE,
with its own crash-rate column); the orchestrator starts a fresh gated
segment with seed + k to run the remaining trial slots and renumbers by
global execution order at merge; any other death shape fails the whole
level closed. Signature whitelist, two entries: `illegal memory access`
and `operation not supported on global/shared address space` (first seen
on Swin L5: same mechanism, different driver error string). Supporting
work: a host preprocessed-image cache (5.6 GiB per preprocessing spec,
SHA-256 key + header + blob verification, any mismatch falls back to
recomputation; per-restart overhead ~4.5 → ~1.5 min; the five models span
three specs); a merge/summary bookkeeping fix for crashed segments that
never produce a final map; a crash-rate panel in the plotter, with the
formal-level filter switched to attempted trials. DeiT-S / Swin-T / ViT-B
all run full-surface with the restart protocol from the start. Paper
point: the ctx pool's share of residency varies wildly by engine
(ResNet-50 0.08% / MobileNetV3 2.6% / Swin-T 28% / EfficientNet-B0 58%),
and together with the pool's fatal density it decides the protocol — scope
when the pool is small and deadly (mobilenet), restart when it is large
and sparse (efficientnet and after).

Status (2026-09-29): **G7-v2 six-model campaign complete** (GPU0; 100
trials per level per model, all VERIFIED; all 61 PROCESS_FATAL events
absorbed by the restart protocol, zero lost and zero fabricated trials).
Summary `docs/G7V2_RESULTS.md`; unified figure
`artifacts/g7/campaign/fig_six_models/` (`tools/g5_faultinj/
plot_six_models.py`, every number recomputed from the CSVs at run time).
Top-1 accuracy (%):

| Model | clean | 1e-7 | 5e-7 | 1e-6 | 3e-6 | 5e-6 | 7e-6 | 1e-5 | Δ pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50 | 78.42 | 78.34 | 77.13 | 76.68 | 73.08 | 66.62 | 67.14 | 61.23 | −17.19 |
| MobileNetV3-L | 75.04 | 74.58 | 74.34 | 71.60 | 67.13 | 53.38 | 51.58 | 47.27 | −27.77 |
| EfficientNet-B0 | 77.36 | 77.24 | 76.50 | 75.21 | 73.63 | 71.08 | 66.20 | 61.34 | −16.02 |
| DeiT-S | 78.73 | 78.51 | 78.47 | 78.35 | 75.54 | 68.48 | 64.63 | 61.11 | −17.62 |
| Swin-T | 81.30 | 81.25 | 79.62 | 77.92 | 74.20 | 74.82 | 62.25 | 58.84 | −22.46 |
| ViT-B | 78.21 | 76.80 | 77.72 | 77.54 | 77.08 | 73.85 | 70.86 | 71.73 | −6.48 |

Frozen injection surface R (B): ResNet-50 28,832,268 / MobileNetV3
9,087,912 (after exclusion) / EfficientNet-B0 17,144,232 / DeiT-S
26,010,832 / Swin-T 43,799,616 / ViT-B 93,867,728. Reliability (DUE
trials / PROCESS_FATAL events, summed over the seven levels): ResNet-50
4/0, MobileNetV3 16/0, EfficientNet-B0 9/24, DeiT-S 0/1, Swin-T 1/33,
ViT-B 3/3; the crash rate rises with the ctx-pool share (Swin-T reaches
17/100 at L7). SDC-top1: the CNNs and Swin-T grow nearly linearly with
log(BER), converging to 28–45% at 1e-5; DeiT-S has a ~9–10% plateau over
L1–L3; ViT-B is almost BER-independent (L1 11.86% → L7 18.50%), with the
largest per-flip perturbation (L1 mean |ΔP| 0.085, Swin-T 0.018), but
the damage saturates instead of accumulating.

**Verdict on the pre-registered hypotheses**: ① knee position and
collapse slope diverge across architectures — confirmed (ViT-B is the
most tolerant at −6.5 pp and its curve is non-monotonic, L6 70.86 → L7
71.73; ResNet-50 5e-6 → 7e-6 and Swin-T 3e-6 → 5e-6 also rise locally);
② MobileNetV3 is the most fragile — confirmed (−27.8 pp, the largest of
the six); ③ the INT8 DUE firewall holds for Transformers — largely
confirmed (DeiT-S / Swin-T / ViT-B DUE ≤ 2/100 per level), but
MobileNetV3 shows NaN/Inf-type DUE up to 6/100; and G6 §4's conclusion
that "the XOR fault channel itself has no crash path" holds only for
ResNet-50/RESISC45 — flips in the G7 engines' ctx-phase control state kill
the process (PROCESS_FATAL, a new failure class); ④ r2w : w2r evolution —
ResNet-50 v2 goes from L1 1058 : 1274 to a one-directional ~13 : 1 at L9,
while ViT-B only goes from L1 ~1.5 : 1 to L7 ~3.3 : 1; the other four
models are not yet summarized uniformly. **State of the planned outputs**:
the one-figure plot of the six absolute accuracy curves is done; the
clean-normalized version, the knee comparison table, and the second
x-axis (share of corrupted weight bytes) are not yet produced.

---

## 7. Per-experiment mapping snapshot

Because GPU VA / GPU PA change with allocation, every run must rebuild its
tables:

```text
RUN N
 ├── Tensor metadata
 ├── GPU VA map
 ├── GPU PA map
 ├── GDDR coordinate map
 ├── reverse index
 ├── selected faults
 └── inference result
```

Directly reusing a historical:

```text
Tensor → GPU PA
```

mapping is forbidden.

The only thing that may be reused long-term is the verified-stable:

```text
GPU PA → GDDR coordinate
```

memory-controller mapping rule.

(As implemented: every G4/G5/G7 run builds its own gate-time snapshot;
the only long-lived artifact is the G3 table v4, validated identical
across the three cards.)

---

## 8. Cache handling principle

The first phase does not study GPU cache faults.

It explicitly adopts this abstraction:

```text
GDDR fault model
      ↓
select a physical GDDR location
      ↓
map to a GPU device-memory bit
      ↓
software XOR
      ↓
inference
```

So first-phase papers/reports must state explicitly:

> GPU cache hierarchy is outside the current fault model.
> GDDR faults are materialized as application-visible device-memory corruptions.

Later work can separately extend to:

```text
GPU L2 / L1 cache fault injection
```

First-phase results must not be described as a complete:

```text
physical GDDR capacitor upset
→ cache
→ SM
```

dynamic propagation model.

---

## 9. Experimental correctness validation

All four segments must be validated separately; validating only the final
inference is not allowed.

### V1: Tensor ↔ GPU VA

Use known patterns:

```text
0x00
0x55
0xAA
0xFF
```

to check element / bit localization.

### V2: GPU VA ↔ GPU PA

Check:

- repeated queries on the same allocation agree;
- page boundaries;
- multiple allocations;
- free/realloc;
- temporal stability.

### V3: GPU PA ↔ GDDR

Check:

- same-bank;
- different-bank;
- row conflict;
- row adjacency;
- consistency across repeated measurements.

### V4: Fault injection

Check:

```text
before
after
xor_mask
expected
```

Requirement:

```text
after = before XOR mask
```

and non-target data must not change.

### V5: End-to-end

Finally:

```text
GDDR Cell
→ Tensor Bit
→ bit flip
→ inference result
```

must be repeatable.

(Status: V1 = G1; V2 = G2's three validation layers; V3 = the G3 S5-T2 /
T3.0 gates R-a…R-e (row adjacency is only an assumed-tier label, never
consumed by the campaigns); V4 = per-site checks in every G4/G5/G7 trial;
V5 = G4-T2 plus the G5/G7 per-trial restore and sanity checks. All
passed.)

---

## 10. Main risks

### R1: No public GPU VA→PA interface

This is currently the highest risk.

Strategy:

- look for existing research tools first;
- prefer driver-side read-only introspection;
- kernel modification as the last resort;
- keep the mapping module independent so it does not contaminate the main
  fault framework.

Outcome (2026-09-15): resolved by the read-only eBPF PTE observer over the
unmodified driver (`tools/g2_observer/`); no kernel modification. The
observer contract is pinned to driver 580.95.05, so a driver upgrade
requires re-deriving it.

### R2: RTX 4090 memory-controller mapping is not public

This is the second-highest risk.

Strategy:

- characterize the mapping independently;
- do not use an assumed mapping as a final experimental conclusion;
- every mapping rule must have measured evidence.

Outcome (2026-09-17): the closed-form hash could not be recovered (S4 —
an honest negative result); it was replaced by a measured empirical
mapping table (table v4) that passed five validation gates and is
identical across the three cards. Every relation the campaigns inject is
measured.

### R3: GDDR6X DQ-level location is hard to recover fully

Strategy:

The first phase can first reach:

```text
Channel / Bank / Row / Column
```

If the DQ mapping cannot be verified rigorously, then:

- SEU can continue;
- row/column MCU can continue;
- DQ MCU is marked unsupported / unverified;
- DQ physical coordinates are never fabricated.

Outcome: DQ is unsupported, and column adjacency is not measurable by
timing either (folded into "same row, random column"). Bank / row
relations are delivered as measured; channel identity is not exposed as a
coordinate.

### R4: TensorRT internal allocations are opaque

Strategy:

First build the whole chain with controllable CUDA tensors / PyTorch CUDA
tensors.

After the full flow is confirmed correct, extend to TensorRT runtime
allocations.

Outcome: the chain was built directly on the TensorRT runtime allocations
(G1.5 AllocationRegistry). TRT-internal regions carry allocation-level
attribution only (`TENSORRT_INTERNAL_UNKNOWN`). A new risk surfaced in G7:
corrupting TRT's execution-context control state kills the process. It is
handled by the restart protocol, or for MobileNetV3 by surface scoping.

---

## 11. First-phase minimum viable goal (MVP)

The first phase should not aim for a complete GPU memory system.

MVP:

```text
one explicit Tensor
     ↓
GPU VA
     ↓
GPU PA
     ↓
GDDR Bank/Row/Column
     ↓
select one SEU
     ↓
reverse mapping
     ↓
locate Tensor element/bit
     ↓
GPU XOR
     ↓
verify the bit flip
     ↓
run inference
```

MVP acceptance must answer:

> **"If a bit flips at this physical location in RTX 4090 GDDR, which
> Tensor, which element, and which bit of the DNN does it correspond to?"**

and be able to actually flip that bit and observe the inference result.

Outcome (2026-09-20): achieved by G4-T2 on all three cards. Sites were
selected from the GDDR side, driven to live TensorRT bytes, XOR-verified,
and inferred (all SDC_NUMERIC), then restored with zero residue. The
answer is given at allocation/byte/bit granularity: tensor-element
granularity for the public bindings, allocation-level for TRT-internal
data.

---

## 12. Final target architecture

```text
                    GPU-side REMU
┌──────────────────────────────────────────────────┐
│                                                  │
│   DNN Semantic Mapper                            │
│   Tensor / Element / Bit                         │
│              ↕                                   │
│          GPU Virtual Address                     │
│              ↕                                   │
│   GPU VA→PA Mapper                               │
│              ↕                                   │
│          GPU Physical Address                    │
│              ↕                                   │
│   RTX4090 GDDR Address Decoder                   │
│              ↕                                   │
│   Ch / BG / Bank / Row / Column / DQ             │
│              ↕                                   │
│   Bitmap Tree / Reverse Index                    │
│              ↕                                   │
│   SEU / MCU Fault Generator                      │
│              ↓                                   │
│   GPU Device-Memory Fault Injector               │
│              ↓                                   │
│   DNN Inference                                  │
│              ↓                                   │
│   BENIGN / SDC / DUE                             │
│                                                  │
└──────────────────────────────────────────────────┘
```

As built, each box maps to:

| Box | Implementation |
| --- | --- |
| Semantic Mapper | G1 / G1.5 `AllocationRegistry` |
| VA→PA Mapper | G2 eBPF observer |
| GDDR Address Decoder | the G3 EMT (table v4 + `query_table.py`); it yields bank / row relations, not Ch / BG / Column / DQ coordinates |
| Reverse Index | the G4 snapshot plus `fault_model.ResidencyIndex` |
| Fault Generator | `fault_model.py` |
| Injector and outcomes | the campaign runner and orchestrator; outcomes extended with PROCESS_FATAL in G7 |

---

## 13. Positioning of the core research contribution

This project should not be described simply as:

> "randomly flipping DNN bits on a GPU."

It should be defined as:

> **a memory-aware fault injection framework based on real GPU address
> mapping and GDDR physical organization.**

The core contribution chain should be:

```text
DNN semantic bit
↔ GPU virtual address
↔ GPU physical address
↔ GDDR physical location
```

and then, based on real GDDR spatial correlation, implement:

```text
SEU / MCU
→ DNN fault
→ inference impact
```

The final goal is to systematically replace REMU's:

```text
CPU VA ↔ CPU PA ↔ LPDDR
```

with:

```text
GPU VA ↔ GPU PA ↔ GDDR6X
```

while keeping its overall methodological framework of:

```text
Dual Addressing
+
Memory-aware Error Model
+
Reverse Mapping
+
Large-scale Reliability Evaluation
```
