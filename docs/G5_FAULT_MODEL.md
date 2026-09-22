# G5 Fault Model — Frozen Parameter Table (G5-T1)

Status: **FROZEN — core L1–L5 user-confirmed 2026-09-21**; extension levels
**L6–L9 added 2026-09-21** after the L1–L5 campaign completed (same
derivation rule and semantics; single-card execution per the verified
L1–L5 no-card-effect result; see §5). This document freezes every
parameter of the G5 fault-injection campaign; the code re-derives the
level table with the rule in §5 and asserts equality against this
document at every campaign start.

Provenance of the ratios and spatial patterns: the user's G5-T0 literature
survey, done 2026-09-21. The reference list will be inserted by the user in
§9 (left as a placeholder on purpose).

## 1. Event mix (from the survey)

- **SBU 60% / MCU 40%** — shares of flip **events** (not of flipped bits).
- MCU multiplicity: **2-bit and 3-bit events, 1:1 by event count**.
- Spatial patterns (survey figure): 2-bit {same-row adjacent, same-column
  adjacent}; 3-bit {horizontal triple, vertical triple, 4×L-shape}; uniform
  within each multiplicity class.
- **Strict adjacency is NOT enforced** (user decision, 2026-09-21): every
  pattern is realized at the strongest relation we can MEASURE. What is
  dropped: adjacency distance, same-column, DQ. What is kept: same-row and
  same-bank/different-row (§3).

## 2. Implemented sampling classes

The 8 literature patterns map onto 6 sampling classes (the L quartet
collapses — §3):

| class | events | sites (byte granularity) | relation kept |
|---|---|---|---|
| SBU | 1 bit | 1 resident byte, bit uniform 0–7 | none (fully measured chain) |
| MCU2-H | 2 bits | base + 1 other byte in the same 256 B PA block | same row |
| MCU2-V | 2 bits | base, base ⊕ anchor (per-page consensus anchor) | same bank, different row |
| MCU3-H | 3 bits | base + 2 other bytes in the same 256 B block | same row |
| MCU3-V | 3 bits | base, base ⊕ m1, base ⊕ m2 (two distinct anchors) | same bank (rows: unknown pairwise) |
| MCU3-L (×4 labels) | 3 bits | base + 1 block mate + 1 anchor mate | same-row pair + same-bank third |

Within-class weights: 2-bit H/V each 50% of the 2-bit events; 3-bit
H/V/L each 1/6 of the 3-bit events, L's four orientations drawn uniformly as
**labels** (their distributions are identical — §3).

## 3. Spatial relations and provenance (all labels travel per site)

- **same-row (horizontal leg)**: G3 S3/S3b measured that in-page PA bits
  0–7 never leave the bank/row (column region) — any two bytes of one
  256 B-aligned PA block share bank AND row. Provenance label
  `measured-column-region` (GPU0-measured across the 4 GiB pool; cross-card
  by structural identity — the universal kernel masks are identical on all
  three cards). The column relation *inside* the block is UNKNOWN
  (burst/DQ organization below 256 B is unmeasured): mates are drawn as
  distinct random bytes of the block, labeled `unknown-column`.
- **same-bank / different-row (vertical leg)**: per-page anchor consensus of
  the G3 EMT table v4 (`page_anchors.csv`, 11264/11264 pages with valid
  anchors; universal kernel masks 0x1f9dc0 / 0x1fdc80). `base ⊕ m` shares
  the bank with base and differs in row — both measured. The mask is
  < 2 MiB so the mate lands inside the same resident PA page. Labels:
  `measured-kernel-mask`, `unknown-row-distance`, `unknown-column`. For the
  vertical triple, pairwise same-bank holds transitively (bank is a
  function); the row relation between the two anchor mates is unknown and
  labeled as such.
- **L-collapse honesty note**: the four L orientations differ only in the
  vertical cell's geometry relative to the horizontal pair; with the
  vertical mate landed by anchor XOR that geometry is random, so all four
  orientations share ONE sampling distribution. Trials still carry the
  intended orientation label for traceability to the survey model.
- **Dropped on principle (unmeasurable by timing)**: adjacency distance
  (no row-distance signal), same-column (no column signal), DQ (R3 —
  unchanged). The G3 query API's "assumed" tier (GeForge monotonic row
  stripes) is **not consumed** by G5: every relation G5 injects is measured.
- **Retired contingencies**: placement steering (VMM pool steering) and
  row-mining top-up were only needed for same-row pairs at measured large
  offsets (the 5 row-class pages, disjoint from workload residency).
  Block-local same-row needs neither — **zero new GDDR measurement in G5**.

## 4. Site sampling rules (per event, per trial)

- SBU: one byte drawn uniformly over the campaign snapshot's resident bytes;
  bit uniform 0–7 (default; override when the survey supplies a bit profile).
- MCU: base drawn the same way; the pattern's other sites derive from it
  (block offsets / anchor XOR). Every site must land inside a byte-residency
  interval of the snapshot — otherwise redraw (anchor first, then base).
  Sites of one MCU stay inside one 2 MiB PA page by construction and may
  legitimately span two allocations sharing that page (the chain is
  per-site).
- All sites of one trial are distinct (byte, bit) pairs — XOR-ing the same
  cell twice would cancel.
- **Measured sampling property (observed on the L1–L5 campaign data,
  2026-09-21)**: SBU sites are uniform over resident bytes (input-binding
  share 2.18% vs its 2.28% residency share — verified), but V/L placements
  additionally require the anchor mate `base ⊕ mask` to land on a RESIDENT
  byte, so on sparsely resident pages (the input binding hosts 602,112 of
  2,097,152 bytes per page) most V/L bases retry and relocate elsewhere:
  the input-binding share among V/L sites is ~0.5–0.7%. This is a
  deterministic consequence of the anchor-mate residency rule, not a
  sampling bug, and is immaterial for this workload (input flips are
  absorbed by INT8 quantization — measured zero output effect at both L1
  and L5).
- Trial-to-trial randomness (the reason for the 100 repetitions): SBU
  byte/bit, MCU base byte/bit, anchor choice, intra-block offsets,
  orientation label, per-site bits. Frozen within a level: B and (s, d, t)
  (§5). RNG seeds recorded per trial — every trial is reproducible offline
  from the snapshot + seed.

## 5. BER levels and frozen bit budgets

**BER semantics**: fraction of the workload's resident bits flipped in one
trial (= one 1000-image evaluation pass with all faults held in place).

R = 26,428,428 resident bytes × 8 = **211,427,424 bits** (workload-
deterministic: same 7 allocations every run; re-verified from each
campaign's snapshot and asserted equal before sampling — a changed R refuses
the campaign and this table is re-derived).

Bit budget **B = round(BER × R)**. Composition: (s, d, t) = non-negative
integers with s + 2d + 3t = B minimizing Σ(share − 60/20/20)² over event
shares, exhaustive over all feasible tuples, ties broken toward more SBU
then more 2-bit. Machine-verified 2026-09-21; G5-T3 re-derives and asserts.

| level | BER | B | effective BER | s (SBU) | d (2-bit) | t (3-bit) | events | event shares % |
|---|---|---|---|---|---|---|---|---|
| L1 | 1e-8 | 2 | 9.46e-9 | 2 | 0 | 0 | 2 | 100 / 0 / 0 |
| L2 | 5e-8 | 11 | 5.20e-8 | 4 | 2 | 1 | 7 | 57.1 / 28.6 / 14.3 |
| L3 | 1e-7 | 21 | 9.93e-8 | 8 | 2 | 3 | 13 | 61.5 / 15.4 / 23.1 |
| L4 | 5e-7 | 106 | 5.01e-7 | 41 | 13 | 13 | 67 | 61.2 / 19.4 / 19.4 |
| L5 | 1e-6 | 211 | 9.98e-7 | 79 | 27 | 26 | 132 | 59.8 / 20.5 / 19.7 |
| L6 | 5e-6 | 1057 | 5.00e-6 | 397 | 132 | 132 | 661 | 60.1 / 20.0 / 20.0 |
| L7 | 1e-5 | 2114 | 1.00e-5 | 794 | 264 | 264 | 1322 | 60.1 / 20.0 / 20.0 |
| L8 | 5e-5 | 10571 | 5.00e-5 | 3964 | 1322 | 1321 | 6607 | 60.0 / 20.0 / 20.0 |
| L9 | 1e-4 | 21143 | 1.00e-4 | 7928 | 2643 | 2643 | 13214 | 60.0 / 20.0 / 20.0 |

(s, d, t) is frozen for all 100 trials of its level — identical bit count
and identical SBU/MCU composition across a level's trials; only positions
randomize (the user's repetition design, to isolate spatial randomness).

**Extension levels L6–L9 (added 2026-09-21, after the L1–L5 campaign)**:
the L1–L5 campaign measured ≤0.30 pp accuracy loss up to BER 1e-6
(top-1 change rate 0.14%→1.03%, power-law slope ≈0.44 in B; three cards,
all VERIFIED). To map the degradation region the same ×5/×2 ladder
continues to 1e-4 with the SAME derivation rule, BER semantics and trial
protocol; the L1–L5 rows above are untouched. Execution is single-card,
justified by the verified L1–L5 card consistency (L1 bit-identical
outcomes across cards; L2–L5 spreads within trial-clustering noise) —
the design's replication axis is trials, not cards. Extrapolation
predictions recorded before running: accuracy ≈94.7 / 94.5 / 93.7 /
93.1% and a ~50% chance of the first DUE at L9 (output-binding hit
expectation 0.64 per campaign; DUE=0 throughout L1–L5 was constructive —
output bindings are ~8 B of the 26.4 MiB residency). Solver note:
`resolve_composition` switches to a windowed search above B=3000 (the
exhaustive grid would cost ~41 s per campaign start at L9); the frozen
literals were produced by the exhaustive solver, the start-time
re-derivation refuses on any disagreement (fail-closed), and the self-test
cross-checks windowed == exhaustive.

**L1 honesty note (flagged for confirmation)**: B = 2 cannot express the
60/40 mix — a single 2-bit MCU alone would be 100% MCU. The frozen tuple is
2×SBU (the least-squares winner), so L1 measures the pure-SBU
low-intensity point. Rejected alternatives, for the record: raising L1 to
the exact 8-bit block (3 SBU + 1×2-bit + 1×3-bit, effective BER 3.8e-8 —
collapses onto L2) or a single 2-bit MCU (0% SBU).

## 6. Trial protocol and repetition

- **One campaign per BER level** = one runner process = one bootstrap
  (observer attaches pre-context; gate-time registry; online snapshot; the
  100 trials reuse that snapshot; never across processes). 5 campaigns
  total, 100 trials each, 500 trials overall. A campaign failure voids only
  its level (fail-closed isolation), co-tenancy recorded never refused.
- Per trial: sample events → XOR **all** sites on device simultaneously →
  1000-image inference with faults held → classify → restore every site
  byte-exactly (byte + whole-allocation compare + guard bytes) → sanity
  inference must reproduce that image's clean output. The sanity inference
  is single-image (`--sample-index`, the G4-T2 convention): INT8 execution
  is deterministic, so one strictly-reproduced output plus the byte-level
  restore proofs above cover no-residue; re-running all 1000 images per
  trial would add no information. Campaign-close checks inherit the G4-T2
  skeleton (registry consistency, gate-vs-final map diff = mid-run remap
  detector, 0 lost BPF events).
- Classification (vocabulary from G4-T2; final lock in T3): each image is
  compared against the campaign's clean pass — SDC_NUMERIC if any output
  deviates numerically, SDC_TOP1 if any top-1 changes, DUE_INVALID_OUTPUT
  if any invalid output/abort, BENIGN if all outputs bit-identical. Trial
  label = precedence DUE > SDC_TOP1 > SDC_NUMERIC > BENIGN; per-image
  deviation counts are logged for G6.
- Restore verification is per allocation class (held-fault semantics): the
  input binding is compared byte-exact against the LAST image the pass
  evaluated (fail-closed — the pass re-stages it per image); the output
  bindings are skipped (engine-owned: rewritten by every enqueue, the flip
  itself is proven by the pre-pass guard compare); TRT-internal regions get
  an informational compare against pristine (weights return exact; engine
  scratch may legitimately differ — soft-upset semantics, recorded, the
  behavioral no-residue proof is the sanity inference).

## 7. Campaign log (G6 feed)

One row per trial plus one row per site, joined by (level, trial_index,
event_index, site_index): pattern class and orientation label, multiplicity,
per-site chain (PA page, in-page offset, bank component, anchor mask where
used, provenance labels, allocation_id, semantic_label, byte_offset, bit,
before/after/xor_mask), trial outcome + per-image deviation counts, restore
verification flags, sanity flag, RNG seed, and the frozen (B, s, d, t) echo.

## 8. Fixed defaults (user-approved by silence)

- bit-in-byte: uniform over 0–7;
- sanity inference after every trial;
- per-trial restore verification at the G4-T2 level (byte + allocation +
  guards);
- co-tenancy: recorded, never refused (unchanged).

## 9. Sources

Ratios (60/40, 2:3 = 1:1) and the spatial-pattern set come from the user's
G5-T0 literature survey (2026-09-21). **Reference list: to be inserted by
the user.**

## 10. Validity boundary

- GDDR faults are materialized as device-memory bit flips; the cache
  hierarchy is outside the fault model (plan §8).
- TRT-internal sites carry allocation-level attribution only (R4).
- Non-resident cells are unreachable → BENIGN by construction
  (resident-set-only site selection).
- Every relation G5 injects is measured; the assumed tier is unused;
  unknowns (row distance, column identity) are labeled per site, never
  guessed.
