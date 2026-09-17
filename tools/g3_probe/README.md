# G3 Probe — S1 timing-channel calibration

Implements step S1 of the G3 plan (`G3_PA_TO_GDDR_PLAN.md`): prove that the
row-buffer hit / different-bank / row-conflict latency regimes are
statistically separable on this GPU, before any address-mapping collection
starts. The method and its provenance are cataloged in
[docs/G3_SURVEY.md](../../docs/G3_SURVEY.md); the pinned reference
repositories are unlicensed, so this is an independent implementation.

S2 adds the PA-annotated timing pool in this directory plus the orchestrator
`tools/g2_observer/run_g3_pool_probe.py` — see "S2 pool" below.

## Method

One warp, two threads; both time one address each in the same warp
instruction, so a same-bank different-row pair pays the row-conflict latency
on both sides. Each timed access is preceded by `discard.global.L2 [addr],
128` (invalidates the line without writeback — the pool is scratch and its
contents are destroyed), then `clock64()` brackets a hand-written PTX load
(`.volatile` by default; six modifiers selectable). Every point is the
minimum over `--iters` kernel launches — noise only adds delay — after a
warm-up loop that stabilizes DVFS, since clocks are not locked (no root).
A `clock64`-vs-`%globaltimer` probe reports the effective SM clock.

The `scan` mode sweeps candidate offsets inside the pool's first 2 MiB —
one GMMU page whose in-page offsets translate 1:1 into framebuffer PA
(G2 finding), so any same-bank periodicity found inside the page is
periodicity in PA, without needing the observer yet.

## Usage

```bash
make -C tools/g3_probe all check

CUDA_VISIBLE_DEVICES=0 tools/g3_probe/g3_timing_probe info   # device + idle clock
CUDA_VISIBLE_DEVICES=0 tools/g3_probe/g3_timing_probe rate   # effective SM clock
CUDA_VISIBLE_DEVICES=0 tools/g3_probe/g3_timing_probe floor  # same-address floor
CUDA_VISIBLE_DEVICES=0 tools/g3_probe/g3_timing_probe scan --file /tmp/s1_gpu0.csv
python3 tools/g3_probe/analyze_scan.py /tmp/s1_gpu0.csv --mhz <measured MHz>
```

`analyze_scan.py` reports the modal baseline, the top-cluster conflict mode,
amplitude, robust spread, separability, and the gcd of conflict-offset
gaps, then evaluates **gate G3-R1**: the top ~0.7% latency cluster must sit
≥ 5× the baseline spread above the modal baseline and be a tight mode
(≤ 3× baseline spread). Exit codes: 0 passed, 2 failed, 1 error (repo
convention). `--self-test` pins the analyzer logic on synthetic
bimodal/unimodal/contaminated scans.

## S1 result on GPU 0 (2026-09-16)

Gate **G3-R1: PASS** (`artifacts/g3/s1_scan_gpu0.csv`, 32768 points, 2 MiB
in-page scan, `.volatile`, unlocked clocks self-boosted and stable at
2520 MHz):

- same-address floor 1017–1027 cycles (~407 ns); modal baseline 1024 cyc;
- conflict cluster: 251 points (0.77%), median 1123 cyc, cluster spread
  only 4.4 cyc — amplitude **98 cycles ≈ 39 ns**, the GDDR6X row-conflict
  delta, consistent with the 43 ns tRC prior measured on GDDR6 (A6000);
- the conflict offsets (852224, 852736, 868352, 868864, …) **match the
  conflict set published for the GA102 A6000**, so GA102 and AD102 share
  the in-page bank structure at these offsets;
- targeted pair checks: (0,0)=1021, (0,8192)=1014, (0,852224)=1145,
  same-row pair (852224,852736)=1043 — row-hit/baseline/conflict regimes
  are programmatically distinguishable. Two addresses that both conflict
  with a third need not conflict with each other ((852224,866304)=1044);
  that structure is S3 material, not an S1 anomaly.

## S2 pool — PA-annotated timing harness

Files: `g3_timing_kernels.cuh` (shared kernels), `g3_pool_harness.cu`
(gated child), `g3_pool.py` (PA query selector), and the orchestrator
`tools/g2_observer/run_g3_pool_probe.py` + `--api g3pool` in
`scripts/run_g2_observer_probe.sh`.

The reference tooling times a VA-only black box; our pool is PA-annotated.
The G2 observer watches the pool allocations, so every 2 MiB GMMU page of
every chunk gets its framebuffer PA **before any timing runs**: the
orchestrator builds the per-chunk ledger with the validated G2 code
(containment matching, complete valid local-VIDEO PTE payloads, gapless
tiling) and only then writes the work CSV and opens the release gate. The
harness warms up (DVFS) while waiting, times the requested pairs, and
writes `result.csv`; after teardown the strict ledger re-runs and
`pool_map.csv` records one row per page (VA page extent ↔ fb PA base).

Because in-page offsets translate 1:1 into PA (G2 finding), `g3_pool.py`
turns page selection into PA-bit steering: PA bits [0:21) via the in-page
offset, bits 21+ via page choice. `select_sanity_queries` emits the
S1-verified in-page triple plus cross-page pairs spread over the observed
PA range; `select_single_bit_pairs` (S3) emits pairs whose PAs differ in
exactly one chosen bit. `g3_pool.py --self-test` pins the arithmetic.

Run (GPU must be idle; the orchestrator refuses co-tenant compute):

```bash
make -C tools/g3_probe all check && make -C tools/g2_observer check
sudo scripts/run_g2_observer_probe.sh --api g3pool --device 0
# artifacts land in artifacts/g3/pool/run_pool_gpu0_*/:
#   pool_map.csv  work.csv  result.csv  events.csv  summary.json
```

Standalone smoke (no observer, no sudo — manual gate/release files) passed
on GPU 0 (2026-09-16): floor/baseline/in-page-conflict = 1010/1010/1122
cycles, cross-chunk pairs 1017/1035, 2520.3 MHz — consistent with S1.

## S2 result on GPU 0 (2026-09-16)

Gate **S2: PASS** (`artifacts/g3/pool/run_pool_gpu0_1789545813750322478`,
64 chunks x 8 MiB, status `G3_POOL_PA_MAP_COMPLETE_OBSERVED`, 0 lost
events, 0 failures):

- `pool_map.csv`: 256 pages, every one valid local-VIDEO, gapless tiling
  per chunk; all 256 framebuffer PAs distinct, forming one **fully
  contiguous 512 MiB PA block** (0x1ee00000..0x3ec00000, every sorted
  step exactly 2 MiB).
- Within every chunk the 4 pages are PA-contiguous, and chunk PA bases
  step exactly 8 MiB in **allocation order** — but chunk **VA** order is
  scrambled (cudaMalloc hands out VAs unordered), so VA→PA is *not*
  monotone: observed PTEs are mandatory, VA arithmetic would mislabel
  pages. This is the premise S3 relies on.
- Timing (min over 10 launches, `.volatile`, 2675.7 MHz): floor 1023 /
  different-bank baseline 1014 / in-page row-conflict 1140 cyc — conflict
  amplitude 126 cyc ≈ 47 ns, consistent with S1's 39 ns at a lower boost
  clock; the in-page conflict offset is again 852224 (0xd0100).
- All 7 cross-page pairs (PA deltas 72–510 MiB) land at 1038–1098 cyc,
  between baseline and conflict — a spread of regimes exactly as
  expected when PA-distant pages sometimes share a bank. These are
  sanity/informational numbers; systematic collection is S3.

## S3 bit-scan — single-bit PA pair collection

`--work-mode bit-scan` turns the pool into the S3 measurement matrix:
after the same gated PA annotation, the work CSV is the calibration triple
(ids 0..2: floor / different-bank / in-page conflict) followed by one
single-bit pair per PA bit — in-page bits [0:21) from several base pages
(votes across bases expose nonlinear hashing), page-level bits from the
page shift up to the pool's top PA, up to `--pairs-per-bit` pairs each
evenly spread over available XOR partners. A 512 MiB pool covers bits
21–28 at page level (599 queries); a 4 GiB pool reaches bit ~31.

```bash
sudo scripts/run_g2_observer_probe.sh --api g3pool --device 0 \
    --work-mode bit-scan --chunks 512          # 4 GiB pool
python3 tools/g3_probe/analyze_bit_scan.py \
    artifacts/g3/pool/<run_dir>                # writes constraints.csv
```

`analyze_bit_scan.py` classifies each pair against the run's own
calibration triple — **low** (no conflict: column bit, or bank/channel
bit — the pair primitive cannot tell these apart yet), **mid** (shoulder),
**conflict** (same bank different row: a row-address bit outside the bank
hash) — flags asymmetric pairs, checks every pair's PA xor really is one
bit against the pool map, and prints the per-bit vote table (SPLIT marks
bits whose votes disagree across bases). `constraints.csv` is the input
for the S4 solver.

## S3b pair-scan — anchored and two-bit probes

The single-bit scan cannot tell a column bit (flip keeps bank and row)
from a bank-hash bit (flip leaves the bank): both time low. S3b fixes
this with the S1 conflict anchor `0xd0100` — an in-page mask whose pairs
are same-bank different-row, so XOR-ing it into any pair makes the row
differ unconditionally. Where the anchor is verified to hold at x, the
anchored probe `(x, x^M^(1<<b))` conflicts exactly when flipping b kept
the bank: column bits stay conflict on every base, bank bits drop to
low, mixed votes across bases are nonlinear-hash evidence. Two-bit pairs
`(x, x^(1<<b1)^(1<<b2))` test additivity (under a linear hash two bank
bits cancel back to conflict).

```bash
sudo scripts/run_g2_observer_probe.sh --api g3pool --device 0 \
    --work-mode pair-scan --chunks 512        # 4 GiB pool, ~1600 queries
python3 tools/g3_probe/analyze_pair_scan.py \
    artifacts/g3/pool/<run_dir>               # writes pair_constraints.csv
```

The analyzer regenerates the query plan from the run's own pool map and
the recorded selection parameters, checks every work.csv row against it,
then prints: the per-in-page-bit anchored vote table
(bank_kept/bank_changed — the column-vs-bank split), the two-bit pair
class summary, the anchor sweep coverage (where `0xd0100` keeps the bank
across the PA range — seed structure), and per page-level bit the
anchored votes plus fresh single-bit votes (a stability check on S3
without a separate run).

## S3b result on GPU 0 (2026-09-16)

Run healthy end to end (`artifacts/g3/pool/run_pool_gpu0_1789550420238116565`,
1587/1587 rows, 0 lost events, 0 asymmetric, 2716.6 MHz, the same 4 GiB PA
hole as S3; calibration floor/baseline/conflict = 1028/1020/1143, amplitude
123 cyc):

- **Anchor validity is position-dependent.** Only 1 of the 4 in-page bases
  keeps the bank under `0xd0100` — the S1 page family (1144/1144) — while
  the other three drop to low/mid. The anchor sweep finds 35/128 pages
  valid (32 mid / 61 low). Under a linear bank hash a fixed mask is either
  always or never bank-preserving, so a 27% mix rules the fixed-support
  linear model out by direct measurement. Caveat: the stride-16 sampling
  leaves PA bits 21–24 constant, so 27% is the rate within that slice; the
  page-triple anchors cover other slices at 3–10 valid of 16.
- **In-page bits 0–20 split at the anchor-valid base**: bank_kept (column
  candidates) 0–7, 9, 11 (conflict stays 1144–1148); bank_changed (hash
  support) 8, 12–14, 16–18, 20, with 8/16/17/18 dropping deepest
  (1022–1023); mid 10 (1068), 19 (1077), 15 (1104), and bit 11's conflict
  vote (1115) clears the threshold by only 9 cyc. The anchor's own bits
  {8,16,18,19} all land in bank_changed/mid — self-consistent, the anchor
  works precisely by flipping row bits without leaving the bank.
- **Two-bit pairs (840: 66 conflict / 659 low / 115 mid, per base
  17/6/25/18)**: conflicts are "column bit × row-carrier bit" pairs whose
  carrier rotates with the base — bit 10 carries at base 2, 11 at base 0,
  16 at base 3, 19 at base 2 — plus a local cancellation cluster at base 0
  ((7,12), (11,13), (11,14), (12,13), (12,14), (13,15), (15,18)). Bit 16
  flipping the bank at base 0 (anchored 1022) yet carrying the row at base
  3 ((0..9,16) all conflict) is direct evidence that the hash support
  moves with the address seed: locally low-degree, coefficients seeded.
- **Page bits resolved**: strong row bits 25/27/28/29 (anchored kept +
  fresh single conflict: 5/5, 6/6, 3/3, 3/3), bank bits 22 (changed 5,
  low 5) and 26 (changed 4), column-like fold 21 (kept 2 + low 2),
  row-leaning mixed 24/30, seed-mixed 23/31. The sweep samples pages with
  PA bit 32 set (top PA 0x114e00000), so the seed includes bit 32 even
  though no query flips it directly.
- `pair_constraints.csv` (1587 rows) + S3 `constraints.csv` (788) are the
  S4 solver input: bank = degree-≤2 GF(2) hash with seeded terms, row
  support learned jointly, mid votes kept soft.

## S4 solver — fit the bank/row model to the pair constraints

`solve_mapping.py` consumes the S3 `constraints.csv` and S3b
`pair_constraints.csv` (run directories or CSV paths — to the solver both
are just (pa_a, pa_b, class) triples) and fits the two functions the
timing primitive measured:

    conflict(x, y)  <=>  same_bank(x, y) AND row(x) != row(y)

Both sides are degree-≤2 GF(2) functional families over the feature
difference phi = mu(x)^mu(y) (a quadratic term `pa_i*pa_k` is exactly
"seed bit i × flipped bit k"): **bank** = every bank functional vanishing
(same bank), **row** = some row functional firing (row differs — real
decoders fold bank bits into the row address, so the row side is seeded
too). Conflicts constrain the bank family homogeneously; lows predicted
same-bank constrain the row family homogeneously; each family must fire
the other side's set. Valid functionals (vanishing on the positive set,
an XOR-closed family) are enumerated exactly to weight 3 via conflict
signatures — equal-signature column pairs, signature-completing triples —
and to weight 4 at stalls by anchoring 1-3 terms inside the uncovered
negative's own feature support (real bits are "linear core ^ seed quads",
often with a single in-support term) completed by signature lookup or a
signature-pair index. A parsimony gate rejects narrow cover (spurious
functionals fire only a stray negative; leftovers stay residuals); bank
weight-1 candidates must be linear terms (output bits have linear cores;
the row side is exempt — row folds can be pure quads). The two stages
alternate; mid pairs are never fitted, only scored.

    python3 tools/g3_probe/solve_mapping.py \
        artifacts/g3/pool/<s3_run> artifacts/g3/pool/<s3b_run> \
        [--degree {1,2}] [--holdout 0.2] [--seed N]
    python3 tools/g3_probe/solve_mapping.py --predict <model.json> \
        0xPA_A 0xPA_B

Output: per-bit term lists for both families, train misclassifications and
unseparable counts, train/holdout accuracy with the ≥95% holdout gate
(the S5 preview), and `mapping_model_d{1,2}.json` (schema v2) for the
prediction API. `--self-test` pins the pipeline on a synthetic seeded
truth whose bank side includes a weight-4 linear-core bit and whose row
side includes a bank-fold quad: degree 2 must reach zero train
misclassifications with ≥99% train/holdout/class accuracy on fresh pairs
(the internal bank/row factorization is only identified up to the labels'
resolving power — predicate-level bars sit at 95%), and degree 1 must
report the insufficiency.

## S4 result on GPU 0 (2026-09-16): model class insufficient — gate FAIL

The solver recovers the synthetic truth exactly (self-test), but on the
real S3+S3b constraints (2375 pairs) **no variant reaches the 95% gate**:
degree 1 degenerates to the low base rate (0.782 — no structure found),
degree 2 with a linear row support reaches 0.774, the full two-family
seeded model 0.758. Diagnosis, in order of discovery:

- **The linear row model is dead by linear algebra**: 43% of lows (619)
  lie inside the GF(2) span of the conflict feature vectors, and every
  functional vanishing on all conflicts vanishes on them — their low
  label can only mean *same bank AND same row*. 216 of them carry
  exactly the anchor mask bits {8,16,18,19} (the anchor-invalid probes).
  A fixed "xor touches R" row predicate cannot express that; hence the
  seeded row family.
- **The labels are reproducible, not noise** — S3b's fresh single-bit
  votes reproduce S3's (5/5, 6/6), and low latency shows no clean drift
  with PA distance from the calibration page.
- **Weight ≤4 degree-2 functionals cannot separate the anchored
  structure**: e16/e18/e19 each fire ~130 anchored in-page conflicts but
  are blocked by ~190 anchor-invalid lows; the strong row bits
  25/27/28/29 fire ~17 page-level conflicts each but are blocked by
  23-38 lows predicted same-bank. The seed structure that would split
  those needs higher weight or degree ≥3 terms.
- 556 mids (23% of pairs) are excluded from fitting — the shoulder band
  eats exactly the informative near-threshold pairs.

Next data step (S4b): region-local calibration triples (per-pool-region
thresholds instead of one global triple) to shrink the mid band, plus
blocker-targeted probes at anchor-invalid pages (the pairs that block the
e16/e18/e19 and page-row-bit functionals) to decide between "row fold is
degree ≥3" and "bank coverage gap". The saved models and this diagnosis
are the honest S4 output; S5 prediction validation stays blocked on a
model that passes the gate.

## S4b-0 result on GPU 0 (2026-09-16): local recalibration + lambda fingerprint

`analyze_local_recal.py` (pure offline, zero new collection) reclassifies
both runs against a per-page baseline and mines the address-dependent
latency fingerprint:

- **The lambda fingerprint is real.** The per-page low-pair latency spans
  1011-1063 cycles (~50 cyc ≈ 19 ns at 2676 MHz), reproducible within a
  page (median within-page spread 7-9 cyc, far below the global spread),
  with no single page-level PA bit explaining more than ~2 cyc — hash-like
  positional variation, consistent with channel/L2-slice path
  differences. No discrete ~12-band structure is resolvable from pair
  data (a pair's single scalar mixes both endpoints; the harness records
  one value per pair — `cycles_a == cycles_b` in every row).
- **The local rule is asymmetric, and that is a measured fact.** The first
  symmetric version (shift both band edges by the page lambda) relabeled
  134 of 139 S3 conflicts to mid — rejected: conflict values sit 30-60
  cycles BELOW lambda+amplitude (conflict-minus-amplitude lands at
  981-1001 while the lambda range is 1011-1063), so the row-conflict
  penalty does not ride on the low-path baseline. The final rule
  localizes only the low/mid boundary (low iff value < max(lambda_a,
  lambda_b) + 0.35*amplitude); the conflict gate stays global — S3b's
  fresh votes had reproduced S3's conflict labels (5/5, 6/6), so
  destroying them is wrong by construction.
- **Outcome**: mids 556 -> 257 (-54%), all 395 conflicts kept (134 S3 +
  139 S3b sit below their own local conflict threshold — flagged as S4b-1
  re-probe candidates, not relabeled), 11 marginal lows corrected to mid.
  The 257 residual mids are saved per run as same-channel candidates
  (`same_channel_candidates.csv`): intra-channel bank-group pipelining
  pays a partial penalty while a cross-channel pair has nothing to pay,
  so a residual mid is positive same-channel evidence.
- **Solver impact**: holdout 0.758 -> 0.809 on the recalibrated labels
  (`mapping_model_local_d2.json`). Real progress, still FAIL against the
  0.95 gate — label quality was a blocker, but the degree-2/weight-4
  model-class insufficiency stands, exactly as the S4 diagnosis predicted.
- **The channel partition is not resolvable offline.** The same-channel
  graph (conflict + residual-mid edges over pages) has 147 small
  components (largest 22 pages): no contradiction with ~12 channels, but
  this edge density cannot merge each channel's ~170 pages. Deciding the
  channel hypothesis needs the S4b-1 per-page census.

## S4b-1 — per-page census, suspect re-probes, row pilot

`--work-mode census` (`plan_census_queries` in `g3_pool.py`, analyzed by
`analyze_census.py`): one run answers the three questions S4b-0 could not.
Sections (contractual order): `calibration` 3 / `self` one (p,p) pair per
pool page — a clean per-page lambda, ONE endpoint per scalar, unlike pair
data / `self_second` the same pair 1 MiB into every 4th page / `repeat`
the first 64 self pairs again late (drift anchor) / `reprobe` the S4b-0
suspect-conflict pairs (PA-level reverse lookup into this run's pool) /
`row_pilot` all unordered pairs of `{0, M, probe, M|probe}` at pages the
S3b anchor sweep classified conflict.

```bash
sudo scripts/run_g2_observer_probe.sh --api g3pool --device 0 \
    --work-mode census --chunks 512 \
    --reprobe-csv <s3_run>/suspect_conflicts.csv \
    --reprobe-csv <s3b_run>/suspect_conflicts.csv \
    --row-pilot-from <s3b_run>
python3 tools/g3_probe/analyze_census.py <census_run> \
    --compare-old <s3_run>
```

The analyzer reconstructs sections from the summary's census_selection
counts, verifies every row's shape against its own PA (fail-closed),
applies the late-section offset (below), and writes `page_lambdas.csv`,
`reprobe_verdicts.csv`, `row_pilot_classes.csv`. The pilot unions banks
only on DEEP conflicts (>= 0.90 amplitude) — the shallow shoulder band
must never be same-bank evidence — then rows on in-bank lows.

## S4b-1 result on GPU 0 (2026-09-16): the conflict class was bimodal

Run healthy end to end (3164 queries, 0 lost events, 2712.9 MHz) and the
same 4 GiB PA hole reproduced a THIRD time — all 273 re-probe pairs and
all 4 pilot pages resolved, 0 drops.

- **Lambda is real, spatial, and NOT a channel observable.** The clean
  per-page lambda spans 1001-1129 cycles with the self block internally
  flat across position (no time trend), so the spread is spatial. No
  ~12 discrete bands exist (gap-4: two clusters, 83%/17%); both clusters
  are fine-grained per-page placement — 249 of 512 chunks mix fast and
  slow pages, no PA bit moves cluster share by more than 0.02 — nothing
  like a coarse channel partition. The S4b-0 lambda-band channel test is
  falsified by measurement. (The S4b-0 pair-derived lambdas barely
  correlate with the census lambda, r=0.11: pair minima were noisy upper
  bounds.)
- **A late-section step, not drift.** Everything measured after the self
  block reads ~-18 cycles vs the same pages inside it (repeat block
  median; the self block's flat bucket medians rule out a ramp). The
  analyzer corrects self_second/reprobe/row_pilot additively.
- **The re-probes split the old conflict class in two.** Of the 273
  S4b-0 suspect conflicts: 228 land in a reproducible SHALLOW band
  (corrected p10-p90 = 1104-1120, ~0.70-0.85 amplitude), 36 fall to low
  (not reproducible), 9 mid — and ZERO reach deep conflict. Applied
  retroactively with a 0.90 gate: S3's 139 "conflicts" were ALL shallow;
  S3b splits 185 shallow / 71 deep (>= 1139). The two-band classifier
  conflated a partial-penalty regime with full row conflicts, and the
  S4 solver's same-bank GF(2) constraints were that mixture — a concrete
  cause of the gate FAIL, now removed. Bonus structure: the deep tail
  1200-1233 (22 pairs) clusters at pages = 7 mod 16 (2 MiB pages) —
  32 MiB-periodic super-conflicts, S4b-2 material.
- **Row pilot (0 contradictions).** At the 2 deep-anchor pages the
  transitive classes are exactly `{0, 0x200}` vs `{M, M|0x200}` per row:
  bit 9 is confirmed a column bit at the hardware level, the anchor
  flips the row, and the non-anchor hash probes leave the bank group.
  At the 2 shallow pages the anchor pair pays only the shoulder — their
  S3b "conflict" was same-channel different-bank evidence, and "anchor
  validity" was the deep/shallow split all along, not a linear-hash
  property.
- **Updated physical model** (the S4b-2 solver input): low = different
  channel or same row; shoulder ~1114 = same channel, different bank /
  bank group; deep >= 1139 = same bank, different row. The shoulder band
  gives same-channel candidates a large positive set (228 re-probed +
  185 S3b shallow + residual mids): the channel partition is now a graph
  question over shoulder edges, not a lambda question.

## S4b-2 — three-band relabel, channel graph, re-solve

`analyze_three_band.py` relabels the S3/S3b constraints with the S4b-1
census per-page lambdas (same PA hole — the census page table joins
1:1; any uncovered constraint page refuses the run, exit 2) and the
three-band rule (deep >= 0.90*amp global; shoulder >= 0.70*amp global;
low < max(lambda_a, lambda_b) + 0.35*amp). It writes
`<name>_bands.csv` (four classes) and `<name>_solver3.csv` (solver
view: deep -> conflict, shoulder -> mid).

```bash
python3 tools/g3_probe/analyze_three_band.py <s3_or_s3b_run> \
    --census <census_run>
python3 tools/g3_probe/analyze_channel_graph.py \
    <s3_run>/constraints_bands.csv <s3b_run>/pair_constraints_bands.csv \
    --census <census_run>
```

`analyze_channel_graph.py` builds the same-channel page graph (shoulder
+ deep cross-page edges both prove same channel; deep = same bank
implies same channel). A cross-page low inside a component is a
contradiction: two distinct 2 MiB pages cannot share channel+bank+row
(they differ in PA bits >= 21, which the in-page column field cannot
absorb), so same-channel implies not-low. The tool reports component
count/sizes vs ~12 channels, the contradiction rate, per-component PA
bit signatures, super-tail (7 mod 16) membership, census-lambda mixing,
and writes `channel_components.csv`.

## S4b-2 result on GPU 0 (2026-09-16): clean labels, honest solver FAIL

- **The retroactive split reproduces the census estimate exactly**: S3's
  139 conflicts are ALL shoulder; S3b is 185 shoulder + 70 deep. And
  the census lambda nearly erases the old mid band (S3 241 -> 5, S3b
  315 -> 124): most S4/S4b-0 "mids" were lambda smear around a low that
  the noisy pair-lambda estimate could not localize — they are now
  correctly low (1851 lows of 2375 pairs).
- **The channel partition signal exists but is under-determined.** 380
  cross-page same-channel edges (379 shoulder + 1 deep; the census
  re-probe shoulders fold in via `--census`) touch 250 pages and give
  82 components, largest 13 pages (5%): consistent edges (only 28/845
  = 3.3% of cross-page lows contradict a component, and those mark
  individual bad edges, not a broken rule) but far below the density
  needed to merge each channel's ~170 of 2048 pages. The two largest
  components carry strong high-bit signatures (bit 27/28 100%, bit 31
  92%; bit 24-26 exclusive), and the census lambdas MIX inside them —
  the partition is address-structural, not lambda-structural, closing
  the loop on S4b-1's lambda finding.
- **Deep is in-page, shoulder is page-level**: 69/70 deep conflicts have
  both endpoints inside one 2 MiB page (the bank hash is fed by PA bits
  < 21), while flipping a page-level bit yields shoulder (same channel)
  or low (different channel). Channel selection lives at high PA bits;
  bank hashing lives at in-page bits — one clean architectural fact the
  bimodal split bought.
- **The solver still cannot fit the deep set — and the gate now says so
  honestly.** On the 3-band labels (71 conflicts / 1851 lows / 453
  mid-excluded) degree 2 finds 18 bank functionals (all vanishing on
  train conflicts) but leaves 47 of ~57 train conflicts unexplained by
  any row functional; holdout accuracy 0.9557 would numerically PASS —
  but holdout conflict recall is 0/14: with conflicts at 3.6% of hard
  pairs, all-low clears 0.95 on base rate alone. The report now prints
  per-class recall and flags this as a DEGENERATE PASS (added to
  `solve_mapping.py`; S5 stays blocked). The S4 verdict stands with
  clean labels: the degree-2/weight-4 model class is insufficient for
  the seeded row fold, and 70 structurally-similar deep conflicts do
  not span it.

## Validity boundary

- Latencies are cycle counts from one SM; conversion to ns uses the
  measured clock of the run and is only indicative without locked clocks.
- S1 measures latency structure only. It does not observe physical
  addresses and claims nothing about the mapping function.
- The pool must be alone on the GPU: co-tenant traffic corrupts timing
  (check `nvidia-smi` first; the probe prints pool/device state).
- `discard.global.L2` destroys pool contents by design; never point this
  tool at buffers whose contents matter.
- S2 timing numbers in `summary.json` are informational: S2 proves the
  pool map and the gated timing path, not mapping rules. The PA in
  `pool_map.csv` is the G2-observed page-table claim for that run only —
  never reused across runs.
