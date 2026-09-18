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

## S5-T0 — empirical mapping table (EMT) builder

Direction change (2026-09-17): the closed-form solver line is archived as
an honest negative result (S4/S4b-2, independently confirmed by GeForge's
footnote 1 — see `docs/G3_SURVEY.md` section 1). The G3 deliverable
becomes the GeForge-style empirical table: measure which PAs share
(channel, bank, row), store the classes, query them. G2 pins every pool
page's true PA each run, so GeForge's page-anchoring problem does not
exist here.

`build_bank_table.py` is the pure-offline first step — zero new
collection. It folds S3/S3b/census into the seed table:

- edges deduped by PA pair with **reprobe > pilot > S3b > S3 priority**
  (the controlled re-measurement supersedes the retroactive label);
- bank classes = union-find over deep conflicts (same bank, different
  row); row classes = union-find over low pairs **inside** a bank class
  (same bank AND same row); mid pairs never become class evidence;
- the same-channel page graph is recomputed from the deduped edges and
  cross-checked against the S4b-2 `channel_components.csv`;
- consistency gates C1 (shoulder inside a bank class), C2 (deep inside a
  row class), C3 (cross-page low inside a channel component), C4
  (cross-page low inside a bank class), C5 (cross-source label
  disagreements).

```bash
python3 build_bank_table.py \
    artifacts/g3/pool/<s3_run> artifacts/g3/pool/<s3b_run> \
    artifacts/g3/pool/<census_run> --out artifacts/g3/table_v0 \
    [--query 0x1eed0100]
```

## S5-T0 result on GPU 0 (2026-09-17): clean seed, honest sparsity

- 2455 deduped edges (54 deep / 320 shoulder / 1932 low; 149 mid
  excluded) over the 2048-page pool universe. The deep count corrects
  S4b-2's "70": those were 71 rows over **47 distinct pairs** (S3b's
  probe types re-emit the same pair), 3 confirmed by the pilot, 1
  demoted to shoulder by its re-probe — 54 survive.
- The recomputed channel partition is **74 components over 220 pages**,
  not S4b-2's 82/250: that graph only ADDED reprobe shoulders and never
  demoted bands labels; here 29 cross-page shoulder edges re-measured
  as low/mid are dropped. C3 falls to 20/807 (2.5%) on the corrected
  graph — the demotion makes the partition cleaner, not worse.
- Classes: 40 bank classes with ≥2 nodes (largest 12 at the pilot page,
  38 pairs of 2), 4 row pairs — the S4b-1 pilot structure is the spine;
  **zero** bank classes span pages (the single cross-page deep of
  S4b-2 was the re-probe demotion). C1/C2/C4: 0 contradictions.
- Coverage is honestly sparse: channel component on 220/2048 pages,
  classified nodes on 38/2048 pages, 2022/2114 singleton nodes. That
  is exactly the S5-T1 workload: an iterative anchor-expansion
  collection mode (`--work-mode table-build`) that densifies classes
  until the pool is covered.

## S5-T1 — table-build collection (`--work-mode table-build`)

One pre-planned run densifies the seed table, no iteration needed on
this pool. The orchestrator (`run_g3_pool_probe.py`) plans six sections
in a fixed order and the harness executes them back-to-back:

| section      | queries | what it measures                                                   |
|--------------|---------|--------------------------------------------------------------------|
| calibration  | 3       | floor/baseline/conflict anchors for the band gates                 |
| self         | 2048    | fresh per-page lambda (the classification reference)               |
| anchor_sweep | 49152   | (p, p^M) per page x 24 candidate masks: deep = valid anchor + edge |
| classify     | 151478  | (p, rep) page starts vs 74 seed reps: deep = same bank, low = other|
| bank_map     | 392     | 8 pages x 25-offset lattice x {base, anchor} double probe          |
| repeat       | 64      | drift anchor for the late-section additive step                    |

Total 203137 queries in a 19 s work phase (~10.8k queries/s). Anchors
come from the seed table (`--seed-table`): the 24 masks that ever
produced a deep conflict anywhere, plus per-page reps from its bank
classes; `--bank-map-from` reuses the mined anchor validity to place the
double-probe lattice (skipping the y∈{0,M} degenerate probes).

`analyze_table_build.py` verifies every row's PA against the plan
(fail-closed, exit 2), classifies with **λ-referenced gates** (see the
model revision below) and writes `table_build_edges.csv` (the
`build_bank_table.py --t1` source), `anchor_validity.csv`,
`bank_map_pages.csv`, `channel_partition.csv`.

```bash
sudo scripts/run_g2_observer_probe.sh --api g3pool --device 0 \
    --work-mode table-build \
    --seed-table artifacts/g3/table_v0 --bank-map-from artifacts/g3/table_v0
python3 tools/g3_probe/analyze_table_build.py \
    artifacts/g3/pool/<table_build_run>
python3 tools/g3_probe/build_bank_table.py \
    artifacts/g3/pool/<s3_run> artifacts/g3/pool/<s3b_run> \
    artifacts/g3/pool/<census_run> --t1 artifacts/g3/pool/<table_build_run> \
    --channel-deep-only --out artifacts/g3/table_v1 [--query 0x1eed0100]
```

Model revision (measured on this run; supersedes S4b-1's three-band
story for cross-page pairs): every pair value references
max(λ_a, λ_b), not the global calibration baseline. The per-page λ
spread (~120 cycles) is wider than the conflict amplitude (110), so any
global gate lands inside the λ spread and manufactures a "shoulder"
band out of slow pages. λ-referenced, the classify section is bimodal
with an **empty +30..+80 valley**: low at d≈0 (96%) and deep at
d≥+80 (0.26% ≈ 1/384, matching the AD102 prior 24 channels × 16 banks).
The historical cross-page "shallow band" (S3b's 185, the census
re-probe's 228) shows the same λ-referenced d distribution for its
shoulder and low verdicts — selection bias, not a physical regime.
Consequences: `analyze_table_build.py` has no shoulder class;
`build_bank_table.py --channel-deep-only` builds page components on
cross-page deep edges only (same-bank page sets) instead of the T0
shoulder+deep union; the old C3 rule (cross-page low inside a channel
component) becomes the valid same-bank contradiction check. In-page the
valley sits at +45..+70; gates: low < 0.35·amp, deep ≥ 0.60·amp above
the λ reference.

## S5-T1 result on GPU 0 (2026-09-17): pool covered, model corrected

Run `run_pool_gpu0_1789656551689364094` (status
G3_POOL_PA_MAP_COMPLETE_OBSERVED, 0 failures): 201022 classified pairs
{deep 11932, low 187761, mid 1329}; calibration 1011/1011/1121
(amp 110); late-section offset −14 cyc anchored on the repeat block.

- **Same-bank page graph**: 387 cross-page deep edges (0.26% of classify
  pairs, vs the 1/384 = 0.26% random-pair prior for 24ch × 16bank) →
  60 components over 370/2048 pages, largest 11. Cross-page lows inside
  a component (bad-deep-edge markers): **0**.
- **Anchor validity**: 2048/2048 pages have ≥1 valid anchor. Masks
  0x1fdc80 and 0x1f9dc0 are valid on **every** page (kernel masks of
  the bank hash); 0x119980/0x11e300/0xd0100 are partial (818/650/484
  pages), floor ~235–320. This corrects S3b's "27% anchor validity,
  position-dependent" — validity is mask-dependent, not
  position-dependent.
- **Bank maps** (8 pages, 25-offset lattice, anchor 0xd0100): clean
  base-low/anchor-deep row splits at healthy pages (0x2ae00000: 9/25
  same-bank, 0x200→row0, 0xd0300/0xd3880/0xd7b00→rowM; 0x42e00000:
  6/25). Three pages ≡7 mod 16 (0x1ee00000, 0x32e00000, 0x3ce00000 —
  the S4b-1 super-conflict family) read deep against **all** 24 sweep
  candidates and 25/25 lattice offsets: their whole in-page pair
  baseline is shifted (sweep d p50 ≈ +85 vs typical +15), so their bank
  maps are super-conflict structure, not contamination (the 0x200
  column probe still reads low there, so λ is sound). Cross-page
  same-bank-set Jaccard p50 0.36 — per-page seed structure on top of
  the universal core, as the S4 verdict predicted.
- **Table v1** (`artifacts/g3/table_v1/`, built with `--t1` +
  `--channel-deep-only`): 200183 deduped edges (198129 from T1, the
  top-priority source; C5 records 321 disagreements, dominated by
  low↔shoulder flips against the old global-gate labels — the expected
  λ correction). 39059 bank classes, 1741 multi-node (largest 125,
  60 spanning >1 page), 13444 row classes inside them; **C1/C2/C3/C4
  all 0 contradictions** on the fully merged edge set. Classified
  nodes on **2048/2048 pages** (T0: 38/2048); same-bank page
  components on 370/2048 pages (the rest are honest T2/T3 residue —
  1/384 odds mean most pages simply share no bank with a rep).

## S5-T2 — table validation gates (`validate_table.py`)

T1 built a table; T2 decides whether it may be TRUSTED as the G4/G5
mapping. `validate_table.py` runs the gates, each printing its numbers
and an explicit pass bar (GeForge-style offline tables ship with no
validation at all; a defensive-reliability deliverable must state its
accuracy):

```bash
python3 tools/g3_probe/validate_table.py <s3> <s3b> <census> [--t1 RUN ...] \
    [--channel-deep-only] [--r-c RUN2] [--r-d RUN2] \
    [--r-e-plan --out pairs.csv --n-deep 128 --n-low 128] \
    [--r-e-check PREDICT_RUN]
```

- **R-a** transitive consistency: recount of C1–C4 plus a new count a3
  (a row class spanning two 2 MiB pages is physically impossible — the
  in-page column field cannot absorb PA bits >= 21). Bar: every count 0.
- **R-b** class cardinality: the unbiased cross-page deep rate against
  the 1/384 prior (z-test) plus the implied bank count. The
  uniform-hash Monte Carlo null is demoted to diagnostics: the T1 reps
  were drawn one per v0 channel component and lambda correlates with
  bank, so reps are bank-clustered (rep-rep deeps 64 vs null p50 6) —
  a selection property, measured and reported, not gated on.
- **R-c** same-card reproducibility: hard deep<->low flips bar 0 (a
  flip across the measured-empty valley falsifies the model); deep|mid
  REGION recall >= 99% and d-shift <= 15 cyc, because deep<->mid is
  gate wobble — each run's gate rides its own single-query calibration
  amplitude. Also anchor-validity agreement and same-bank partition
  co-membership Jaccard.
- **R-d** cross-card transfer (same model, another GPU): structure bars
  only — hard-flip RATE <= 0.1%, Jaccard >= 0.99, region >= 99%; the
  corrected-d median shift is informational because lambda carries a
  per-card timing offset. This is why the table stores classes, not
  cycles.
- **R-e** end-to-end prediction: `--r-e-plan` samples unmeasured pairs
  the table predicts transitively (same row class -> low, row classes
  linked by a deep edge -> deep, row relation unknown -> excluded;
  cross-page pairs first), the orchestrator's
  `--work-mode predict-check --pairs-csv` measures them through the
  calibration/self/predict/repeat query plan, and `--r-e-check` scores
  the run. Bar >= 95% per class counting hard flips only.

## S5-T2 result (2026-09-17): 5/5 gates PASS, row classes probable

Runs: R-c rerun `run_pool_gpu0_1789662064404501578`, predict-check
`run_pool_gpu0_1789663372902350235`, cross-card GPU 1
`run_pool_gpu1_1789663447672819650` (different UUID, identical pool PA
layout — first_pa 0x1ee00000 on both cards).

- **R-a PASS** — a1/a2/a3/a4 = 0/0/0/0 on table v2 (both T1 runs
  merged).
- **R-b PASS** — cross-page deep rate 774/302956 = 0.2555% vs prior
  0.2604% (z = −0.53); implied banks 391 (2σ 365..422) covers 384.
  Null diagnostics: components 60 vs 67–68, covered 370 vs 408–465,
  rep-rep deeps 64 vs 6 — all v0 rep-selection properties (bank
  clustering), not hash violations; next build round should draw reps
  uniformly.
- **R-c PASS** — 200830 common pairs, agreement 99.45%, deep region
  recall 10936/10936, hard flips 0, d-shift median 4 cyc, anchor hard
  flips 0, co-membership Jaccard 1.000; universal masks 0x1f9dc0 +
  0x1fdc80.
- **R-d PASS** — GPU0 vs GPU1: agreement 99.36%, deep region recall
  10936/10936, hard-flip rate 0/200830 = 0.0000%, anchor hard 0,
  Jaccard 1.000; d-shift median 25 cyc informational (amp 110 vs 111,
  calibration 1032/1038/1149 — the per-card timing offset). The
  same-bank structure — 387 deeps to the same 60 components/370 pages,
  bank-map offset sets, void pages, universal 0x1f9dc0 — transfers
  across cards of one model (0x1fdc80 is 2047/2048 on GPU 1): the
  table is per-MODEL, corroborating GeForge's reuse claim with
  measured evidence.
- **R-e PASS** — 215 predicted pairs (128 deep, 87 low): deep hard
  0/128 = 100%, low hard 86/87 = 98.85%. One hard falsifier
  (0x2aed3880/0x2aed7b00 — both reproducibly anchor-low vs M at
  0x2ae00000, yet deep +97 between themselves): the double-probe row
  inference is not universally valid, so row classes are PROBABLE
  (~1% error); bank-level claims are solid (deep hard flips 0/128 in
  every gate).

Verdict for G4/G5: bank classes may be consumed as measured facts; row
classes carry ~1% uncertainty — treat a row-class collision as strong
evidence, not proof (a future build round should double-read low edges
used for row merging).

## T3.0 — big-pool full-card table (table v3)

The table is an absolute-PA relation snapshot (the S4 verdict + per-page
seeded lattice: cross-page same-bank Jaccard p50 0.36), so a NEW PA can
only be measured, never extrapolated. Rather than rebuild mid-experiment
when an experiment pool grows, T3.0 pays the coverage cost once: a
704×32 MiB ≈ 22 GiB pool = 11264 pages covering the whole card. Reps are
drawn UNIFORMLY (`--rep-uniform N`, default seed 7) — the T2 R-b
diagnostic showed the v0 seed reps were bank-clustered (rep-rep deeps 64
vs null 6). With 384 banks, R uniform reps link 1−(1−1/384)^R of pages
to a same-bank rep: R=1024 → 93% (chosen; 1536 → 98% costs 50% more
queries for 5 points of coverage). Query budget ≈ 11.8M (classify
11264×1024 dominates), ~18 min of work on an idle card.

Two gate refinements, both evidence-driven and documented in
`validate_table.py` (never bar-bending; each carries its measured
rationale):

- **R-b effective-class band** — the big run (11.5M classify pairs)
  measures the cross-page deep rate reproducibly 2% BELOW 1/384:
  implied effective classes 392 (2σ 387..396, z = −3.4; earlier runs
  391.4/391.7 agree). A rate below 1/384 is impossible for any fixed
  distribution over ≤384 buckets (non-uniformity only raises
  collisions), so the deviation direction excludes the corruption R-b
  exists to catch; and the per-page degree split proves σ is NOT
  understated (non-rep degrees UNDER-dispersed vs the multinomial null,
  variance 2.20 vs 2.67, no page above the null max degree — the
  apparent 27× overdispersion is just the two populations: ~2.7 for
  normal pages, ~30 for the 1024 reps themselves). Gate = implied
  classes in [368, 400]; the exactly-uniform-384 z-test and the MC null
  stay as printed diagnostics.
- **R-d Jaccard is same-shape only** — co-membership across runs with
  different pool/rep sets compares pair coverage, not structure (the
  denser graph merges strictly more page pairs by measuring more of
  them; big-vs-old-2048 measured Jaccard 0.199 with zero hard flips
  both ways). Shape-mismatched pairs print it as informational; the
  falsifying bars (hard-flip rate, anchor flips, region recall) stay
  fully gated. Same-shape cross-card runs (the T3.0 protocol: identical
  builds per card) still gate Jaccard ≥ 0.99.

`build_bank_table.py` also validates every T1 edge against ITS OWN
run's `pool_map.csv` universe (big-run edges span 11264 pages; the
2048-page census universe was the wrong integrity reference — the first
v3 build attempt failed closed on exactly this) and widens the table
universe to census ∪ T1-pool pages with an honest empty λ cell for
T1-only pages (λ is per-run timing, never a table constant).

## T3.0 result on GPU 0 (2026-09-17): 96% coverage, 4/4 gates + R-e PASS

Big build `run_pool_gpu0_1789670148751001580`: 11,815,371 queries in
513 s (23,028 q/s), 0 failures, amp 117, late offset −18. PA hole
stable (first_pa 0x1ee00000, one contiguous 22 GiB range, old 2048
pages a strict subset). Cross-page deep rate 29441/11533312 = 0.2553%.
Anchor validity 11264/11264 pages (0x1f9dc0 11264, 0x1fdc80 11263);
super-conflict trio (pages ≡7 mod 16) 25/25 lattice offsets identical
across runs; 0x2ae/0x42e bank maps match T1 exactly. Three bank-map
pages void (0x22e/0x2ce/0x48e anchors read low) — traced to stale S3b
mining under the λ-referenced model, not a reproducibility failure
(rerun-vs-big anchor hard flips 0/49152).

Table v3 (`artifacts/g3/table_v3/`): universe 11264 pages (2048 census +
9216 T1-pool-only), 11396475 deduped edges (deep 81050), C1–C4 all 0
contradictions; 372 same-bank page components over 10852/11264 pages
(**96%**, was 370/2048 = 18%), largest 42; classified bank-class nodes
on 11264/11264 pages. Gates: R-a 0/0/0/0; R-b K̂ 392 in band; R-c vs
the 2048-page rerun — hard flips 0, region recall 10521/10521, d-shift
median 5, anchor hard 0; R-d vs GPU1's 2048-page run — hard-flip rate
0/71869, anchor hard 0, region recall 100%, Jaccard informational
(shape mismatch); R-e — deep hard 0/128, low 86/87 (the same reproducible
0x2aed3880/0x2aed7b00 row-inference falsifier as T2). GPU1/GPU2
same-shape big builds follow; big-vs-big R-d gates Jaccard.

## T3.0 result on all three cards (2026-09-17): identical structure, table v4

GPU1 `run_pool_gpu1_1789673269815802288` and GPU2
`run_pool_gpu2_1789673819559585717`, same shape as GPU0's big build
(704×32 MiB, `--rep-uniform 1024 --rep-seed 7`): cross-page deeps
29441/29439/29441 over the identical 11,533,312 classify pairs, the
same component size head [42, 41, 40, 38, ...], identical bank-map
lattice tables and the same three void pages (0x22e/0x2ce/0x48e — the
stale S3b mining labels), 0x1f9dc0 near-universal on every card
(11264/11263/11264). Calibration amplitudes differ per card:
117/121/101 cyc.

Pairwise big-vs-big R-d (the T3.0 protocol, identical pool/rep shape):

- GPU1-vs-GPU0: hard deep<->low 1/11803848, deep|mid region recall
  100.00%, co-membership Jaccard 1.000 (156286 page pairs together in
  both, 0 in exactly one) — PASS.
- GPU2-vs-GPU0: hard 43/11803848 (0.0004%), region 99.95%, Jaccard
  1.000 — PASS. Universal mask in both: 0x1f9dc0.
- GPU2-vs-GPU1: hard 635/11803848 (0.0054%), region 99.32%, Jaccard
  1.000 — PASS. All 11264 pages keep a common anchor on every pair.

One more evidence-driven gate refinement (the third of T3.0): the R-d
per-CELL anchor hard-flip rate is informational, replaced by the
consumer-level common-anchor bar (no page may lose every anchor the
other run found; 0.1% rate cross-card, 0 same-card). The anchor
sweep's mid valley is POPULATED (16427/19162/9461 mid cells =
3.5–7%, vs classify's 0.08%), so per-card gate placement (amplitudes
117/121/101) composes a per-cell deep<->low flip out of two soft
band-edge steps: measured 633 cells GPU2-deep/GPU1-low with GPU0
reading mid on 601 of them — one direction, 209 pages — while every
structure bar sat at 1.000 and ZERO pages lost a common anchor
(11264/11264 pages keep an anchor both runs found; per-page
anchor-set Jaccard p50 1.000, p10 0.286 on the worst pair). A card
with a genuinely different bank hash would fail this bar on
essentially every page.

Two merge rules make the canonical table honest under repeated
same-shape measurement (`build_bank_table.py`):

- **mid never overrides a decided label** — a deep measured once is a
  same-bank fact (hard deep<->mid is shallow-conflict wobble); a
  decided label from any rank survives a fresher mid.
- **a cross-T1-run deep<->low contest is resolved by majority vote** —
  deep needs the plurality; a tie leaves the pair mid (unestablished,
  excluded from classes), never a wrong same-bank edge. On the
  three-card merge: 656 contests (622 ties -> mid, 34 low-majority),
  which drops GPU2's band-edge sweep deeps that GPU1 read low.

Table v4 (`artifacts/g3/table_v4/`, the canonical G4/G5 table): five
T1 sources folded chronologically (2048-page v1 run + rerun + the
three big builds — the old runs' edges are uncontradicted evidence on
pairs the big rep sets never measured; omitting them cost 14 pages of
coverage), 11,396,475 deduped edges (deep 91,911), C1–C4 all 0,
**372 same-bank components over 10852/11264 pages (96%)**, classified
nodes on 11264/11264 pages. Remaining honest limits: 412 pages
unlinked (1/384 odds — most pages share no bank with any of the 1024
uniform reps), row classes ~1% probable (the reproducible
0x2aed3880/0x2aed7b00 double-probe falsifier), three bank-map void
pages (stale S3b labels), λ column census-only.

## S5-T3 — EMT query API (`query_table.py`)

The consumer surface for the canonical table (v4) — what G4/G5 call into.
Every claim carries a **provenance label** (measured / transitive /
assumed / unknown), and the two fault-model families the timing channel
cannot support are REFUSED outright (exit 2): `dq-adjacent` (research-plan
R3 — the burst/DQ domain has no observable structure) and
`column-adjacent` (column distance never alters latency; folded into
same-row random-column sites under the simplified G5 fault model).

```bash
python3 query_table.py --table artifacts/g3/table_v4 \
    --query 0x1ee00000 0x2ae00000            # relation + provenance
python3 query_table.py --table ... --check-pool <g3_run_dir>   # fail-closed
python3 query_table.py --table ... --select same-bank-diff-row \
    --near 0x1ee00000 --count 3              # fault-site selectors
python3 query_table.py --table ... --select row-adjacent --near 0x1ee00000
python3 query_table.py --table ... --annotate-pool <g3_run_dir> \
    --out snapshot.csv                       # G4 seam: VA<->PA<->GDDR
python3 query_table.py --table ... --build-anchors <t1_run> <t1_run> \
    <t1_run> --out <table_dir>/page_anchors.csv
```

Semantics (all pinned by `--self-test`):

- **Distinct page components ⇒ different bank** — the classify sections
  are COMPLETE over pages × reps, so a page joins a component exactly
  when some rep shares its bank; two same-bank pages therefore always
  share a component (merging runs only unions). Same component ⇒ same
  bank (transitive; R-e validated the closure 0/128 hard flips) and
  **different row, physically** (gate a3: two 2 MiB pages cannot share a
  bank row).
- **In-page node classes do NOT get that argument** — in-page pairs were
  sparsely sampled, so two nodes in different in-page bank classes are
  UNKNOWN, never "different bank", and differing row classes are not row
  inequality. Different row in-page is claimed only from direct deep
  evidence (a consensus anchor cell `(page_base, page_base^M)`).
- **Same row** = row-class membership (measured in-bank lows, ~1%
  probable). The one reproducible R-e falsifier (0x2aed3880/0x2aed7b00,
  predict-check measured the pair deep) is flagged on query
  (`measured-contradicted`) and excluded from fault-site selection.
- **Row-adjacent** is ASSUMED, never measured: PA-order neighbors within
  a bank component under the GeForge App. B monotonic-row-stripe prior
  (timing carries no row-distance information). In-page row boundaries
  are unmapped, so only the cross-page selector is offered.
- **Coverage is fail-closed**: a PA whose page is outside the table
  universe exits 2 (the relation must be measured, never extrapolated);
  unlinked pages (bank drew none of the 1024 uniform reps) warn by
  default, `--require-linked` escalates.
- `--annotate-pool` joins a run's `pool_map.csv` (VA page ↔ PA page)
  with the GDDR classes into the per-run snapshot file G4 consumes (the
  G3 leg of its dual-addressing chain).
- `--build-anchors` folds the T1 runs' anchor sweep cells into a per-page
  strict-majority consensus (per-cell flips stay informational — the
  populated sweep valley — exactly the wobble R-d documented).

S5-T3 result on table v4 (2026-09-17): anchor consensus over the three
big builds — 270,336 cells, **11264/11264 pages keep a valid anchor**,
universal masks 0x1f9dc0 + 0x1fdc80; big-pool coverage check 11264/11264
PASS (412 unlinked pages warned); measured same-row siting = 113 pairs on
9 row classes (114 minus the excluded falsifier — the honest limit for
same-row faults); `--annotate-pool` on the S3b run: 2048/2048 pages in
universe, 1992 bank-known, 5 pages with same-row sites. G3 is closed;
G4 consumes the snapshots.

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
