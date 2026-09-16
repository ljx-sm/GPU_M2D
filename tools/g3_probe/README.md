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

with bank(x) one degree-≤2 GF(2) polynomial per bank output bit (the
linearized form of a row-seeded hash: a quadratic term `pa_i*pa_k` is
exactly "seed bit i × flipped bit k") and the row difference the pair xor
touching a learned row support R. Conflicts give homogeneous equations —
every bank functional must vanish on their feature difference; lows whose
xor touches R must be fired by at least one functional. Valid functionals
(those vanishing on all conflicts, an XOR-closed set) are enumerated
exactly up to weight 3 via conflict signatures — equal-signature pairs and
signature-completing triples — then grown by beam XOR; R is re-solved as
monotone clauses (hit every conflict xor) plus units (avoid every
same-bank low xor) with violation counting. The stages alternate; mid
pairs are never fitted, only scored.

    python3 tools/g3_probe/solve_mapping.py \
        artifacts/g3/pool/<s3_run> artifacts/g3/pool/<s3b_run> \
        [--degree {1,2}] [--holdout 0.2] [--seed N]
    python3 tools/g3_probe/solve_mapping.py --predict <model.json> \
        0xPA_A 0xPA_B

Output: per-bank-bit term lists (`pa8`, `pa16*pa24`, ...), the row
support, residual (unseparable low) and row-violation counts, train and
holdout accuracy with the ≥95% holdout gate (the S5 preview), and
`mapping_model_d{1,2}.json` for the prediction API. Residuals are the
honest measure of what the model class cannot express; `--degree 1`
quantifies the linear baseline S3/S3b ruled out. `--self-test` pins the
pipeline on a synthetic seeded truth: degree 2 must hit zero residuals and
violations with ≥99% holdout accuracy (the true functionals live at
weight 2–3), and degree 1 must report the insufficiency.

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
