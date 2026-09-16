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
