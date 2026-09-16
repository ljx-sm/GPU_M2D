# G3 Probe — S1 timing-channel calibration

Implements step S1 of the G3 plan (`G3_PA_TO_GDDR_PLAN.md`): prove that the
row-buffer hit / different-bank / row-conflict latency regimes are
statistically separable on this GPU, before any address-mapping collection
starts. The method and its provenance are cataloged in
[docs/G3_SURVEY.md](../../docs/G3_SURVEY.md); the pinned reference
repositories are unlicensed, so this is an independent implementation.

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

## Validity boundary

- Latencies are cycle counts from one SM; conversion to ns uses the
  measured clock of the run and is only indicative without locked clocks.
- S1 measures latency structure only. It does not observe physical
  addresses and claims nothing about the mapping function.
- The pool must be alone on the GPU: co-tenant traffic corrupts timing
  (check `nvidia-smi` first; the probe prints pool/device state).
- `discard.global.L2` destroys pool contents by design; never point this
  tool at buffers whose contents matter.
