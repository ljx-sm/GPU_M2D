# G8 L2-cache tooling

Tooling for the G8 L2 cache fault plan
([docs/G8_CACHE_FAULT_PLAN.md](../../docs/G8_CACHE_FAULT_PLAN.md)).
**G8-T0 status (2026-10-01, GPU 0): V6 probe calibration PASS, V7 hook
neutrality PASS. G8-T1 status (2026-10-02): residency maps built and
verified, V9 reproducibility PASS (last section).**

## Probe kernel

`include/gpu_m2d/l2_probe_kernels.cuh` holds the kernel and is shared by
the calibration tool and the runner. `include/gpu_m2d/l2_probe.hpp` and
`src/injector/l2_probe.cu` hold the host API (`L2Prober`).

The probe times ONE L2-cached load (`ld.global.cg`) per unit, where a unit
is a 128-B line or a 32-B sector, and compares the latency with a
calibrated threshold.

- **Read-only.** It never stores to, or discards, probed memory.
- **Load completion is forced.** The loaded value is stored to shared
  memory before the second `clock64` read. The SASS order is
  `CS2R → LDG.STRONG.GPU → STS → CS2R`.
- **One active lane per warp.** A warp's `LDG` completes only when all of
  its lanes' data has returned, so with 32 active lanes every lane would
  time the *slowest* of 32 units. Concurrency therefore comes from many
  single-lane warps: `probes_per_sm`, 16 in production.
- **Compact output.** The k-th probed unit's latency goes to `out[k]`,
  and only those entries are copied back. A scattered `out[u]` layout
  dirtied one L2 sector per probed unit (T1 finding).
- **Staggered sub-sampling** (`stride`, `phase`). Sweep *s* probes units
  `u % stride == s % stride`, so one sweep's own L2 fill traffic is about
  1/stride of the surface. `reverse` probes in descending order and serves
  as a contamination detector.

## T0 part 1 — calibration (`g8_l2_calib`, `analyze_l2_calib.py`)

```bash
make -C tools/g8_cache all check
for r in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 tools/g8_cache/g8_l2_calib --lanes 1 --probe-tps 16 \
      --rounds 10 --out artifacts/g8/t0/calib1lane_r$r.csv
done
python3 tools/g8_cache/analyze_l2_calib.py artifacts/g8/t0/calib1lane_r{1,2,3}.csv \
    --probe-tps 16 --json artifacts/g8/t0/calib1lane_report.json
```

The tool works only on its own scratch buffers. It measures:

| Test | What it does | Expected |
| --- | --- | --- |
| `cold` | thrash L2, then probe | miss |
| `warm` | touch the buffer, then probe | hit |
| `reprobe` | probe again right after a probe | hit |
| `mixed` | only the even lines resident | even = hit, odd = miss (tests per-line resolution) |
| capacity | touch W twice, then probe W | hits until W exceeds L2 |
| sector | touch sector 0 only, then probe sectors 0–3 | shows the fill granularity |

Gate V6 requires, at the production concurrency:

- an empty hit/miss latency gap;
- ≥ 99.9 % correct classification for every state, including both mixed
  parities;
- a threshold that is stable across runs.

### Result — GPU 0, idle, 3 runs × 10 rounds, 1 lane/warp: **PASS**

| Probes/SM | Hit (cyc) | Miss (cyc) | Gap | Classification (incl. mixed) |
| --- | --- | --- | --- | --- |
| 4 | 208–352 | 496– | +144…+160 | 100 % |
| 8 | 208–384 | 480– | +112 | 100 % |
| **16 (production)** | **208–400** | **480–** | **+80…+112** | **100 %** |
| 32 | 208–1280 | 496– | overlap | 99.67–99.99 % |
| 48 | 208–1950 | 480– | overlap | 98.4–99.9 % |

- **Threshold: 440 / 440 / 448 cycles**; production uses **440**. The
  common gap of the three runs is **[400, 480)** cycles.
- **Throughput** at 16 probes/SM is ~5 M units/ms. A full sweep takes:

  | Surface | Per 128-B line | Per 32-B sector |
  | --- | --- | --- |
  | ResNet-50 v2 | 0.044 ms | 0.18 ms |
  | ViT-B | 0.14 ms | 0.57 ms |

- **Capacity:** hit fraction after touching W twice (identical in all
  three runs). The cliff sits at the 72 MiB L2, so the probe measures L2
  residency itself.

  | W (MiB) | ≤64 | 68 | 72 | 76 | 80 | ≥96 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Hit fraction | 1.000 | 0.903 | 0.617 | 0.267 | 0.060 | 0.000 |

- **Sector**: after touching only sector 0, sectors 1–3 hit **0 %**. L2
  fills per **32-B sector**, so residency is a per-sector property.

### The warp-max effect (documented failure mode of the first design)

The first kernel probed with all 32 lanes of a warp. It passed the
cold/warm/reprobe tests, with a gap of 144–176 cycles at 32 threads/SM,
but `--lanes 32` on the **mixed** test reads **0.000 %** of the even
(resident) lines as hits. Every warp contained an odd (missing) line, and
every lane timed the warp's slowest load. Its residency readings were
therefore per group of 32 consecutive lines, not per line. That design is
retained only behind `--lanes 32` to reproduce the effect
(`artifacts/g8/t0/calib_lanes32.csv`).

## T0 part 2 — runner hook (`--l2-probe-*`, `analyze_l2_probe_pass.py`)

`apps/resnet50_int8_g1_5.cpp`, mode `--l2-probe-out PREFIX`:

1. The runner runs the strict clean pass over every image.
2. It then runs a second, **faultless** pass:
   - every `--l2-probe-every` images, at the image boundary (before the
     image's input is staged), it sweeps every registered allocation with
     `L2Prober`;
   - it times every inference with CUDA events, excluding probe time;
   - it compares every output with the clean pass bit-for-bit.
3. The mode is measure-only and excludes the campaign and injection modes.
   It needs no observer.

```bash
build-g1.5/gpu_m2d_resnet50_int8_g1_5 --engine <clean.engine> \
    --sample-csv /data1/luojx/datasets/imagenet1k/splits/g7_eval_10000_perclass10.csv \
    --device 0 --output-prefix <run> --image-cache-dir artifacts/g8/image_cache \
    <G7 preprocessing flags from the model's bootstrap.json> \
    --l2-probe-out <run>_l2 --l2-probe-threshold 440 --l2-probe-per-sm 16 \
    --l2-probe-unit 128 [--l2-probe-every 1] [--l2-probe-stride 16] [--l2-probe-reverse 0|1]
python3 tools/g8_cache/analyze_l2_probe_pass.py <run>_l2
```

The pass writes four outputs:

- `_ranges.csv`: the probed allocations;
- `_images.csv`: per-image inference time and bit-identical flags;
- `_boundaries.csv`: units probed and hits per sweep × allocation;
- `_hist.csv`: in-situ latency histograms.

### Result — GPU 0, idle, 2026-10-01: neutrality **V7 PASS**

Neutrality was checked on 30 probe passes (ResNet-50 v2 and ViT-B,
10,000 images each, every probe design and mode tried):

- **27 passes reproduced the clean pass bit-identically, with 0
  mismatches**, and none failed.
- **3 passes are unverified.** They were accidental repeats of earlier
  control runs by a reused driver script; their output was discarded
  before it was read. Their result directories, which also carried
  superseded threshold settings under their original names, were
  deleted. Results with the production probe (1 lane, 16 probes/SM,
threshold 440, k = 1):

| Run | GPU inference ms/image | Probe ms/sweep | In-gap samples | Weights resident | Scratch resident | Input binding | R_eff preview |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-50, stride 1 | 0.254 | 0.034 | 0.011 % | **1.000** | 1.000 | 1.000 | 28.83 MB = **100 %** of R |
| ViT-B, stride 1, fwd | 0.635 | 0.114 | 0.000 % | 0.784 | **0.000** | 0.000 | 73.5 % — *contaminated* |
| ViT-B, stride 1, rev | 0.659 | 0.150 | 0.001 % | **0.000** | 0.774 | 0.000 | 4.4 % — *contaminated* |
| ViT-B, stride 16, fwd | 0.653 | 0.013 | 0.038 % | 0.795 | 0.681 | 0.000 | 73.5 MB = 78.3 % |
| ViT-B, stride 16, rev | 0.654 | 0.013 | 0.122 % | 0.791 | 0.774 | 0.000 | 73.6 MB = 78.5 % |
| ViT-B, stride 64, fwd | 0.654 | 0.008 | 0.182 % | 0.795 | 0.691 | 0.000 | 73.6 MB = 78.4 % |
| ViT-B, stride 64, rev | 0.655 | 0.008 | 0.011 % | 0.797 | 0.767 | 0.000 | 74.2 MB = 79.0 % |

### Findings

1. **Neutrality holds.** The probe pass never changed an output.
2. **The threshold transfers to real TensorRT memory.** ≤ 0.2 % of in-situ
   latencies fall inside the calibration gap.
3. **ResNet-50 v2 on an idle GPU 0: everything is resident for the whole
   run.** Every unit of every allocation is in L2 at every one of the
   10,000 boundaries, confirmed again with probes every 100 images and per
   32-B sector. Consequences:
   - read-only cache flips last until the trial ends;
   - R_eff = R.
4. **Full sweeps contaminate surfaces larger than L2** (ViT-B, 94 MB).
   - **Cause:** a sweep's own fills evict whatever it probes last, so the
     forward and reverse orders give opposite pictures.
   - **Fix:** staggered sub-sampling. At stride 16 and 64, forward and
     reverse agree on weights within 0.4 pp and on R_eff within 0.8 pp
     (73.5–74.2 MB). That is the L2's effective capacity, as expected for a
     model larger than L2.
5. **Open: residual scratch order effect for ViT-B.** Scratch reads ~68 %
   in forward order and ~77 % in reverse, at both stride 16 and stride 64.
   - It is not fill-volume driven: it does not shrink with the stride.
   - It amounts to ~0.4 MB, ≈0.6 % of R_eff.
   - It is recorded as a measurement uncertainty, without a claimed
     explanation.
6. **Inference time.** The pure GPU inference time is 0.25 ms (ResNet-50)
   and 0.65 ms (ViT-B) per image. The G7 figure of 35–53 s per 10K pass
   includes host overhead. A probe sweep therefore costs 13 % of a
   ResNet-50 inference at stride 1, and 1–2 % of a ViT-B inference at
   stride 16–64.

Artifacts (untracked) are under `artifacts/g8/t0/`. The host image cache
is under `artifacts/g8/image_cache/`; its key includes the runner binary,
so stale files are deleted after rebuilds.

## T1 — residency map (`--l2-probe-map 1`, `residency_map.py`)

**Status (2026-10-02, GPU 0 idle): residency maps built and verified for
ResNet-50 v2 and ViT-B; reproducibility gate V9 PASS.**

### What it produces

The runner's probe pass, with `--l2-probe-map 1`, feeds every probed
unit's hit/miss into `gpu_m2d::ResidencyMapBuilder`
(`include/gpu_m2d/residency_map.hpp`, unit test `ctest -R
residency_map`). The builder turns the observations into **residency
periods** `[start, end]` per unit:

- a hit opens a period (back to image 0 on a unit's first observation);
- a miss at boundary *b* closes it at *b − 1*;
- open periods close at the last image.

Each pass writes `PREFIX_residency.bin` and `PREFIX_residency.json`:

- per-image GPU inference times;
- per-unit resident bytes;
- the periods;
- T_total and R_eff_bits = Σ bits_ℓ·T_ℓ / T_total;
- the bin's sha256.

`--l2-probe-passes N` repeats the pass N times in one process.

`tools/g8_cache/residency_map.py` is the stdlib reader the T2
orchestrator will use. It is fail-closed:

- it verifies the sha256, the structure and the neutrality;
- it **independently recomputes T_total and R_eff_bits** and requires them
  to equal the runner's values.

```bash
python3 tools/g8_cache/residency_map.py summary  <prefix>...
python3 tools/g8_cache/residency_map.py compare  <prefix_a> <prefix_b>
python3 tools/g8_cache/residency_map.py diagnose <prefix>...   # probe-order artifact check
```

Production settings (user decisions 2026-10-02):

- **probe unit:** 32-B sector;
- **probing:** every image (k = 1), with the direction alternating per
  unit observation;
- **stride:** 1 for surfaces smaller than L2, **64 for ViT-B** (see
  below);
- **probe:** 1 lane/warp, 16 probes/SM, threshold 440 cycles. V6 was
  re-checked after the kernel change: gap [400, 464), PASS.

### Two probe-footprint artifacts found and handled in T1

T1 measures per sector, and that resolution exposed artifacts that T0's
allocation-level counts had hidden.

1. **Scattered output writes (fixed).** The probe wrote each latency at
   `out[u]`. At stride 16 those 2-B writes were 32 B apart, so every probed
   unit dirtied its own L2 sector: 5.9 MB per ViT-B sweep, interleaved with
   the timed loads.
   - **Fix:** the output is now compact (`out[k]`), and only the probed
     entries are copied back.
   - **Effect on ViT-B:** reverse-locked weight sectors fell from 36,218 to
     88, and R_eff rose from 69.05 to 72.18 MB.
2. **The sweep's own miss fills (bounded by the stride).** In a full L2
   (ViT-B) every probe miss fills a sector and evicts another, sometimes
   one the sweep has not probed yet. With alternating direction, such a
   unit oscillates: it reads resident only on forward, or only on reverse,
   observations. `diagnose` counts these units, and they shrink with
   sweep size:

   | ViT-B stride | Units per sweep | Weights direction-locked | Scratch direction-locked | R_eff |
   | --- | --- | --- | --- | --- |
   | 16 | 183 K | 4.97 % | 12.7 % | 72.18 MB |
   | **64** | 46 K | **0.77 %** | **5.3 %** | **74.21 MB** |
   | 256 | 11 K | 0.25 % | 0.79 % | 74.56 MB |

   - **Stride 64 is the chosen trade-off.** Each unit is observed every 64
     images (156 observations per run), and R_eff is within 0.5 % of the
     stride-256 value.
   - **The remaining bias is in one direction.** A hit is reliable
     evidence, while a miss can be an artifact, so the true R_eff is at or
     slightly above the measured value.
   - **ResNet-50 is not affected.** Its whole surface is resident, so it
     has almost no misses and the stride-1 maps carry no artifact.

### Result — final maps, GPU 0 idle, 2026-10-02

Each model ran 2 passes in one process plus a second process, all
neutral, with R_eff recomputed and matching the runner.

| | ResNet-50 v2 (stride 1) | ViT-B (stride 64) |
| --- | --- | --- |
| Units (32-B sectors) | 901,011 | 2,933,369 |
| R_eff | **28.832 MB = 100.00 %** of R | **74.212 MB = 79.06 %** of R (≈ the L2's effective capacity) |
| Weights (`trt-internal-0`) | 100 % of sectors always resident | 79.1 % always, 19.7 % never, 1.1 % partial (79.9 % of the time) |
| Scratch | 99.3 % always, 0.7 % partial (~2 periods/unit) | 71.2 % always, 22.5 % never, 6.2 % partial |
| Input binding, small buffers | resident | never resident at image boundaries |
| **V9** same process: same class / R_eff difference | 99.98 % / 0.00002 % | 99.77 % / 0.0001 % |
| **V9** cross process: same class / R_eff difference | 99.99 % / 0.0001 % | 99.77 % / 0.0004 % |
| Map size (bin) | 17 MB | 64 MB |
| Wall time per pass (incl. clean pass) | ~2.3 min | ~4 min |

**What the maps mean for T2:**

- **ResNet-50 v2** on an idle GPU 0:
  `n_cache = round(BER_cache × R_bits)`, flips land uniformly over the
  surface, and every read-only flip lasts until the trial ends.
- **ViT-B**:
  - flips concentrate on the ~80 % of weight sectors that stay resident;
  - the ~20 % that are never resident at image boundaries receive none;
  - the input binding, which is streamed in and evicted within each
    inference, also receives none. That is the plan's stated image-level
    time-resolution limit.

Artifacts (untracked) are under `artifacts/g8/t1/`, including the
superseded stride-16/64/256 comparison maps `vit_compact*`.
