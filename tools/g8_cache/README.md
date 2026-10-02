# G8 L2-cache tooling

Tooling for the G8 L2 cache fault plan
([docs/G8_CACHE_FAULT_PLAN.md](../../docs/G8_CACHE_FAULT_PLAN.md)).
**G8-T0 status (2026-10-01, GPU 0): V6 probe calibration PASS, V7 hook
neutrality PASS.**

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
