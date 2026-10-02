# G8 L2-cache tooling

Tooling for the G8 L2 cache fault plan
([docs/G8_CACHE_FAULT_PLAN.md](../../docs/G8_CACHE_FAULT_PLAN.md)).

## Probe kernel

`include/gpu_m2d/l2_probe_kernels.cuh` is shared with the G1.5 runner. It
times ONE L2-cached load (`ld.global.cg`) per 128-B line of a set of device
ranges and reports the latency in cycles. An L2 hit is fast; a GDDR fetch
is not.

- **Read-only.** It never stores to, or discards, probed memory.
- **Load completion is forced.** The loaded value is stored to shared
  memory before the second `clock64` read, so the clock waits for the load.
  The SASS order is `CS2R clock → LDG.STRONG.GPU → STS → CS2R clock`, so
  the default `-O3` backend is fine; G3's `-O0` workaround is not needed.

## T0 calibration (`g8_l2_calib`, `analyze_l2_calib.py`)

```bash
make -C tools/g8_cache all check
CUDA_VISIBLE_DEVICES=0 tools/g8_cache/g8_l2_calib --rounds 10 \
    --out artifacts/g8/t0/calib_gpu0_r1.csv          # repeat r2, r3
python3 tools/g8_cache/analyze_l2_calib.py artifacts/g8/t0/calib_gpu0_r{1,2,3}.csv \
    --json artifacts/g8/t0/calib_gpu0_report.json
```

The tool works only on its own scratch buffers. It measures:

| Test | What it does | Expected |
| --- | --- | --- |
| `cold` | thrash L2, then probe | miss |
| `warm` | touch the buffer, then probe | hit |
| `reprobe` | probe again right after a probe | hit |
| capacity | touch a working set twice, then probe it; sizes 8–128 MiB | hits until the working set exceeds L2 |
| sector | touch only sector 0 of each line, then probe sector 0..3 | shows the L2 fill granularity |

The parallelism (threads per SM) is swept for every test.

Gate V6 (plan §7) requires, at the probe parallelism:

- an empty gap between all hit samples and all miss samples;
- ≥ 99.9 % of each known class classified correctly;
- a threshold that is stable across runs.

### Result — GPU 0, idle, 2026-10-01 (3 runs × 10 rounds): **PASS**

| Threads/SM | Hit range (cyc) | Miss range (cyc) | Gap | Classification |
| --- | --- | --- | --- | --- |
| **32** | 288–448 | 592–1664 | **+144…+176** | **100 %** all classes |
| 64 | 288–608 | 608–2000 | +16 | 100 % |
| 128 | 288–864 | 608–2992 | −240 (overlap) | 98.4–99.96 % |
| 512 | 304–3520 | 640–8192 | −2,880 (overlap) | 93–99.8 % |

- **Probe parallelism: one warp per SM (32 threads/SM).** At higher
  parallelism, queueing slows hits into the miss range.
- **Threshold:** 520 / 520 / 528 cycles across the three runs (SM clock
  2.52–2.68 GHz, not locked).
- **Cost:** ~13 M lines/ms. A full-surface probe takes:

  | Surface | Per 128-B line | Per 32-B sector |
  | --- | --- | --- |
  | ResNet-50 v2 | 0.017 ms | 0.067 ms |
  | ViT-B | 0.054 ms | 0.22 ms |

  One inference takes ~4 ms, so **probing at every image boundary
  (k = 1) is affordable**.
- **Capacity**: hit fraction after touching W twice:

  | W (MiB) | ≤64 | 68 | 72 | 76 | 80 | ≥96 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Hit fraction | 1.000 | 0.874 | 0.527 | 0.097 | 0.002 | 0.000 |

  The cliff sits at the 72 MiB L2 size, so the probe measures L2 residency
  itself. Identical in all three runs.
- **Sector granularity**: after touching only sector 0 of each line,
  sector 0 hits 100 % and sectors 1–3 hit **0 %**. L2 fills per **32-B
  sector**, not per 128-B line. Residency is therefore a per-sector
  property; this affects the plan's 128-B "line" unit and is reported to
  the user as a T0 finding.

Artifacts (untracked): `artifacts/g8/t0/calib_gpu0_r{1,2,3}.csv`,
`calib_gpu0_report.json`.
