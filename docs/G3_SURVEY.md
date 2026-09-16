# G3 Survey — Prior Art and Platform Facts (S0)

Status: **complete** on 2026-09-15. This is the S0 deliverable of the G3
execution plan (`G3_PA_TO_GDDR_PLAN.md`): what the open timing-side-channel
tooling actually contains, which techniques we port into our own probe, the
verified platform facts about RTX 4090 / GDDR6X, and the unknowns that S1+
must measure. Nothing here claims a mapping rule; every rule must come from
measurement (research-plan requirement R2).

## 1. Reference repositories (pinned, study-only)

| Repo | Commit | Date | License |
| --- | --- | --- | --- |
| `sith-lab/gpuhammer` | `2bfd290cdfa289370fc41152d552eb78cdacc51d` | 2026-03-11 | none |
| `heelsec/GDDRHammer` | `30673e5433440fcc39dd27833563a29326b61add` | 2026-08-03 | none |

Both are cloned to `/data1/luojx/g3_refs/` (outside this repository). They
carry **no license**, so nothing may be copied into GPU_M2D; we study the
methodology and write our own implementation (which we must do anyway,
because our probe is PA-annotated while theirs is a VA-only black box).

- GPUHammer (USENIX Security '25, U Toronto): first practical Rowhammer on
  GPU GDDR6; its `src/re_gddr/` is the address-mapping RE toolchain
  (`drama_conflict_prober`, `conf_set`/`row_set`/`bank_set`/`gen_time`/
  `load_modifiers`).
- GDDRHammer (IEEE S&P '26): fork of the same RE toolchain plus multibank
  synchronized hammering and multi-GPU orchestration
  (`rowhammer/util/run_timing_task_multi_gpus.py`, `lock_freq.sh`).

## 2. Technique catalog to port into our S1/S3 probe

Everything below is from reading `src/re_gddr/drama_conflict_prober.{cu,cuh}`
and `util/run_timing_task.py` (GPUHammer, verified identical in GDDRHammer
modulo logging and a removed store-back).

1. **Timing primitive.** One block, ≤ 32 threads (one warp). Per thread,
   per access: `discard.global.L2 [addr], 128` → `clock64()` → single PTX
   load → `clock64()`. The discard invalidates the 128-byte L2 line without
   writeback, so the load must go to GDDR (PTX ISA 7.4+, sm_80+; sm_89 OK).
   No capacity flush needed for the prober (their hammer path additionally
   has a capacity evict kernel touching `l2CacheSize × 8` bytes).
2. **Load modifiers.** The load is parameterized over `ld.u8.global`
   {default, `.ca`, `.cg`, `.cs`, `.cv`, `.volatile`}; their default is
   `.volatile` (`load_modifiers_main.cu` characterizes the choice per GPU).
3. **Conflict signal without ping-pong.** N addresses are timed in the same
   warp instruction (all threads issue together). Two addresses in the same
   bank, different rows → both threads see the row-conflict latency; same
   row → both see row-hit; different bank → baseline. Classic DRAMA
   alternating-access is not required.
4. **Noise discipline.** Repeat each point ≥ 10 kernel launches and take the
   **minimum** ("noise only ever adds delay"); before any measurement, warm
   up with 100 000 iterations to stabilize DVFS when clocks are not locked.
5. **Clock conversion.** `toNS()` divides `clock64()` cycles by
   `cudaDeviceProp.clockRate` (max clock, static). Only meaningful with
   locked clocks (`nvidia-smi -lgc/-lmc`, root) or for relative deltas.
6. **Compiler flags.** Loads are hand-written PTX; they compile with
   `-O3 -Xcicc -O0 -Xptxas -O0` — above `-O0` in either backend the
   measured access can be optimized away.
7. **Two-step conflict confirmation.** (a) prefilter: pair delay >
   same-address base + threshold; (b) sanity: the candidate timed against
   itself must sit within ± 10 ns of the base delay.
8. **Set construction.** `gt` sweeps a fixed anchor against an 8 MiB range
   (32 B step) to produce the latency histogram and pick the threshold;
   `conf_set` collects same-bank offsets of an anchor; `row_set` groups them
   into rows (no-conflict with a representative ⇒ same row; only the last 5
   rows are checked); `bank_set` collects first-appearances of new banks.
9. **Measured priors on RTX A6000 (GA102, GDDR6, 16 Gb ×24, 384-bit).**
   Row-hit vs conflict delta tRC ≈ 43 ns; working threshold 25–30 ns;
   tREFI = 1407 ns; **bank interleave at 256 B granularity** (bank-set
   offsets 0/256/1024/1280/2048…). Row-set generation took ~1 day there.
   GA102 is the direct 384-bit/12-MC predecessor of AD102, so these are the
   best available priors — hypotheses only until measured on our cards.

## 3. Our adaptation: PA-annotated instead of VA-black-box

Their RE never knows a physical address: it discovers same-bank/same-row
sets empirically in a 15 GiB VA layout. We change the information structure:

- G2 already gives us the framebuffer PA of every 2 MiB page a process maps.
  A measurement run allocates a large pool **once**, observes every page's
  PTE (existing gated observer flow), and keeps the process alive.
- In-page offsets are untranslated (G2-validated), so PA bits [0:21) are
  freely steerable inside a page; PA bits 21+ are steered by **selecting
  pages by their observed PA**. The probe therefore emits constraints
  directly in PA space, and the solver (S4) works on true PA bits instead
  of empirical offset classes.
- Pool sizing: A6000 used 15 GiB of 48 GiB. Our cards are 24 GiB and
  shared; plan ~4–8 GiB, only after confirming the GPU is idle.
- Expected win: targeted pairs instead of blind sweeps (we can *choose* PA
  pairs that differ in exactly one bit), fewer measurements, cleaner
  equations, plus cross-checks unavailable to them (a known-PA pair's
  conflict outcome is a directly testable prediction).

## 4. Platform facts (verified)

### RTX 4090 / AD102

| Fact | Value | Source |
| --- | --- | --- |
| Memory bus | 384-bit = 12 × 32-bit MCs | NVIDIA Ada whitepaper; TechPowerUp |
| Memory | 24 GB GDDR6X @ 21 Gbps (1008 GB/s) | TechPowerUp |
| L2 cache | 72 MiB enabled (96 MiB on full die) | whitepaper; Chips and Cheese microbenchmark |
| SMs / arch | 128 of 144, sm_89 | whitepaper |
| ECC | none on GeForce (A6000 needed `nvidia-smi -e 0`; N/A for us) | GPUHammer README |

### Micron GDDR6X 16 Gb ×32 (package family MT61K512M32)

| Fact | Value | Source |
| --- | --- | --- |
| Package | 2 independent x16 channels (CA shared) | Micron 16 Gb GDDR6/X brief |
| Banks | 16 banks, 4 bank groups (per x16 channel) | Micron brief/datasheet |
| Per-channel capacity arithmetic | 16 banks × 32 768 rows × 2 KiB row = 1 GiB ⇒ 12 packages × 2 GiB = 24 GB ✓ | derived; datasheet row/column counts to be pinned |

Capacity arithmetic closes, so the expectation is: one 32-bit MC = one
package = 2 x16 channels in lockstep ⇒ an MC-level row buffer of ~4 KiB.
**This is an expectation, not a fact** — the bank count, row stride, and
channel interleave on AD102 must come out of S3/S4 measurement.

### What has and has not been done publicly

- Done: timing-based PA→bank mapping RE on GDDR6 GPUs (A6000 and, per its
  paper, 25 GDDR6 parts in GDDRHammer); bank-conflict timing shown to work.
- Not done anywhere public: **GDDR6X / AD102**. Our measured delta, bank
  count, and hash will be new results.

## 5. Risks and unknowns (feeding S1+)

| # | Risk / unknown | Mitigation |
| --- | --- | --- |
| U1 | GDDR6X hit/conflict delta unknown (A6000 GDDR6: ≈ 43 ns) | S1 calibrates the histogram before anything else (gate G3-R1) |
| U2 | `discard.global.L2` on sm_89 | compile check in S1 (expected fine, sm_80+) |
| U3 | Shared-server co-tenancy noise | idle-check before every run; min-statistics; repeat runs |
| U4 | Clock locking needs root (not in sudo whitelist) | ask admin for `nvidia-smi -lgc/-lmc` on the three GPUs, else 100k-iteration warm-up + relative deltas (their proven fallback) |
| U5 | 12 channels is not a power of two → channel hash may be 4 bits with 12/16 used, or linear+hash mix | treat channel function as unknown arity in S4; let constraints decide |
| U6 | Bank hash may be non-linear / seeded | GF(2) first, Z3 fallback, residual analysis |
| U7 | Column/burst/DQ below our observability | per research plan R3: report what is provable, mark the rest unsupported |
| U8 | Unlicensed reference code | study only; clean-room implementation in this repo |

## 6. Requirements derived for S1 (calibration kernel)

- Single-warp probe with: `discard.global.L2` + timed `ld.global.volatile`
  (plus a modifier sweep), ≥ 10 reps min-statistics, 100k-iteration warm-up,
  `-O3 -Xcicc -O0 -Xptxas -O0` build flags, output = latency histogram.
- Calibration targets: same-address delay (row-hit floor), far-pair delay
  (different-bank baseline), and a first same-bank scan (in-page offsets
  only — no PA knowledge needed yet) to see whether a conflict bump exists.
- Gate G3-R1: the three regimes are statistically separable on GPU 0
  (bump ≥ 5× point-to-point spread), else stop and re-plan (e.g. locked
  clocks, different modifier).

## 7. Reproduce the study

```bash
mkdir -p /data1/luojx/g3_refs && cd /data1/luojx/g3_refs
git clone https://github.com/sith-lab/gpuhammer.git      # 2bfd290c
git clone https://github.com/heelsec/GDDRHammer.git      # 30673e54
# key files:
#   gpuhammer/src/re_gddr/drama_conflict_prober.cu   (timing primitive)
#   gpuhammer/src/re_gddr/{conf,row,bank}_set_main.cu (set builders)
#   gpuhammer/util/run_timing_task.py                 (orchestration/defaults)
#   GDDRHammer/rowhammer/util/run_timing_task_multi_gpus.py
```

Sources: [NVIDIA Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf),
[TechPowerUp RTX 4090](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889),
[Chips and Cheese RTX 4090 microbenchmark](https://chipsandcheese.com/p/microbenchmarking-nvidias-rtx-4090),
[Micron 16 Gb GDDR6 SGRAM brief](https://www.mouser.com/datasheet/2/671/Micron_08232024_gddr6_sgram_16gb_brief_1578719_348-3554756.pdf),
[sith-lab/gpuhammer](https://github.com/sith-lab/gpuhammer),
[heelsec/GDDRHammer](https://github.com/heelsec/GDDRHammer),
[gddr.fail](http://gddr.fail/).
