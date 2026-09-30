# GPU_M2D

GPU-side memory-aware fault injection for DNN inference. The project extends
REMU's dual-addressing method from CPU/LPDDR memory to discrete NVIDIA GPU
device memory and RTX 4090 GDDR6X:

```text
Tensor / allocation byte / bit <-> GPU VA <-> GPU PA <-> GDDR6X bank / row
```

Workloads: ResNet-50 INT8 on RESISC45 (the G5 campaign) and six ImageNet-1K
INT8 TensorRT models (the G7-v2 campaign). GDDR faults are materialized as
device-memory bit flips; the GPU cache hierarchy is outside the fault model
and DQ-level placement is unsupported.

## Status (2026-09-29)

| Phase | Scope | Status | Details |
| --- | --- | --- | --- |
| G1 / G1.5 | Tensor bit ↔ GPU VA, CUDA XOR injector, TensorRT integration | PASS 2026-09-14, 3 GPUs | [G1](docs/G1_VALIDATION.md), [G1.5](docs/G1_5_VALIDATION.md) |
| G2 | GPU VA → framebuffer PA (read-only eBPF observer) | PASS 2026-09-15, 3 GPUs | [G2](docs/G2_VALIDATION.md) |
| G3 | PA → GDDR6X bank/row (timing channel → empirical mapping table v4) | closed 2026-09-17, 3 GPUs | [G3 probe](tools/g3_probe/README.md) |
| G4 | Dual-addressing snapshot + live gated XOR through the chain | PASS 2026-09-20, 3 GPUs | [G4](docs/G4_VALIDATION.md) |
| G5 | Fault-injection campaign, ResNet-50/RESISC45, 9 BER levels | complete 2026-09-22, 19 campaigns VERIFIED | [fault model](docs/G5_FAULT_MODEL.md) |
| G6 | Attribution analysis of the G5 campaign | complete 2026-09-22 | [G6](docs/G6_ANALYSIS.md) |
| G7-v2 | Six ImageNet-1K INT8 models, 7 BER levels each | complete 2026-09-29 | [results](docs/G7V2_RESULTS.md), [prep](tools/g7_prep/README.md) |

The research plan with the full dated status log is
[GPU_SIDE_REMU_RESEARCH_PLAN.md](GPU_SIDE_REMU_RESEARCH_PLAN.md).

## G1 scope

Phase G1 is a strictly scoped foundation:

```text
Tensor / Element / Bit <-> GPU Virtual Address -> CUDA XOR bit flip
Allocation ID / Byte Offset / Bit <-> Active GPU Virtual Address
```

G1 does **not** claim that a CUDA device pointer is a GPU physical address or
that a software bit flip models cache propagation from a physical GDDR cell.

## Build and test G1

Requirements: CMake 3.22+, a C++17 compiler, CUDA Toolkit, and an NVIDIA GPU.
The default CUDA architecture is Ada SM 8.9 and can be overridden through
`CMAKE_CUDA_ARCHITECTURES`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To run the CUDA validation on every visible GPU:

```bash
scripts/run_g1_validation.sh
```

The REMU reference implementation is pinned as a Git submodule under
`third_party/radiation-error-emulator`. Clone this repository with:

```bash
git clone --recurse-submodules https://github.com/ljx-sm/GPU_M2D.git
```

See [docs/G1_VALIDATION.md](docs/G1_VALIDATION.md) for the G1 results.

## Audit the G2 public address interfaces

```bash
scripts/run_g2_capability_probe.sh
```

This records CUDA VMM, GPUDirect RDMA, DMA-BUF, allocation range, and buffer-ID
capabilities without treating an opaque handle or CUDA pointer as a physical
address. See [docs/G2_PLATFORM_AUDIT.md](docs/G2_PLATFORM_AUDIT.md).

## Map GPU VA to local framebuffer PA with the G2 observer

The G2 primary route is a read-only eBPF observer over the unmodified NVIDIA
modules; it records the PTE payload RM hands to UVM at CUDA allocation map
time and decodes it under a version-pinned AD102 GMMU v2 contract:

```bash
make -C tools/g2_observer all check
scripts/run_g1_5_validation.sh                       # prebuild the TensorRT runner

sudo scripts/run_g2_observer_probe.sh --api device   # cudaMalloc scratch
sudo scripts/run_g2_observer_probe.sh --api vmm      # CUDA VMM scratch
sudo scripts/run_g2_observer_probe.sh --api alias    # VMM alias double-mapping
sudo scripts/run_g2_observer_probe.sh --api tensorrt # full G1.5 workload map
```

Root is needed only to attach the read-only probes; the CUDA child runs as
the invoking user and no other GPU process is touched. The tensorrt mode
maps every active G1.5 allocation page-by-page and regenerates
`artifacts/g2/gpu_va_pa_map.csv` after all GPUs pass. Run artifacts are
written under `artifacts/g2/observer/` and excluded from Git. G2 passed on
all three GPUs on 2026-09-15 (scratch, VMM alias, and TensorRT workload
map); see [docs/G2_VALIDATION.md](docs/G2_VALIDATION.md) and
[tools/g2_observer/README.md](tools/g2_observer/README.md). Lookups and
their rejection cases: `tools/g2_observer/va_pa_lookup.py --self-test`.

## Survey the G3 PA-to-GDDR landscape

S0 of the G3 plan audited the open timing-side-channel tooling (GPUHammer,
GDDRHammer — pinned by commit, study-only, both unlicensed) and collected the
verified RTX 4090 / GDDR6X platform facts and measured A6000 priors that our
own probe will be calibrated against; see [docs/G3_SURVEY.md](docs/G3_SURVEY.md).

## Calibrate the G3 timing channel (S1)

```bash
make -C tools/g3_probe all check
CUDA_VISIBLE_DEVICES=0 tools/g3_probe/g3_timing_probe scan --file /tmp/s1.csv
python3 tools/g3_probe/analyze_scan.py /tmp/s1.csv --mhz 2520
```

S1 passed on GPU 0 on 2026-09-16: the GDDR6X row-conflict delta is ~39 ns
(98 cycles at a self-boosted, stable 2520 MHz), the conflict cluster is
tight (4.4 cycles spread), and the in-page conflict offsets match the ones
published for GA102; see [tools/g3_probe/README.md](tools/g3_probe/README.md).

## Build and query the G3 empirical mapping table

Solving for a closed-form PA→bank hash failed honestly (S4/S4b: the function
is non-linear and mixes nearly all address bits, consistent with GeForge).
G3 therefore follows GeForge's methodology and builds an **empirical mapping
table (EMT)** from measured pairs: a 22 GiB pool (11,264 × 2 MiB pages) with
~11.8 M timing queries per card, built on all three cards. Structure was
identical across cards and all five validation gates passed. Table v4
(`artifacts/g3/table_v4/`, untracked) is the canonical table G4/G5 consume.
It covers 11,264/11,264 pool pages, and 96% of them are linked to a
same-bank component. Row classes are probabilistic (~1%).

```bash
python3 tools/g3_probe/query_table.py --self-test
python3 tools/g3_probe/query_table.py --table artifacts/g3/table_v4 --query 0x1ee00000 0x2ae00000
```

Every answer carries a provenance label (measured / transitive / assumed /
unknown). PAs outside the table universe are refused, and DQ-adjacent and
column-adjacent queries are rejected by design. See the S5 sections of
[tools/g3_probe/README.md](tools/g3_probe/README.md).

## Join the chain and inject through it (G4)

`tools/g4_dualaddr/build_snapshot.py` folds one run's G1.5 registry, its G2
VA→PA map, and table v4 into a fail-closed per-run snapshot.
`--api g4t2` then picks sites from the GDDR side and flips them in a live,
gated TensorRT run:

```bash
python3 tools/g4_dualaddr/build_snapshot.py --self-test
sudo scripts/run_g2_observer_probe.sh --api g4t2 [--device N]
```

See [docs/G4_VALIDATION.md](docs/G4_VALIDATION.md) and
[tools/g4_dualaddr/README.md](tools/g4_dualaddr/README.md).

## Run a fault-injection campaign (G5 / G7)

One invocation runs one frozen BER level: 100 trials by default, each XOR-holding
B bits on device through a full evaluation pass, then restoring and
re-verifying them. The fault model (60 % single-bit / 40 % 2- and 3-bit MCU;
same-row = same 256 B PA block, same-bank/different-row = per-page anchor
mask) and each workload's frozen level table live in
[docs/G5_FAULT_MODEL.md](docs/G5_FAULT_MODEL.md) and
`tools/g5_faultinj/fault_model.py`.

```bash
python3 tools/g5_faultinj/fault_model.py --self-test
python3 tools/g5_faultinj/run_g5_campaign.py --self-test
sudo scripts/run_g2_observer_probe.sh --api g5campaign --device 0 --level L3 \
     [--workload NAME] [--trials 100] [--seed 7]
```

A new G7 workload is run in `--bootstrap` mode first to measure its resident
bytes R. Its level table is then frozen with
`tools/g7_prep/freeze_g7v2_workload.py` (G7 runner passthrough flags —
`--engine`, `--sample-csv`, preprocessing — are listed in the wrapper's
header). If the runner process dies mid-trial with a known fatal CUDA
signature, that trial is recorded as PROCESS_FATAL and fresh gated segments
finish the remaining trials; this happens when a flip lands in TensorRT's
execution-context control state. Every other failure fails the level closed.
See [tools/g5_faultinj/README.md](tools/g5_faultinj/README.md) and
[tools/g7_prep/README.md](tools/g7_prep/README.md).

Analysis and figures (run artifacts are untracked):

```bash
python3 tools/g5_faultinj/analyze_campaign.py --min-trials 100          # G5
python3 tools/g5_faultinj/analyze_g6_attribution.py                     # G6
python3 tools/g5_faultinj/analyze_campaign.py --root artifacts/g7/campaign \
        --workload g7v2_imagenet1k_resnet50 --min-trials 100            # one G7 model
/data1/luojx/miniforge3/envs/vit_fault/bin/python \
        tools/g5_faultinj/plot_six_models.py                            # G7-v2 figure
```

Headline results:

- **G5 (ResNet-50/RESISC45):** accuracy holds within 0.3 pp up to BER 1e-6,
  then collapses superlinearly to 36.1 % at 1e-4. No trial produced a DUE
  (invalid output) at any level
  ([G6 analysis](docs/G6_ANALYSIS.md)).
- **G7-v2 (six ImageNet-1K models, BER 1e-7…1e-5):** by 1e-5 top-1 drops
  16–28 pp for five models. MobileNetV3-L is the most fragile (−27.8 pp) and
  ViT-B the outlier (−6.5 pp). Across all runs, 61 PROCESS_FATAL crashes were
  absorbed by the restart protocol with no lost trials
  ([results](docs/G7V2_RESULTS.md)).

## Run the optional ResNet-50 INT8 G1.5 integration

On the reference host, the existing TensorRT/OpenCV environment, engine, and
RESISC45 split can be reused without copying large assets:

```bash
scripts/run_g1_5_validation.sh
```

This registers both the public TensorRT I/O bindings and TensorRT-owned device
allocations in a lifetime-aware allocation registry, injects one selected input
Tensor bit, verifies the complete device buffer, and runs clean/injected
inference. Internal allocations remain semantically labeled
`TENSORRT_INTERNAL_UNKNOWN`, but are still exactly addressable by allocation ID,
byte offset, and bit while active.
See [docs/G1_5_VALIDATION.md](docs/G1_5_VALIDATION.md).
