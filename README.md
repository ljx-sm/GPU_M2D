# GPU_M2D

GPU-side memory-aware fault injection for DNN inference. The project extends
REMU's dual-addressing method from CPU/LPDDR memory to discrete NVIDIA GPU
device memory and RTX 4090 GDDR6X.

The target evaluation workload is ResNet-50 INT8 on RESISC45. Phase G1 is a
strictly scoped foundation:

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

See [GPU_SIDE_REMU_RESEARCH_PLAN.md](GPU_SIDE_REMU_RESEARCH_PLAN.md) for the
full research plan and [docs/G1_VALIDATION.md](docs/G1_VALIDATION.md) for the
current milestone status.

## Audit the G2 public address interfaces

```bash
scripts/run_g2_capability_probe.sh
```

This records CUDA VMM, GPUDirect RDMA, DMA-BUF, allocation range, and buffer-ID
capabilities without treating an opaque handle or CUDA pointer as a physical
address. See [docs/G2_PLATFORM_AUDIT.md](docs/G2_PLATFORM_AUDIT.md).

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
