# G2 Platform Address-Translation Audit

Status: **PUBLIC PA PATH UNAVAILABLE** on 2026-09-14.

## Purpose

G2 must obtain a local framebuffer physical address. A CUDA pointer, opaque
CUDA VMM allocation handle, CPU pagemap PFN, PCI BAR address, or IOMMU DMA
address is not accepted as a GPU physical address.

The `gpu_m2d_g2_capability_probe` records the public CUDA capabilities that
could support a minimally invasive translation path. It also validates an
actual `cudaMalloc` allocation and captures the CUDA driver buffer ID needed
for later lifetime checks.

## Run

```bash
scripts/run_g2_capability_probe.sh
```

Per-device output is written under `artifacts/g2/capability/` and is excluded
from Git because CUDA VAs and buffer IDs are allocation-specific.

## Results

All three RTX 4090 devices running driver 580.95.05 and CUDA runtime 12.4
reported the same capabilities:

| Capability | Result |
| --- | ---: |
| CUDA virtual memory management | 1 |
| Exportable POSIX FD VMM handle | 1 |
| GPUDirect RDMA | 0 |
| GPUDirect RDMA with CUDA VMM | 0 |
| DMA-BUF | 0 |
| Minimum VMM allocation granularity | 2 MiB |

For a real 2 MiB `cudaMalloc` allocation on each device:

- memory type was `CU_MEMORYTYPE_DEVICE`;
- device ordinal, allocation range, and buffer ID queries succeeded;
- `CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE` returned 0;
- `CU_POINTER_ATTRIBUTE_P2P_TOKENS` returned
  `CUDA_ERROR_INVALID_DEVICE`;
- the resulting `public_gpudirect_pa_candidate` decision was 0.

The NVIDIA kernel module exports `nvidia_p2p_get_pages`, but the CUDA device
and pointer capability gates explicitly reject the required GPUDirect path.
Calling the kernel entry point despite those gates would not be a supported or
defensible physical-address method, so G2 does not do that.

## Decision

CUDA VMM will be retained for controlled alias validation: two VAs mapped to
one opaque physical allocation must translate to the same framebuffer page.
The opaque handle itself is not a numeric PA.

The primary G2 implementation must instead use a read-only Ada GMMU page-table
walker. That component will be isolated from the inference runner and will
return an aperture-qualified framebuffer address, page size, and PTE metadata.
