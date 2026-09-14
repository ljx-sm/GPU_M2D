# G1 Validation

Status: **PASS** on 2026-09-14.

## Acceptance criteria

- Tensor element/bit maps to the expected GPU byte virtual address and mask.
- GPU byte virtual address/bit reverses to the original Tensor element/bit.
- INT8 and FP32 cross-byte cases are covered.
- Unknown tensors, invalid bits, out-of-range elements, overlapping ranges,
  undersized allocations, and non-contiguous views are rejected.
- CUDA injection satisfies `after == before XOR mask`.
- Every non-target byte remains identical to the baseline pattern.
- A new mapping snapshot is constructed after `cudaFree` / `cudaMalloc`; no
  assumption is made that the allocator must return a different GPU VA.
- The CUDA validation passes independently on all three RTX 4090 devices.

## Observed results

The Release build completed with CUDA 12.4.131 and GCC 11.4.0. CTest reported:

```text
100% tests passed, 0 tests failed out of 2
```

The standalone CUDA test then passed on each GPU:

```text
GPU_M2D_G1_CUDA_PASS device=0 name="NVIDIA GeForce RTX 4090" compute_capability=8.9
GPU_M2D_G1_CUDA_PASS device=1 name="NVIDIA GeForce RTX 4090" compute_capability=8.9
GPU_M2D_G1_CUDA_PASS device=2 name="NVIDIA GeForce RTX 4090" compute_capability=8.9
```

On all three devices, the second `cudaMalloc(16)` reused the virtual address
returned by the first allocation after it was freed. This is permitted CUDA
allocator behavior. The test intentionally creates a new `MappingSnapshot`
for the second allocation even when the numeric GPU VA is unchanged, proving
that allocation identity/lifetime cannot be inferred from address equality.

Validated patterns include `0x00`, `0x55`, `0xAA`, `0xFF`, first/last bytes,
several INT8 bit positions, FP32 element 1 bit 31, exact before/after XOR, and
full-buffer comparison to protect every non-target byte.

CUDA Compute Sanitizer `memcheck` was also run independently on devices 0, 1,
and 2. Each run reported:

```text
ERROR SUMMARY: 0 errors
```

The pure C++ semantic test passed under AddressSanitizer and
UndefinedBehaviorSanitizer. The CUDA validation was additionally repeated ten
times per GPU (30 runs total), with all runs passing.

## Terminology boundary

The address reported in G1 is the CUDA process-visible GPU virtual address.
G1 neither obtains nor infers the GPU physical/VRAM address, GDDR6X coordinate,
memory-controller mapping, or cache residency.
