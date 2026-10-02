// G8 L2 residency probe kernels (docs/G8_CACHE_FAULT_PLAN.md §4).
//
// A probe answers, for each 128-B line of a set of device ranges, "is
// this line resident in L2 right now?" by timing ONE load per line:
// an L2 hit returns in a few hundred cycles, a GDDR fetch takes clearly
// longer. Thresholds are calibrated on the target GPU by
// tools/g8_cache/g8_l2_calib (G8-T0); nothing here hard-codes them.
//
// Properties that matter for the residency measurement:
//   - READ-ONLY on the probed memory: the kernel never stores to, and never
//     discards (discard.global.L2), a probed line, so it cannot change
//     workload data or drop a dirty line.
//   - ld.global.cg: cached in L2 only (bypasses L1), so the timing reflects
//     L2, not a per-SM L1 copy.
//   - Completion is forced before the second clock read by a dependent
//     shared-memory store of the loaded value (in-order issue + scoreboard
//     wait), the standard pointer-chase-free microbenchmark pattern.
//   - One probe per thread at a time, so per-thread latency is a single
//     load's latency; parallelism comes from many threads.
//   - Range descriptors travel as a by-value kernel parameter (no device
//     table to read), so the probe's own L2 footprint is the output array.
//   - A probe that MISSES loads the line into L2 (unavoidable for any
//     load-based probe); see the plan's perturbation note.

#ifndef GPU_M2D_L2_PROBE_KERNELS_CUH
#define GPU_M2D_L2_PROBE_KERNELS_CUH

#include <cstdint>

namespace gpu_m2d
{
namespace l2probe
{

inline constexpr std::uint64_t kLineBytes = 128;
inline constexpr int kMaxRanges = 32;

// One contiguous device byte range [base, base + bytes). Lines are the
// 128-B-aligned blocks overlapping the range; each line is probed at the
// first byte of the line that lies inside the range (so a partial edge
// line is probed inside its own allocation).
struct ProbeRange
{
  const std::uint8_t *base;
  std::uint64_t bytes;
  std::uint64_t first_line; // global output index of this range's line 0
  std::uint64_t lines;      // number of 128-B lines overlapping the range
};

struct ProbeRanges
{
  ProbeRange r[kMaxRanges];
  int count;
  std::uint64_t total_lines;
};

__forceinline__ __device__ std::uint64_t
read_clock64 ()
{
  std::uint64_t c;
  asm volatile ("mov.u64 %0, %%clock64;" : "=l"(c)::"memory");
  return c;
}

// Address of global line `g` (caller guarantees g < total_lines).
__forceinline__ __device__ const std::uint8_t *
line_address (const ProbeRanges &ranges, std::uint64_t g)
{
  int k = 0;
  while (k + 1 < ranges.count && g >= ranges.r[k + 1].first_line)
    ++k;
  const ProbeRange &pr = ranges.r[k];
  const std::uint64_t local = g - pr.first_line;
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t> (pr.base);
  const std::uintptr_t aligned = base & ~static_cast<std::uintptr_t> (kLineBytes - 1);
  std::uintptr_t addr = aligned + local * kLineBytes;
  if (addr < base)
    addr = base; // partial first line: probe inside the range
  return reinterpret_cast<const std::uint8_t *> (addr);
}

// Times one L2-cached load of `p`. `sink` is a per-thread shared-memory
// slot; storing the loaded value there before the second clock read makes
// the clock wait for the load to complete.
__forceinline__ __device__ std::uint32_t
timed_load_cg (const std::uint8_t *p, volatile std::uint32_t *sink)
{
  std::uint32_t v;
  const std::uint64_t c0 = read_clock64 ();
  asm volatile ("ld.global.cg.u8 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  *sink = v;
  const std::uint64_t c1 = read_clock64 ();
  const std::uint64_t dt = c1 - c0;
  return dt > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t> (dt);
}

// Grid-stride probe of every line; latency (cycles, saturated to 16 bits)
// per global line index. Launch with blockDim.x <= 1024 and dynamic shared
// memory = blockDim.x * sizeof(uint32_t).
__global__ void
probe_latency_kernel (ProbeRanges ranges, std::uint16_t *__restrict__ out)
{
  extern __shared__ std::uint32_t probe_sink[];
  volatile std::uint32_t *sink = probe_sink + threadIdx.x;
  const std::uint64_t stride
      = static_cast<std::uint64_t> (gridDim.x) * blockDim.x;
  for (std::uint64_t g
       = static_cast<std::uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
       g < ranges.total_lines; g += stride)
    {
      const std::uint32_t dt = timed_load_cg (line_address (ranges, g), sink);
      out[g] = dt > 0xffffu ? 0xffffu : static_cast<std::uint16_t> (dt);
    }
}

} // namespace l2probe
} // namespace gpu_m2d

#endif // GPU_M2D_L2_PROBE_KERNELS_CUH
