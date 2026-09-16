// Shared G3 timing kernels (S1 calibration probe and S2 pool harness).
//
// One warp, two threads: both time one address each in the same warp
// instruction, so a same-bank different-row pair pays the row-conflict
// latency on both sides. Every timed access is preceded by
// `discard.global.L2 [addr], 128` (invalidates the L2 line without
// writeback; pool contents are scratch by contract), then clock64()
// brackets a hand-written PTX load. Compile with -Xcicc -O0 -Xptxas -O0
// or the loads may be elided (docs/G3_SURVEY.md section 2).

#ifndef GPU_M2D_G3_TIMING_KERNELS_CUH
#define GPU_M2D_G3_TIMING_KERNELS_CUH

#include <cstdint>

namespace g3_probe
{

// Load-modifier numbering kept identical to the reference tooling so
// results are comparable: 0 plain, 1 .ca, 2 .cg, 3 .cs, 4 .cv, 5 .volatile.
inline constexpr int kModifierCount = 6;
inline constexpr const char *kModifierNames[kModifierCount] = {
  "plain", ".ca", ".cg", ".cs", ".cv", ".volatile"
};

__forceinline__ __device__ uint64_t
read_clock64 ()
{
  uint64_t c;
  asm volatile ("mov.u64 %0, %%clock64;" : "=l" (c));
  return c;
}

__forceinline__ __device__ uint64_t
read_globaltimer ()
{
  uint64_t g;
  asm volatile ("mov.u64 %0, %%globaltimer;" : "=l" (g));
  return g;
}

__forceinline__ __device__ uint64_t
timed_load (const uint8_t *addr, int modifier)
{
  uint64_t temp, c0, c1;
  asm volatile ("discard.global.L2 [%0], 128;" ::"l" (addr) : "memory");
  __syncwarp ();
  c0 = read_clock64 ();
  switch (modifier)
    {
    case 0:
      asm volatile ("ld.global.u8 %0, [%1];" : "=l" (temp) : "l" (addr) : "memory");
      break;
    case 1:
      asm volatile ("ld.global.ca.u8 %0, [%1];" : "=l" (temp) : "l" (addr) : "memory");
      break;
    case 2:
      asm volatile ("ld.global.cg.u8 %0, [%1];" : "=l" (temp) : "l" (addr) : "memory");
      break;
    case 3:
      asm volatile ("ld.global.cs.u8 %0, [%1];" : "=l" (temp) : "l" (addr) : "memory");
      break;
    case 4:
      asm volatile ("ld.global.cv.u8 %0, [%1];" : "=l" (temp) : "l" (addr) : "memory");
      break;
    default:
      asm volatile ("ld.volatile.global.u8 %0, [%1];" : "=l" (temp) : "l" (addr) : "memory");
      break;
    }
  c1 = read_clock64 ();
  /* Consume the loaded value in a provably-almost-dead branch so it
     cannot be folded away without touching memory. */
  if (temp == 0xDEADBEEFu)
    asm volatile ("" ::: "memory");
  return c1 - c0;
}

// Thread 0 times addr_a, thread 1 times addr_b. times[2*r + tid] per
// repetition r: the caller launches with times advanced by 2*r.
__global__ void time_pair_addrs_kernel (const uint8_t *__restrict__ addr_a,
                                        const uint8_t *__restrict__ addr_b,
                                        uint64_t *__restrict__ times,
                                        int modifier)
{
  uint64_t dt = timed_load (threadIdx.x == 0 ? addr_a : addr_b, modifier);
  times[threadIdx.x] = dt;
}

// Derives the effective SM clock: spins ~2M cycles and compares the
// clock64 delta against the nanosecond %globaltimer delta.
__global__ void clock_rate_kernel (uint64_t *__restrict__ out)
{
  uint64_t c0 = read_clock64 ();
  uint64_t g0 = read_globaltimer ();
  uint64_t c1;
  do
    {
      c1 = read_clock64 ();
    }
  while (c1 - c0 < 2000000ull);
  uint64_t g1 = read_globaltimer ();
  out[0] = c1 - c0;
  out[1] = g1 - g0;
}

} // namespace g3_probe

#endif /* GPU_M2D_G3_TIMING_KERNELS_CUH */
