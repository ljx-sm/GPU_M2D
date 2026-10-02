// G8-T0 L2 probe calibration (docs/G8_CACHE_FAULT_PLAN.md §4, §6 T0).
//
// Question: does the shared probe kernel (include/gpu_m2d/l2_probe_kernels.cuh)
// separate "line resident in L2" from "line must come from GDDR" cleanly
// on this GPU, at what parallelism, and how fast is a full-surface probe?
//
// Everything runs on a private scratch buffer (no workload memory):
//   cold     thrash L2 with a large separate buffer, then probe every
//            line of the test buffer      -> expected: misses
//   warm     touch every sector of the test buffer, then probe
//                                          -> expected: hits
//   reprobe  probe again immediately after a probe
//                                          -> expected: hits (the first
//            probe loaded every line; also proves probes are not destructive)
//   capacity for working sets W: touch W twice, probe W
//                                          -> hit fraction should collapse
//            once W exceeds the effective L2 capacity (72 MiB on AD102)
//   sector   touch only sector 0 of every line, then probe sector s
//            (s = 0..3)                    -> does one sector access bring
//            the whole 128-B line into L2?
//
// Output: one CSV of latency histograms (16-cycle bins) + per-probe kernel
// times, analyzed by analyze_l2_calib.py (threshold, separation, gate).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu_m2d/l2_probe_kernels.cuh"

#define CHECK_CUDA(expr)                                                     \
  do                                                                         \
    {                                                                        \
      cudaError_t _err = (expr);                                             \
      if (_err != cudaSuccess)                                               \
        {                                                                    \
          std::fprintf (stderr, "CUDA error %s at %s:%d: %s\n", #expr,       \
                        __FILE__, __LINE__, cudaGetErrorString (_err));      \
          std::exit (1);                                                     \
        }                                                                    \
    }                                                                        \
  while (0)

namespace
{

using gpu_m2d::l2probe::kLineBytes;
using gpu_m2d::l2probe::ProbeRanges;

constexpr int kBinCycles = 16;
constexpr int kBins = 512; // 0 .. 8191 cycles; last bin = overflow

struct Options
{
  int device = 0;
  std::uint64_t buf_mib = 32;
  std::uint64_t thrash_mib = 512;
  int rounds = 5;
  std::string out = "g8_l2_calib.csv";
  // Concurrent probes per SM. With lanes == 1 (production) each probe is
  // one warp with one active lane; with lanes == 32 every lane probes and
  // times the warp-max (kept to reproduce that effect).
  std::vector<int> threads_per_sm = { 4, 8, 16, 32, 48 };
  int probe_tps = 32; // concurrency for the capacity and sector tests
  int lanes = 1;
  std::vector<std::uint64_t> capacity_mib
      = { 8, 16, 32, 48, 56, 64, 68, 72, 76, 80, 96, 128 };
};

// Reads one 32-bit word per 32-B sector (all four sectors of every line),
// L2-cached, and folds them into one value so nothing is elided.
__global__ void
touch_kernel (const std::uint32_t *__restrict__ p, std::uint64_t words,
              std::uint32_t *__restrict__ sink)
{
  std::uint32_t acc = 0;
  const std::uint64_t stride
      = static_cast<std::uint64_t> (gridDim.x) * blockDim.x * 8;
  for (std::uint64_t i
       = (static_cast<std::uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x)
         * 8;
       i < words; i += stride)
    {
      std::uint32_t v;
      asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p + i));
      acc ^= v;
    }
  if (acc == 0x9e3779b9u)
    sink[0] = acc;
}

// Touches only byte 0 of every 128-B line (sector 0).
__global__ void
touch_sector0_kernel (const std::uint8_t *__restrict__ p, std::uint64_t lines,
                      std::uint32_t *__restrict__ sink)
{
  std::uint32_t acc = 0;
  const std::uint64_t stride
      = static_cast<std::uint64_t> (gridDim.x) * blockDim.x;
  for (std::uint64_t l
       = static_cast<std::uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
       l < lines; l += stride)
    {
      std::uint32_t v;
      asm volatile ("ld.global.cg.u8 %0, [%1];"
                    : "=r"(v)
                    : "l"(p + l * kLineBytes));
      acc ^= v;
    }
  if (acc == 0x9e3779b9u)
    sink[0] = acc;
}

// Touches all four sectors of every EVEN line only.
__global__ void
touch_even_lines_kernel (const std::uint8_t *__restrict__ p,
                         std::uint64_t lines, std::uint32_t *__restrict__ sink)
{
  std::uint32_t acc = 0;
  const std::uint64_t stride
      = static_cast<std::uint64_t> (gridDim.x) * blockDim.x;
  for (std::uint64_t l
       = (static_cast<std::uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x)
         * 2;
       l < lines; l += 2 * stride)
    for (int sct = 0; sct < 4; ++sct)
      {
        std::uint32_t v;
        asm volatile ("ld.global.cg.u32 %0, [%1];"
                      : "=r"(v)
                      : "l"(p + l * kLineBytes + 32 * sct));
        acc ^= v;
      }
  if (acc == 0x9e3779b9u)
    sink[0] = acc;
}

// Times sector `s` (byte offset 32*s) of every line.  Same lane policy as
// the production kernel: lanes == 1 -> one active lane per warp.
__global__ void
probe_sector_kernel (const std::uint8_t *__restrict__ p, std::uint64_t lines,
                     int s, int lanes, std::uint16_t *__restrict__ out)
{
  extern __shared__ std::uint32_t sector_sink[];
  volatile std::uint32_t *sink = sector_sink + threadIdx.x;
  std::uint64_t worker
      = static_cast<std::uint64_t> (blockIdx.x) * blockDim.x + threadIdx.x;
  std::uint64_t workers = static_cast<std::uint64_t> (gridDim.x) * blockDim.x;
  if (lanes == 1)
    {
      if ((threadIdx.x & 31u) != 0)
        return;
      worker /= 32;
      workers /= 32;
    }
  for (std::uint64_t l = worker; l < lines; l += workers)
    {
      const std::uint32_t dt = gpu_m2d::l2probe::timed_load_cg (
          p + l * kLineBytes + 32 * s, sink);
      out[l] = dt > 0xffffu ? 0xffffu : static_cast<std::uint16_t> (dt);
    }
}

__global__ void
clock_rate_kernel (std::uint64_t *out)
{
  std::uint64_t c0 = gpu_m2d::l2probe::read_clock64 ();
  std::uint64_t g0;
  asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  std::uint64_t c1;
  do
    c1 = gpu_m2d::l2probe::read_clock64 ();
  while (c1 - c0 < 20000000ull);
  std::uint64_t g1;
  asm volatile ("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  out[0] = c1 - c0;
  out[1] = g1 - g0;
}

struct Context
{
  Options opt;
  int sms = 0;
  std::uint8_t *thrash = nullptr;
  std::uint8_t *buf = nullptr; // max(buf, capacity) bytes
  std::uint64_t buf_bytes = 0;
  std::uint16_t *lat = nullptr;
  std::uint32_t *sink = nullptr;
  cudaEvent_t e0{}, e1{};
  FILE *csv = nullptr;
};

void
touch (Context &c, const std::uint8_t *p, std::uint64_t bytes)
{
  touch_kernel<<<c.sms * 4, 256>>> (
      reinterpret_cast<const std::uint32_t *> (p), bytes / 4, c.sink);
  CHECK_CUDA (cudaGetLastError ());
  CHECK_CUDA (cudaDeviceSynchronize ());
}

void
thrash (Context &c)
{
  touch (c, c.thrash, c.opt.thrash_mib << 20);
}

// Launch geometry for `probes_per_sm` concurrent probes per SM.
void
geometry (const Context &c, int probes_per_sm, int &blocks, int &block)
{
  const int threads_per_sm
      = c.opt.lanes == 1 ? probes_per_sm * 32 : probes_per_sm;
  block = std::min (threads_per_sm, 512);
  blocks = std::max (1, c.sms * threads_per_sm / block);
}

// Full-range probe with the PRODUCTION kernel; returns kernel ms.
float
probe (Context &c, const std::uint8_t *p, std::uint64_t bytes,
       int threads_per_sm)
{
  ProbeRanges ranges{};
  ranges.count = 1;
  ranges.unit = kLineBytes;
  ranges.stride = 1;
  ranges.phase = 0;
  ranges.reverse = 0;
  ranges.lanes = c.opt.lanes;
  ranges.r[0].base = p;
  ranges.r[0].bytes = bytes;
  ranges.r[0].first_line = 0;
  ranges.r[0].lines = (bytes + kLineBytes - 1) / kLineBytes;
  ranges.total_lines = ranges.r[0].lines;
  int blocks = 0, block = 0;
  geometry (c, threads_per_sm, blocks, block);
  CHECK_CUDA (cudaEventRecord (c.e0));
  gpu_m2d::l2probe::probe_latency_kernel<<<blocks, block,
                                            block * sizeof (std::uint32_t)>>> (
      ranges, c.lat);
  CHECK_CUDA (cudaEventRecord (c.e1));
  CHECK_CUDA (cudaGetLastError ());
  CHECK_CUDA (cudaEventSynchronize (c.e1));
  float ms = 0.f;
  CHECK_CUDA (cudaEventElapsedTime (&ms, c.e0, c.e1));
  return ms;
}

void
emit (Context &c, const char *mode, int threads_per_sm, const char *state,
      int round, std::uint64_t param, std::uint64_t lines, float ms)
{
  std::vector<std::uint16_t> host (lines);
  CHECK_CUDA (cudaMemcpy (host.data (), c.lat, lines * sizeof (std::uint16_t),
                          cudaMemcpyDeviceToHost));
  std::vector<std::uint64_t> hist (kBins, 0);
  for (std::uint16_t v : host)
    hist[std::min<int> (v / kBinCycles, kBins - 1)]++;
  std::fprintf (c.csv, "probe,%s,%d,%s,%d,%llu,%llu,%.6f\n", mode,
                threads_per_sm, state, round,
                static_cast<unsigned long long> (param),
                static_cast<unsigned long long> (lines), ms);
  for (int b = 0; b < kBins; ++b)
    if (hist[b])
      std::fprintf (c.csv, "hist,%s,%d,%s,%d,%llu,%d,%llu\n", mode,
                    threads_per_sm, state, round,
                    static_cast<unsigned long long> (param), b * kBinCycles,
                    static_cast<unsigned long long> (hist[b]));
}

// Like emit(), but histograms even and odd lines as separate states.
void
emit_parity (Context &c, const char *mode, int threads_per_sm, int round,
             std::uint64_t param, std::uint64_t lines, float ms)
{
  std::vector<std::uint16_t> host (lines);
  CHECK_CUDA (cudaMemcpy (host.data (), c.lat, lines * sizeof (std::uint16_t),
                          cudaMemcpyDeviceToHost));
  for (int parity = 0; parity < 2; ++parity)
    {
      const char *state = parity == 0 ? "mixed_even" : "mixed_odd";
      std::vector<std::uint64_t> hist (kBins, 0);
      std::uint64_t n = 0;
      for (std::uint64_t l = parity; l < lines; l += 2, ++n)
        hist[std::min<int> (host[l] / kBinCycles, kBins - 1)]++;
      std::fprintf (c.csv, "probe,%s,%d,%s,%d,%llu,%llu,%.6f\n", mode,
                    threads_per_sm, state, round,
                    static_cast<unsigned long long> (param),
                    static_cast<unsigned long long> (n), ms);
      for (int b = 0; b < kBins; ++b)
        if (hist[b])
          std::fprintf (c.csv, "hist,%s,%d,%s,%d,%llu,%d,%llu\n", mode,
                        threads_per_sm, state, round,
                        static_cast<unsigned long long> (param),
                        b * kBinCycles,
                        static_cast<unsigned long long> (hist[b]));
    }
}

std::vector<int>
parse_int_list (const char *s)
{
  std::vector<int> v;
  std::string str (s);
  std::size_t pos = 0;
  while (pos <= str.size ())
    {
      std::size_t next = str.find (',', pos);
      if (next == std::string::npos)
        next = str.size ();
      v.push_back (std::atoi (str.substr (pos, next - pos).c_str ()));
      pos = next + 1;
    }
  return v;
}

} // namespace

int
main (int argc, char **argv)
{
  Context c;
  for (int i = 1; i < argc; ++i)
    {
      const std::string k = argv[i];
      auto need = [&] () -> const char * {
        if (i + 1 >= argc)
          {
            std::fprintf (stderr, "missing value for %s\n", k.c_str ());
            std::exit (2);
          }
        return argv[++i];
      };
      if (k == "--device")
        c.opt.device = std::atoi (need ());
      else if (k == "--buf-mib")
        c.opt.buf_mib = std::strtoull (need (), nullptr, 10);
      else if (k == "--thrash-mib")
        c.opt.thrash_mib = std::strtoull (need (), nullptr, 10);
      else if (k == "--rounds")
        c.opt.rounds = std::atoi (need ());
      else if (k == "--out")
        c.opt.out = need ();
      else if (k == "--threads-per-sm")
        c.opt.threads_per_sm = parse_int_list (need ());
      else if (k == "--probe-tps")
        c.opt.probe_tps = std::atoi (need ());
      else if (k == "--lanes")
        {
          c.opt.lanes = std::atoi (need ());
          if (c.opt.lanes != 1 && c.opt.lanes != 32)
            {
              std::fprintf (stderr, "--lanes must be 1 or 32\n");
              return 2;
            }
        }
      else
        {
          std::fprintf (stderr,
                        "usage: %s [--device N] [--buf-mib N] [--thrash-mib N] "
                        "[--rounds N] [--threads-per-sm a,b,..] [--probe-tps N] "
                        "[--lanes 1|32] "
                        "[--out CSV]\n",
                        argv[0]);
          return 2;
        }
    }

  CHECK_CUDA (cudaSetDevice (c.opt.device));
  cudaDeviceProp prop{};
  CHECK_CUDA (cudaGetDeviceProperties (&prop, c.opt.device));
  c.sms = prop.multiProcessorCount;

  const std::uint64_t cap_max = *std::max_element (c.opt.capacity_mib.begin (),
                                                   c.opt.capacity_mib.end ());
  c.buf_bytes = std::max (c.opt.buf_mib, cap_max) << 20;
  CHECK_CUDA (cudaMalloc (&c.thrash, c.opt.thrash_mib << 20));
  CHECK_CUDA (cudaMalloc (&c.buf, c.buf_bytes));
  CHECK_CUDA (cudaMalloc (&c.lat, (c.buf_bytes / kLineBytes) * sizeof (std::uint16_t)));
  CHECK_CUDA (cudaMalloc (&c.sink, sizeof (std::uint32_t)));
  CHECK_CUDA (cudaMemset (c.thrash, 1, c.opt.thrash_mib << 20));
  CHECK_CUDA (cudaMemset (c.buf, 2, c.buf_bytes));
  CHECK_CUDA (cudaEventCreate (&c.e0));
  CHECK_CUDA (cudaEventCreate (&c.e1));
  c.csv = std::fopen (c.opt.out.c_str (), "w");
  if (!c.csv)
    {
      std::perror ("open output");
      return 1;
    }

  // DVFS warm-up (clocks are not locked: no root), then clock report.
  for (int w = 0; w < 20; ++w)
    thrash (c);
  std::uint64_t *dclk = nullptr;
  CHECK_CUDA (cudaMalloc (&dclk, 2 * sizeof (std::uint64_t)));
  clock_rate_kernel<<<1, 1>>> (dclk);
  CHECK_CUDA (cudaDeviceSynchronize ());
  std::uint64_t hclk[2];
  CHECK_CUDA (cudaMemcpy (hclk, dclk, sizeof (hclk), cudaMemcpyDeviceToHost));
  const double mhz = static_cast<double> (hclk[0]) * 1e3 / hclk[1];
  std::fprintf (c.csv, "meta,%s,%d,%llu,%llu,%d,%.1f,%llu,%d\n", prop.name,
                c.sms, static_cast<unsigned long long> (prop.l2CacheSize),
                static_cast<unsigned long long> (c.opt.buf_mib),
                c.opt.rounds, mhz,
                static_cast<unsigned long long> (c.opt.thrash_mib),
                c.opt.lanes);
  std::printf ("G8_L2_CALIB device=%d name=\"%s\" sms=%d l2=%d MiB "
               "sm_clock=%.0f MHz\n",
               c.opt.device, prop.name, c.sms, prop.l2CacheSize >> 20, mhz);

  const std::uint64_t buf = c.opt.buf_mib << 20;
  const std::uint64_t buf_lines = buf / kLineBytes;

  // 1. cold / warm / reprobe, per parallelism level.
  for (int tps : c.opt.threads_per_sm)
    for (int r = 0; r < c.opt.rounds; ++r)
      {
        thrash (c);
        float ms = probe (c, c.buf, buf, tps);
        emit (c, "calib", tps, "cold", r, buf >> 20, buf_lines, ms);
        ms = probe (c, c.buf, buf, tps);
        emit (c, "calib", tps, "reprobe", r, buf >> 20, buf_lines, ms);
        thrash (c);
        touch (c, c.buf, buf);
        ms = probe (c, c.buf, buf, tps);
        emit (c, "calib", tps, "warm", r, buf >> 20, buf_lines, ms);
        // mixed: only even lines resident -> a per-line probe must read
        // even = hit, odd = miss (a warp-max probe reads both as miss)
        thrash (c);
        touch_even_lines_kernel<<<c.sms * 4, 256>>> (c.buf, buf_lines, c.sink);
        CHECK_CUDA (cudaGetLastError ());
        CHECK_CUDA (cudaDeviceSynchronize ());
        ms = probe (c, c.buf, buf, tps);
        emit_parity (c, "calib", tps, r, buf >> 20, buf_lines, ms);
      }
  std::printf ("calib done (%zu parallelism levels x %d rounds)\n",
               c.opt.threads_per_sm.size (), c.opt.rounds);

  // 2. capacity sweep at the probe parallelism level.
  const int cap_tps = c.opt.probe_tps;
  for (std::uint64_t w : c.opt.capacity_mib)
    for (int r = 0; r < c.opt.rounds; ++r)
      {
        const std::uint64_t bytes = w << 20;
        thrash (c);
        touch (c, c.buf, bytes);
        touch (c, c.buf, bytes);
        const float ms = probe (c, c.buf, bytes, cap_tps);
        emit (c, "capacity", cap_tps, "touched2x", r, w, bytes / kLineBytes,
              ms);
      }
  std::printf ("capacity sweep done\n");

  // 3. sector fill granularity.
  for (int s = 0; s < 4; ++s)
    for (int r = 0; r < c.opt.rounds; ++r)
      {
        thrash (c);
        touch_sector0_kernel<<<c.sms * 4, 256>>> (c.buf, buf_lines, c.sink);
        CHECK_CUDA (cudaGetLastError ());
        CHECK_CUDA (cudaDeviceSynchronize ());
        CHECK_CUDA (cudaEventRecord (c.e0));
        int sblocks = 0, sblock = 0;
        geometry (c, c.opt.probe_tps, sblocks, sblock);
        probe_sector_kernel<<<sblocks, sblock,
                              sblock * sizeof (std::uint32_t)>>> (
            c.buf, buf_lines, s, c.opt.lanes, c.lat);
        CHECK_CUDA (cudaEventRecord (c.e1));
        CHECK_CUDA (cudaGetLastError ());
        CHECK_CUDA (cudaEventSynchronize (c.e1));
        float ms = 0.f;
        CHECK_CUDA (cudaEventElapsedTime (&ms, c.e0, c.e1));
        emit (c, "sector", c.opt.probe_tps, "touch_s0", r,
              static_cast<std::uint64_t> (s),
              buf_lines, ms);
      }
  std::printf ("sector test done\n");

  std::fclose (c.csv);
  std::printf ("G8_L2_CALIB_DONE out=%s\n", c.opt.out.c_str ());
  return 0;
}
