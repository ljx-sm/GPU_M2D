// G3 S1 calibration probe: measures whether row-buffer hit / different-bank /
// row-conflict latency regimes are statistically separable on this GPU.
//
// Method (ported in spirit from the pinned, unlicensed GPUHammer/GDDRHammer
// references -- see docs/G3_SURVEY.md; no code copied):
//   - one warp, two threads time two addresses in the same warp instruction,
//     so a same-bank different-row pair pays the row-conflict latency on both
//     sides (no A-B-A-B ping-pong needed);
//   - every timed access is preceded by `discard.global.L2 [addr], 128`,
//     which invalidates the L2 line without writeback, forcing the load to
//     GDDR (the pool is scratch: discard destroys its contents);
//   - each point repeats `iters` kernel launches and keeps the minimum,
//     because noise only ever adds delay;
//   - a warm-up loop stabilizes DVFS when clocks are not locked (no root
//     here); a clock-rate probe (clock64 vs %globaltimer) reports the
//     effective SM clock before/after warm-up.
//
// S1 never claims anything about physical addresses. The scan sweeps
// candidate offsets inside the pool's first 2 MiB, which G2 proved is one
// GMMU page with untranslated in-page offsets, so any conflict periodicity
// observed inside one page is periodicity in framebuffer PA. Interpretation
// of the histogram is done by analyze_scan.py (gate G3-R1).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "g3_timing_kernels.cuh"

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

using g3_probe::kModifierNames;

struct Options
{
  uint64_t pool_mib = 1024;
  uint64_t warmup = 200000;
  uint64_t iters = 10;
  uint64_t range = 2ull << 20; /* stay inside the first 2 MiB GMMU page */
  uint64_t step = 64;
  uint64_t anchor = 0;
  uint64_t ofs_a = 0;
  uint64_t ofs_b = 4096;
  int modifier = 5;
  const char *file = nullptr;
};

uint64_t
parse_u64 (const char *s)
{
  return std::strtoull (s, nullptr, 0);
}

void
usage (const char *prog)
{
  std::fprintf (stderr,
                "usage: %s <info|rate|floor|pair|scan>\n"
                "             [--pool-mib N] [--warmup N] [--iters N]\n"
                "             [--range B] [--step B] [--anchor B]\n"
                "             [--ofs-a B] [--ofs-b B] [--modifier 0-5]\n"
                "             [--file PATH]\n"
                "  info    device summary\n"
                "  rate    effective SM clock (cycles per ns vs globaltimer)\n"
                "  floor   same-address latency floor (row-hit reference)\n"
                "  pair    one address pair, detailed stats\n"
                "  scan    sweep candidate offsets, CSV output\n",
                prog);
}

class Probe
{
public:
  Probe (const Options &opt) : m_opt (opt)
  {
    CHECK_CUDA (cudaSetDevice (0));
    CHECK_CUDA (cudaMalloc (&mp_pool, m_opt.pool_mib << 20));
    CHECK_CUDA (cudaMalloc (&mp_times, sizeof (uint64_t) * (2 * m_opt.iters + 8)));
    CHECK_CUDA (cudaMemset (mp_pool, 0x5A, m_opt.pool_mib << 20));
    m_h.resize (2 * m_opt.iters + 8);
    cudaDeviceProp prop{};
    CHECK_CUDA (cudaGetDeviceProperties (&prop, 0));
    m_name = prop.name;
    m_max_clock_khz = prop.clockRate;
    m_l2_bytes = prop.l2CacheSize;
    m_align = (reinterpret_cast<uintptr_t> (mp_pool) & ((2ull << 20) - 1));
    std::fprintf (stderr,
                  "[probe] device=%s max_clock=%.0f MHz l2=%.1f MiB pool=%llu"
                  " MiB base=%p align2mib=%llu\n",
                  m_name.c_str (), m_max_clock_khz / 1000.0,
                  m_l2_bytes / 1048576.0,
                  (unsigned long long)m_opt.pool_mib, (void *)mp_pool,
                  (unsigned long long)m_align);
    if (m_opt.range - m_opt.step >= (2ull << 20) - m_align)
      std::fprintf (stderr,
                    "[probe] WARNING: range exceeds one 2 MiB page; offsets"
                    " past %llu mix GMMU pages\n",
                    (unsigned long long)((2ull << 20) - m_align));
  }

  ~Probe ()
  {
    cudaFree (mp_pool);
    cudaFree (mp_times);
  }

  double
  measure_rate_mhz ()
  {
    uint64_t out[2];
    uint64_t *d_out;
    CHECK_CUDA (cudaMalloc (&d_out, sizeof (out)));
    CHECK_CUDA (cudaMemset (d_out, 0, sizeof (out)));
    g3_probe::clock_rate_kernel<<<1, 1>>> (d_out);
    CHECK_CUDA (cudaGetLastError ());
    CHECK_CUDA (cudaDeviceSynchronize ());
    CHECK_CUDA (cudaMemcpy (out, d_out, sizeof (out), cudaMemcpyDeviceToHost));
    cudaFree (d_out);
    /* cycles-per-ns x 1000 == MHz */
    return 1000.0 * (double)out[0] / (double)out[1];
  }

  void
  warm_up ()
  {
    const uint8_t *anchor = mp_pool + m_opt.anchor;
    for (uint64_t i = 0; i < m_opt.warmup; i++)
      g3_probe::time_pair_addrs_kernel<<<1, 2>>> (anchor, anchor,
                                                  mp_times, m_opt.modifier);
    CHECK_CUDA (cudaDeviceSynchronize ());
  }

  // Runs the pair kernel `iters` times; returns per-thread minima.
  std::pair<uint64_t, uint64_t>
  run_pair (uint64_t ofs_a, uint64_t ofs_b)
  {
    const uint8_t *a = mp_pool + ofs_a;
    const uint8_t *b = mp_pool + ofs_b;
    for (uint64_t r = 0; r < m_opt.iters; r++)
      g3_probe::time_pair_addrs_kernel<<<1, 2>>> (a, b,
                                                  mp_times + 2 * r, m_opt.modifier);
    CHECK_CUDA (cudaGetLastError ());
    CHECK_CUDA (cudaDeviceSynchronize ());
    CHECK_CUDA (cudaMemcpy (m_h.data (), mp_times,
                            sizeof (uint64_t) * 2 * m_opt.iters,
                            cudaMemcpyDeviceToHost));
    uint64_t min_a = UINT64_MAX, min_b = UINT64_MAX;
    for (uint64_t r = 0; r < m_opt.iters; r++)
      {
        min_a = std::min (min_a, m_h[2 * r]);
        min_b = std::min (min_b, m_h[2 * r + 1]);
      }
    return { min_a, min_b };
  }

private:
  Options m_opt;
  uint8_t *mp_pool = nullptr;
  uint64_t *mp_times = nullptr;
  std::vector<uint64_t> m_h;
  std::string m_name;
  uint64_t m_max_clock_khz = 0;
  uint64_t m_l2_bytes = 0;
  uintptr_t m_align = 0;
};

} // namespace

int
main (int argc, char **argv)
{
  if (argc < 2)
    {
      usage (argv[0]);
      return 1;
    }
  std::string mode = argv[1];
  Options opt;
  for (int i = 2; i < argc; i++)
    {
      std::string flag = argv[i];
      auto next = [&]() -> const char * {
        if (i + 1 >= argc)
          {
            std::fprintf (stderr, "missing value for %s\n", flag.c_str ());
            std::exit (1);
          }
        return argv[++i];
      };
      if (flag == "--pool-mib")
        opt.pool_mib = parse_u64 (next ());
      else if (flag == "--warmup")
        opt.warmup = parse_u64 (next ());
      else if (flag == "--iters")
        opt.iters = parse_u64 (next ());
      else if (flag == "--range")
        opt.range = parse_u64 (next ());
      else if (flag == "--step")
        opt.step = parse_u64 (next ());
      else if (flag == "--anchor")
        opt.anchor = parse_u64 (next ());
      else if (flag == "--ofs-a")
        opt.ofs_a = parse_u64 (next ());
      else if (flag == "--ofs-b")
        opt.ofs_b = parse_u64 (next ());
      else if (flag == "--modifier")
        opt.modifier = (int)parse_u64 (next ());
      else if (flag == "--file")
        opt.file = next ();
      else
        {
          std::fprintf (stderr, "unknown flag %s\n", flag.c_str ());
          usage (argv[0]);
          return 1;
        }
    }

  if (mode == "info")
    {
      Probe probe (opt);
      double before = probe.measure_rate_mhz ();
      std::printf ("rate_before_warmup_mhz=%.1f\n", before);
      return 0;
    }

  if (mode == "rate")
    {
      Probe probe (opt);
      for (int r = 0; r < 5; r++)
        std::printf ("rate_mhz=%.1f\n", probe.measure_rate_mhz ());
      return 0;
    }

  Probe probe (opt);
  double r0 = probe.measure_rate_mhz ();
  probe.warm_up ();
  double r1 = probe.measure_rate_mhz ();
  std::fprintf (stderr, "[probe] rate before/after warm-up: %.1f -> %.1f"
               " MHz (modifier=%s)\n",
               r0, r1, kModifierNames[opt.modifier]);

  if (mode == "floor")
    {
      std::vector<uint64_t> a, b;
      for (uint64_t i = 0; i < 200; i++)
        {
          auto res = probe.run_pair (opt.ofs_a, opt.ofs_a);
          a.push_back (res.first);
          b.push_back (res.second);
        }
      std::sort (a.begin (), a.end ());
      std::sort (b.begin (), b.end ());
      std::printf ("# same-address floor, cycles (min/median/max) over 200"
                   " points x %llu reps\n",
                   (unsigned long long)opt.iters);
      std::printf ("thread0 %llu %llu %llu\n", (unsigned long long)a.front (),
                   (unsigned long long)a[100], (unsigned long long)a.back ());
      std::printf ("thread1 %llu %llu %llu\n", (unsigned long long)b.front (),
                   (unsigned long long)b[100], (unsigned long long)b.back ());
      return 0;
    }

  if (mode == "pair")
    {
      auto res = probe.run_pair (opt.ofs_a, opt.ofs_b);
      std::printf ("# ofs_a=%llu ofs_b=%llu modifier=%s\n",
                   (unsigned long long)opt.ofs_a,
                   (unsigned long long)opt.ofs_b,
                   kModifierNames[opt.modifier]);
      std::printf ("anchor_cycles=%llu candidate_cycles=%llu\n",
                   (unsigned long long)res.first,
                   (unsigned long long)res.second);
      return 0;
    }

  if (mode == "scan")
    {
      FILE *out = stdout;
      if (opt.file)
        {
          out = std::fopen (opt.file, "w");
          if (!out)
            {
              std::fprintf (stderr, "cannot open %s\n", opt.file);
              return 1;
            }
        }
      std::fprintf (out, "# g3 s1 scan: anchor=%llu range=%llu step=%llu"
                         " iters=%llu modifier=%s rate_mhz=%.1f\n",
                    (unsigned long long)opt.anchor,
                    (unsigned long long)opt.range,
                    (unsigned long long)opt.step,
                    (unsigned long long)opt.iters,
                    kModifierNames[opt.modifier], r1);
      std::fprintf (out, "# offset,anchor_cycles,candidate_cycles\n");
      for (uint64_t ofs = 0; ofs < opt.range; ofs += opt.step)
        {
          auto res = probe.run_pair (opt.anchor, ofs);
          std::fprintf (out, "%llu,%llu,%llu\n", (unsigned long long)ofs,
                        (unsigned long long)res.first,
                        (unsigned long long)res.second);
        }
      if (out != stdout)
        std::fclose (out);
      return 0;
    }

  usage (argv[0]);
  return 1;
}
