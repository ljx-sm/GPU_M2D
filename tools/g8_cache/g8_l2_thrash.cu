// G8-T2 controlled co-tenant: an L2 "thrasher" for the shared-GPU test.
//
// Runs as its own process on the target GPU and keeps streaming a private
// working set through L2 (one 32-bit load per 32-B sector, L2-cached), so
// the residency pass of a concurrently running campaign sees a realistic
// competing tenant. Duty cycle: each period of --period-ms, the GPU
// streams for duty * period and the host sleeps for the rest.
//
//   g8_l2_thrash [--device N] [--mib 64] [--duty 1.0] [--period-ms 10]
//                [--seconds 0 (= until killed)]
//
// Prints one status line per second (passes, achieved GB/s). It touches
// only its own allocation.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <cuda_runtime.h>

#define CHECK_CUDA(expr)                                                     \
  do                                                                         \
    {                                                                        \
      cudaError_t _err = (expr);                                             \
      if (_err != cudaSuccess)                                               \
        {                                                                    \
          std::fprintf (stderr, "CUDA error %s: %s\n", #expr,                \
                        cudaGetErrorString (_err));                          \
          std::exit (1);                                                     \
        }                                                                    \
    }                                                                        \
  while (0)

__global__ void
stream_kernel (const std::uint32_t *__restrict__ p, std::uint64_t words,
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

int
main (int argc, char **argv)
{
  int device = 0;
  std::uint64_t mib = 64;
  double duty = 1.0;
  double period_ms = 10.0;
  double seconds = 0.0;
  for (int i = 1; i < argc; ++i)
    {
      const std::string k = argv[i];
      if (i + 1 >= argc)
        {
          std::fprintf (stderr, "missing value for %s\n", k.c_str ());
          return 2;
        }
      const char *v = argv[++i];
      if (k == "--device")
        device = std::atoi (v);
      else if (k == "--mib")
        mib = std::strtoull (v, nullptr, 10);
      else if (k == "--duty")
        duty = std::atof (v);
      else if (k == "--period-ms")
        period_ms = std::atof (v);
      else if (k == "--seconds")
        seconds = std::atof (v);
      else
        {
          std::fprintf (stderr, "unknown option %s\n", k.c_str ());
          return 2;
        }
    }
  if (duty <= 0.0 || duty > 1.0 || mib == 0 || period_ms <= 0.0)
    {
      std::fprintf (stderr, "need 0 < duty <= 1, mib > 0, period-ms > 0\n");
      return 2;
    }
  CHECK_CUDA (cudaSetDevice (device));
  int sms = 0;
  CHECK_CUDA (cudaDeviceGetAttribute (&sms, cudaDevAttrMultiProcessorCount,
                                      device));
  const std::uint64_t bytes = mib << 20;
  std::uint32_t *buf = nullptr;
  std::uint32_t *sink = nullptr;
  CHECK_CUDA (cudaMalloc (&buf, bytes));
  CHECK_CUDA (cudaMalloc (&sink, sizeof (std::uint32_t)));
  CHECK_CUDA (cudaMemset (buf, 3, bytes));
  std::printf ("G8_L2_THRASH device=%d mib=%llu duty=%.2f period_ms=%.1f\n",
               device, static_cast<unsigned long long> (mib), duty, period_ms);
  std::fflush (stdout);

  using clock = std::chrono::steady_clock;
  const auto start = clock::now ();
  auto report = start;
  std::uint64_t passes = 0, passes_since = 0;
  for (;;)
    {
      const auto period_start = clock::now ();
      const auto busy_until
          = period_start
            + std::chrono::duration_cast<clock::duration> (
                std::chrono::duration<double, std::milli> (duty * period_ms));
      do
        {
          stream_kernel<<<sms * 4, 256>>> (buf, bytes / 4, sink);
          CHECK_CUDA (cudaGetLastError ());
          CHECK_CUDA (cudaDeviceSynchronize ());
          ++passes;
          ++passes_since;
        }
      while (clock::now () < busy_until);
      const auto period_end
          = period_start
            + std::chrono::duration_cast<clock::duration> (
                std::chrono::duration<double, std::milli> (period_ms));
      if (duty < 1.0)
        std::this_thread::sleep_until (period_end);
      const auto now = clock::now ();
      const double since = std::chrono::duration<double> (now - report).count ();
      if (since >= 1.0)
        {
          std::printf ("thrash passes=%llu rate=%.1f GB/s\n",
                       static_cast<unsigned long long> (passes),
                       passes_since * (bytes / 8.0) / since / 1e9);
          std::fflush (stdout);
          report = now;
          passes_since = 0;
        }
      if (seconds > 0.0
          && std::chrono::duration<double> (now - start).count () >= seconds)
        break;
    }
  cudaFree (buf);
  cudaFree (sink);
  return 0;
}
