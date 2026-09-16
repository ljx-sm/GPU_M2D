// GPU_M2D G3 S2 pool harness for the eBPF PTE observer.
//
// The harness is started by the G3 pool orchestrator and blocks before any
// CUDA context or allocation exists, exactly like the G2 scratch harness.
// Once the gate file appears (probes are attached by then), it allocates N
// chunks (plain cudaMalloc, reported as ALLOCATED events), announces
// POOL_READY, then warms up the timing kernel while waiting for the
// release file -- the orchestrator builds the PA-annotated pool map from
// the observer in the meantime and only releases work once every chunk is
// fully covered by valid local-VIDEO PTEs. After release it executes the
// pair-timing workload from the work CSV and appends the per-query minima
// to the result CSV, then frees every chunk and exits.
//
// The timing kernels are shared with the S1 calibration probe
// (g3_timing_kernels.cuh): one warp, two threads, discard.global.L2 +
// clock64-bracketed PTX load, minimum over `iters` launches. The pool is
// scratch -- discard destroys its contents by design.
//
// Fail-closed rules: a pre-existing gate file is refused (stale gate); any
// malformed work-CSV row, out-of-range chunk index, or offset that would
// let the 128-byte discard line cross the chunk end aborts the run before
// WORK_BEGIN, so a partial result CSV always means a failed run.

#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

#include "g3_timing_kernels.cuh"

struct HarnessConfig {
    int device = 0;
    long chunks = 64;
    long chunk_mib = 8;
    long iters = 10;
    long warmup = 100000;
    int modifier = 5;
    int gate_timeout_seconds = 30;
    int release_timeout_seconds = 600;
    std::string gate_file;
    std::string release_file;
    std::string work_file;
    std::string result_file;
};

struct Query {
    long query_id = 0;
    long chunk_a = 0;
    long offset_a = 0;
    long chunk_b = 0;
    long offset_b = 0;
};

static uint64_t wall_time_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

static uint64_t monotonic_time_ns()
{
    struct timespec value {};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        std::perror("clock_gettime");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(value.tv_nsec);
}

static long parse_long(const char *value, const char *option)
{
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 0);
    if (!value[0] || !end || *end != '\0' || parsed < 0) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", option, value);
        std::exit(EXIT_FAILURE);
    }
    return parsed;
}

static void cuda_check(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess) {
        std::fprintf(stderr,
                     "GPU_M2D_ERROR,operation=%s,cuda_code=%d,cuda_error=%s\n",
                     operation,
                     static_cast<int>(result),
                     cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

static void emit_event(const char *event)
{
    std::printf("GPU_M2D_EVENT,event=%s,pid=%d,tgid=%d,wall_time_ns=%" PRIu64
                ",monotonic_ns=%" PRIu64 "\n",
                event,
                static_cast<int>(getpid()),
                static_cast<int>(getpid()),
                wall_time_ns(),
                monotonic_time_ns());
}

static std::string format_uuid(const cudaUUID_t &uuid)
{
    char output[37];
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(uuid.bytes);
    std::snprintf(output,
                  sizeof(output),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  bytes[0], bytes[1], bytes[2], bytes[3],
                  bytes[4], bytes[5], bytes[6], bytes[7],
                  bytes[8], bytes[9], bytes[10], bytes[11],
                  bytes[12], bytes[13], bytes[14], bytes[15]);
    return output;
}

// Effective SM clock via the shared spin probe (cycles vs %globaltimer).
static double measure_rate_mhz()
{
    uint64_t out[2] = {0, 0};
    uint64_t *device_out = nullptr;
    cuda_check(cudaMalloc(&device_out, sizeof(out)), "cudaMalloc(rate probe)");
    cuda_check(cudaMemset(device_out, 0, sizeof(out)), "cudaMemset(rate probe)");
    g3_probe::clock_rate_kernel<<<1, 1>>>(device_out);
    cuda_check(cudaGetLastError(), "clock_rate_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "clock_rate_kernel synchronize");
    cuda_check(cudaMemcpy(out, device_out, sizeof(out), cudaMemcpyDeviceToHost),
               "cudaMemcpy(rate probe)");
    cudaFree(device_out);
    if (out[1] == 0)
        return 0.0;
    // cycles per nanosecond x 1000 == MHz
    return 1000.0 * static_cast<double>(out[0]) / static_cast<double>(out[1]);
}

class PairTimer {
public:
    explicit PairTimer(long iters)
        : m_iters(iters),
          m_host(2 * static_cast<size_t>(iters) + 8, 0)
    {
        cuda_check(cudaMalloc(&m_times, sizeof(uint64_t) * m_host.size()),
                   "cudaMalloc(times)");
    }

    ~PairTimer()
    {
        cudaFree(m_times);
    }

    // Launches `launches` untimed pair kernels at the same address (used to
    // stabilize DVFS while the orchestrator builds the pool map).
    void warm_pair(const uint8_t *address, int modifier, long launches)
    {
        for (long i = 0; i < launches; ++i)
            g3_probe::time_pair_addrs_kernel<<<1, 2>>>(address, address,
                                                       m_times, modifier);
        cuda_check(cudaGetLastError(), "warmup launch");
        cuda_check(cudaDeviceSynchronize(), "warmup synchronize");
    }

    // Minimum over m_iters launches for both threads (noise only adds).
    std::pair<uint64_t, uint64_t> time_pair(const uint8_t *address_a,
                                            const uint8_t *address_b,
                                            int modifier)
    {
        for (long repetition = 0; repetition < m_iters; ++repetition) {
            g3_probe::time_pair_addrs_kernel<<<1, 2>>>(
                address_a, address_b,
                m_times + 2 * static_cast<size_t>(repetition), modifier);
        }
        cuda_check(cudaGetLastError(), "time_pair launch");
        cuda_check(cudaDeviceSynchronize(), "time_pair synchronize");
        cuda_check(cudaMemcpy(m_host.data(), m_times,
                              sizeof(uint64_t) * 2 * static_cast<size_t>(m_iters),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(times)");
        uint64_t min_a = UINT64_MAX;
        uint64_t min_b = UINT64_MAX;
        for (long repetition = 0; repetition < m_iters; ++repetition) {
            min_a = std::min(min_a, m_host[2 * static_cast<size_t>(repetition)]);
            min_b = std::min(min_b, m_host[2 * static_cast<size_t>(repetition) + 1]);
        }
        return {min_a, min_b};
    }

private:
    long m_iters;
    std::vector<uint64_t> m_host;
    uint64_t *m_times = nullptr;
};

static bool file_exists(const std::string &path)
{
    return access(path.c_str(), F_OK) == 0;
}

// Warms up in batches while polling for the release file, so the DVFS
// stabilization overlaps the orchestrator's pool-map construction.
static bool wait_for_release_warming_up(const std::string &path,
                                        int timeout_seconds,
                                        PairTimer &timer,
                                        const uint8_t *warmup_address,
                                        int modifier,
                                        long warmup_iterations)
{
    const long batch = 2000;
    long done = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(timeout_seconds);
    emit_event("WARMUP_BEGIN");
    while (std::chrono::steady_clock::now() < deadline) {
        if (file_exists(path)) {
            std::printf("GPU_M2D_EVENT,event=WARMUP_END,warmup_iterations=%ld"
                        ",pid=%d,monotonic_ns=%" PRIu64 "\n",
                        done,
                        static_cast<int>(getpid()),
                        monotonic_time_ns());
            std::fflush(stdout);
            return true;
        }
        const long remaining = warmup_iterations - done;
        if (remaining > 0) {
            const long launch = std::min(batch, remaining);
            timer.warm_pair(warmup_address, modifier, launch);
            done += launch;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
    return false;
}

static bool parse_work_file(const char *path,
                            long chunk_count,
                            size_t chunk_bytes,
                            std::vector<Query> &queries)
{
    FILE *input = std::fopen(path, "r");
    if (!input) {
        std::fprintf(stderr, "GPU_M2D_ERROR,operation=open_work_file,path=%s\n", path);
        return false;
    }
    std::vector<Query> parsed;
    char line[512];
    bool header_seen = false;
    while (std::fgets(line, sizeof(line), input)) {
        std::string trimmed(line);
        while (!trimmed.empty() &&
               (trimmed.back() == '\n' || trimmed.back() == '\r' || trimmed.back() == ' '))
            trimmed.pop_back();
        if (trimmed.empty() || trimmed[0] == '#')
            continue;
        if (!header_seen) {
            if (trimmed != "query_id,chunk_a,ofs_a,chunk_b,ofs_b") {
                std::fprintf(stderr, "GPU_M2D_ERROR,operation=work_header,got=%s\n",
                             trimmed.c_str());
                std::fclose(input);
                return false;
            }
            header_seen = true;
            continue;
        }
        Query query;
        char tail[2];
        const int fields = std::sscanf(trimmed.c_str(), "%ld,%ld,%ld,%ld,%ld%1s",
                                       &query.query_id, &query.chunk_a, &query.offset_a,
                                       &query.chunk_b, &query.offset_b, tail);
        if (fields != 5) {
            std::fprintf(stderr, "GPU_M2D_ERROR,operation=work_row,got=%s\n",
                         trimmed.c_str());
            std::fclose(input);
            return false;
        }
        if (query.chunk_a >= chunk_count || query.chunk_b >= chunk_count ||
            query.offset_a + 128 > static_cast<long>(chunk_bytes) ||
            query.offset_b + 128 > static_cast<long>(chunk_bytes)) {
            std::fprintf(stderr,
                         "GPU_M2D_ERROR,operation=work_range,query_id=%ld"
                         ",chunk_a=%ld,ofs_a=%ld,chunk_b=%ld,ofs_b=%ld\n",
                         query.query_id, query.chunk_a, query.offset_a,
                         query.chunk_b, query.offset_b);
            std::fclose(input);
            return false;
        }
        parsed.push_back(query);
    }
    std::fclose(input);
    if (!header_seen || parsed.empty()) {
        std::fprintf(stderr, "GPU_M2D_ERROR,operation=work_empty,path=%s\n", path);
        return false;
    }
    queries = std::move(parsed);
    return true;
}

static HarnessConfig parse_config(int argc, char **argv)
{
    HarnessConfig config;
    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (index + 1 >= argc && option != "--help")
            std::exit(EXIT_FAILURE);
        if (option == "--device") {
            config.device = static_cast<int>(parse_long(argv[++index], option.c_str()));
        }
        else if (option == "--chunks") {
            config.chunks = parse_long(argv[++index], option.c_str());
        }
        else if (option == "--chunk-mib") {
            config.chunk_mib = parse_long(argv[++index], option.c_str());
        }
        else if (option == "--iters") {
            config.iters = parse_long(argv[++index], option.c_str());
        }
        else if (option == "--warmup") {
            config.warmup = parse_long(argv[++index], option.c_str());
        }
        else if (option == "--modifier") {
            const long modifier = parse_long(argv[++index], option.c_str());
            if (modifier >= g3_probe::kModifierCount)
                std::exit(EXIT_FAILURE);
            config.modifier = static_cast<int>(modifier);
        }
        else if (option == "--gate-timeout-seconds") {
            config.gate_timeout_seconds =
                static_cast<int>(parse_long(argv[++index], option.c_str()));
        }
        else if (option == "--release-timeout-seconds") {
            config.release_timeout_seconds =
                static_cast<int>(parse_long(argv[++index], option.c_str()));
        }
        else if (option == "--gate-file") {
            config.gate_file = argv[++index];
        }
        else if (option == "--release-file") {
            config.release_file = argv[++index];
        }
        else if (option == "--work-file") {
            config.work_file = argv[++index];
        }
        else if (option == "--result-file") {
            config.result_file = argv[++index];
        }
        else if (option == "--help") {
            std::printf("Usage: %s --gate-file PATH --release-file PATH"
                        " --work-file PATH --result-file PATH [--device N]"
                        " [--chunks N] [--chunk-mib N] [--iters N] [--warmup N]"
                        " [--modifier 0-5] [--gate-timeout-seconds N]"
                        " [--release-timeout-seconds N]\n",
                        argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", option.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    if (config.gate_file.empty() || config.release_file.empty() ||
        config.work_file.empty() || config.result_file.empty() ||
        config.chunks <= 0 || config.chunk_mib <= 0 || config.iters <= 0 ||
        config.warmup < 0)
        std::exit(EXIT_FAILURE);
    return config;
}

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const HarnessConfig config = parse_config(argc, argv);
    if (file_exists(config.gate_file)) {
        std::fprintf(stderr, "Gate file already exists; refusing stale gate: %s\n",
                     config.gate_file.c_str());
        return EXIT_FAILURE;
    }

    emit_event("PROCESS_READY");
    emit_event("WAIT_PRE_ALLOC_GATE");
    std::fflush(stdout);
    const auto gate_deadline = std::chrono::steady_clock::now() +
                               std::chrono::seconds(config.gate_timeout_seconds);
    while (!file_exists(config.gate_file)) {
        if (std::chrono::steady_clock::now() > gate_deadline) {
            std::fprintf(stderr, "GPU_M2D_ERROR,operation=wait_for_pre_alloc_gate,error=timeout\n");
            return EXIT_FAILURE;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    emit_event("PRE_ALLOC_GATE_OPEN");

    emit_event("CONTEXT_BEGIN");
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (config.device < 0 || config.device >= device_count)
        return EXIT_FAILURE;
    cuda_check(cudaSetDevice(config.device), "cudaSetDevice");
    cuda_check(cudaFree(nullptr), "cudaFree(context initialization)");
    cudaDeviceProp device_properties {};
    cuda_check(cudaGetDeviceProperties(&device_properties, config.device),
               "cudaGetDeviceProperties");
    const std::string gpu_uuid = format_uuid(device_properties.uuid);
    emit_event("CONTEXT_READY");

    emit_event("ALLOCATION_BEGIN");
    const size_t chunk_bytes = static_cast<size_t>(config.chunk_mib) * 1024ULL * 1024ULL;
    std::vector<uint8_t *> chunks(static_cast<size_t>(config.chunks), nullptr);
    std::vector<std::string> allocation_ids(static_cast<size_t>(config.chunks));
    const uint64_t allocation_time = wall_time_ns();
    for (long index = 0; index < config.chunks; ++index) {
        cuda_check(cudaMalloc(&chunks[static_cast<size_t>(index)], chunk_bytes),
                   "cudaMalloc(chunk)");
        cuda_check(cudaMemset(chunks[static_cast<size_t>(index)], 0x5A, chunk_bytes),
                   "cudaMemset(chunk)");
        char allocation_id[160];
        std::snprintf(allocation_id, sizeof(allocation_id),
                      "g3-pool-chunk-%ld-pid-%d-ns-%" PRIu64,
                      index, static_cast<int>(getpid()), allocation_time);
        allocation_ids[static_cast<size_t>(index)] = allocation_id;
        std::printf("GPU_M2D_EVENT,event=ALLOCATED,allocation_api=device,pid=%d,tgid=%d"
                    ",device=%d,gpu_uuid=%s,allocation_id=%s,chunk_index=%ld"
                    ",wall_time_ns=%" PRIu64 ",monotonic_ns=%" PRIu64
                    ",base_va=0x%" PRIxPTR ",size_bytes=%zu\n",
                    static_cast<int>(getpid()), static_cast<int>(getpid()),
                    config.device, gpu_uuid.c_str(), allocation_id, index,
                    wall_time_ns(), monotonic_time_ns(),
                    reinterpret_cast<uintptr_t>(chunks[static_cast<size_t>(index)]),
                    chunk_bytes);
    }
    emit_event("ALLOCATION_END");
    std::printf("GPU_M2D_EVENT,event=POOL_READY,chunks=%ld,chunk_mib=%ld"
                ",total_bytes=%zu,gpu_uuid=%s,rate_mhz=%.1f,pid=%d,monotonic_ns=%" PRIu64 "\n",
                config.chunks, config.chunk_mib,
                chunk_bytes * static_cast<size_t>(config.chunks), gpu_uuid.c_str(),
                measure_rate_mhz(), static_cast<int>(getpid()), monotonic_time_ns());
    std::fflush(stdout);

    PairTimer timer(config.iters);
    if (!wait_for_release_warming_up(config.release_file,
                                     config.release_timeout_seconds,
                                     timer, chunks[0],
                                     config.modifier, config.warmup)) {
        std::fprintf(stderr, "GPU_M2D_ERROR,operation=wait_for_release,error=timeout\n");
        return EXIT_FAILURE;
    }
    emit_event("WORK_RELEASED");

    std::vector<Query> queries;
    if (!parse_work_file(config.work_file.c_str(), config.chunks, chunk_bytes, queries))
        return EXIT_FAILURE;

    FILE *results = std::fopen(config.result_file.c_str(), "w");
    if (!results) {
        std::fprintf(stderr, "GPU_M2D_ERROR,operation=open_result_file,path=%s\n",
                     config.result_file.c_str());
        return EXIT_FAILURE;
    }
    std::fprintf(results, "# g3 pool query results: cycles are the minimum over"
                          " %ld launches, modifier=%s\n",
                 config.iters, g3_probe::kModifierNames[config.modifier]);
    std::fprintf(results, "query_id,chunk_a,ofs_a,chunk_b,ofs_b,cycles_a,cycles_b\n");

    const double start_rate = measure_rate_mhz();
    std::printf("GPU_M2D_EVENT,event=WORK_BEGIN,queries=%zu,rate_mhz=%.1f,pid=%d"
                ",monotonic_ns=%" PRIu64 "\n",
                queries.size(), start_rate, static_cast<int>(getpid()),
                monotonic_time_ns());
    std::fflush(stdout);

    for (const Query &query : queries) {
        const uint8_t *address_a =
            chunks[static_cast<size_t>(query.chunk_a)] + query.offset_a;
        const uint8_t *address_b =
            chunks[static_cast<size_t>(query.chunk_b)] + query.offset_b;
        const auto measured = timer.time_pair(address_a, address_b, config.modifier);
        std::fprintf(results, "%ld,%ld,%ld,%ld,%ld,%" PRIu64 ",%" PRIu64 "\n",
                     query.query_id, query.chunk_a, query.offset_a,
                     query.chunk_b, query.offset_b,
                     measured.first, measured.second);
        std::fflush(results);
    }
    std::fclose(results);

    const double end_rate = measure_rate_mhz();
    std::printf("GPU_M2D_EVENT,event=WORK_DONE,queries=%zu,rate_mhz=%.1f,pid=%d"
                ",monotonic_ns=%" PRIu64 "\n",
                queries.size(), end_rate, static_cast<int>(getpid()),
                monotonic_time_ns());
    std::fflush(stdout);

    emit_event("FREE_BEGIN");
    for (long index = config.chunks - 1; index >= 0; --index) {
        cuda_check(cudaFree(chunks[static_cast<size_t>(index)]), "cudaFree(chunk)");
        std::printf("GPU_M2D_EVENT,event=FREE,allocation_id=%s,pid=%d,tgid=%d"
                    ",wall_time_ns=%" PRIu64 ",monotonic_ns=%" PRIu64
                    ",base_va=0x%" PRIxPTR ",size_bytes=%zu\n",
                    allocation_ids[static_cast<size_t>(index)].c_str(),
                    static_cast<int>(getpid()), static_cast<int>(getpid()),
                    wall_time_ns(), monotonic_time_ns(),
                    reinterpret_cast<uintptr_t>(chunks[static_cast<size_t>(index)]),
                    chunk_bytes);
    }
    emit_event("FREE_END");
    emit_event("TEARDOWN_BEGIN");
    emit_event("PROCESS_END");
    std::printf("GPU_M2D_G3_POOL_PASS,queries=%zu,chunks=%ld\n",
                queries.size(), config.chunks);
    return EXIT_SUCCESS;
}
