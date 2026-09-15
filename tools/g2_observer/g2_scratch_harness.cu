// GPU_M2D G2 scratch allocation harness for the eBPF PTE observer.
//
// The harness is started by the observer orchestrator and blocks before
// any CUDA context or allocation exists. Once the gate file appears
// (probes are attached by then), it creates one allocation (plain
// cudaMalloc "device" or CUDA VMM "vmm"), prints its VA range, UUID and
// sample VAs, optionally performs the one-bit XOR closeout, holds the
// allocation, and frees it. Timeline markers use GPU_M2D_EVENT lines so
// the orchestrator can correlate the user-space lifecycle with kernel
// MAP/PTE/FREE events.
//
// Ported from the REMU gpu_va_pa_mapping harness
// (allocation_premap_trace_test.cu + test_common.cuh, frozen 2026-09-08);
// made self-contained and renamed to GPU_M2D markers.

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

struct HarnessConfig {
    int device = 0;
    size_t size_bytes = 8ULL * 1024ULL * 1024ULL;
    int hold_seconds = 2;
    std::string gate_file;
    std::string api;
    int gate_timeout_seconds = 30;
    bool xor_closeout = false;
};

struct Allocation {
    void *pointer = nullptr;
    size_t size_bytes = 0;
    CUdeviceptr reserved_va = 0;
    CUmemGenericAllocationHandle vmm_handle = 0;
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
    const long parsed = std::strtol(value, &end, 10);
    if (!value[0] || !end || *end != '\0') {
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

static void driver_check(CUresult result, const char *operation)
{
    if (result != CUDA_SUCCESS) {
        const char *name = "unknown";
        const char *description = "unknown";
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &description);
        std::fprintf(stderr,
                     "GPU_M2D_ERROR,operation=%s,driver_code=%d,driver_name=%s,driver_error=%s\n",
                     operation,
                     static_cast<int>(result),
                     name,
                     description);
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

static void print_samples(const std::string &allocation_id, const void *base, size_t size_bytes)
{
    const size_t offsets[] = {
        0,
        4ULL * 1024ULL,
        64ULL * 1024ULL,
        2ULL * 1024ULL * 1024ULL,
        4ULL * 1024ULL * 1024ULL,
    };
    const auto base_address = reinterpret_cast<uintptr_t>(base);
    for (size_t index = 0; index < sizeof(offsets) / sizeof(offsets[0]); ++index) {
        if (offsets[index] >= size_bytes)
            continue;
        std::printf("GPU_M2D_SAMPLE,allocation_id=%s,index=%zu,offset_bytes=%zu,va=0x%" PRIxPTR "\n",
                    allocation_id.c_str(),
                    index,
                    offsets[index],
                    base_address + offsets[index]);
    }
}

static void hold_allocation(const std::string &allocation_id, int hold_seconds)
{
    std::printf("GPU_M2D_EVENT,event=HOLD_BEGIN,allocation_id=%s,timestamp_ns=%" PRIu64
                ",hold_seconds=%d\n",
                allocation_id.c_str(),
                wall_time_ns(),
                hold_seconds);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(hold_seconds));
    std::printf("GPU_M2D_EVENT,event=HOLD_END,allocation_id=%s,timestamp_ns=%" PRIu64 "\n",
                allocation_id.c_str(),
                wall_time_ns());
}

static long required_value(int argc, char **argv, int &index, const std::string &option)
{
    if (index + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", option.c_str());
        std::exit(EXIT_FAILURE);
    }
    return parse_long(argv[++index], option.c_str());
}

static HarnessConfig parse_config(int argc, char **argv)
{
    HarnessConfig config;
    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (option == "--api") {
            if (index + 1 >= argc)
                std::exit(EXIT_FAILURE);
            config.api = argv[++index];
        }
        else if (option == "--device") {
            config.device = static_cast<int>(required_value(argc, argv, index, option));
        }
        else if (option == "--size-mib") {
            const long size_mib = required_value(argc, argv, index, option);
            if (size_mib <= 0 || size_mib > 64)
                std::exit(EXIT_FAILURE);
            config.size_bytes = static_cast<size_t>(size_mib) * 1024ULL * 1024ULL;
        }
        else if (option == "--hold-seconds") {
            config.hold_seconds = static_cast<int>(required_value(argc, argv, index, option));
            if (config.hold_seconds < 0)
                std::exit(EXIT_FAILURE);
        }
        else if (option == "--gate-file") {
            if (index + 1 >= argc)
                std::exit(EXIT_FAILURE);
            config.gate_file = argv[++index];
        }
        else if (option == "--gate-timeout-seconds") {
            config.gate_timeout_seconds = static_cast<int>(required_value(argc, argv, index, option));
            if (config.gate_timeout_seconds <= 0)
                std::exit(EXIT_FAILURE);
        }
        else if (option == "--xor-closeout") {
            config.xor_closeout = true;
        }
        else if (option == "--help") {
            std::printf("Usage: %s --api device|vmm --gate-file PATH [--device N] "
                        "[--size-mib N] [--hold-seconds N] [--gate-timeout-seconds N] "
                        "[--xor-closeout]\n",
                        argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", option.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    if ((config.api != "device" && config.api != "vmm") || config.gate_file.empty())
        std::exit(EXIT_FAILURE);
    return config;
}

__global__ static void xor_one_byte(unsigned char *base, size_t offset, unsigned char mask)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        base[offset] ^= mask;
}

static uint64_t hamming_distance(const std::vector<unsigned char> &left,
                                 const std::vector<unsigned char> &right,
                                 size_t *changed_bytes)
{
    if (left.size() != right.size())
        return UINT64_MAX;
    uint64_t bits = 0;
    size_t bytes = 0;
    for (size_t index = 0; index < left.size(); ++index) {
        const unsigned char difference = left[index] ^ right[index];
        if (difference != 0)
            ++bytes;
        bits += static_cast<uint64_t>(__builtin_popcount(static_cast<unsigned int>(difference)));
    }
    if (changed_bytes != nullptr)
        *changed_bytes = bytes;
    return bits;
}

static bool execute_xor_closeout(const Allocation &allocation)
{
    constexpr size_t target_offset = 4096;
    constexpr unsigned int bit_index = 3;
    constexpr unsigned char mask = static_cast<unsigned char>(1U << bit_index);
    if (allocation.size_bytes <= target_offset)
        return false;

    std::vector<unsigned char> baseline(allocation.size_bytes);
    std::vector<unsigned char> mutated(allocation.size_bytes);
    std::vector<unsigned char> restored(allocation.size_bytes);
    cuda_check(cudaMemcpy(baseline.data(), allocation.pointer, allocation.size_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(xor baseline D2H)");

    xor_one_byte<<<1, 1>>>(static_cast<unsigned char *>(allocation.pointer), target_offset, mask);
    cuda_check(cudaGetLastError(), "xor_one_byte(forward) launch");
    cuda_check(cudaDeviceSynchronize(), "xor_one_byte(forward) synchronize");
    cuda_check(cudaMemcpy(mutated.data(), allocation.pointer, allocation.size_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(xor mutated D2H)");

    xor_one_byte<<<1, 1>>>(static_cast<unsigned char *>(allocation.pointer), target_offset, mask);
    cuda_check(cudaGetLastError(), "xor_one_byte(restore) launch");
    cuda_check(cudaDeviceSynchronize(), "xor_one_byte(restore) synchronize");
    cuda_check(cudaMemcpy(restored.data(), allocation.pointer, allocation.size_bytes,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(xor restored D2H)");

    size_t forward_changed_bytes = 0;
    size_t restore_changed_bytes = 0;
    size_t final_changed_bytes = 0;
    const uint64_t forward_hamming = hamming_distance(baseline, mutated, &forward_changed_bytes);
    const uint64_t restore_hamming = hamming_distance(mutated, restored, &restore_changed_bytes);
    const uint64_t final_hamming = hamming_distance(baseline, restored, &final_changed_bytes);
    const unsigned int before = baseline[target_offset];
    const unsigned int after = mutated[target_offset];
    const unsigned int final_value = restored[target_offset];
    const bool pass = forward_hamming == 1 && restore_hamming == 1 && final_hamming == 0 &&
                      forward_changed_bytes == 1 && restore_changed_bytes == 1 &&
                      final_changed_bytes == 0 && after == (before ^ mask) && final_value == before;

    std::printf("GPU_M2D_XOR,status=%s,target_offset=%zu,bit_index=%u,mask=0x%02x,"
                "before=0x%02x,after=0x%02x,restored=0x%02x,"
                "forward_hamming_bits=%" PRIu64 ",restore_hamming_bits=%" PRIu64 ","
                "final_hamming_bits=%" PRIu64 ",forward_changed_bytes=%zu,"
                "restore_changed_bytes=%zu,final_changed_bytes=%zu,size_bytes=%zu\n",
                pass ? "PASS" : "FAIL",
                target_offset,
                bit_index,
                static_cast<unsigned int>(mask),
                before,
                after,
                final_value,
                forward_hamming,
                restore_hamming,
                final_hamming,
                forward_changed_bytes,
                restore_changed_bytes,
                final_changed_bytes,
                allocation.size_bytes);
    return pass;
}

static bool wait_for_gate(const std::string &path, int timeout_seconds)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (access(path.c_str(), F_OK) == 0)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return false;
}

static size_t round_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

static Allocation allocate_device(size_t size_bytes)
{
    Allocation allocation;
    allocation.size_bytes = size_bytes;
    cuda_check(cudaMalloc(&allocation.pointer, size_bytes), "cudaMalloc");
    return allocation;
}

static Allocation allocate_vmm(int device, size_t requested_size)
{
    driver_check(cuInit(0), "cuInit");
    CUmemAllocationProp properties {};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t granularity = 0;
    driver_check(cuMemGetAllocationGranularity(&granularity,
                                               &properties,
                                               CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                 "cuMemGetAllocationGranularity");
    const size_t size_bytes = round_up(requested_size, granularity);

    Allocation allocation;
    allocation.size_bytes = size_bytes;
    driver_check(cuMemAddressReserve(&allocation.reserved_va, size_bytes, granularity, 0, 0),
                 "cuMemAddressReserve");
    driver_check(cuMemCreate(&allocation.vmm_handle, size_bytes, &properties, 0), "cuMemCreate");
    driver_check(cuMemMap(allocation.reserved_va, size_bytes, 0, allocation.vmm_handle, 0), "cuMemMap");
    CUmemAccessDesc access {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    driver_check(cuMemSetAccess(allocation.reserved_va, size_bytes, &access, 1), "cuMemSetAccess");
    allocation.pointer = reinterpret_cast<void *>(allocation.reserved_va);
    return allocation;
}

static void release_allocation(const std::string &api, Allocation &allocation)
{
    if (api == "device") {
        cuda_check(cudaFree(allocation.pointer), "cudaFree");
        return;
    }
    driver_check(cuMemUnmap(allocation.reserved_va, allocation.size_bytes), "cuMemUnmap");
    driver_check(cuMemRelease(allocation.vmm_handle), "cuMemRelease");
    driver_check(cuMemAddressFree(allocation.reserved_va, allocation.size_bytes), "cuMemAddressFree");
}

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const HarnessConfig config = parse_config(argc, argv);
    if (access(config.gate_file.c_str(), F_OK) == 0) {
        std::fprintf(stderr, "Gate file already exists; refusing stale gate: %s\n", config.gate_file.c_str());
        return EXIT_FAILURE;
    }

    emit_event("PROCESS_READY");
    emit_event("WAIT_PRE_ALLOC_GATE");
    std::fflush(stdout);
    if (!wait_for_gate(config.gate_file, config.gate_timeout_seconds)) {
        std::fprintf(stderr, "GPU_M2D_ERROR,operation=wait_for_pre_alloc_gate,error=timeout\n");
        return EXIT_FAILURE;
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
    cuda_check(cudaGetDeviceProperties(&device_properties, config.device), "cudaGetDeviceProperties");
    emit_event("CONTEXT_READY");

    emit_event("ALLOCATION_BEGIN");
    const uint64_t allocation_time = wall_time_ns();
    Allocation allocation = config.api == "device"
        ? allocate_device(config.size_bytes)
        : allocate_vmm(config.device, config.size_bytes);

    char allocation_id[128];
    std::snprintf(allocation_id,
                  sizeof(allocation_id),
                  "g2-scratch-%s-%d-pid-%d-ns-%" PRIu64,
                  config.api.c_str(),
                  config.device,
                  static_cast<int>(getpid()),
                  allocation_time);
    std::printf("GPU_M2D_EVENT,event=ALLOCATED,allocation_api=%s,pid=%d,tgid=%d,device=%d,"
                "gpu_uuid=%s,allocation_id=%s,wall_time_ns=%" PRIu64 ",monotonic_ns=%" PRIu64
                ",base_va=0x%" PRIxPTR ",size_bytes=%zu\n",
                config.api.c_str(),
                static_cast<int>(getpid()),
                static_cast<int>(getpid()),
                config.device,
                format_uuid(device_properties.uuid).c_str(),
                allocation_id,
                wall_time_ns(),
                monotonic_time_ns(),
                reinterpret_cast<uintptr_t>(allocation.pointer),
                allocation.size_bytes);
    print_samples(allocation_id, allocation.pointer, allocation.size_bytes);

    emit_event("ACCESS_BEGIN");
    cuda_check(cudaMemset(allocation.pointer, 0x5A, allocation.size_bytes), "cudaMemset");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after memset)");
    const bool xor_closeout_passed = !config.xor_closeout || execute_xor_closeout(allocation);
    emit_event("ACCESS_END");
    hold_allocation(allocation_id, config.hold_seconds);
    emit_event("FREE_BEGIN");
    release_allocation(config.api, allocation);
    emit_event("FREE_END");
    return xor_closeout_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
